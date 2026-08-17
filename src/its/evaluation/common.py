"""
The shared primitives of the cross-baseline session evaluation.

The tutor-facing protocol itself, meaning the prompts, the action tags, the tool-call format and
the rule-based checks, comes from `its.prompts` and `its.protocol`, which the training side reads
too, so a tutor is evaluated under exactly the conventions it was trained under. What lives here
is the rest: loading a checkpoint, generating from it greedily, and turning a finished session
into the student-visible view a judge reads.

Nothing here imports from `its.training`, so the evaluation still runs without the training stack.
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from its.protocol import strip_leading_tag

# torch is imported inside the two methods that need it, so `eval-report` and the API judges
# never pay for it
if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


# All four scenarios in one prompt, for the interactive chat where the student can ask for anything
# T4 was trained on the scenarios separately, so this combined prompt is out of distribution for it
UNIFIED_TUTOR_SYSTEM = """\
You are a knowledgeable, friendly math tutor. Read what the student needs each turn and \
respond in the matching mode:

- SOLVE a problem: guide the student to the answer themselves - acknowledge their attempt, ask a \
reflective question, give a minimal hint only if they are stuck, and NEVER reveal the final answer.
- CHECK their reasoning: when the student shows their work and asks you to check it, use your \
verifier tool to assess whether the reasoning is sound, then tell them what it found - affirm if \
correct, or point out where it goes wrong and guide them to fix it. Do NOT reveal the answer.
- Give a PRACTICE PROBLEM: when the student asks for a problem to work on, use your planner tool to \
fetch a suitable one from the problem bank and present it in full with the [present_problem] tag. \
Do not invent a problem yourself.
- EXPLAIN a concept: when the student asks you to explain a math concept, use your retriever tool to \
look up accurate reference material, then explain it clearly and directly (grounded in what you \
retrieved) with the [explain] tag. Here it is GOOD to explain directly - do not withhold.

Keep replies concise (2-4 sentences) and check the student is following."""


# The two model interfaces, described here rather than imported so no ML library is needed to read this
class _Model(Protocol):

    @property
    def device(self) -> "torch.device":
        ...

    def generate(self, **kwargs: Any) -> Any:
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


class _Tokenizer(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def encode(self, *args: Any, **kwargs: Any) -> list[int]:
        ...

    def decode(self, *args: Any, **kwargs: Any) -> str:
        ...


# --- Student-visible view + rule metrics ----------------------------------------
# Rule metrics are computed over what the student actually saw


# The dialogue as the student experienced it, as {"role": tutor|student, "content"} turns
def student_view(transcript: list[dict]) -> list[dict]:
    view: list[dict] = []
    for t in transcript:
        if t["role"] == "student":
            view.append({"role": "student", "content": t["content"]})
        elif t["role"] == "tutor" and "<tool_call>" not in t.get("content", ""):
            view.append({"role": "tutor", "content": strip_leading_tag(t["content"])})
    return view


# Plain TUTOR:/STUDENT: lines, which is the form a judge reads
def render_dialogue(view: list[dict]) -> str:
    return "\n".join(f"{t['role'].upper()}: {t['content']}" for t in view)


# --- Model loading + greedy generation ------------------------------------------


# Chat wrapper for the evaluation, greedy unless the interactive chat asks for sampling
@dataclass
class GreedyLLM:
    model: _Model
    tokenizer: _Tokenizer
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    # Above 1.0 this penalises tokens already in the context
    repetition_penalty: float = 1.0
    # Above 0 this blocks any n-gram of that size from repeating, which kills verbatim loops
    no_repeat_ngram_size: int = 0

    # The prompt token ids for these messages
    def template_ids(self, messages: list[dict], tools: list[dict] | None = None) -> list[int]:
        result = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, tools=tools)
        return result["input_ids"] if not isinstance(result, list) else result

    # Continues a tokenised prompt and returns the decoded reply
    def generate_from_ids(self, prompt_ids: list[int], stop: list[str] | None = None) -> str:
        import torch
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.model.device)
        kwargs: dict = {"max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample}
        # Same thing as before
        if self.do_sample:
            kwargs.update(temperature=self.temperature, top_p=self.top_p)
        else:
            kwargs.update(temperature=None, top_p=None, top_k=None)
        if self.repetition_penalty and self.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = self.repetition_penalty
        if self.no_repeat_ngram_size:
            kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        if stop:
            kwargs.update(stop_strings=stop, tokenizer=self.tokenizer)
        with torch.no_grad():
            out = self.model.generate(input_ids=input_ids,
                                      attention_mask=torch.ones_like(input_ids), **kwargs)
        return self.tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

    def __call__(self, messages: list[dict], tools: list[dict] | None = None,
                 stop: list[str] | None = None) -> str:
        return self.generate_from_ids(self.template_ids(messages, tools=tools), stop=stop)

    # The model's context window, or 4096 when the config does not say
    @property
    def context_limit(self) -> int:
        cfg = getattr(self.model, "config", None)
        return int(getattr(cfg, "max_position_embeddings", 4096) or 4096)


# The turn's action tag, every "[action]" continuation scored in one batched pass, as (name, tag ids, prompt ids)
def greedy_action(llm: GreedyLLM, msgs: list[dict], actions: Mapping[str, str],
                  tools: list[dict] | None = None) -> tuple[str, list[int], list[int]]:
    import torch
    tok, model = llm.tokenizer, llm.model
    names = list(actions)
    prompt_ids = llm.template_ids(msgs, tools=tools)
    cand_ids = [tok.encode(f"[{name}]", add_special_tokens=False) for name in names]

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    width = max(len(c) for c in cand_ids)
    batch = [prompt_ids + c + [pad_id] * (width - len(c)) for c in cand_ids]
    attn = [[1] * (len(prompt_ids) + len(c)) + [0] * (width - len(c)) for c in cand_ids]
    input_ids = torch.tensor(batch, dtype=torch.long, device=model.device)
    attention_mask = torch.tensor(attn, dtype=torch.long, device=model.device)
    with torch.no_grad():
        try:
            logits = model(input_ids, attention_mask=attention_mask, logits_to_keep=width + 1).logits
        except TypeError:
            logits = model(input_ids, attention_mask=attention_mask).logits[:, -(width + 1):]

    scores = []
    for k, c in enumerate(cand_ids):
        logps = torch.log_softmax(logits[k, : len(c)].float(), dim=-1)
        tgt = torch.tensor(c, dtype=torch.long, device=model.device)
        scores.append(logps.gather(-1, tgt.unsqueeze(-1)).sum())
    scores = torch.stack(scores)
    if getattr(llm, "do_sample", False):
        temp = max(llm.temperature, 1e-6)
        idx = int(torch.multinomial(torch.softmax(scores / temp, dim=-1), num_samples=1).item())
    else:
        idx = int(scores.argmax().item())
    return names[idx], cand_ids[idx], prompt_ids


# An existing local path becomes absolute, and anything else is left as an HF repo id
def _resolve_ref(ref: str | None) -> str | None:
    if not ref:
        return ref
    p = Path(ref)
    return str(p.resolve()) if p.exists() else ref


# The base model a LoRA adapter records, or None when `model_ref` is not an adapter
def _adapter_base(model_ref: str | None) -> str | None:
    if not model_ref:
        return None
    try:
        from peft import PeftConfig
        return PeftConfig.from_pretrained(model_ref).base_model_name_or_path
    except Exception:
        # No adapter_config.json, so this is a full model rather than an adapter
        return None


# Loads a full checkpoint or a LoRA adapter, local or from the Hub, merging the adapter into its base
def load_chat_model(model_path: str, base_model: str | None = None,
                    device: str = "auto",
                    max_new_tokens: int = 256) -> GreedyLLM:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = _resolve_ref(model_path) or model_path
    adapter_base = _adapter_base(resolved)
    if adapter_base is not None or base_model is not None:
        base = _resolve_ref(base_model) or _resolve_ref(adapter_base)
        if not base:
            raise ValueError(f"{model_path} is a LoRA adapter but no base model was found - pass --base-model")
        log.info("Loading LoRA adapter %s on base %s", resolved, base)
        model = AutoModelForCausalLM.from_pretrained(base, dtype="auto", device_map=device)
        from peft import PeftModel
        adapter: Any = PeftModel.from_pretrained(model, resolved)
        # Merging the adapter in makes inference faster
        model = adapter.merge_and_unload()
        # Prefer the adapter's tokenizer, falling back to the base if it has none
        try:
            tokenizer = AutoTokenizer.from_pretrained(resolved)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base)
    else:
        log.info("Loading model %s", resolved)
        tokenizer = AutoTokenizer.from_pretrained(resolved)
        model = AutoModelForCausalLM.from_pretrained(resolved, dtype="auto", device_map=device)
    model.eval()
    return GreedyLLM(model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens)


# --- JSONL helpers ---------------------------------------------------------------


# Reads a JSONL file into a list, treating a missing file as empty
def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# Appends one record, creating the parent directory if it is not there yet
def append_jsonl(path: Path, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
