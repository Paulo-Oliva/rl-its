"""
Session generation for the cross-baseline evaluation.

    uv run eval-sessions --label T1 --model Qwen/Qwen2.5-Math-7B-Instruct --mode plain
    uv run eval-sessions --label T3 --model <ckpt> --mode actions --base-model <t2>

The mode has to match how the checkpoint was trained. `plain` is free-form turns for T1 and
T2, `actions` is the "[action] message" turns of T3, and `tools` adds the three frozen tools
for T4.

Run `uv run eval-sessions --help` for the options.
"""

import argparse
import json
import logging
import random
from pathlib import Path

from its._jsonparse import lenient_json
from its.config import TUTOR_ACTIONS
from its.evaluation.common import (GreedyLLM, append_jsonl, greedy_action, load_chat_model, read_jsonl, student_view)
from its.prompts import STUDENT_SYSTEM, TUTOR_SYSTEM, VERIFIER_SYSTEM
from its.protocol import (TOOL_CALL_STOP, TOOL_RESULT_ROLE, TOOL_TURN_NOTE, action_instructions, action_stop_strings,
                          count_tutor_turns, detect_leakage, detect_solve, ensure_closed, parse_tool_call,
                          strip_extra_tags, strip_leading_tag, strip_tool_calls, verify_answer)

log = logging.getLogger(__name__)

DEFAULT_BANK = Path("data/preprocessed/problems.jsonl")
DEFAULT_EVAL_SET = Path("data/eval/problems_eval.jsonl")
DEFAULT_OUT_DIR = Path("data/eval/sessions")
DEFAULT_KB = Path("data/preprocessed/math_kb.jsonl")
DEFAULT_MODELS_DIR = Path("data/models")


def discover_training_logs(models_dir: Path = DEFAULT_MODELS_DIR) -> list[Path]:
    return sorted(p for p in models_dir.rglob("sessions.jsonl") if "eval" not in p.parts)


def build_eval_set(bank: Path, exclude_logs: list[Path], n_problems: int, seed: int, out: Path) -> list[dict]:

    bank_texts: set[str] = set()
    for line in bank.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            bank_texts.add(json.loads(line)["problem"])

    seen: set[str] = set()
    for logf in exclude_logs:
        if not logf.exists():
            log.warning("Exclusion log %s not found - skipping", logf)
            continue
        n0 = len(seen)
        with open(logf, encoding="utf-8") as f:
            for line in f:
                try:
                    p = json.loads(line).get("problem")
                except json.JSONDecodeError:
                    continue
                if p:
                    seen.add(p)
        log.info("Scanned %s: +%d distinct trained problems", logf, len(seen) - n0)

    unmatched = seen - bank_texts
    if unmatched:
        log.warning(
            "%d logged training problems do NOT exact-match any bank entry - "
            "text-based exclusion may be unreliable (bank rebuilt?). Example: %.120s", len(unmatched),
            next(iter(unmatched)))
    trained_in_bank = seen & bank_texts
    log.info("Trained-on problems: %d total in logs, %d matched in bank; "
             "held-out pool = %d of %d bank problems", len(seen), len(trained_in_bank),
             len(bank_texts) - len(trained_in_bank), len(bank_texts))

    by_diff: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
    added: set[str] = set()
    for line in bank.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["problem"] in seen or r["problem"] in added:
            continue
        added.add(r["problem"])
        by_diff.setdefault(r.get("difficulty", "medium"), []).append({
            "id": r.get("id"),
            "problem": r["problem"],
            "answer": str(r["answer"]),
            "difficulty": r.get("difficulty", "medium")
        })

    rng = random.Random(seed)
    rows: list[dict] = []

    if n_problems == -1:
        for diff in sorted(by_diff):
            pool = by_diff[diff]
            rng.shuffle(pool)
            rows.extend(pool)
    else:
        per = -(-n_problems // len(by_diff))
        for diff in sorted(by_diff):
            pool = by_diff[diff]
            rng.shuffle(pool)
            rows.extend(pool[:per])
        rows = rows[:n_problems]

    leaked = [r for r in rows if r["problem"] in seen]
    assert not leaked, f"{len(leaked)} trained problems leaked into the eval set"

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Built eval set: %d problems (%s), disjoint from all training logs → %s", len(rows),
             ", ".join(f"{d}={sum(1 for r in rows if r['difficulty'] == d)}" for d in sorted(by_diff)), out)
    return rows


# --- Frozen tools (eval replicas of its.training.tools) -------------------------

_LEVEL_TO_DIFFICULTY = {"weak": "easy", "medium": "medium", "strong": "hard"}


# The three tools as the evaluation serves them, with the schemas and phrasing the tutor was trained against
class EvalTools:
    def __init__(self, verifier_llm, bank: list[dict], kb: dict[str, str], seed: int = 0):
        self.verifier_llm = verifier_llm
        self.bank = bank
        self.kb = kb
        self.rng = random.Random(seed)

    @property
    def schemas(self) -> list[dict]:

        def fn(name, desc, params):
            return {"type": "function", "function": {"name": name, "description": desc, "parameters": params}}

        return [
            fn(
                "call_verifier", "Check whether the student's mathematical reasoning/steps are valid.", {
                    "type": "object",
                    "properties": {
                        "work": {
                            "type": "string",
                            "description": "the student's worked steps to check"
                        }
                    },
                    "required": ["work"]
                }),
            fn(
                "call_planner", "Fetch a practice problem from the problem bank, matched to the student.", {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "optional topic to focus on"
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": ["easy", "medium", "hard"]
                        }
                    },
                    "required": []
                }),
            fn(
                "call_retriever", "Look up reference material about a math concept before explaining it.", {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "the concept to look up"
                        }
                    },
                    "required": ["topic"]
                }),
        ]

    # Runs one tool call and returns (result text, ok)
    def run(self, name: str, args: dict, problem: str, answer: str, transcript: list[dict],
            student_level: str) -> tuple[str, bool]:
        if name == "call_verifier":
            return self._verifier(args, problem, answer, transcript)
        if name == "call_planner":
            return self._planner(args, student_level)
        if name == "call_retriever":
            return self._retriever(args)
        return f"error: unknown tool {name!r}", False

    def _verifier(self, args, problem, answer, transcript) -> tuple[str, bool]:
        work = str(args.get("work", "")).strip()
        # Falls back to the student's latest message, as training does
        if not work:
            work = next((str(t.get("content", "")).strip() for t in reversed(transcript) if t.get("role") == "student"),
                        "")
        if not work:
            return 'error: provide the student\'s work to check in {"work": "..."}', False
        user = (f"[Problem] {problem}\n"
                f"[Correct answer - PRIVATE, do not reveal] {answer}\n\n"
                f"[Student's work]\n{work}")
        try:
            reply = self.verifier_llm([{
                "role": "system",
                "content": VERIFIER_SYSTEM
            }, {
                "role": "user",
                "content": user
            }])
            import re
            m = re.search(r"\{.*\}", reply, re.DOTALL)
            obj = lenient_json(m.group(0)) if m else None
            if obj is None:
                raise ValueError("no parseable JSON in verifier reply")
            valid = bool(int(obj.get("valid", 0)))
            feedback = str(obj.get("feedback", "")).strip()
        except Exception as e:
            log.warning("Eval verifier failed (%s)", e)
            return "error: could not verify the reasoning", False
        verdict = "valid" if valid else "flawed"
        return f"The student's reasoning appears {verdict}. {feedback}".strip(), True

    def _planner(self, args, student_level) -> tuple[str, bool]:
        diff = str(args.get("difficulty", "")).lower().strip()
        if diff not in ("easy", "medium", "hard"):
            diff = _LEVEL_TO_DIFFICULTY.get(student_level, "medium")
        candidates = [p for p in self.bank if p.get("difficulty") == diff] or self.bank
        if not candidates:
            return "error: no problems available", False
        p = self.rng.choice(candidates)
        return f"Practice problem ({diff}): {p['problem']}", True

    def _retriever(self, args) -> tuple[str, bool]:
        topic = str(args.get("topic", "")).strip().lower()
        if not topic:
            return 'error: provide the concept in {"topic": "..."}', False
        if topic in self.kb:
            return f"Reference on {topic}: {self.kb[topic]}", True
        for k, v in self.kb.items():
            if topic in k or k in topic:
                return f"Reference on {k}: {v}", True
        qwords = set(topic.split())
        best, best_k = 0, None
        for k in self.kb:
            overlap = len(qwords & set(k.split()))
            if overlap > best:
                best, best_k = overlap, k
        if best_k is not None:
            return f"Reference on {best_k}: {self.kb[best_k]}", True
        return f"error: no reference material found for {topic!r}", False


# Assembles the three tools from the bank and KB files
def build_eval_tools(verifier_llm, bank_path: Path, kb_path: Path, seed: int = 0) -> EvalTools:
    kb: dict[str, str] = {}
    if kb_path.exists():
        for r in read_jsonl(kb_path):
            kb[str(r["concept"]).lower()] = r["snippet"]
    else:
        log.warning("KB %s not found - retriever will miss on every topic", kb_path)
    bank = read_jsonl(bank_path) if bank_path.exists() else []
    return EvalTools(verifier_llm=verifier_llm, bank=bank, kb=kb, seed=seed)


# --- Session loops (greedy; one per mode) ---------------------------------------


def _context_ok(tutor: GreedyLLM, msgs: list[dict], tools: list[dict] | None, headroom: int) -> bool:
    return len(tutor.template_ids(msgs, tools=tools)) + headroom <= tutor.context_limit


# The tutor system prompt for one mode, where `base` replaces the Socratic preamble
def tutor_system_prompt(answer: str, actions, schemas, base: str | None = None) -> str:
    sys_prompt = (base or TUTOR_SYSTEM)
    if answer:
        sys_prompt += (f"\n\nThe correct answer is: {answer}\n"
                       "Use this to guide the student, but never reveal it directly.")
    if actions:
        sys_prompt += action_instructions(actions)
    if schemas:
        sys_prompt += TOOL_TURN_NOTE
    return sys_prompt


# The scaffold that opens a native call to tool `name`, a frozen copy of the training one
def _force_prefix(name: str) -> str:
    return f' <tool_call>\n{{"name": "{name}", "arguments": {{'


def generate_tutor_turn(
    tutor: GreedyLLM,
    tutor_msgs: list[dict],
    transcript: list[dict],
    actions,
    schemas,
    tools: EvalTools | None,
    problem: str,
    answer: str,
    student_level: str,
    max_tool_calls: int,
    constrain: bool = True,
    force_tool: str | None = None,
) -> str:
    # The plain mode of T1 and T2
    if actions is None:
        visible = tutor(tutor_msgs)
        tutor_msgs.append({"role": "assistant", "content": visible})
        transcript.append({"role": "tutor", "content": visible})
        return visible

    def segment(force: str | None = None, fixed_tag: str | None = None) -> tuple[str | None, str]:
        stop = action_stop_strings(actions) + ([TOOL_CALL_STOP] if schemas else [])
        prefix = _force_prefix(force) if (force and schemas) else ""
        force_ids = tutor.tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
        if constrain:
            if fixed_tag:
                action = fixed_tag
                prompt_ids = tutor.template_ids(tutor_msgs, tools=schemas)
                tag_ids = tutor.tokenizer.encode(f"[{fixed_tag}]", add_special_tokens=False)
            else:
                action, tag_ids, prompt_ids = greedy_action(tutor, tutor_msgs, actions, tools=schemas)
            raw = prefix + tutor.generate_from_ids(prompt_ids + tag_ids + force_ids, stop=stop)
            return action, f"[{action}] {strip_extra_tags(raw, actions)}"
        # In free mode with a forced tool, the scaffold is seeded right after the prompt
        if prefix:
            raw = prefix + tutor.generate_from_ids(tutor.template_ids(tutor_msgs, tools=schemas) + force_ids,
                                                   stop=[TOOL_CALL_STOP])
            return None, raw
        raw = tutor(tutor_msgs, tools=schemas, stop=[TOOL_CALL_STOP] if schemas else None)
        # Unconstrained, so there is no forced tag and whatever the model emitted stands
        return None, raw

    # After a planner or retriever result, pin that tool's signature action instead of re-sampling
    signature = {"call_planner": "present_problem", "call_retriever": "explain"}

    action, full = segment(force=force_tool)
    n_calls = 0
    while schemas and tools is not None and n_calls < max_tool_calls:
        call = parse_tool_call(full)
        if call is None:
            break
        name, args = call
        full = ensure_closed(full)
        if name == "call_invalid" or args is None:
            result, ok = "error: tool-call arguments were not valid JSON", False
        else:
            result, ok = tools.run(name, args, problem, answer, transcript, student_level)
        tutor_msgs.append({"role": "assistant", "content": full})
        tutor_msgs.append({"role": TOOL_RESULT_ROLE, "content": result})
        transcript.append({"role": "tutor", "action": name, "content": full})
        transcript.append({"role": "tool", "name": name, "content": result, "ok": ok})
        n_calls += 1
        # Continue the turn; pin the tool's signature tag when this menu offers it
        sig = signature.get(name)
        action, full = segment(fixed_tag=sig if (constrain and sig in actions) else None)

    # This comes out the same on both paths
    visible = strip_tool_calls(strip_leading_tag(full))
    tutor_msgs.append({"role": "assistant", "content": full})
    transcript.append({"role": "tutor", "action": action, "content": full})
    return visible


def run_eval_session(
    problem: str,
    answer: str,
    tutor: GreedyLLM,
    student: GreedyLLM,
    student_level: str,
    mode: str,
    max_turns: int,
    tools: EvalTools | None = None,
    max_tool_calls: int = 2,
) -> list[dict]:
    actions = TUTOR_ACTIONS if mode in ("actions", "tools") else None
    schemas = tools.schemas if (mode == "tools" and tools) else None

    tutor_msgs: list[dict] = [{"role": "system", "content": tutor_system_prompt(answer, actions, schemas)}]
    student_msgs: list[dict] = [
        {
            "role": "system",
            "content": STUDENT_SYSTEM[student_level]
        },
        {
            "role": "user",
            "content": f"Problem: {problem}"
        },
    ]
    transcript: list[dict] = []

    student_reply = student(student_msgs)
    student_msgs.append({"role": "assistant", "content": student_reply})
    transcript.append({"role": "student", "content": student_reply})
    tutor_msgs.append({"role": "user", "content": f"Problem: {problem}\n\nStudent: {student_reply}"})

    headroom = tutor.max_new_tokens * (max_tool_calls + 1 if schemas else 1) + 512

    for _ in range(max_turns):
        if not _context_ok(tutor, tutor_msgs, schemas, headroom):
            break

        visible = generate_tutor_turn(tutor, tutor_msgs, transcript, actions, schemas, tools, problem, answer,
                                      student_level, max_tool_calls)

        student_msgs.append({"role": "user", "content": visible})
        student_reply = student(student_msgs)
        student_msgs.append({"role": "assistant", "content": student_reply})
        transcript.append({"role": "student", "content": student_reply})
        tutor_msgs.append({"role": "user", "content": student_reply})

        if verify_answer(student_reply, answer):
            break

    return transcript


# --- Entry point -----------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    p = argparse.ArgumentParser(description="Generate evaluation sessions for one tutor baseline")
    p.add_argument("--label", required=True, help="Baseline name (T1/T2/T3/T4) - output file + report key")
    p.add_argument("--model", required=True, help="Tutor checkpoint: HF ID, full model dir, or LoRA adapter dir")
    p.add_argument("--mode",
                   required=True,
                   choices=["plain", "actions", "tools"],
                   help="Turn format the checkpoint was trained with (T1/T2=plain, T3=actions, T4=tools)")
    p.add_argument("--base-model",
                   default=None,
                   help="Base model for a LoRA adapter (default: adapter_config.json; "
                   "use data/models/runpod to avoid re-downloading T2)")
    p.add_argument("--student-model",
                   default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="FIXED student simulator (keep identical across baselines; "
                   "for the final eval prefer a held-out, stronger model)")
    p.add_argument("--eval-set",
                   type=Path,
                   default=DEFAULT_EVAL_SET,
                   help="Held-out problem file; built on first run, reused afterwards")
    p.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    p.add_argument("--exclude-log",
                   type=Path,
                   action="append",
                   default=None,
                   help="RL training sessions.jsonl whose problems are excluded (repeatable; "
                   "default: auto-discover every data/models/**/sessions.jsonl)")
    p.add_argument("--n-problems", type=int, default=60, help="Eval set size (stratified over difficulty)")
    p.add_argument("--levels", default="weak,medium,strong", help="Comma-separated student levels")
    p.add_argument("--seed", type=int, default=42, help="Problem-sampling seed (eval set build only)")
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-tool-calls", type=int, default=2, help="Per-turn tool budget (tools mode)")
    p.add_argument("--kb", type=Path, default=DEFAULT_KB, help="Retriever KB JSONL (tools mode)")
    p.add_argument("--device", default="auto")
    p.add_argument("--student-device", default="auto")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--resume", action="store_true", help="Skip session_ids already in the output file")
    p.add_argument("--max-sessions", type=int, default=-1, help="Cap total sessions (smoke tests)")
    p.add_argument("--shard",
                   default=None,
                   metavar="I/N",
                   help="Run only the I-th of N interleaved slices of the session list "
                   "(0-based, e.g. 0/4 … 3/4) and write to <label>.shardI.jsonl. "
                   "Launch N processes in parallel on one big GPU - each loads its own "
                   "model copies (~30GB per tutor+student pair in fp16), sessions are "
                   "disjoint, and eval-judge/eval-report merge the shard files by label.")
    # Greedy by default, with an ngram guard so a collapsed policy cannot loop a turn verbatim
    p.add_argument("--temperature",
                   type=float,
                   default=0.0,
                   help="Tutor sampling temperature (0 = greedy/deterministic, the default)")
    p.add_argument("--top-p", type=float, default=0.9, help="Nucleus cutoff (used only if temperature > 0)")
    p.add_argument("--repetition-penalty",
                   type=float,
                   default=1.05,
                   help="Mild anti-repetition on the tutor (1.0 = off)")
    p.add_argument("--no-repeat-ngram",
                   type=int,
                   default=4,
                   help="Block verbatim n-gram repeats in tutor turns (0 = off)")
    args = p.parse_args()

    levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    bad = [l for l in levels if l not in STUDENT_SYSTEM]
    if bad:
        raise SystemExit(f"Unknown student level(s): {bad}")

    if args.eval_set.exists():
        problems = read_jsonl(args.eval_set)
        log.info("Reusing existing eval set: %d problems from %s", len(problems), args.eval_set)
    else:
        exclude = args.exclude_log if args.exclude_log else discover_training_logs()
        if not exclude:
            raise SystemExit(f"{args.eval_set} not found and NO training logs exist on this machine to "
                             "exclude trained problems from a fresh build. Copy the blessed eval set here "
                             f"(scp data/eval/problems_eval.jsonl <this-machine>:{args.eval_set}) or build "
                             "one where the logs live: uv run eval-build-set")
        log.warning(
            "%s not found - building a NEW eval set (n=%d). If other baselines used a "
            "different set, their sessions are NOT comparable with these!", args.eval_set, args.n_problems)
        log.info("Excluding problems from %d training log(s): %s", len(exclude), [str(p) for p in exclude])
        problems = build_eval_set(args.bank, exclude, args.n_problems, args.seed, args.eval_set)

    shard_i, shard_n = 0, 1
    if args.shard:
        try:
            shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        except ValueError:
            raise SystemExit(f"--shard must look like 0/4 (got {args.shard!r})")
        if not (0 <= shard_i < shard_n):
            raise SystemExit(f"--shard index out of range: {args.shard}")

    out_file = args.out_dir / (f"{args.label}.shard{shard_i}.jsonl" if args.shard else f"{args.label}.jsonl")
    done: set[str] = set()
    if args.resume:
        done = {r["session_id"] for r in read_jsonl(out_file)}
        log.info("Resuming: %d sessions already in %s", len(done), out_file)
    elif out_file.exists():
        raise SystemExit(f"{out_file} exists - pass --resume to continue or delete it to restart")

    log.info("Loading student simulator %s ...", args.student_model)
    student = load_chat_model(args.student_model, device=args.student_device, max_new_tokens=args.max_new_tokens)
    tutor = load_chat_model(args.model,
                            base_model=args.base_model,
                            device=args.device,
                            max_new_tokens=args.max_new_tokens)
    # Anti-loop decoding on the tutor only, with the student greedy as in training
    tutor.do_sample = args.temperature > 0
    tutor.temperature = args.temperature if args.temperature > 0 else 1.0
    tutor.top_p = args.top_p
    tutor.repetition_penalty = args.repetition_penalty
    tutor.no_repeat_ngram_size = args.no_repeat_ngram
    decoding = {
        "temperature": args.temperature,
        "top_p": args.top_p if args.temperature > 0 else None,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram": args.no_repeat_ngram
    }

    tools = None
    if args.mode == "tools":
        tools = build_eval_tools(student, args.bank, args.kb, seed=args.seed)

    todo = [(prob, lvl) for prob in problems for lvl in levels]
    # An interleaved slice of the work, which is all of it when sharding is off
    todo = todo[shard_i::shard_n]
    todo = [t for t in todo if f"p{t[0]['id']}-{t[1]}" not in done]
    if args.max_sessions != -1:
        todo = todo[:args.max_sessions]
    log.info("Running %d sessions (%d problems × %s%s) for %s [%s] ...", len(todo), len(problems), levels,
             f", shard {shard_i}/{shard_n}" if args.shard else "", args.label, args.mode)

    n_solve = n_leak = 0
    for i, (prob, lvl) in enumerate(todo, 1):
        transcript = run_eval_session(prob["problem"],
                                      prob["answer"],
                                      tutor,
                                      student,
                                      lvl,
                                      args.mode,
                                      args.max_turns,
                                      tools=tools,
                                      max_tool_calls=args.max_tool_calls)
        view = student_view(transcript)
        solve = detect_solve(view, prob["answer"])
        leak = detect_leakage(view, prob["answer"])
        n_solve += solve
        n_leak += leak
        append_jsonl(
            out_file, {
                "session_id": f"p{prob['id']}-{lvl}",
                "label": args.label,
                "problem_id": prob["id"],
                "problem": prob["problem"],
                "answer": prob["answer"],
                "difficulty": prob["difficulty"],
                "student_level": lvl,
                "mode": args.mode,
                "model": str(args.model),
                "student_model": args.student_model,
                "decoding": decoding,
                "transcript": transcript,
                "solve": solve,
                "leakage": leak,
                "tutor_turns": count_tutor_turns(view),
            })
        log.info("[%d/%d] p%s-%s  solve=%d leak=%d turns=%d  (running: solve %.2f, leak %.2f)", i, len(todo),
                 prob["id"], lvl, solve, leak, count_tutor_turns(view), n_solve / i, n_leak / i)

    log.info("Done: %d sessions → %s", len(todo), out_file)


if __name__ == "__main__":
    main()
