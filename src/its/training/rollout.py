"""
The multi-turn session rollout GRPO trains on.

A completion here is a whole tutoring session, where the tutor policy and the frozen student
simulator alternate until the student solves or the turn cap is reached. The finished session
is turned into what TRL's rollout_func contract expects, which is prompt_ids, completion_ids,
per-token logprobs and an env_mask marking the tutor's tokens so only those get a gradient.

With `actions` every tutor turn becomes "[action] message". The tag is chosen by constrained
decoding, stays in the assistant turn we train on, and is stripped from the student's view.
"""

import logging
import random
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from its.config import MAX_TURNS, STUDENT_LEVEL_WEIGHTS
from its.llm import ChatModel, VLLMChatModel
from its.llm import apply_template as _apply_template
from its.prompts import STUDENT_SYSTEM, TUTOR_SYSTEM
from its.protocol import (TOOL_CALL_STOP, action_instructions, action_stop_strings, strip_extra_tags,
                          strip_tool_calls, verify_answer)
from its.training.settings import DecodeSettings

log = logging.getLogger(__name__)


@dataclass
class SessionTokens:
    prompt_ids: list[int]
    completion_ids: list[int]
    env_mask: list[int]
    turn_spans: list[tuple[int, int]]


def build_session_tokens(tokenizer, tutor_msgs: list[dict], tools: list[dict] | None = None) -> SessionTokens:
    """
    Splits a finished tutor conversation into prompt and completion, and builds the env_mask
    over the completion. The prompt runs up to the first assistant turn's generation prompt,
    and the completion is everything after it.

    Args:
        tokenizer: The policy tokenizer, which applies the chat template.
        tutor_msgs (list[dict]): The finished session as the tutor's own message list, so the
            tutor is the assistant and the student is the user.
        tools (list[dict] | None, optional): Tool schemas to template in, so the ids line up
            with what generation actually conditioned on. Defaults to None.

    Raises:
        ValueError: If the session has no tutor turn, which would leave nothing to train on.

    Returns:
        SessionTokens: The prompt and completion ids, the env_mask marking which completion
            tokens are the tutor's, and one (start, end) span per tutor turn.
    """
    # The first tutor turn is the split point, since everything before it is context the policy
    # never generated
    first_asst = next((i for i, m in enumerate(tutor_msgs) if m["role"] == "assistant"), None)
    if first_asst is None:
        raise ValueError("Tutor conversation has no assistant (tutor) turns.")

    full_ids = _apply_template(tokenizer, tutor_msgs, add_generation_prompt=False, tools=tools)
    prompt_ids = _apply_template(tokenizer, tutor_msgs[:first_asst], add_generation_prompt=True, tools=tools)
    boundary = len(prompt_ids)
    # Each tutor turn gets templated twice, once with the turns before it and once including it
    full_mask = [0] * len(full_ids)
    turn_spans: list[tuple[int, int]] = []
    for i, msg in enumerate(tutor_msgs):
        if msg["role"] != "assistant":
            continue
        prefix_ids = _apply_template(tokenizer, tutor_msgs[:i], add_generation_prompt=True, tools=tools)
        incl_ids = _apply_template(tokenizer, tutor_msgs[:i + 1], add_generation_prompt=False, tools=tools)
        for j in range(len(prefix_ids), len(incl_ids)):
            full_mask[j] = 1
        turn_spans.append((len(prefix_ids) - boundary, len(incl_ids) - boundary))

    return SessionTokens(
        prompt_ids=full_ids[:boundary],
        completion_ids=full_ids[boundary:],
        env_mask=full_mask[boundary:],
        turn_spans=turn_spans,
    )


@torch.no_grad()
def sample_action(llm: ChatModel,
                  msgs: list[dict],
                  actions: Mapping[str, str],
                  tools: list[dict] | None = None) -> tuple[str, list[int], list[int]]:
    """
    Picks the next pedagogical action as a K-way categorical over the tag token sequences.
    One batched forward pass scores every tag, and the draw is proportional to
    softmax(score / T), or the argmax when decoding greedily.

    Args:
        llm (ChatModel): The tutor. Scoring goes through `llm.model` directly, so it stays on
            the HF policy even when vLLM is generating the messages.
        msgs (list[dict]): The tutor's messages so far, ending on a student turn.
        actions (Mapping[str, str]): The action menu, tag name to description.
        tools (list[dict] | None, optional): Tool schemas to template in. Defaults to None.

    Returns:
        tuple[str, list[int], list[int]]: the action name, its tag token ids, and the prompt
                        token ids the tag was scored against.
    """
    tok, model = llm.tokenizer, llm.model
    names = list(actions)
    prompt_ids = _apply_template(tok, msgs, add_generation_prompt=True, tools=tools)
    # One candidate continuation per action, each being that action's tag as tokens
    cand_ids = [tok.encode(f"[{name}]", add_special_tokens=False) for name in names]

    # Every candidate is scored in one pass, as a batch of K rows that share the prompt and
    # differ only in the tag glued to the end. The tags are not all the same length, so the
    # short ones get padding that attention then ignores
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    width = max(len(c) for c in cand_ids)
    batch = [prompt_ids + c + [pad_id] * (width - len(c)) for c in cand_ids]
    attn = [[1] * (len(prompt_ids) + len(c)) + [0] * (width - len(c)) for c in cand_ids]
    input_ids = torch.tensor(batch, dtype=torch.long, device=model.device)
    attention_mask = torch.tensor(attn, dtype=torch.long, device=model.device)

    # A forward pass would return a logit for every position, and K by prompt length by vocab
    # is big enough to run out of memory on a long session. logits_to_keep asks for only the
    # last few positions, which are the ones the tag tokens sit at
    try:
        logits = model(input_ids, attention_mask=attention_mask, logits_to_keep=width + 1).logits
    except TypeError:
        # An older transformers without logits_to_keep support, so take the same tail by hand
        logits = model(input_ids, attention_mask=attention_mask).logits[:, -(width + 1):]

    # A tag's score is the logprob the model assigns to generating exactly that tag
    norm = getattr(llm, "length_norm_actions", False)
    scores = []
    for k, c in enumerate(cand_ids):
        logps = torch.log_softmax(logits[k, :len(c)].float(), dim=-1)
        tgt = torch.tensor(c, dtype=torch.long, device=model.device)
        s = logps.gather(-1, tgt.unsqueeze(-1)).sum()
        scores.append(s / len(c) if norm else s)
    scores = torch.stack(scores)

    if llm.do_sample:
        action_temp = getattr(llm, "action_temperature", None) or llm.temperature
        probs = torch.softmax(scores / action_temp, dim=-1)
        idx = int(torch.multinomial(probs, num_samples=1).item())
    else:
        idx = int(scores.argmax().item())
    return names[idx], cand_ids[idx], prompt_ids


@torch.no_grad()
def _generate_message(llm: ChatModel, prompt_ids: list[int], stop: list[str]) -> str:
    """
    Generates one tutor message, stopping at any of `stop`.

    Args:
        llm (ChatModel): The tutor. A VLLMChatModel goes through the engine, anything else
            through HF generate.
        prompt_ids (list[int]): The templated prompt, already ending in the chosen action tag.
        stop (list[str]): Strings that cut the generation short, which is how a turn is held to
            a single action.

    Returns:
        str: The decoded message, with the prompt dropped.
    """
    if isinstance(llm, VLLMChatModel):
        llm.stop = stop
        out_ids = llm.generate_ids(prompt_ids)
        return llm.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=llm.model.device)
    gen_kwargs: dict = {"max_new_tokens": llm.max_new_tokens, "do_sample": llm.do_sample}
    # Greedy decoding has to clear the checkpoint's own sampling defaults, same as in llm.py
    if llm.do_sample:
        gen_kwargs.update(temperature=llm.temperature, top_p=llm.top_p)
    else:
        gen_kwargs.update(temperature=None, top_p=None, top_k=None)
    gen_kwargs.update(stop_strings=stop, tokenizer=llm.tokenizer)
    output_ids = llm.model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **gen_kwargs)
    return llm.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def generate_action_turn(
    llm: ChatModel,
    msgs: list[dict],
    actions: Mapping[str, str],
    tools: list[dict] | None = None,
    tool_stop: bool = False,
    force_prefix: str | None = None,
) -> tuple[str, str, str]:
    """
    Generates one whole tutor turn, which is an action tag and then the message that goes with
    it. The tag is drawn first and the message is generated conditioned on it.

    Args:
        llm (ChatModel): The tutor.
        msgs (list[dict]): The tutor's messages so far.
        actions (Mapping[str, str]): The action menu.
        tools (list[dict] | None, optional): Tool schemas to template in. Defaults to None.
        tool_stop (bool, optional): Stop at `</tool_call>` too, so a tool call ends the turn and
            its result can be fed back before the tutor carries on. Defaults to False.
        force_prefix (str | None, optional): Text the message has to open with, which is how a
            tool call gets forced past the cold start. Defaults to None.

    Returns:
        tuple[str, str, str]: the action name, the message as the student sees it with tool
                        calls stripped, and the tagged text that gets trained on.
    """
    action, tag_ids, prompt_ids = sample_action(llm, msgs, actions, tools=tools)

    stop = action_stop_strings(actions) + ([TOOL_CALL_STOP] if tool_stop else [])
    if force_prefix:
        force_ids = llm.tokenizer.encode(force_prefix, add_special_tokens=False)
        raw = force_prefix + _generate_message(llm, prompt_ids + tag_ids + force_ids, stop)
    else:
        raw = _generate_message(llm, prompt_ids + tag_ids, stop)
    message = strip_extra_tags(raw, actions)
    return action, strip_tool_calls(message), f"[{action}] {message}"


# --- Interactive session rollout -------------------------------------------------


@dataclass
class Session:
    # The tutor's own view of the conversation, which is what gets tokenized
    tutor_msgs: list[dict]
    # The session as {"role": "tutor" or "student", "content": ...}, for the reward and the logs
    transcript: list[dict]
    problem: str
    answer: str


# Qwen2.5-Math has a 4096 token window
def _tutor_ctx_budget(tutor) -> int:
    cfg = getattr(getattr(tutor, "model", None), "config", None)
    limit = getattr(cfg, "max_position_embeddings", None) or 4096
    return limit - getattr(tutor, "max_new_tokens", 256) - 16


# True if another turn cannot fit the context
def _ctx_overflow(tutor, tutor_msgs: list[dict], budget: int) -> bool:
    tok = getattr(tutor, "tokenizer", None)
    if tok is None or len(tutor_msgs) <= 2:
        return False
    return len(_apply_template(tok, tutor_msgs, True)) > budget


def run_tutor_session(
    problem: str,
    answer: str,
    tutor: ChatModel,
    student: ChatModel,
    student_level: str = "medium",
    max_turns: int = MAX_TURNS,
    actions: Mapping[str, str] | None = None,
) -> Session:
    """
    Runs one tutor and student-simulator GRPO session to the end.

    The two models never share a message list. Each sees itself as the assistant and the other
    as the user, and the tutor is told the answer while the student never is.

    Args:
        problem (str): The problem text.
        answer (str): The gold answer, injected into the tutor's system prompt and used to check
            whether the student got there.
        tutor (ChatModel): The policy being trained.
        student (ChatModel): The frozen student simulator.
        student_level (str, optional): Which ability prompt the student gets, one of weak,
            medium or strong. Defaults to "medium".
        max_turns (int, optional): Cap on tutor turns, since a session that never solves has to
            stop somewhere. Defaults to MAX_TURNS.
        actions (Mapping[str, str] | None, optional): The action menu, which turns on the tagged
            turns. Defaults to None, which is the free-form ablation.

    Returns:
        Session: The tutor's message list for tokenizing, plus the transcript both roles saw.
    """
    tutor_system = (
        TUTOR_SYSTEM +
        f"\n\nThe correct answer is: {answer}\nUse this to guide the student, but never reveal it directly.")
    if actions:
        tutor_system += action_instructions(actions)
    tutor_msgs: list[dict] = [{"role": "system", "content": tutor_system}]
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

    # The student answers the problem cold, before the tutor has said anything
    student_reply = student(student_msgs)
    student_msgs.append({"role": "assistant", "content": student_reply})
    transcript.append({"role": "student", "content": student_reply})
    tutor_msgs.append({"role": "user", "content": f"Problem: {problem}\n\nStudent: {student_reply}"})
    # Generation loop
    ctx_budget = _tutor_ctx_budget(tutor)
    for _ in range(max_turns):
        if _ctx_overflow(tutor, tutor_msgs, ctx_budget):
            break
        if actions:
            action, student_visible, tutor_reply = generate_action_turn(tutor, tutor_msgs, actions)
            transcript.append({"role": "tutor", "action": action, "content": tutor_reply})
        else:
            tutor_reply = tutor(tutor_msgs)
            student_visible = tutor_reply
            transcript.append({"role": "tutor", "content": tutor_reply})
        tutor_msgs.append({"role": "assistant", "content": tutor_reply})

        student_msgs.append({"role": "user", "content": student_visible})
        student_reply = student(student_msgs)
        student_msgs.append({"role": "assistant", "content": student_reply})
        transcript.append({"role": "student", "content": student_reply})
        tutor_msgs.append({"role": "user", "content": student_reply})
        # Solved, so the session ends here
        if verify_answer(student_reply, answer):
            break

    return Session(tutor_msgs=tutor_msgs, transcript=transcript, problem=problem, answer=answer)


# Batched generation for running multiple sessions in a group at the same time
@torch.no_grad()
def _hf_generate_batch(llm: ChatModel, prompts_ids: list[list[int]], stop: list[str] | None = None) -> list[str]:
    tok, model = llm.tokenizer, llm.model
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    maxlen = max(len(p) for p in prompts_ids)
    # The prompts are all different lengths, so they get padded up to the longest one
    input_ids, attn = [], []
    for p in prompts_ids:
        pad = maxlen - len(p)
        # Padding goes on the left, so every row's new tokens start at the same index
        input_ids.append([pad_id] * pad + list(p))
        attn.append([0] * pad + [1] * len(p))

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device)
    attn = torch.tensor(attn, dtype=torch.long, device=model.device)
    gen_kwargs: dict = {"max_new_tokens": llm.max_new_tokens, "do_sample": llm.do_sample}

    if llm.do_sample:
        gen_kwargs.update(temperature=llm.temperature, top_p=llm.top_p)
    else:
        gen_kwargs.update(temperature=None, top_p=None, top_k=None)

    if stop:
        gen_kwargs.update(stop_strings=stop, tokenizer=tok)

    out = model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
    return [tok.decode(row[maxlen:], skip_special_tokens=True).strip() for row in out]


def _batch_tutor_messages(tutor, prompts_ids: list[list[int]], stop: list[str] | None = None) -> list[str]:
    if isinstance(tutor, VLLMChatModel):
        tutor.stop = stop
        return [
            tutor.tokenizer.decode(o, skip_special_tokens=True).strip() for o in tutor.generate_ids_batch(prompts_ids)
        ]
    return _hf_generate_batch(tutor, prompts_ids, stop=stop)


def run_tutor_group(
    problems: list[str],
    answers: list[str],
    tutor: ChatModel,
    student: ChatModel,
    levels: list[str],
    max_turns: int = MAX_TURNS,
    actions: Mapping[str, str] | None = None,
) -> list[Session]:
    """
    Runs a whole group of sessions in lockstep, which is the `--batched-rollout` path.

    `run_tutor_session` finishes one session before starting the next, so the GPU generates a
    single short reply at a time. This version advances every session by one turn instead, and
    each turn is one batched call covering all the sessions still going. A session that solves
    drops out of the batch, so the later turns only generate for the ones still running.

    Args:
        problems (list[str]): One problem per session.
        answers (list[str]): The gold answer for each, in the same order.
        tutor (ChatModel): The policy being trained.
        student (ChatModel): The frozen student simulator.
        levels (list[str]): The ability prompt for each session, drawn once per group.
        max_turns (int, optional): Cap on tutor turns. Defaults to MAX_TURNS.
        actions (Mapping[str, str] | None, optional): The action menu, which turns on the tagged
            turns. Defaults to None, which is the free-form ablation.

    Returns:
        list[Session]: One finished session per problem, in the order given. The shape matches
            what `run_tutor_session` returns, so tokenizing and masking are unchanged.
    """
    n = len(problems)
    st: list[dict] = []
    for i in range(n):
        tutor_system = (
            TUTOR_SYSTEM +
            f"\n\nThe correct answer is: {answers[i]}\nUse this to guide the student, but never reveal it directly.")
        if actions:
            tutor_system += action_instructions(actions)
        st.append({
            "tutor_msgs": [{
                "role": "system",
                "content": tutor_system
            }],
            "student_msgs": [{
                "role": "system",
                "content": STUDENT_SYSTEM[levels[i]]
            }, {
                "role": "user",
                "content": f"Problem: {problems[i]}"
            }],
            "transcript": [],
            "active":
            True,
        })

    # The cold student replies, in one batched call
    cold = _hf_generate_batch(student, [_apply_template(student.tokenizer, s["student_msgs"], True) for s in st])
    for i, s in enumerate(st):
        s["student_msgs"].append({"role": "assistant", "content": cold[i]})
        s["transcript"].append({"role": "student", "content": cold[i]})
        s["tutor_msgs"].append({"role": "user", "content": f"Problem: {problems[i]}\n\nStudent: {cold[i]}"})

    ctx_budget = _tutor_ctx_budget(tutor)
    for _ in range(max_turns):
        for s in st:
            if s["active"] and _ctx_overflow(tutor, s["tutor_msgs"], ctx_budget):
                s["active"] = False
        active = [i for i, s in enumerate(st) if s["active"]]
        if not active:
            break

        # The action is still drawn one session at a time, since only the message generation is batched
        if actions:
            acts, prompts = [], []
            for i in active:
                a, tag_ids, prompt_ids = sample_action(tutor, st[i]["tutor_msgs"], actions)
                acts.append(a)
                prompts.append(prompt_ids + tag_ids)
            messages = _batch_tutor_messages(tutor, prompts, stop=action_stop_strings(actions))
            visibles = []
            # Strip the tag for the student and append stuff to history
            for k, i in enumerate(active):
                msg = strip_extra_tags(messages[k], actions)
                full = f"[{acts[k]}] {msg}"
                visibles.append(msg)
                st[i]["transcript"].append({"role": "tutor", "action": acts[k], "content": full})
                st[i]["tutor_msgs"].append({"role": "assistant", "content": full})
        # In case of no action tags
        else:
            prompts = [_apply_template(tutor.tokenizer, st[i]["tutor_msgs"], True) for i in active]
            messages = _batch_tutor_messages(tutor, prompts)
            visibles = messages
            for k, i in enumerate(active):
                st[i]["transcript"].append({"role": "tutor", "content": messages[k]})
                st[i]["tutor_msgs"].append({"role": "assistant", "content": messages[k]})

        # The student turns, also batched
        for k, i in enumerate(active):
            st[i]["student_msgs"].append({"role": "user", "content": visibles[k]})
        replies = _hf_generate_batch(
            student,
            [_apply_template(student.tokenizer, st[i]["student_msgs"], True) for i in active],
        )

        for k, i in enumerate(active):
            r = replies[k]
            st[i]["student_msgs"].append({"role": "assistant", "content": r})
            st[i]["transcript"].append({"role": "student", "content": r})
            st[i]["tutor_msgs"].append({"role": "user", "content": r})
            if verify_answer(r, answers[i]):
                st[i]["active"] = False

    return [Session(s["tutor_msgs"], s["transcript"], problems[i], answers[i]) for i, s in enumerate(st)]


@torch.no_grad()
def completion_logprobs(model,
                        prompt_ids: list[int],
                        completion_ids: list[int],
                        temperature: float = 1.0) -> list[float]:
    """
    Scores an already-generated session, giving the logprob of each completion token.

    GRPO needs the policy's own logprob for every token it is about to take a gradient on. Those
    come from a single forward pass over the finished session rather than from generate(), so
    there is no chance of the generated ids and the re-templated ids disagreeing.

    Args:
        model: The HF policy. This never runs on the vLLM engine, which produces tokens but not
            probabilities.
        prompt_ids (list[int]): The session prompt, which is only context here.
        completion_ids (list[int]): The tokens to score.
        temperature (float, optional): Divides the logits before the softmax, matching whatever
            generation sampled at. Defaults to 1.0.

    Returns:
        list[float]: One logprob per completion token, in order.
    """
    seq = torch.tensor(prompt_ids + completion_ids, dtype=torch.long, device=model.device).unsqueeze(0)
    # Shaped (tokens, vocab)
    logits = model(seq).logits[0]
    # The logits at position t predict token t+1, so the first completion token is predicted
    # by the logits one position before it
    start = len(prompt_ids)
    sel = logits[start - 1:start - 1 + len(completion_ids)].float() / temperature
    logps = torch.log_softmax(sel, dim=-1)
    tgt = torch.tensor(completion_ids, dtype=torch.long, device=model.device)
    return logps.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).tolist()


def make_rollout_func(
    student: ChatModel,
    decode: DecodeSettings | None = None,
    student_levels: tuple[str, ...] = ("weak", "medium", "strong"),
    max_turns: int = MAX_TURNS,
    actions: Mapping[str, str] | None = None,
    use_vllm: bool = False,
    batched: bool = False,
):
    """
    Builds the `rollout_func` TRL calls to produce completions.

    TRL hands it the prompts for a step and expects prompt_ids, completion_ids and logprobs
    back. Each prompt becomes one full session, and TRL has already repeated every prompt
    `num_generations` times, so those repeats are what form the GRPO group.

    Args:
        student (ChatModel): The frozen student simulator.
        decode (DecodeSettings | None, optional): How the tutor generates. Defaults to None,
            which takes the DecodeSettings defaults.
        student_levels (tuple[str, ...], optional): The abilities to draw from, once per group so
            a group is comparable. Defaults to ("weak", "medium", "strong").
        max_turns (int, optional): Cap on tutor turns. Defaults to MAX_TURNS.
        actions (Mapping[str, str] | None, optional): The action menu. Defaults to None.
        use_vllm (bool, optional): Generate the messages through the trainer's colocated engine.
            Defaults to False.
        batched (bool, optional): Advance the group in lockstep through `run_tutor_group`
            instead of one session at a time. Defaults to False.

    Returns:
        The rollout_func itself, ready to pass to GRPOTrainer.
    """
    decode = decode if decode is not None else DecodeSettings()

    def rollout_func(prompts: list, trainer) -> dict:
        model = trainer.model
        tokenizer = trainer.processing_class
        # Sampling rather than greedy, so repeated prompts in a group give different sessions
        tutor = decode.build_tutor(model, tokenizer, vllm_generation=trainer.vllm_generation if use_vllm else None)

        prompt_ids_out: list[list[int]] = []
        completion_ids_out: list[list[int]] = []
        logprobs_out: list[list[float]] = []
        env_mask_out: list[list[int]] = []
        answers_out: list[str] = []
        problems_out: list[str] = []
        transcripts_out: list[list[dict]] = []
        levels_out: list[str] = []
        difficulties_out: list[str] = []

        # Tokenises one finished session, recomputes logprobs and records the outputs
        def emit(session: "Session", problem: str, answer: str, level: str, difficulty: str) -> None:
            st = build_session_tokens(tokenizer, session.tutor_msgs)
            logps = completion_logprobs(model, st.prompt_ids, st.completion_ids, temperature=decode.temperature)
            prompt_ids_out.append(st.prompt_ids)
            completion_ids_out.append(st.completion_ids)
            logprobs_out.append(logps)
            env_mask_out.append(st.env_mask)
            answers_out.append(answer)
            problems_out.append(problem)
            transcripts_out.append(session.transcript)
            levels_out.append(level)
            difficulties_out.append(difficulty)

        was_training = model.training
        old_cache = getattr(model.config, "use_cache", True)
        model.eval()
        model.config.use_cache = True
        n_gen = max(1, getattr(trainer, "num_generations", 1) or 1)
        # Choose a student level
        level_w = [STUDENT_LEVEL_WEIGHTS.get(l, 1.0) for l in student_levels]
        levels = []
        cur = student_levels[0]
        for i in range(len(prompts)):
            if i % n_gen == 0:
                cur = random.choices(student_levels, weights=level_w, k=1)[0]
            levels.append(cur)
        probs = [p["problem"] if isinstance(p, dict) else p for p in prompts]
        answs = [p["answer"] if isinstance(p, dict) else "" for p in prompts]
        diffs = [p.get("difficulty", "medium") if isinstance(p, dict) else "medium" for p in prompts]
        try:
            if batched:
                sessions = run_tutor_group(probs, answs, tutor, student, levels, max_turns, actions)
                for sess, prob, ans, lvl, diff in zip(sessions, probs, answs, levels, diffs):
                    emit(sess, prob, ans, lvl, diff)
            else:
                for prob, ans, lvl, diff in zip(probs, answs, levels, diffs):
                    session = run_tutor_session(prob, ans, tutor, student, lvl, max_turns, actions)
                    emit(session, prob, ans, lvl, diff)
        finally:
            model.config.use_cache = old_cache
            if was_training:
                model.train()

        return {
            "prompt_ids": prompt_ids_out,
            "completion_ids": completion_ids_out,
            "logprobs": logprobs_out,
            "env_mask": env_mask_out,
            # The extra fields below are forwarded to the reward functions
            "answer": answers_out,
            "problem": problems_out,
            "transcript": transcripts_out,
            "student_level": levels_out,
            "difficulty": difficulties_out,
        }

    return rollout_func
