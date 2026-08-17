"""
The T4 rollout, which is T3 plus the frozen tools the tutor can call mid-session.
"""

import logging
from collections.abc import Mapping

from its.llm import ChatModel
from its.prompts import STUDENT_SYSTEM, TUTOR_SYSTEM
from its.protocol import (ALL_TUTOR_ACTIONS, TOOL_CALL_STOP, TOOL_RESULT_ROLE, TOOL_TURN_NOTE, action_instructions,
                          action_stop_strings, ensure_closed, parse_tool_call, strip_extra_tags, strip_tool_calls,
                          verify_answer)
from its.training.rollout import (Session, _apply_template, _batch_tutor_messages, _hf_generate_batch,
                                  generate_action_turn, sample_action)
from its.training.settings import ScenarioStart, SessionSettings
from its.training.tools.base import Tool, ToolContext, ToolResult

log = logging.getLogger(__name__)


# The tutor's context window in tokens, from the HF model config
def _context_limit(tutor) -> int:
    cfg = getattr(getattr(tutor, "model", None), "config", None)
    return int(getattr(cfg, "max_position_embeddings", 4096) or 4096)


# --- Tools ---------------------------------------------------------------------


# The native function schemas for `apply_chat_template(tools=...)`, or None without any tools
def _tool_schemas(tools: list[Tool]) -> list[dict] | None:
    return [t.tool_schema for t in tools] or None


# True once tool `name` has been called
def _tool_was_called(transcript: list[dict], name: str) -> bool:
    return any(t.get("role") == "tutor" and t.get("action") == name for t in transcript)


# Forces a turn to open a native call to tool `name`, seeding the arguments brace
def _force_prefix(name: str) -> str:
    return f' <tool_call>\n{{"name": "{name}", "arguments": {{'


# --- Tool-augmented session loop ---------------------------------------------


# One tutor turn, which is the pedagogical move plus any tool calls it makes
def _run_tutor_turn(
    tutor: ChatModel,
    tutor_msgs: list[dict],
    transcript: list[dict],
    registry: Mapping[str, Tool],
    schemas: list[dict] | None,
    ped_actions: Mapping[str, str],
    max_tool_calls: int,
    problem: str,
    answer: str,
    student_level: str,
    force_tool: str | None = None,
) -> tuple[str, str]:
    can_call = bool(registry)
    force_prefix = _force_prefix(force_tool) if (force_tool and force_tool in registry) else None
    action, visible, full = generate_action_turn(tutor,
                                                 tutor_msgs,
                                                 ped_actions,
                                                 tools=schemas,
                                                 tool_stop=can_call,
                                                 force_prefix=force_prefix)

    n_tool = 0
    # Whether the first segment is the forced one
    seg_forced = force_prefix is not None
    while can_call and n_tool < max_tool_calls:
        call = parse_tool_call(full)
        # No tool call, so this segment is the student-facing move
        if call is None:
            break
        name, args = call
        full = ensure_closed(full)
        if name in registry:
            if args is None:
                result = ToolResult("error: tool-call arguments were not valid JSON", ok=False)
            else:
                result = registry[name].run(
                    args, ToolContext(problem=problem,
                                      answer=answer,
                                      transcript=transcript,
                                      student_level=student_level))
            rec = name
        else:
            result = ToolResult(f"error: unknown tool {name!r}", ok=False)
            rec = "call_invalid"
        # Plain text rather than a structured tool_calls field, so the template renders it back
        tutor_msgs.append({"role": "assistant", "content": full})
        tutor_msgs.append({"role": TOOL_RESULT_ROLE, "content": result.text})
        transcript.append({"role": "tutor", "action": rec, "content": full})
        transcript.append({
            "role": "tool",
            "name": rec,
            "content": result.text,
            "data": result.data,
            "ok": result.ok,
            "forced": seg_forced
        })
        n_tool += 1
        # Only force the first call
        seg_forced = False
        # A fresh action turn
        action, visible, full = generate_action_turn(
            tutor,
            tutor_msgs,
            ped_actions,
            tools=schemas,
            tool_stop=(n_tool < max_tool_calls),
        )

    # The student-facing move, which is the segment that made no tool call
    tutor_msgs.append({"role": "assistant", "content": full})
    transcript.append({"role": "tutor", "action": action, "content": full})
    return action, visible


def run_tutor_session_with_tools(
    problem: str,
    answer: str,
    tutor: ChatModel,
    student: ChatModel,
    tools: list[Tool],
    start: ScenarioStart | None = None,
    settings: SessionSettings | None = None,
) -> Session:
    """
    One T4 session, which is `rollout.run_tutor_session` except that each turn the tutor may
    call tools before its student-facing move.

    Args:
        problem (str): The problem text.
        answer (str): The gold answer, which the tutor only sees when the scenario allows it.
        tutor (ChatModel): The policy being trained.
        student (ChatModel): The frozen student simulator.
        tools (list[Tool]): The tools on offer, keyed by name into a registry.
        start (ScenarioStart | None, optional): How the session opens and what the tutor is
            told. Defaults to None, which is the plain solve session T3 used.
        settings (SessionSettings | None, optional): The turn budget, the action menu and the
            forcing. Defaults to None, which takes the SessionSettings defaults.

    Returns:
        Session: The tutor's message list for tokenizing, plus the transcript.
    """
    start = start if start is not None else ScenarioStart()
    settings = settings if settings is not None else SessionSettings()
    student_level = settings.student_level
    max_turns = settings.max_turns
    ped_actions = settings.ped_actions
    max_tool_calls = settings.max_tool_calls
    terminate_fn = settings.terminate_fn
    force_tool = settings.force_tool
    opening = start.opening
    solve_terminates = start.solve_terminates
    tutor_sees_answer = start.tutor_sees_answer
    student_directive = start.student_directive
    tutor_system = start.tutor_system
    show_problem = start.show_problem

    registry: dict[str, Tool] = {t.name: t for t in tools}
    # The tools are advertised natively, through the chat template
    schemas = _tool_schemas(tools)

    sys_prompt = tutor_system if tutor_system is not None else TUTOR_SYSTEM
    if tutor_sees_answer:
        sys_prompt += (f"\n\nThe correct answer is: {answer}\n"
                       "Use this to guide the student, but never reveal it directly.")
    sys_prompt += action_instructions(ped_actions)
    if schemas:
        sys_prompt += TOOL_TURN_NOTE
    tutor_msgs: list[dict] = [{"role": "system", "content": sys_prompt}]
    # The dataset problem only seeds the context for scenarios where it is the task
    student_first = f"Problem: {problem}" if show_problem else "You are chatting with your math tutor."
    if opening is None and student_directive:
        student_first += f"\n\n{student_directive}"
    student_msgs: list[dict] = [
        {
            "role": "system",
            "content": STUDENT_SYSTEM[student_level]
        },
        {
            "role": "user",
            "content": student_first
        },
    ]
    transcript: list[dict] = []

    # Headroom for a whole turn, which is every tool segment plus the final move
    ctx_limit = _context_limit(tutor)
    gen = getattr(tutor, "max_new_tokens", 256)
    # The extra 512 is roughly the tool result and the student's text
    turn_headroom = gen * (max_tool_calls + 1) + 512
    tok = getattr(tutor, "tokenizer", None)

    # The student opens, with either the scenario's message or a generated cold reply
    student_reply = opening if opening is not None else student(student_msgs)
    student_msgs.append({"role": "assistant", "content": student_reply})
    transcript.append({"role": "student", "content": student_reply})
    first_tutor_msg = (f"Problem: {problem}\n\nStudent: {student_reply}"
                       if show_problem else f"Student: {student_reply}")
    tutor_msgs.append({"role": "user", "content": first_tutor_msg})

    for _ in range(max_turns):
        # Stop before the next turn would overflow the window
        if tok is not None:
            prompt_len = len(_apply_template(tok, tutor_msgs, add_generation_prompt=True, tools=schemas))
            if prompt_len + turn_headroom > ctx_limit:
                log.debug("ending session early: prompt %d + headroom %d > ctx %d", prompt_len, turn_headroom,
                          ctx_limit)
                break
        # The forcing only lasts until the tool has been called once this session
        force = force_tool if (force_tool and not _tool_was_called(transcript, force_tool)) else None
        _action, student_visible = _run_tutor_turn(
            tutor,
            tutor_msgs,
            transcript,
            registry,
            schemas,
            ped_actions,
            max_tool_calls,
            problem,
            answer,
            student_level,
            force_tool=force,
        )

        # The student replies to the move
        student_msgs.append({"role": "user", "content": student_visible})
        student_reply = student(student_msgs)
        student_msgs.append({"role": "assistant", "content": student_reply})
        transcript.append({"role": "student", "content": student_reply})
        tutor_msgs.append({"role": "user", "content": student_reply})

        # A scenario's own predicate decides when its goal is met, otherwise the solve check
        if terminate_fn is not None:
            if terminate_fn(transcript):
                break
        elif solve_terminates and verify_answer(student_reply, answer):
            break

    return Session(tutor_msgs=tutor_msgs, transcript=transcript, problem=problem, answer=answer)


def run_scenario_group(
    scenarios: list,
    problems: list[str],
    answers: list[str],
    difficulties: list[str],
    student_levels: list[str],
    force_tools: list[str | None],
    tutor,
    student,
    tools: list[Tool],
    rng,
    settings: SessionSettings | None = None,
) -> list[tuple[Session, dict]]:
    """
    A whole batch of T4 sessions in lockstep, the tool-aware counterpart of
    `rollout.run_tutor_group`.

    Only the generations are batched, so action scoring, tool dispatch and logprobs stay per
    session and the appended messages match the sequential path. A T4 turn has no fixed length,
    since a move can be followed by tool calls and their continuations, so an inner loop
    generates for whichever sessions are still mid-turn while the rest wait.

    Args:
        scenarios (list): One Scenario per session, which sets the opening and the scoring.
        problems (list[str]): One problem per session.
        answers (list[str]): The gold answer for each, in the same order.
        difficulties (list[str]): The bank's difficulty label for each.
        student_levels (list[str]): The ability prompt for each session.
        force_tools (list[str | None]): The tool to force per session, or None to leave it free.
        tutor: The policy being trained.
        student: The frozen student simulator.
        tools (list[Tool]): The tools on offer.
        rng: Seeded generator the scenarios draw their openings from.
        settings (SessionSettings | None, optional): The turn budget and the tool budget.
            Defaults to None, which takes the SessionSettings defaults.

    Returns:
        list[tuple[Session, dict]]: One finished session per problem, in the order given, each
            with its scenario's meta dict for the reward to read.
    """
    settings = settings if settings is not None else SessionSettings()
    max_turns = settings.max_turns
    max_tool_calls = settings.max_tool_calls
    registry: dict[str, Tool] = {t.name: t for t in tools}
    schemas = _tool_schemas(tools)
    n = len(problems)

    # per-session init, mirroring run_tutor_session_with_tools
    starts = [scenarios[i].start(problems[i], answers[i], difficulties[i], rng) for i in range(n)]
    sess: list[dict] = []
    for i in range(n):
        start = starts[i]
        sys_prompt = start.tutor_system if start.tutor_system is not None else TUTOR_SYSTEM
        if start.tutor_sees_answer:
            sys_prompt += (f"\n\nThe correct answer is: {answers[i]}\n"
                           "Use this to guide the student, but never reveal it directly.")
        # Each scenario brings its own action vocabulary
        sys_prompt += action_instructions(scenarios[i].actions)
        if schemas:
            sys_prompt += TOOL_TURN_NOTE
        student_first = (f"Problem: {problems[i]}" if start.show_problem else "You are chatting with your math tutor.")
        if start.opening is None and start.student_directive:
            student_first += f"\n\n{start.student_directive}"
        sess.append({
            "tutor_msgs": [{
                "role": "system",
                "content": sys_prompt
            }],
            "student_msgs": [{
                "role": "system",
                "content": STUDENT_SYSTEM[student_levels[i]]
            }, {
                "role": "user",
                "content": student_first
            }],
            "transcript": [],
            "actions":
            scenarios[i].actions,
            "active":
            True,
        })

    # The scenario's own opening where it has one, otherwise one batched student call
    need_cold = [i for i in range(n) if starts[i].opening is None]
    cold: dict[int, str] = {}
    if need_cold:
        replies = _hf_generate_batch(
            student, [_apply_template(student.tokenizer, sess[i]["student_msgs"], True) for i in need_cold])
        cold = dict(zip(need_cold, replies))

    for i in range(n):
        opening = starts[i].opening if starts[i].opening is not None else cold[i]
        sess[i]["student_msgs"].append({"role": "assistant", "content": opening})
        sess[i]["transcript"].append({"role": "student", "content": opening})
        first = (f"Problem: {problems[i]}\n\nStudent: {opening}" if starts[i].show_problem else f"Student: {opening}")
        sess[i]["tutor_msgs"].append({"role": "user", "content": first})

    ctx_limit = _context_limit(tutor)
    gen = getattr(tutor, "max_new_tokens", 256)
    turn_headroom = gen * (max_tool_calls + 1) + 512
    tok = getattr(tutor, "tokenizer", None)
    # A batch can mix scenarios with different vocabularies, so generation halts at any tag
    stop = action_stop_strings(ALL_TUTOR_ACTIONS) + ([TOOL_CALL_STOP] if schemas else [])

    for _ in range(max_turns):
        active = [i for i in range(n) if sess[i]["active"]]
        # Any session that would overflow the window before this turn is dropped here
        if tok is not None:
            for i in active:
                plen = len(_apply_template(tok, sess[i]["tutor_msgs"], True, tools=schemas))
                if plen + turn_headroom > ctx_limit:
                    sess[i]["active"] = False
            active = [i for i in range(n) if sess[i]["active"]]
        if not active:
            break

        # The first action and any forcing prefix, cheap enough not to batch
        for i in active:
            a, tag_ids, prompt_ids = sample_action(tutor, sess[i]["tutor_msgs"], sess[i]["actions"], tools=schemas)
            ft = force_tools[i]
            fp = (_force_prefix(ft) if
                  (ft and ft in registry and not _tool_was_called(sess[i]["transcript"], ft)) else None)
            p0 = prompt_ids + tag_ids
            if fp:
                p0 = p0 + tutor.tokenizer.encode(fp, add_special_tokens=False)
            sess[i].update(_action=a, _force=fp, _prompt=p0, _budget=max_tool_calls)

        # The tool-call rounds, batching whatever sessions are still mid-turn
        in_turn = list(active)
        first = True
        while in_turn:
            if not first:
                for i in in_turn:
                    a, tag_ids, prompt_ids = sample_action(tutor,
                                                           sess[i]["tutor_msgs"],
                                                           sess[i]["actions"],
                                                           tools=schemas)
                    sess[i].update(_action=a, _force=None, _prompt=prompt_ids + tag_ids)

            raws = _batch_tutor_messages(tutor, [sess[i]["_prompt"] for i in in_turn], stop=stop)

            next_in_turn = []
            for k, i in enumerate(in_turn):
                raw = raws[k]

                if sess[i]["_force"]:
                    raw = sess[i]["_force"] + raw

                message = strip_extra_tags(raw, sess[i]["actions"])
                full = f"[{sess[i]['_action']}] {message}"
                call = parse_tool_call(full) if schemas else None

                if call is not None and sess[i]["_budget"] > 0:
                    name, args = call
                    full = ensure_closed(full)

                    if name in registry:
                        result = (ToolResult("error: tool-call arguments were not valid JSON", ok=False)
                                  if args is None else registry[name].run(
                                      args,
                                      ToolContext(problem=problems[i],
                                                  answer=answers[i],
                                                  transcript=sess[i]["transcript"],
                                                  student_level=student_levels[i])))
                        rec = name
                    else:
                        result = ToolResult(f"error: unknown tool {name!r}", ok=False)
                        rec = "call_invalid"
                    # Only the seeded first call counts as forced
                    seg_forced = sess[i]["_force"] is not None
                    sess[i]["tutor_msgs"].append({"role": "assistant", "content": full})
                    sess[i]["tutor_msgs"].append({"role": TOOL_RESULT_ROLE, "content": result.text})
                    sess[i]["transcript"].append({"role": "tutor", "action": rec, "content": full})
                    sess[i]["transcript"].append({
                        "role": "tool",
                        "name": rec,
                        "content": result.text,
                        "data": result.data,
                        "ok": result.ok,
                        "forced": seg_forced,
                    })
                    sess[i]["_budget"] -= 1
                    next_in_turn.append(i)
                else:
                    # No call, or no budget left, so this segment is the student-facing move
                    sess[i]["tutor_msgs"].append({"role": "assistant", "content": full})
                    sess[i]["transcript"].append({"role": "tutor", "action": sess[i]["_action"], "content": full})
                    sess[i]["_visible"] = strip_tool_calls(message)
            in_turn = next_in_turn
            first = False

        # The student replies for every active session, in one batched call
        for i in active:
            sess[i]["student_msgs"].append({"role": "user", "content": sess[i]["_visible"]})
        replies = _hf_generate_batch(
            student, [_apply_template(student.tokenizer, sess[i]["student_msgs"], True) for i in active])
        for k, i in enumerate(active):
            r = replies[k]
            sess[i]["student_msgs"].append({"role": "assistant", "content": r})
            sess[i]["transcript"].append({"role": "student", "content": r})
            sess[i]["tutor_msgs"].append({"role": "user", "content": r})
            if scenarios[i].is_complete(sess[i]["transcript"], answers[i]):
                sess[i]["active"] = False

    return [(Session(tutor_msgs=sess[i]["tutor_msgs"],
                     transcript=sess[i]["transcript"],
                     problem=problems[i],
                     answer=answers[i]), starts[i].meta) for i in range(n)]
