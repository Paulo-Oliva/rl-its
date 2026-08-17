import pytest
from conftest import as_chat_model

from its._jsonparse import lenient_json
from its.protocol import (TOOL_RESULT_ROLE, _parse_tool_args, ensure_closed, parse_tool_call, strip_tool_calls)
from its.training.rollout import build_session_tokens
from its.training.rollout_tools import (_force_prefix, run_tutor_session_with_tools)
from its.training.settings import SessionSettings
from its.training.tools.base import EchoTool, ToolContext

_TURN_SEQUENCE = [
    ("check_understanding", "", '[check_understanding] Let me check your work. '
     '<tool_call>{"name": "call_echo", "arguments": {"msg": "check 3x=30"}}</tool_call>'),
    ("flag_mistake", "", '[flag_mistake] <tool_call>{"name": "call_echo"}</tool_call>'),
    ("ask_question", "What undoes times 3?", "[ask_question] What undoes times 3?"),
]


def _patch_generation(monkeypatch):
    step = {"i": 0}

    def fake_action(*a, **k):
        s = _TURN_SEQUENCE[min(step["i"], len(_TURN_SEQUENCE) - 1)]
        step["i"] += 1
        return s

    monkeypatch.setattr("its.training.rollout_tools.generate_action_turn", fake_action)


# Solves immediately, which ends the session after one turn
class _FakeStudent:

    def __call__(self, msgs):
        return "x = 10"


def _run_loop(monkeypatch):
    _patch_generation(monkeypatch)
    return run_tutor_session_with_tools(
        "3x=30",
        "10",
        tutor=as_chat_model(None),
        student=as_chat_model(_FakeStudent()),
        tools=[EchoTool()],
        settings=SessionSettings(max_tool_calls=2),
    )


# --- pure logic (default) ----------------------------------------------------


def test_tool_schema_is_native_function_format():
    sch = EchoTool().tool_schema
    assert sch["type"] == "function"
    assert sch["function"]["name"] == "call_echo"
    assert sch["function"]["parameters"]["properties"]["msg"]["type"] == "string"


# (text, expected result of parse_tool_call)
_PARSE_CASES = [
    # prose before the call
    ('thinking ... <tool_call>{"name": "call_echo", "arguments": {"msg": "hi"}}</tool_call>', ("call_echo", {
        "msg": "hi"
    })),
    # unterminated, because the stop string ate the closing tag
    ('<tool_call>{"name": "call_retriever", "arguments": {"topic": "slopes"}}', ("call_retriever", {
        "topic": "slopes"
    })),
    # no arguments object at all
    ('<tool_call>{"name": "call_echo"}</tool_call>', ("call_echo", None)),
    # arguments is not an object
    ('<tool_call>{"name": "call_retriever", "arguments": 2}</tool_call>', ("call_retriever", None)),
    # keys arrive with stray spaces and mixed case
    ('<tool_call>{"name": "call_retriever", "arguments": {" Topic ": "lines"}}</tool_call>', ("call_retriever", {
        "topic": "lines"
    })),
    # an object with no name is dispatched as an invalid call
    ('<tool_call>{"arguments": {"x": 1}}</tool_call>', ("call_invalid", None)),
    # a plain pedagogical move, so there is nothing to dispatch
    ("[ask_question] What undoes times 3?", None),
    # an opening tag with nothing parseable after it
    ("<tool_call> garbage", None),
]


@pytest.mark.parametrize("text, expected", _PARSE_CASES)
def test_parse_tool_call(text, expected):
    assert parse_tool_call(text) == expected


def test_strip_tool_calls_and_ensure_closed():
    assert strip_tool_calls('Sure. <tool_call>{"name": "x"}</tool_call> Done.') == "Sure.  Done."
    assert strip_tool_calls('text <tool_call>{"name": "x"}') == "text"
    assert ensure_closed("a <tool_call>{...}").endswith("</tool_call>")
    assert ensure_closed("a <tool_call>{...}</tool_call>").count("</tool_call>") == 1


def test_tool_io():
    ctx = ToolContext(problem="3x=30", answer="10")
    r = EchoTool().run({"msg": "hi"}, ctx)
    assert r.text == "echo: hi" and r.data == {"echo": "hi"} and r.ok is True
    assert EchoTool().run({}, ctx).ok is False, "empty args should not be ok"


# (text, expected arguments dict)
_ARG_CASES = [
    ('here you go {"msg": "hi"} ok', {
        "msg": "hi"
    }),
    ('{ topic: "parabolas", difficulty: "medium" }', {
        "topic": "parabolas",
        "difficulty": "medium"
    }),
    ("{'topic': 'number theory'}", {
        "topic": "number theory"
    }),
    ('{"work": "x = 10",}', {
        "work": "x = 10"
    }),
    # an apostrophe inside a valid object must survive the strict path
    ('{"work": "Newton\'s method gives x=2"}', {
        "work": "Newton's method gives x=2"
    }),
    # forcing seeds an open brace the model may leave unclosed
    ('{"topic": "slope"', {
        "topic": "slope"
    }),
    ('{"name": "call_retriever", "arguments": {"topic": "slope"}', {
        "name": "call_retriever",
        "arguments": {
            "topic": "slope"
        }
    }),
    # LaTeX backslashes are invalid JSON escapes
    (r'{"work": "x = \frac{1}{2} \( so \) ..."}', {
        "work": r"x = \frac{1}{2} \( so \) ..."
    }),
    ("not json at all", None),
    ("[1, 2]", None),
    ("Let's use a practice problem about lines.", None),
]


@pytest.mark.parametrize("text, expected", _ARG_CASES)
def test_parse_tool_args(text, expected):
    assert _parse_tool_args(text) == expected


_JSON_CASES = [
    ('{"feedback": "Newton\'s method"}', {
        "feedback": "Newton's method"
    }),
    (r'{"valid": 0, "feedback": "you wrote \frac{a}{b} but \(x\) is wrong"}', {
        "valid": 0,
        "feedback": r"you wrote \frac{a}{b} but \(x\) is wrong"
    }),
    ("{topic: 'lines', difficulty: 'easy',}", {
        "topic": "lines",
        "difficulty": "easy"
    }),
    ("just prose", None),
    ("[1, 2]", None),
]


@pytest.mark.parametrize("blob, expected", _JSON_CASES)
def test_lenient_json(blob, expected):
    assert lenient_json(blob) == expected


def test_tool_loop_caps_and_terminates(monkeypatch):
    sess = _run_loop(monkeypatch)
    tool_entries = [t for t in sess.transcript if t["role"] == "tool"]
    assert len(tool_entries) == 2, f"per-turn cap should allow exactly 2 calls, got {len(tool_entries)}"
    assert tool_entries[0]["content"] == "echo: check 3x=30" and tool_entries[0]["data"] == {"echo": "check 3x=30"}
    assert tool_entries[0]["ok"] is True, "valid call → ok"
    assert tool_entries[1]["ok"] is False and tool_entries[1]["content"].startswith("error"), "no-args call → ok=False"
    # the student only sees the final pedagogical prose (its own action tag, <tool_call> gone)
    assert sess.transcript[-2] == {
        "role": "tutor",
        "action": "ask_question",
        "content": "[ask_question] What undoes times 3?"
    }
    assert sess.transcript[-1] == {"role": "student", "content": "x = 10"}, "solve ends the session"


# Forcing seeds the first turn with the tool's prefix, then stops once the tool has been called
def test_tool_forcing_seeds_call_then_stops(monkeypatch):
    seen_prefixes = []

    def fake_action(tutor, msgs, ped_actions, tools=None, tool_stop=False, force_prefix=None):
        seen_prefixes.append(force_prefix)
        # The prefix already opens the arguments object, so only the value and braces are left
        if force_prefix:
            full = f'[check_understanding] {force_prefix}"msg": "check"}}}}</tool_call>'
            return "check_understanding", "", full
        return "ask_question", "What undoes times 3?", "[ask_question] What undoes times 3?"

    monkeypatch.setattr("its.training.rollout_tools.generate_action_turn", fake_action)

    class _NeverSolves:

        def __call__(self, msgs):
            return "still thinking"

    sess = run_tutor_session_with_tools("3x=30",
                                        "10",
                                        tutor=as_chat_model(None),
                                        student=as_chat_model(_NeverSolves()),
                                        tools=[EchoTool()],
                                        settings=SessionSettings(max_turns=3, max_tool_calls=1, force_tool="call_echo"))

    # turn 1 forced with the echo prefix; turns 2+ unforced (tool already called)
    assert seen_prefixes[0] == _force_prefix("call_echo")
    assert all(p is None for p in seen_prefixes[1:]), seen_prefixes
    assert any(t["role"] == "tutor" and t.get("action") == "call_echo" for t in sess.transcript), \
        "the forced call must be recorded as a call_echo tutor move"


# --- batched group rollout (model-free: generation primitives stubbed) ---------


# The smallest thing run_scenario_group accepts as a scenario
def _fake_scenario(opening=None, sees=True):
    import types

    from its.config import TUTOR_ACTIONS
    s = types.SimpleNamespace(force_tool=None, actions=TUTOR_ACTIONS)
    s.start = lambda problem, answer, difficulty, rng, _o=opening, _se=sees: types.SimpleNamespace(
        opening=_o,
        solve_terminates=True,
        tutor_sees_answer=_se,
        student_directive=None,
        tutor_system=None,
        show_problem=True,
        meta={})
    s.is_complete = lambda transcript, answer: False
    return s


def _fake_tutor_student():
    import types
    tok = types.SimpleNamespace(encode=lambda s, add_special_tokens=False: [0])
    tutor = types.SimpleNamespace(tokenizer=tok, max_new_tokens=32)
    student = types.SimpleNamespace(tokenizer=tok, max_new_tokens=32)
    return tutor, student


# Stubs the batched generation so run_scenario_group runs without a model, with
# `tutor_messages(round)` giving the message every prompt gets that round
def _patch_batched(monkeypatch, tutor_messages):
    import its.training.rollout_tools as rt
    monkeypatch.setattr(rt, "_apply_template", lambda *a, **k: [0, 1, 2])
    monkeypatch.setattr(rt, "sample_action", lambda *a, **k: ("ask_question", [9], [3, 4]))
    monkeypatch.setattr(rt, "_hf_generate_batch", lambda llm, prompts, stop=None: ["x = 11"] * len(prompts))
    state = {"n": 0}

    def fake_batch(tutor, prompts, stop=None):
        msg = tutor_messages(state["n"])
        state["n"] += 1
        return [msg] * len(prompts)

    monkeypatch.setattr(rt, "_batch_tutor_messages", fake_batch)


def test_batched_scenario_group_well_formed(monkeypatch):
    import its.training.rollout_tools as rt

    # Plain moves, with no tool calls
    _patch_batched(monkeypatch, lambda n: "Let me help you.")
    tutor, student = _fake_tutor_student()
    # One session opens with a cold reply and the other with a fixed opening
    scns = [_fake_scenario(), _fake_scenario(opening="hi tutor")]
    results = rt.run_scenario_group(scns, ["3x=30", "x+5=12"], ["10", "7"], ["medium", "easy"], ["medium", "weak"],
                                    [None, None],
                                    tutor,
                                    student, [EchoTool()],
                                    rng=None,
                                    settings=SessionSettings(max_turns=2))
    assert len(results) == 2, "one (session, meta) per input"
    for session, meta in results:
        assert meta == {}
        assert session.transcript[0]["role"] == "student", "session starts with the cold/opening student turn"
        tutor_turns = [t for t in session.transcript if t["role"] == "tutor"]
        assert len(tutor_turns) == 2, "max_turns=2, no early termination → 2 tutor moves"
        assert all(t["content"].startswith("[ask_question] ") for t in tutor_turns), "action tag on each move"


def test_batched_group_runs_tool_call(monkeypatch):
    import its.training.rollout_tools as rt

    # round 0 -> a native call_echo; the continuation -> the student-facing move
    _patch_batched(
        monkeypatch,
        lambda n: ('<tool_call>{"name": "call_echo", "arguments": {"msg": "hi"}}</tool_call>'
                   if n == 0 else "Here is a hint."),
    )
    tutor, student = _fake_tutor_student()
    results = rt.run_scenario_group([_fake_scenario()], ["3x=30"], ["10"], ["medium"], ["medium"], [None],
                                    tutor,
                                    student, [EchoTool()],
                                    rng=None,
                                    settings=SessionSettings(max_turns=1, max_tool_calls=2))
    session, _ = results[0]
    tools_used = [t for t in session.transcript if t["role"] == "tool"]
    assert len(tools_used) == 1, "the inner round-loop ran exactly one tool call"
    assert tools_used[0]["name"] == "call_echo" and tools_used[0]["ok"] is True
    assert any(t["role"] == "tutor" and t.get("action") == "call_echo" for t in session.transcript)
    # the tool RESULT is spliced as a masked (non-assistant) tutor_msgs entry
    assert any(m["role"] == TOOL_RESULT_ROLE and m["content"] == "echo: hi" for m in session.tutor_msgs)


_TOOL_CALL_TURN = ('[check_understanding] Let me check. '
                   '<tool_call>{"name": "call_echo", "arguments": {"msg": "is 3x=30 right?"}}</tool_call>')

# A session the tutor could have produced, with the tool call trained and the result masked
_HAND_BUILT = [
    {
        "role": "system",
        "content": "You are a Socratic tutor. Answer: 10"
    },
    {
        "role": "user",
        "content": "Problem: 3x=30\n\nStudent: I'm stuck."
    },
    {
        "role": "assistant",
        "content": _TOOL_CALL_TURN
    },
    {
        "role": TOOL_RESULT_ROLE,
        "content": "echo: is 3x=30 right?"
    },
    {
        "role": "assistant",
        "content": "[ask_question] What undoes times 3?"
    },
    {
        "role": "user",
        "content": "Divide by 3?"
    },
]


@pytest.mark.model
def test_tool_result_is_masked_hand_built(tokenizer):
    st = build_session_tokens(tokenizer, _HAND_BUILT, tools=[EchoTool().tool_schema])
    trained = tokenizer.decode([t for t, m in zip(st.completion_ids, st.env_mask) if m == 1], skip_special_tokens=True)
    assert "<tool_call>" in trained and "call_echo" in trained, "tool CALL must be trained"
    assert "[ask_question] What undoes times 3?" in trained, "pedagogical move must be trained"
    assert "echo: is 3x=30 right?" not in trained, "tool RESULT must be masked"
    assert "Divide by 3?" not in trained, "student turn must be masked"


@pytest.mark.model
def test_tool_result_masked_on_generated_session(tokenizer, monkeypatch):
    sess = _run_loop(monkeypatch)
    st = build_session_tokens(tokenizer, sess.tutor_msgs, tools=[EchoTool().tool_schema])
    trained = tokenizer.decode([t for t, m in zip(st.completion_ids, st.env_mask) if m == 1], skip_special_tokens=True)
    assert "<tool_call>" in trained, "tool call (incl. native JSON args) trained"
    assert "echo:" not in trained and "x = 10" not in trained, "tool results + student masked"
