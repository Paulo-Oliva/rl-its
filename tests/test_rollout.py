import pytest

from conftest import as_chat_model

from its.config import TUTOR_ACTIONS
from its.protocol import action_stop_strings, action_tag_regex, strip_extra_tags
from its.training.rollout import build_session_tokens, run_tutor_session


class _Scripted:

    def __init__(self, replies):
        self.replies = list(replies)

    def __call__(self, messages):
        return self.replies.pop(0)


def test_strip_extra_tags_keeps_one_action():
    salad = "Can you confirm this? [ give hint] Think about the next step. [check_understanding] What did you get?"
    assert strip_extra_tags(salad, TUTOR_ACTIONS) == "Can you confirm this?"
    salad2 = "Good start. [flag_mistake] Recheck the sign."
    assert strip_extra_tags(salad2, TUTOR_ACTIONS) == "Good start."
    # a maths interval is not an action tag
    math = "Substitute into the interval [0, 1] and simplify [a, b] too."
    assert strip_extra_tags(math, TUTOR_ACTIONS) == math
    assert strip_extra_tags("Just one clean question here?", TUTOR_ACTIONS) == "Just one clean question here?"


def test_action_stop_strings_cover_name_variants():
    stops = action_stop_strings(TUTOR_ACTIONS)
    assert "[give_hint" in stops and "[give hint" in stops


def test_session_terminates_when_student_states_answer():
    tutor = _Scripted(["What undoes multiplying by 3?"] * 10)
    student = _Scripted(["No idea.", "Dividing, maybe?", "So x = 10.", "SHOULD NEVER BE REACHED"])
    session = run_tutor_session("3x = 30", "10", as_chat_model(tutor), as_chat_model(student), max_turns=10)
    n_tutor = sum(1 for t in session.transcript if t["role"] == "tutor")
    assert n_tutor == 2, f"expected 2 tutor turns after early stop, got {n_tutor}"
    assert session.transcript[-1]["content"] == "So x = 10.", "session must end on the solving reply"


_TUTOR_MSGS = [
    {
        "role": "system",
        "content": "You are a Socratic tutor. Answer: 10"
    },
    {
        "role": "user",
        "content": "Problem: 3x=30\n\nStudent: I don't know where to start."
    },
    {
        "role": "assistant",
        "content": "[ask_question] What operation undoes multiplication by 3?"
    },
    {
        "role": "user",
        "content": "Division?"
    },
    {
        "role": "assistant",
        "content": "[give_hint] Right. So what is 30 divided by 3?"
    },
    {
        "role": "user",
        "content": "x = 10"
    },
    {
        "role": "assistant",
        "content": "[affirm] Exactly — well done."
    },
]


@pytest.mark.model
def test_env_mask_isolates_tutor_turns(tokenizer):
    st = build_session_tokens(tokenizer, _TUTOR_MSGS)
    assert len(st.completion_ids) == len(st.env_mask), "mask/ids length mismatch"
    decoded = tokenizer.decode([tid for tid, m in zip(st.completion_ids, st.env_mask) if m == 1],
                               skip_special_tokens=True)
    for turn in [m["content"] for m in _TUTOR_MSGS if m["role"] == "assistant"]:
        assert turn in decoded, f"tutor turn missing from masked spans: {turn!r}"
    for student_turn in ["I don't know where to start.", "Division?", "x = 10"]:
        assert student_turn not in decoded, f"student text leaked into tutor mask: {student_turn!r}"


@pytest.mark.model
def test_turn_spans_isolate_each_tutor_turn(tokenizer):
    st = build_session_tokens(tokenizer, _TUTOR_MSGS)
    expected = [m["content"] for m in _TUTOR_MSGS if m["role"] == "assistant"]
    assert len(st.turn_spans) == len(expected), "one span per tutor turn"
    for (a, b), content in zip(st.turn_spans, expected):
        span_text = tokenizer.decode(st.completion_ids[a:b], skip_special_tokens=True)
        assert content in span_text, f"span {a}:{b} does not contain its turn: {content!r}"
        for other in expected:
            if other != content:
                assert other not in span_text, f"span {a}:{b} bleeds into another turn"


@pytest.mark.model
def test_constrained_action_decoding_is_valid():
    import torch

    from its.llm import load_model
    from its.config import SIMULATOR_MODEL_ID
    from its.prompts import TUTOR_SYSTEM
    from its.protocol import action_instructions
    from its.training.rollout import generate_action_turn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm = load_model(SIMULATOR_MODEL_ID, device, max_new_tokens=60)
    llm.do_sample, llm.temperature = True, 1.0
    msgs = [
        {
            "role": "system",
            "content": TUTOR_SYSTEM + action_instructions(TUTOR_ACTIONS)
        },
        {
            "role": "user",
            "content": "Problem: 3x=30\n\nStudent: I don't know where to start."
        },
    ]
    for _ in range(3):
        action, message, full = generate_action_turn(llm, msgs, TUTOR_ACTIONS)
        assert action in TUTOR_ACTIONS, f"invalid action {action!r}"
        assert full.startswith(f"[{action}] "), f"bad turn format: {full!r}"
        assert not action_tag_regex(TUTOR_ACTIONS).search(message), "extra tag leaked into student message"


@pytest.mark.model
def test_batched_group_rollout_well_formed():
    import torch

    from its.llm import load_model
    from its.config import SIMULATOR_MODEL_ID
    from its.training.rollout import run_tutor_group

    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm = load_model(SIMULATOR_MODEL_ID, device, max_new_tokens=48)
    llm.do_sample, llm.temperature = True, 1.0
    student = load_model(SIMULATOR_MODEL_ID, device, max_new_tokens=48)

    sessions = run_tutor_group(["3x = 30", "3x = 30", "x + 5 = 12"], ["10", "10", "7"],
                               llm,
                               student, ["medium", "medium", "weak"],
                               max_turns=3,
                               actions=TUTOR_ACTIONS)
    assert len(sessions) == 3, "one session per input"
    for s in sessions:
        assert s.transcript[0]["role"] == "student", "session starts with the cold student attempt"
        tutor_turns = [t for t in s.transcript if t["role"] == "tutor"]
        assert tutor_turns, "session has at least one tutor turn"
        for t in tutor_turns:
            assert t["action"] in TUTOR_ACTIONS, f"invalid action {t.get('action')!r}"
            assert t["content"].startswith(f"[{t['action']}] "), "tag missing from tutor turn"
        stk = build_session_tokens(llm.tokenizer, s.tutor_msgs)
        assert len(stk.completion_ids) == len(stk.env_mask), "mask/ids length mismatch"
        assert sum(stk.env_mask) > 0, "no tutor tokens marked"
