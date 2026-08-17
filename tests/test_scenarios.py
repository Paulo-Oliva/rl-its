import random

from conftest import MATH_KB

from its.protocol import count_tutor_turns
from its.training.reward import Judge
from its.training.rollout import Session
from its.training.scenarios.kinds import (CheckMyWorkScenario, ExplainConceptScenario, RequestProblemScenario,
                                          SolveScenario, build_scenarios)
from its.training.scenarios.rollout import make_scenario_reward_func
from its.training.scenarios.signals import (bad_tool_use, contradicts_verdict, gibberish_score)

SCENARIOS = build_scenarios(MATH_KB)


# Answers every relay check with a fixed value and refuses to score anything else
class _StubRelayJudge(Judge):

    def __init__(self, val: float) -> None:
        self.val = val

    def score(self, *a, **k):
        raise AssertionError("relay-check scenarios must not call full score()")

    def relay_check(self, *a, **k) -> float:
        return self.val


# --- scenario starts ---------------------------------------------------------


def test_solve_start_matches_the_plain_session():
    st = SolveScenario().start("3x=30", "10", "medium", random.Random(0))
    assert st.opening is None and st.solve_terminates and st.tutor_sees_answer


def test_check_my_work_start_hides_answer():
    st = CheckMyWorkScenario().start("3x=30", "10", "medium", random.Random(0))
    assert st.opening is None and not st.tutor_sees_answer and not st.solve_terminates
    assert st.meta == {}


# --- check_my_work scoring ---------------------------------------------------


def _cmw_called():
    return Session([], [
        {
            "role": "student",
            "content": "I solved it: 3x=30 so x=10 by dividing both sides by 3."
        },
        {
            "role": "tutor",
            "action": "call_verifier",
            "content": '[call_verifier] {"work": "x = 10 by dividing by 3"}'
        },
        {
            "role": "tool",
            "name": "call_verifier",
            "content": "The student's reasoning appears valid. Each step follows.",
            "data": {
                "valid": True,
                "feedback": "Each step follows."
            },
            "ok": True
        },
        {
            "role": "tutor",
            "action": "affirm",
            "content": "[affirm] Your reasoning is sound - nicely done."
        },
        {
            "role": "student",
            "content": "great, thanks"
        },
    ], "3x=30", "10")


def _cmw_not_called():
    return Session([], [
        {
            "role": "student",
            "content": "I solved it: 3x=30 so x=10 by dividing both sides by 3."
        },
        {
            "role": "tutor",
            "action": "affirm",
            "content": "[affirm] Looks right to me."
        },
        {
            "role": "student",
            "content": "ok"
        },
    ], "3x=30", "10")


def test_check_my_work_rewards_verifier_use():
    cmw = CheckMyWorkScenario()
    b_called = cmw.score(_cmw_called(), {}, judge=None)
    b_not = cmw.score(_cmw_not_called(), {}, judge=None)
    assert b_called.components["verifier_used"] == 1.0 and b_not.components["verifier_used"] == 0.0
    assert b_called.reward > b_not.reward
    # a turn spent calling the verifier is not a pedagogical turn
    assert count_tutor_turns(_cmw_called().transcript) == 1


def test_tool_call_args_are_not_leakage():
    cmw = CheckMyWorkScenario()
    leaky_call = Session([], [
        {
            "role": "student",
            "content": "Here is my work, can you check it?"
        },
        {
            "role": "tutor",
            "action": "call_verifier",
            "content": '[call_verifier] {"work": "so x = 10"}'
        },
        {
            "role": "tool",
            "name": "call_verifier",
            "content": "The student's reasoning appears valid.",
            "data": {
                "valid": True,
                "feedback": "sound"
            },
            "ok": True
        },
        {
            "role": "tutor",
            "action": "affirm",
            "content": "[affirm] Your reasoning is sound."
        },
        {
            "role": "student",
            "content": "ok"
        },
    ], "3x=30", "10")
    assert cmw.score(leaky_call, {}, judge=None).leakage == 0


def test_check_my_work_relay_check():
    cmw = CheckMyWorkScenario()
    ok = cmw.score(_cmw_called(), {}, judge=_StubRelayJudge(1.0))
    bad = cmw.score(_cmw_called(), {}, judge=_StubRelayJudge(0.0))
    assert ok.components["verdict_relayed"] == 1.0 and bad.components["verdict_relayed"] == 0.0
    assert ok.reward > bad.reward, "accurately relaying the verifier's finding must out-earn not"
    # nothing to relay when the verifier was never called
    assert "verdict_relayed" not in cmw.score(_cmw_not_called(), {}, judge=_StubRelayJudge(1.0)).components


# --- request_problem scoring -------------------------------------------------


def _planned():
    return Session([], [
        {
            "role": "student",
            "content": "Can I have a problem?"
        },
        {
            "role": "tutor",
            "action": "call_planner",
            "content": '[call_planner] {"difficulty": "easy"}'
        },
        {
            "role": "tool",
            "name": "call_planner",
            "content": "Practice problem (easy): ...",
            "data": {
                "problem": "...",
                "answer": "1",
                "difficulty": "easy"
            },
            "ok": True
        },
        {
            "role": "tutor",
            "action": "ask_question",
            "content": "[ask_question] Here's one - give it a try?"
        },
        {
            "role": "student",
            "content": "ok"
        },
    ], "3x=30", "10")


def _no_plan():
    return Session([], [
        {
            "role": "student",
            "content": "Can I have a problem?"
        },
        {
            "role": "tutor",
            "action": "ask_question",
            "content": "[ask_question] Sure, what is 2+2?"
        },
        {
            "role": "student",
            "content": "4"
        },
    ], "3x=30", "10")


def test_request_problem_start_and_planner_use():
    rp = RequestProblemScenario()
    rps = rp.start("3x=30", "10", "medium", random.Random(0))
    assert rps.opening and not rps.tutor_sees_answer and not rps.solve_terminates
    assert rp.score(_planned(), {}, judge=None).reward > rp.score(_no_plan(), {}, judge=None).reward


def test_request_problem_relay_check():
    rp = RequestProblemScenario()
    assert rp.score(_planned(), {}, _StubRelayJudge(1.0)).components["problem_presented"] == 1.0
    assert rp.score(_planned(), {}, _StubRelayJudge(1.0)).reward > \
        rp.score(_planned(), {}, _StubRelayJudge(0.0)).reward
    assert "problem_presented" not in rp.score(_no_plan(), {}, _StubRelayJudge(1.0)).components


# --- explain_concept scoring -------------------------------------------------


def _retrieved():
    snippet = MATH_KB["the quadratic formula"]
    return Session([], [
        {
            "role": "student",
            "content": "Can you explain the quadratic formula?"
        },
        {
            "role": "tutor",
            "action": "call_retriever",
            "content": '[call_retriever] {"topic": "the quadratic formula"}'
        },
        {
            "role": "tool",
            "name": "call_retriever",
            "content": f"Reference on ...: {snippet}",
            "data": {
                "topic": "the quadratic formula",
                "snippet": snippet
            },
            "ok": True
        },
        {
            "role": "tutor",
            "action": "give_hint",
            "content": "[give_hint] It's x = (-b ± sqrt(b^2-4ac))/(2a)."
        },
        {
            "role": "student",
            "content": "thanks"
        },
    ], "3x=30", "10")


def _no_retrieve():
    return Session([], [
        {
            "role": "student",
            "content": "Can you explain the quadratic formula?"
        },
        {
            "role": "tutor",
            "action": "give_hint",
            "content": "[give_hint] Sure, it's some formula with a, b, c."
        },
        {
            "role": "student",
            "content": "ok"
        },
    ], "3x=30", "10")


def test_explain_concept_start_picks_kb_concept():
    ec = ExplainConceptScenario(MATH_KB)
    ecs = ec.start("3x=30", "10", "medium", random.Random(0))
    assert ecs.opening and not ecs.tutor_sees_answer and not ecs.solve_terminates
    assert ecs.meta["concept"] in MATH_KB, "asked concept must be retrievable"


def test_explain_concept_rewards_retrieval_and_faithfulness():
    ec = ExplainConceptScenario(MATH_KB)
    assert ec.score(_retrieved(), {}, judge=None).reward > ec.score(_no_retrieve(), {}, judge=None).reward
    assert ec.score(_retrieved(), {}, _StubRelayJudge(1.0)).components["explanation_faithful"] == 1.0
    assert ec.score(_retrieved(), {}, _StubRelayJudge(1.0)).reward > \
        ec.score(_retrieved(), {}, _StubRelayJudge(0.0)).reward, "faithful out-earns hallucinated"
    assert "explanation_faithful" not in ec.score(_no_retrieve(), {}, _StubRelayJudge(1.0)).components


# --- BadToolUse + dispatch ---------------------------------------------------

_MALFORMED = [
    {
        "role": "tutor",
        "action": "call_verifier",
        "content": '[call_verifier] not json'
    },
    {
        "role": "tool",
        "name": "call_verifier",
        "content": "error: ...",
        "ok": False
    },
]


def test_bad_tool_use_counts():
    assert bad_tool_use(_cmw_called().transcript) == 0, "clean single call is free"
    assert bad_tool_use(_MALFORMED) == 1, "malformed call is charged"
    dup = [
        {
            "role": "tutor",
            "action": "call_verifier",
            "content": '[call_verifier] {"work": "x=10"}'
        },
        {
            "role": "tool",
            "name": "call_verifier",
            "content": "Correct",
            "ok": True
        },
        {
            "role": "tutor",
            "action": "call_verifier",
            "content": '[call_verifier] {"work": "x=10"}'
        },
        {
            "role": "tool",
            "name": "call_verifier",
            "content": "Correct",
            "ok": True
        },
    ]
    assert bad_tool_use(dup) == 1, "duplicate call charged once (repeat only)"
    # a forced call that fails is the scaffold's doing, not the tutor's
    forced_bad = [
        {
            "role": "tutor",
            "action": "call_verifier",
            "content": "[call_verifier] <tool_call>..."
        },
        {
            "role": "tool",
            "name": "call_verifier",
            "content": "error: ...",
            "ok": False,
            "forced": True
        },
    ]
    assert bad_tool_use(forced_bad) == 0, "forced failed call is exempt from BadToolUse"
    forced_bad[1]["forced"] = False
    assert bad_tool_use(forced_bad) == 1, "the same call unforced IS charged"


def test_reward_dispatch_routes_and_penalizes():
    rf = make_scenario_reward_func(judge=None, scenarios=SCENARIOS)
    rewards = rf(completions=[None, None],
                 transcript=[_planned().transcript, _no_plan().transcript],
                 answer=["10", "10"],
                 problem=["3x=30", "3x=30"],
                 student_level=["medium", "medium"],
                 difficulty=["medium", "medium"],
                 scenario=["request_problem", "request_problem"],
                 meta=[{}, {}])
    assert rewards[0] > rewards[1]
    # the same session with a malformed call must score lower
    bad_planned = _planned().transcript + _MALFORMED
    rb = rf(completions=[None, None],
            transcript=[_planned().transcript, bad_planned],
            answer=["10", "10"],
            problem=["3x=30", "3x=30"],
            student_level=["medium", "medium"],
            difficulty=["medium", "medium"],
            scenario=["request_problem", "request_problem"],
            meta=[{}, {}])
    assert rb[1] < rb[0]


def test_correct_tag_reward():
    rf = make_scenario_reward_func(judge=None, scenarios=SCENARIOS)
    presented = Session([], [
        {
            "role": "student",
            "content": "Can I have a problem?"
        },
        {
            "role": "tutor",
            "action": "call_planner",
            "content": "[present_problem] <tool_call>"
        },
        {
            "role": "tool",
            "name": "call_planner",
            "content": "Practice problem (easy): ...",
            "data": {},
            "ok": True
        },
        {
            "role": "tutor",
            "action": "present_problem",
            "content": "[present_problem] Here's your problem: 2x=8."
        },
        {
            "role": "student",
            "content": "ok"
        },
    ], "3x=30", "10")

    r = rf(completions=[None, None],
           transcript=[presented.transcript, _planned().transcript],
           answer=["10", "10"],
           problem=["3x=30", "3x=30"],
           student_level=["medium", "medium"],
           difficulty=["medium", "medium"],
           scenario=["request_problem", "request_problem"],
           meta=[{}, {}])
    assert r[0] > r[1], "using [present_problem] earns the correct_tag bonus; [ask_question] does not"


def test_gibberish_detection_and_penalty():
    clean = [{
        "role": "tutor",
        "action": "explain",
        "content": "[explain] A derivative measures the rate of change of a function. For f(x)=x^2 it is 2x."
    }]
    salad = [{
        "role": "tutor",
        "action": "explain",
        "content": "[explain] ulos do.Re 주의顾客 GNOME讓我 иностранн taraf disturbed 尽早"
    }]
    greek = [{
        "role": "tutor",
        "action": "explain",
        "content": "[explain] With \\(\\tan\\alpha=2\\), \\(\\sin 2\\alpha=\\frac{4}{5}\\)."
    }]
    assert gibberish_score(clean) < 0.05 and gibberish_score(greek) < 0.05, "clean math (incl. Greek) not flagged"
    assert gibberish_score(salad) > 0.05, "script salad flagged"
    rf = make_scenario_reward_func(judge=None, scenarios=SCENARIOS)
    r = rf(completions=[None, None],
           transcript=[clean, salad],
           answer=["1", "1"],
           problem=["p", "p"],
           student_level=["medium", "medium"],
           difficulty=["medium", "medium"],
           scenario=["explain_concept", "explain_concept"],
           meta=[{}, {}])
    assert r[0] > r[1], "gibberish session penalized below the clean one"


def test_verdict_contradiction_penalty():
    flawed = {"role": "tool", "name": "call_verifier", "content": "flawed", "data": {"valid": False}, "ok": True}
    sound = {"role": "tool", "name": "call_verifier", "content": "sound", "data": {"valid": True}, "ok": True}
    affirm = {"role": "tutor", "action": "affirm", "content": "[affirm] on track!"}
    flag = {"role": "tutor", "action": "flag_mistake", "content": "[flag_mistake] step 2 is wrong"}
    neutral = {"role": "tutor", "action": "check_understanding", "content": "[check_understanding] why?"}
    assert contradicts_verdict([flawed, affirm]) is True, "affirm after flawed = contradiction"
    assert contradicts_verdict([sound, flag]) is True, "flag after sound = contradiction"
    assert contradicts_verdict([flawed, flag]) is False, "flag after flawed = correct"
    assert contradicts_verdict([flawed, neutral]) is False, "neutral tag is not a contradiction"
    sc = CheckMyWorkScenario()
    call = {"role": "tutor", "action": "call_verifier", "content": "[check_understanding] <tool_call>"}
    stu = {"role": "student", "content": "w"}
    good = sc.score(Session([], [stu, call, flawed, flag], "p", "a"), {}, judge=None)
    bad = sc.score(Session([], [stu, call, flawed, affirm], "p", "a"), {}, judge=None)
    assert bad.reward < good.reward and bad.components.get("verdict_contradiction") == -1.5


def test_scenario_registry():
    assert set(SCENARIOS) == {"solve", "check_my_work", "request_problem", "explain_concept"}
    assert SCENARIOS["request_problem"].signature_action == "present_problem"
    assert SCENARIOS["explain_concept"].signature_action == "explain"
    assert SCENARIOS["solve"].signature_action is None


# --- early termination (is_complete) -----------------------------------------


def _student(c):
    return {"role": "student", "content": c}


def _tutor(action, c="..."):
    return {"role": "tutor", "action": action, "content": f"[{action}] {c}"}


def _tool(name):
    return {"role": "tool", "name": name, "content": "...", "data": {}, "ok": True}


def test_solve_complete_when_answer_stated():
    sc = SolveScenario()
    assert sc.is_complete([_student("I'm stuck")], "10") is False
    assert sc.is_complete([_student("I'm stuck"), _student("So x = 10.")], "10") is True


def test_check_my_work_complete_needs_verifier_and_answer():
    cmw = CheckMyWorkScenario()
    # the answer is there but was never verified
    assert cmw.is_complete([_student("x = 10")], "10") is False
    # verified, but the student has not reached the right answer yet
    assert cmw.is_complete(
        [_student("x = 27"),
         _tutor("call_verifier"),
         _tool("call_verifier"),
         _tutor("flag_mistake")], "10") is False
    # verified and then corrected to the right answer
    assert cmw.is_complete([
        _student("x = 27"),
        _tutor("call_verifier"),
        _tool("call_verifier"),
        _tutor("flag_mistake"),
        _student("oh, x = 10")
    ], "10") is True


def test_explain_and_request_complete_after_delivery():
    ec = ExplainConceptScenario(MATH_KB)
    # retrieved but not yet explained
    assert ec.is_complete([_tutor("call_retriever"), _tool("call_retriever")], "") is False
    # retrieved and then explained
    assert ec.is_complete([_tutor("call_retriever"),
                           _tool("call_retriever"),
                           _tutor("give_hint"),
                           _student("thanks")], "") is True
    rp = RequestProblemScenario()
    assert rp.is_complete([_tutor("call_planner"), _tool("call_planner")], "") is False
    assert rp.is_complete(
        [_tutor("call_planner"), _tool("call_planner"),
         _tutor("ask_question"), _student("ok")], "") is True
