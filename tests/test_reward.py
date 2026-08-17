import json
from pathlib import Path

import pytest

from its.config import REWARD_WEIGHTS
from its.protocol import detect_leakage, detect_solve, verify_answer
from its.training.reward import Judge, JudgeScores, LLMJudge, _parse_judge_scores, score_session


# Agrees with the rule checks (so the gates stay 1.0) and gives no pedagogy credit
class AgreeJudge(Judge):

    def score(self, transcript, problem, answer):
        return JudgeScores(student_solved=float(detect_solve(transcript, answer)),
                           tutor_revealed=float(detect_leakage(transcript, answer)))


@pytest.fixture
def agree():
    return AgreeJudge()


_FIXTURES = {
    f["category"]: f
    for f in json.loads((Path(__file__).parent / "fixtures" / "t3_sessions.json").read_text(encoding="utf-8"))
}


# (transcript, problem, answer) from a real T3 training session
def _real(category):
    f = _FIXTURES[category]
    return f["transcript"], f["problem"], f["answer"]


# The student solves it independently over two turns, with no leak
GOOD, GOOD_PROB, GOOD_ANS = _real("good")
# The tutor reveals the answer
LEAKY, LEAKY_PROB, LEAKY_ANS = _real("leaky")
# The student never solves it, and nothing leaks
FAILED, FAILED_PROB, FAILED_ANS = _real("failed")
# The student states the answer cold and the tutor only confirms it
COLD_SOLVE, COLD_PROB, COLD_ANS = _real("cold_solve")

# --- Reward arithmetic ------------------------------------------------------------


@pytest.mark.parametrize("transcript, answer, problem", [
    (GOOD, GOOD_ANS, GOOD_PROB),
    (LEAKY, LEAKY_ANS, LEAKY_PROB),
    (FAILED, FAILED_ANS, FAILED_PROB),
])
def test_components_sum_to_reward(transcript, answer, problem, agree):
    b = score_session(transcript, answer, problem, agree)
    assert abs(sum(b.components.values()) - b.reward) < 1e-9


def test_reward_ordering(agree):
    g = score_session(GOOD, GOOD_ANS, GOOD_PROB, agree)
    le = score_session(LEAKY, LEAKY_ANS, LEAKY_PROB, agree)
    assert g.solve_success == 1 and g.leakage == 0, "good session is an independent solve"
    assert le.leakage == 1 and le.solve_success == 0, "leaky session: flagged, no solve credit"
    assert g.reward > le.reward, "independent solve beats a leaked solve"


# --- Solve/leakage cross-examination gate -----------------------------------------


# The rule says solved and the judge disagrees, so the solve credit is cut to 0.4
def test_gate_rule_solve_judge_rejects(agree):
    g = score_session(GOOD, GOOD_ANS, GOOD_PROB, agree)
    judge_no_solve = LLMJudge(lambda m: '{"delta_mastery":0,"progress":0,"overhelp":0,'
                              '"pedagogical_match":0,"student_solved":0,"tutor_revealed":0}')
    gn = score_session(GOOD, GOOD_ANS, GOOD_PROB, judge_no_solve)
    assert gn.solve_gate == 0.4 and gn.reward < g.reward


# The rule misses a solve the judge sees, which earns the smaller 0.3 recovery credit
def test_gate_judge_only_recovery(agree):
    judge_yes_solve = LLMJudge(lambda m: '{"delta_mastery":0,"progress":0,"overhelp":0,'
                               '"pedagogical_match":0,"student_solved":1,"tutor_revealed":0}')
    fr = score_session(FAILED, FAILED_ANS, FAILED_PROB, judge_yes_solve)
    ff = score_session(FAILED, FAILED_ANS, FAILED_PROB, agree)
    assert fr.solve_gate == 0.3 and fr.reward > ff.reward


# --- Ability x difficulty solve factor --------------------------------------------


def test_solve_factor_scales_credit(agree):
    g = score_session(GOOD, GOOD_ANS, GOOD_PROB, agree)
    wh = score_session(GOOD, GOOD_ANS, GOOD_PROB, agree, student_level="weak", difficulty="hard")
    se = score_session(GOOD, GOOD_ANS, GOOD_PROB, agree, student_level="strong", difficulty="easy")
    assert wh.solve_factor > 1.0 > se.solve_factor
    assert wh.reward > g.reward > se.reward, "weak+hard solve > neutral > strong+easy solve"


def test_solve_factor_irrelevant_when_unsolved(agree):
    fhard = score_session(FAILED, FAILED_ANS, FAILED_PROB, agree, student_level="weak", difficulty="hard")
    assert fhard.components["solve_success"] == 0.0


# --- LLM judge parsing + fallback -------------------------------------------------


def test_judge_scores_parse_and_clamp():
    js = _parse_judge_scores('Here you go: {"delta_mastery": 2.0, "progress": 0.5, "overhelp": -3, '
                             '"pedagogical_match": 0.8, "student_solved": 1, "tutor_revealed": 0}')
    assert js.delta_mastery == 1.0 and js.progress == 0.5 and js.overhelp == 0.0
    assert js.pedagogical_match == 0.8 and js.student_solved == 1.0 and js.tutor_revealed == 0.0


def test_positive_judge_raises_reward(agree):
    g = score_session(GOOD, GOOD_ANS, GOOD_PROB, agree)
    judge_ok = LLMJudge(lambda msgs: '{"delta_mastery": 0.5, "progress": 1.0, "overhelp": 0.0, '
                        '"pedagogical_match": 1.0, "student_solved": 1, "tutor_revealed": 0}')
    assert score_session(GOOD, GOOD_ANS, GOOD_PROB, judge_ok).reward > g.reward


def test_malformed_judge_trusts_rule(agree):
    g = score_session(GOOD, GOOD_ANS, GOOD_PROB, agree)
    judge_bad = LLMJudge(lambda msgs: "I think the tutor did a great job!")
    g3 = score_session(GOOD, GOOD_ANS, GOOD_PROB, judge_bad)
    assert g3.solve_gate == 1.0, "garbage judge → gate trusts the rule at full strength"
    # Progress and overhelp cancel at the 0.5 fallback, leaving pedagogical_match on its own
    assert g3.reward == pytest.approx(g.reward + REWARD_WEIGHTS["pedagogical_match"] * 0.5)


# --- verify_answer ----------------------------------------------------------------


def test_small_int_needs_answer_context():
    assert not verify_answer("So we have 4x on the left side.", "4"), "coefficient is not a match"
    assert not verify_answer("Step 4: simplify both sides.", "4"), "step number is not a match"
    assert not verify_answer("Try dividing 30 by 3 now.", "10"), "operands are not a match"
    assert verify_answer("So the answer is 4.", "4")
    assert verify_answer("x = 4", "4")
    assert verify_answer("That gives 10.", "10")
    assert verify_answer("\\boxed{4}", "4")
    assert verify_answer("4", "4"), "bare value as the whole message counts"


def test_digit_boundaries_and_symbolic():
    assert not verify_answer("I got 1200 somehow", "120"), "digit boundaries respected"
    assert verify_answer("the result equals 120", "120")
    assert verify_answer("so it's 3/4 of the total", "3/4"), "symbolic answers keep containment"


def test_bare_latex_gold_dollar_wrap():
    assert verify_answer("So I get \\(\\frac{-4\\sqrt{2}}{9}\\).", "-\\frac{4\\sqrt{2}}{9}")
    assert verify_answer("the area is $\\frac{54\\sqrt{3} a^2}{16}$", "\\frac{27a^2\\sqrt{3}}{8}"), \
        "equivalent unsimplified form"
    assert verify_answer("so the answer is $9a$", "9a"), "bare parse truncated this to 9 before"


def test_echo_exemption(agree):
    cs = score_session(COLD_SOLVE, COLD_ANS, COLD_PROB, agree)
    assert cs.leakage == 0 and cs.solve_success == 1, "confirming a cold solve is not leakage"
