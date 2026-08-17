import random

from conftest import MATH_KB
from its.training.rollout_tools import ToolContext
from its.training.tools.planner import PlannerTool
from its.training.tools.retriever import RetrieverTool
from its.training.tools.verifier import VerifierTool


# Replies with a fixed string and keeps the messages, so a test can read what was asked
class _StubLLM:

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last: list[dict] = []

    def __call__(self, messages):
        self.last = messages
        return self.reply


def test_verifier_judges_reasoning_validity():
    cx = ToolContext(problem="3x = 30", answer="10")

    valid = VerifierTool(_StubLLM('{"valid": 1, "feedback": "Each step follows correctly."}'))
    r = valid.run({"work": "3x = 30, so dividing both sides by 3 gives x = 10."}, cx)
    assert r.ok and r.data["valid"] is True and "valid" in r.text

    flawed = VerifierTool(_StubLLM('{"valid": 0, "feedback": "Step 2 divides incorrectly."}'))
    r2 = flawed.run({"work": "3x = 30, so x = 90."}, cx)
    assert r2.ok and r2.data["valid"] is False
    assert "flawed" in r2.text and "Step 2" in r2.text


def test_verifier_passes_gold_privately_with_no_reveal_instruction():
    llm = _StubLLM('{"valid": 1, "feedback": "ok"}')
    VerifierTool(llm).run({"work": "x = 10"}, ToolContext(problem="3x=30", answer="10"))
    system = llm.last[0]["content"]
    user = llm.last[1]["content"]
    assert "10" in user and "PRIVATE" in user, "gold answer passed to the model as a private anchor"
    assert "x = 10" in user, "the student's work is included"
    assert "NEVER reveal" in system or "do not reveal" in system.lower()


def test_verifier_missing_work_and_malformed_reply():
    cx = ToolContext(problem="3x=30", answer="10")
    assert VerifierTool(_StubLLM("{}")).run({}, cx).ok is False, "no work + no transcript → ok=False"
    bad = VerifierTool(_StubLLM("not json at all")).run({"work": "x=10"}, cx)
    assert bad.ok is False and bad.text.startswith("error"), "unparseable reply → ok=False"


def test_verifier_falls_back_to_student_turn_when_work_arg_missing():
    cx = ToolContext(
        problem="3x=30",
        answer="10",
        transcript=[{
            "role": "tutor",
            "content": "[ask_question] show me"
        }, {
            "role": "student",
            "content": "I divided both sides by 3 to get x=10"
        }],
    )
    r = VerifierTool(_StubLLM('{"valid": 1, "feedback": "sound"}')).run({}, cx)
    assert r.ok is True and r.data["valid"] is True, "empty work arg → read work from transcript"


def test_planner_level_match_and_no_leak():
    bank = [
        {
            "problem": "easy Q",
            "answer": "1",
            "difficulty": "easy"
        },
        {
            "problem": "hard Q",
            "answer": "\\frac{1}{2}",
            "difficulty": "hard"
        },
    ]
    p = PlannerTool(bank, rng=random.Random(0))
    # the difficulty the tutor asked for
    r_hard = p.run({"difficulty": "hard"}, ToolContext(student_level="weak"))
    assert r_hard.data["problem"] == "hard Q" and r_hard.data["difficulty"] == "hard"
    assert "\\frac{1}{2}" not in r_hard.text, "planner must not show the answer"
    # no difficulty asked for, so the student's level decides
    r_weak = p.run({}, ToolContext(student_level="weak"))
    assert r_weak.data["difficulty"] == "easy" and r_weak.data["problem"] == "easy Q"


def test_retriever_lookup_hits_and_misses():
    rt = RetrieverTool(MATH_KB)
    exact = rt.run({"topic": "the quadratic formula"}, ToolContext())
    assert exact.ok and exact.data["topic"] == "the quadratic formula"
    assert "discriminant" in exact.data["snippet"] and exact.data["snippet"] in exact.text
    assert rt.run({"topic": "pythagorean"}, ToolContext()).data["topic"] == "the pythagorean theorem"
    assert rt.run({
        "topic": "how does completing the square work"
    }, ToolContext()).data["topic"] == "completing the square"
    assert rt.run({"topic": "tensor calculus"}, ToolContext()).ok is False
    assert rt.run({}, ToolContext()).ok is False
