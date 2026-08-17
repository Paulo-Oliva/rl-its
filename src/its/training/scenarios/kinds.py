"""
The four scenarios. `solve` is the plain tutoring episode, and the other three make a tool the
actual task by changing how the session opens.
"""

from its.config import EXPLAIN_ACTIONS, REQUEST_ACTIONS, REWARD_WEIGHTS
from its.protocol import count_tutor_turns, detect_leakage
from its.training.reward import RewardBreakdown, score_session
from its.training.scenarios.base import Scenario
from its.training.scenarios.prompts import (CHECK_TUTOR_SYSTEM, EXPLAIN_TUTOR_SYSTEM, REQUEST_TUTOR_SYSTEM)
from its.training.scenarios.signals import (_student_facing, _student_reached_answer, _tool_called,
                                            _tutor_move_after_last_tool, contradicts_verdict)
from its.training.settings import ScenarioStart


class SolveScenario(Scenario):
    name = "solve"

    def start(self, problem, answer, difficulty, rng) -> ScenarioStart:
        # A cold reply, the solve ends it, and the tutor is shown the answer
        return ScenarioStart()

    def score(self, session, meta, judge, weights=None, student_level="", difficulty="") -> RewardBreakdown:
        # Unlike the tool scenarios, every term here comes from the judge, so it is required
        if judge is None:
            raise ValueError("the solve scenario needs a judge")
        return score_session(session.transcript,
                             session.answer,
                             session.problem,
                             judge,
                             weights,
                             student_level=student_level,
                             difficulty=difficulty)

    def is_complete(self, transcript, answer) -> bool:
        return _student_reached_answer(transcript, answer)


class CheckMyWorkScenario(Scenario):
    """
    The student shows their working and the tutor has to check the reasoning. The tutor is not
    given the gold answer, so it cannot judge correctness itself and has to call the verifier,
    which is the oracle here.

    There are two rewards, one for making the call and one for relaying what came back.
    """

    name = "check_my_work"
    force_tool = "call_verifier"
    VERIFIER_REWARD = 1.0
    RELAY_REWARD = 1.0
    LEAKAGE_PENALTY = 2.0

    def start(self, problem, answer, difficulty, rng) -> ScenarioStart:
        # The student asks for a check, which is the cue to call the verifier
        directive = ("Attempt this problem yourself and show your full working step by step, "
                     "then ask the tutor to check whether your reasoning is correct. "
                     "Do not ask the tutor to solve it for you.")
        # The tutor still needs the problem text, framed as checking rather than solving
        return ScenarioStart(opening=None,
                             solve_terminates=False,
                             tutor_sees_answer=False,
                             student_directive=directive,
                             tutor_system=CHECK_TUTOR_SYSTEM,
                             meta={})

    def score(self, session, meta, judge, weights=None, student_level="", difficulty="") -> RewardBreakdown:
        used_verifier = any(t["role"] == "tutor" and t.get("action") == "call_verifier" for t in session.transcript)
        leak = detect_leakage(_student_facing(session.transcript), session.answer)
        components = {
            "verifier_used": self.VERIFIER_REWARD * (1.0 if used_verifier else 0.0),
            "leakage": -self.LEAKAGE_PENALTY * leak,
        }
        if judge is not None and used_verifier:
            findings = " ".join(
                t.get("data", {}).get("feedback") or t.get("content", "") for t in session.transcript
                if t["role"] == "tool" and t.get("name") == "call_verifier" and t.get("ok", True))
            if findings:
                criterion = (f"The student showed their working and the tutor used a reasoning-checking tool, "
                             f"which reported: \"{findings}\". Did the tutor accurately convey this assessment to "
                             f"the student - telling them whether their reasoning was sound and, if not, where it "
                             f"went wrong - rather than ignoring it, contradicting it, or staying vague?")
                components["verdict_relayed"] = self.RELAY_REWARD * judge.relay_check(
                    _student_facing(session.transcript), session.problem, session.answer, criterion)
        # Charged above the call credit, so contradicting the verifier is net negative
        if contradicts_verdict(session.transcript):
            w = (weights or REWARD_WEIGHTS)["verdict_contradiction"]
            components["verdict_contradiction"] = -w
        return RewardBreakdown(
            reward=float(sum(components.values())),
            solve_success=0,
            leakage=leak,
            tutor_turns=count_tutor_turns(session.transcript),
            components=components,
        )

    def is_complete(self, transcript, answer) -> bool:
        # Checked, and the student has reached the answer
        return _tool_called(transcript, "call_verifier") and _student_reached_answer(transcript, answer)


class RequestProblemScenario(Scenario):
    """
    The student asks for a practice problem. The tutor cannot conjure a real one out of the
    bank, so it has to call the planner, and the dataset problem is hidden since it is not the
    task here.

    Two rewards again, one for the call and one for actually putting a concrete problem in
    front of the student rather than inventing one or just chatting.
    """

    name = "request_problem"
    force_tool = "call_planner"
    actions = REQUEST_ACTIONS
    signature_action = "present_problem"
    PLANNER_REWARD = 1.0
    PRESENT_REWARD = 1.0

    def start(self, problem, answer, difficulty, rng) -> ScenarioStart:
        opening = "I'd like to practice. Can you give me a problem to work on?"
        # Hide the dataset problem and frame the tutor around fetch-and-present
        return ScenarioStart(opening=opening,
                             solve_terminates=False,
                             tutor_sees_answer=False,
                             tutor_system=REQUEST_TUTOR_SYSTEM,
                             show_problem=False,
                             meta={})

    def score(self, session, meta, judge, weights=None, student_level="", difficulty="") -> RewardBreakdown:
        used_planner = any(t["role"] == "tutor" and t.get("action") == "call_planner" for t in session.transcript)
        components = {"planner_used": self.PLANNER_REWARD * (1.0 if used_planner else 0.0)}
        if judge is not None and used_planner:
            criterion = ("The student asked for a practice problem. Did the tutor actually present a "
                         "concrete, complete math problem for the student to work on (a specific "
                         "problem statement), rather than not giving one or only making small talk?")
            components["problem_presented"] = self.PRESENT_REWARD * judge.relay_check(
                _student_facing(session.transcript), session.problem, session.answer, criterion)
        return RewardBreakdown(
            reward=float(sum(components.values())),
            solve_success=0,
            leakage=0,
            tutor_turns=count_tutor_turns(session.transcript),
            components=components,
        )

    def is_complete(self, transcript, answer) -> bool:
        # Done once the planner has delivered and the tutor has presented the problem
        return _tutor_move_after_last_tool(transcript, "call_planner")


class ExplainConceptScenario(Scenario):
    """
    The student asks for an explanation of a concept. Rather than wing it from memory, the
    tutor should retrieve reference material and explain from that.

    One reward is for the call and one for staying faithful to what came back.
    """

    name = "explain_concept"
    force_tool = "call_retriever"
    actions = EXPLAIN_ACTIONS
    signature_action = "explain"
    RETRIEVER_REWARD = 1.0
    FAITHFUL_REWARD = 1.0

    def __init__(self, kb: dict[str, str]) -> None:
        # Has to be the same kb the Retriever serves, or the tutor is asked the unlookable
        self.kb = kb

    def start(self, problem, answer, difficulty, rng) -> ScenarioStart:
        concept = rng.choice(list(self.kb))
        opening = f"Can you explain {concept} to me?"
        # The concept is the task, so the problem is hidden and the framing overridden
        return ScenarioStart(opening=opening,
                             solve_terminates=False,
                             tutor_sees_answer=False,
                             tutor_system=EXPLAIN_TUTOR_SYSTEM,
                             show_problem=False,
                             meta={"concept": concept})

    def score(self, session, meta, judge, weights=None, student_level="", difficulty="") -> RewardBreakdown:
        used_retriever = any(t["role"] == "tutor" and t.get("action") == "call_retriever" for t in session.transcript)
        components = {"retriever_used": self.RETRIEVER_REWARD * (1.0 if used_retriever else 0.0)}
        if judge is not None and used_retriever:
            snippets = [
                t.get("data", {}).get("snippet") or t.get("content", "") for t in session.transcript
                if t["role"] == "tool" and t.get("name") == "call_retriever" and t.get("ok", True)
            ]
            reference = " ".join(s for s in snippets if s)
            # This is only empty if every retrieval failed, which bad tool use has charged already
            if reference:
                criterion = (f"The student asked the tutor to explain a concept. The reference material the "
                             f"tutor retrieved says: \"{reference}\". Did the tutor's explanation stay faithful "
                             f"to this reference - conveying it accurately, without contradicting it or making "
                             f"up facts it does not support?")
                components["explanation_faithful"] = self.FAITHFUL_REWARD * judge.relay_check(
                    _student_facing(session.transcript), session.problem, session.answer, criterion)
        return RewardBreakdown(
            reward=float(sum(components.values())),
            solve_success=0,
            leakage=0,
            tutor_turns=count_tutor_turns(session.transcript),
            components=components,
        )

    def is_complete(self, transcript, answer) -> bool:
        # Done once the retriever has delivered and the tutor has explained the concept
        return _tutor_move_after_last_tool(transcript, "call_retriever")


def build_scenarios(kb: dict[str, str]) -> dict[str, Scenario]:
    return {
        s.name: s
        for s in [SolveScenario(),
                  CheckMyWorkScenario(),
                  RequestProblemScenario(),
                  ExplainConceptScenario(kb)]
    }
