"""
The session-level reward GRPO trains against.

One scalar comes out of each finished session:

    R = 3*Solve + 2*DeltaMastery - 2*Leakage
      + 0.8*Progress - 0.2*TurnCount - 0.8*Overhelp + 0.3*PedMatch

Solve, Leakage and TurnCount are rule-based, and the other four come from an LLM judge in one call per session on the
frozen student-simulator model.
Solve and Leakage are also cross-examined against the judge's own verdicts and gated by whether the two agree.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from its._jsonparse import lenient_json
from its.config import (REWARD_WEIGHTS, SOLVE_DIFFICULTY_FACTOR, SOLVE_LEVEL_FACTOR)
from its.protocol import count_tutor_turns, detect_leakage, detect_solve

log = logging.getLogger(__name__)

# --- Rule-based session signals --------------------------------------------------


def count_hints(transcript: list[dict]) -> int:
    return sum(1 for t in transcript if t["role"] == "tutor" and t.get("action") == "give_hint")


# --- LLM-judge interface ---------------------------------------------------------


@dataclass
class JudgeScores:
    """
    What one judge call returns for a session.
    """
    # Mastery after the session minus before it, in [-1, 1]
    delta_mastery: float = 0.0
    # The fraction of tutor turns that moved the student forward, in [0, 1]
    progress: float = 0.0
    # How much of the work the tutor did for the student, in [0, 1]
    overhelp: float = 0.0
    # The fraction of turns whose action fitted what the student needed, in [0, 1]
    pedagogical_match: float = 0.0
    # The judge's own read on the two rule-based facts, or None when the call failed
    student_solved: float | None = 0.0
    tutor_revealed: float | None = 0.0


class Judge(ABC):
    """
    Scores the judge-derived reward signals from a session transcript.
    """

    @abstractmethod
    def score(self, transcript: list[dict], problem: str, answer: str) -> JudgeScores:
        ...

    def relay_check(self, transcript: list[dict], problem: str, answer: str, criterion: str) -> float:
        """1.0 if `criterion` is satisfied, which is how the scenarios ask whether the tutor actually
        used a tool's result rather than merely calling it. This default fails closed, and
        LLMJudge overrides it."""
        return 0.0


_JUDGE_SYSTEM = """\
You are a STRICT, critical evaluator of Socratic math tutoring. Most tutoring is mediocre - \
keep scores low by default and reserve the top of each scale for genuinely excellent work. \
Judge the TUTOR's contribution, NOT the student's innate ability: if the student basically \
solved it themselves, the tutor deserves little credit no matter how the dialogue looks. \
When a session falls between two anchors, pick the LOWER one.

You will be shown a math problem, its correct answer, and a tutoring dialogue. Tutor turns are \
tagged with the action the tutor chose, e.g. "[give_hint] ...".

Reply with ONLY a JSON object, no other text, with exactly these keys:
{"delta_mastery": <float>, "progress": <float>, "overhelp": <float>, "pedagogical_match": <float>, \
"student_solved": <0 or 1>, "tutor_revealed": <0 or 1>}

Score each field by choosing the closest anchor:

delta_mastery - how much the student's OWN understanding grew BECAUSE OF the tutoring:
  -1.0 = the tutor confused the student or led them toward a wrong method
   0.0 = the student already knew how - the tutor taught nothing new
   0.5 = the student gained partial new understanding with the tutor's help
   1.0 = a genuinely stuck or confused student reached real understanding through the tutor's guidance

progress - fraction of tutor turns that ACTUALLY moved the student forward because of the tutor \
(a turn that only validates or cheers - "Good, keep going!" - while the student does the real work \
does NOT count):
   0.0 = no turn produced a real step
   0.5 = about half the turns produced a step the student would not have taken alone
   1.0 = every tutor turn produced such a step

overhelp - how much the tutor did the work FOR the student:
   0.0 = always elicited the student's own reasoning
   0.5 = gave away some key steps
   1.0 = essentially solved the problem for them

pedagogical_match - fraction of tutor turns whose tagged action was the RIGHT move for that moment \
AND whose content matches the tag (a poorly-timed move, a generic/unhelpful question, or content \
that doesn't match its tag - e.g. a hint tagged [ask_question] - does NOT count):
   0.0 = no turn was a well-matched, tag-consistent move
   0.5 = about half the turns were
   1.0 = every turn was the right move and matched its tag

student_solved (0 or 1): 1 ONLY if the STUDENT themselves clearly stated the correct final answer \
by the end. An answer the TUTOR supplied, a muddled near-miss, or the wrong quantity does NOT count. \
When unsure, answer 0.

tutor_revealed (0 or 1): 1 if the TUTOR gave away the correct final answer (or all but the last \
trivial step) BEFORE the student had reached it. Confirming an answer the student already stated is \
NOT revealing. When unsure, answer 0.\
"""


# The clamped signals from a judge reply, raising on anything malformed
def _parse_judge_scores(text: str) -> JudgeScores:
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m is None:
        raise ValueError(f"no JSON object in judge reply: {text[:120]!r}")
    # Parse the json
    raw = lenient_json(m.group(0))
    if raw is None:
        raise ValueError(f"unparseable JSON object in judge reply: {text[:120]!r}")

    def clamp(key: str, lo: float, hi: float) -> float:
        return min(hi, max(lo, float(raw.get(key, 0.0))))

    return JudgeScores(
        delta_mastery=clamp("delta_mastery", -1.0, 1.0),
        progress=clamp("progress", 0.0, 1.0),
        overhelp=clamp("overhelp", 0.0, 1.0),
        pedagogical_match=clamp("pedagogical_match", 0.0, 1.0),
        student_solved=clamp("student_solved", 0.0, 1.0),
        tutor_revealed=clamp("tutor_revealed", 0.0, 1.0),
    )


_RELAY_JUDGE_SYSTEM = """\
You are a strict evaluator checking ONE factual yes/no question about a math tutoring \
dialogue. You will be shown the problem, its correct answer, a specific CRITERION, and the \
dialogue. Judge ONLY whether the criterion is satisfied - ignore overall tutoring quality.

Reply with ONLY a JSON object, no other text:
{"satisfied": 0 or 1}

Answer 1 only if the criterion is clearly and fully satisfied. When unsure, answer 0.\
"""


class LLMJudge(Judge):
    """
    Scores sessions with the frozen student-simulator model under a judge prompt, in one
    greedy call per session after the session ends, so the two roles never contend.
    """

    def __init__(self, llm) -> None:
        # Anything callable that takes a list of message dicts and returns a string
        self.llm = llm

    @staticmethod
    def _dialogue(transcript: list[dict]) -> str:
        return "\n".join(f"{turn['role'].upper()}: {turn['content']}" for turn in transcript)

    def score(self, transcript: list[dict], problem: str, answer: str) -> JudgeScores:
        user = (f"[Problem] {problem}\n[Correct answer] {answer}\n\n"
                f"[Dialogue]\n{self._dialogue(transcript)}")
        try:
            reply = self.llm([
                {
                    "role": "system",
                    "content": _JUDGE_SYSTEM
                },
                {
                    "role": "user",
                    "content": user
                },
            ])
            return _parse_judge_scores(reply)
        # A judge failure must never kill a training step
        except Exception as e:
            log.warning("LLM judge failed (%s), so pedagogy is neutral and the gate falls back "
                        "to the rule", e)
            return JudgeScores(progress=0.5,
                               overhelp=0.5,
                               pedagogical_match=0.5,
                               student_solved=None,
                               tutor_revealed=None)

    def relay_check(self, transcript, problem, answer, criterion: str) -> float:
        """
        1.0 if `criterion` holds for this session, which is how the scenarios check that a tool's
        result was actually relayed rather than merely fetched.

        Args:
            transcript: The session transcript.
            problem: The problem text.
            answer: The gold answer.
            criterion (str): The yes-or-no question put to the judge.

        Returns:
            float: 1.0 if satisfied, otherwise 0.0.
        """
        user = (f"[Problem] {problem}\n[Correct answer] {answer}\n\n"
                f"[Criterion] {criterion}\n\n[Dialogue]\n{self._dialogue(transcript)}")
        try:
            reply = self.llm([
                {
                    "role": "system",
                    "content": _RELAY_JUDGE_SYSTEM
                },
                {
                    "role": "user",
                    "content": user
                },
            ])
            m = re.search(r"\{.*?\}", reply, re.DOTALL)
            if m is None:
                return 0.0
            obj = lenient_json(m.group(0))
            return 1.0 if obj and int(obj.get("satisfied", 0)) == 1 else 0.0
        except Exception as e:
            log.warning("Relay-check judge failed (%s), scoring 0.0", e)
            return 0.0


# --- Reward aggregation ----------------------------------------------------------

# The credit multiplier for a rule check, indexed by (rule says yes, judge says yes)
AGREEMENT_GATE = {
    (True, True): 1.0,
    (True, False): 0.4,
    (False, True): 0.3,
    (False, False): 0.0,
}


# The credit multiplier from cross-examining a rule check against the judge
def _agreement_gate(rule: bool, judge: float | None) -> float:
    if judge is None:
        return 1.0 if rule else 0.0
    return AGREEMENT_GATE[(bool(rule), judge >= 0.5)]


# The ability and difficulty multiplier on solve credit, or 1.0 for an unknown label
def _solve_factor(student_level: str, difficulty: str) -> float:
    return (SOLVE_LEVEL_FACTOR.get(student_level, 1.0) * SOLVE_DIFFICULTY_FACTOR.get(difficulty, 1.0))


@dataclass
class RewardBreakdown:
    reward: float
    solve_success: int
    leakage: int
    tutor_turns: int
    hints: int = 0
    # The agreement multiplier applied to the solve credit
    solve_gate: float = 0.0
    # The agreement multiplier applied to the leakage penalty
    leak_gate: float = 0.0
    # The ability and difficulty weighting on the solve credit
    solve_factor: float = 1.0
    judge: JudgeScores = field(default_factory=JudgeScores)
    # Each term's signed contribution, which sums back to the reward
    components: dict[str, float] = field(default_factory=dict)


def score_session(
    transcript: list[dict],
    answer: str,
    problem: str,
    judge: Judge,
    weights: dict[str, float] | None = None,
    student_level: str = "",
    difficulty: str = "",
) -> RewardBreakdown:
    """
    The reward for one finished session, with every term kept separately.

    Args:
        transcript (list[dict]): The session transcript.
        answer (str): The gold answer.
        problem (str): The problem text, which the judge is shown.
        judge (Judge): Scores the four pedagogy signals and the two cross-examination flags.
        weights (dict[str, float] | None, optional): Per-term weights. Defaults to None, which
            uses config.REWARD_WEIGHTS.
        student_level (str, optional): The student's ability, which weights solve credit.
            Defaults to "".
        difficulty (str, optional): The problem's difficulty, which also weights solve credit.
            Defaults to "".

    Returns:
        RewardBreakdown: The scalar reward plus every component that went into it.
    """
    w = weights or REWARD_WEIGHTS

    solve = detect_solve(transcript, answer)
    leak = detect_leakage(transcript, answer)
    turns = count_tutor_turns(transcript)
    hints = count_hints(transcript)

    js = judge.score(transcript, problem, answer)

    # A leaked answer voids the solve credit regardless of what the gate says
    solve_gate = _agreement_gate(bool(solve), js.student_solved)
    if leak:
        solve_gate = 0.0
    solve_factor = _solve_factor(student_level, difficulty)
    leak_gate = _agreement_gate(bool(leak), js.tutor_revealed)

    # Logged but not charged, since Overhelp already covers doing too much
    components = {
        "solve_success": w["solve_success"] * solve_gate * solve_factor,
        "delta_mastery": w["delta_mastery"] * js.delta_mastery,
        "leakage": -w["leakage"] * leak_gate,
        "progress": w["progress"] * js.progress,
        "turn_cost": -w["turn_cost"] * turns,
        "overhelp": -w["overhelp"] * js.overhelp,
        "pedagogical_match": w["pedagogical_match"] * js.pedagogical_match,
    }
    return RewardBreakdown(
        reward=float(sum(components.values())),
        solve_success=int(solve and not leak),
        leakage=leak,
        tutor_turns=turns,
        hints=hints,
        solve_gate=solve_gate,
        leak_gate=leak_gate,
        solve_factor=solve_factor,
        judge=js,
        components=components,
    )


# --- Logging helpers -------------------------------------------------------------


def append_jsonl(path, record: dict) -> None:
    """
    Append one JSON line to `path`.

    Args:
        path: The file to append to.
        record (dict): The record to write.
    """
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("Could not append to %s (%s)", path, e)


def log_wandb(metrics: dict) -> None:
    """
    Log metrics to the active wandb run.

    Args:
        metrics (dict): The metrics to log, committed with the next step.
    """
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, commit=False)
    except Exception:
        pass


# The batch means of every reward component, plus the tutor's action distribution, for logging
def _component_metrics(breakdowns: list[RewardBreakdown], transcripts: list[list[dict]]) -> dict:
    n = len(breakdowns)
    metrics = {
        "reward/solve_rate": sum(b.solve_success for b in breakdowns) / n,
        "reward/leakage_rate": sum(b.leakage for b in breakdowns) / n,
        "reward/tutor_turns": sum(b.tutor_turns for b in breakdowns) / n,
        "reward/hints": sum(b.hints for b in breakdowns) / n,
        "reward/judge_delta_mastery": sum(b.judge.delta_mastery for b in breakdowns) / n,
        "reward/judge_progress": sum(b.judge.progress for b in breakdowns) / n,
        "reward/judge_overhelp": sum(b.judge.overhelp for b in breakdowns) / n,
        "reward/judge_ped_match": sum(b.judge.pedagogical_match for b in breakdowns) / n,
        # The disagreement rate is a live read on how often verify_answer and the judge differ
        "reward/solve_gate": sum(b.solve_gate for b in breakdowns) / n,
        "reward/leak_gate": sum(b.leak_gate for b in breakdowns) / n,
        "reward/solve_disagree_rate": sum(b.solve_gate in (0.4, 0.3) for b in breakdowns) / n,
    }
    for term in breakdowns[0].components:
        metrics[f"contrib/{term}"] = sum(b.components.get(term, 0.0) for b in breakdowns) / n
    actions = [t.get("action") for tr in transcripts for t in tr if t["role"] == "tutor"]
    actions = [a for a in actions if a]
    for name in set(actions):
        metrics[f"actions/{name}"] = actions.count(name) / len(actions)
    return metrics


# --- TRL reward_func adapter -----------------------------------------------------


def make_reward_func(judge: Judge, weights: dict[str, float] | None = None, log_file=None):
    """
    Builds a reward_func for GRPOTrainer.
    """

    def reward_func(completions,
                    answer=None,
                    transcript=None,
                    problem=None,
                    student_level=None,
                    difficulty=None,
                    trainer_state=None,
                    **kwargs) -> list[float]:
        n = len(completions)
        answers = answer or [""] * n
        transcripts = transcript or [[] for _ in range(n)]
        problems = problem or [""] * n
        levels = student_level or [""] * n
        difficulties = difficulty or [""] * n
        step = getattr(trainer_state, "global_step", -1)

        rewards, breakdowns = [], []
        for tr, ans, prob, lvl, diff in zip(transcripts, answers, problems, levels, difficulties):
            b = score_session(tr, ans, prob, judge, weights, student_level=lvl, difficulty=diff)
            breakdowns.append(b)
            rewards.append(b.reward)

        if log_file is not None:
            for tr, ans, prob, lvl, diff, b in zip(transcripts, answers, problems, levels, difficulties, breakdowns):
                append_jsonl(
                    log_file,
                    {
                        "type": "session",
                        "step": step,
                        "problem": prob,
                        "answer": ans,
                        "student_level": lvl,
                        "difficulty": diff,
                        "reward": b.reward,
                        "solve": b.solve_success,
                        "leakage": b.leakage,
                        "tutor_turns": b.tutor_turns,
                        "hints": b.hints,
                        "solve_gate": b.solve_gate,
                        "leak_gate": b.leak_gate,
                        "solve_factor": b.solve_factor,
                        "judge": {
                            "delta_mastery": b.judge.delta_mastery,
                            "progress": b.judge.progress,
                            "overhelp": b.judge.overhelp,
                            "pedagogical_match": b.judge.pedagogical_match,
                            "student_solved": b.judge.student_solved,
                            "tutor_revealed": b.judge.tutor_revealed
                        },
                        # Each term's signed weighted contribution, summing back to the reward
                        "components": {
                            k: round(v, 4)
                            for k, v in b.components.items()
                        },
                        "transcript": tr,
                    })
        if breakdowns:
            log_wandb(_component_metrics(breakdowns, transcripts))
        return rewards

    reward_func.__name__ = "session_reward"
    return reward_func
