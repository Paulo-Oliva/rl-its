"""
TRL hooks for the other scenarios.
"""

import logging
import random
from dataclasses import replace

from its.config import REWARD_WEIGHTS, STUDENT_LEVEL_WEIGHTS
from its.llm import ChatModel
from its.training.reward import Judge, append_jsonl, log_wandb
from its.training.rollout import (Session, build_session_tokens, completion_logprobs)
from its.training.rollout_tools import (Tool, run_scenario_group, run_tutor_session_with_tools)
from its.training.scenarios.base import Scenario
from its.training.scenarios.signals import bad_tool_use, gibberish_score
from its.training.settings import DecodeSettings, SessionSettings

log = logging.getLogger(__name__)


def run_scenario_session(
    scenario: Scenario,
    problem: str,
    answer: str,
    difficulty: str,
    tutor,
    student,
    tools: list[Tool],
    rng: random.Random,
    settings: SessionSettings | None = None,
    force_tool: str | None = None,
) -> tuple[Session, dict]:
    settings = settings if settings is not None else SessionSettings()
    start = scenario.start(problem, answer, difficulty, rng)
    session = run_tutor_session_with_tools(
        problem,
        answer,
        tutor,
        student,
        tools,
        start=start,
        settings=settings.for_scenario(scenario, answer, force_tool),
    )
    return session, start.meta


# --- TRL adapters (T4 rollout + reward) ----------------------------------------


def make_tool_rollout_func(
    student: ChatModel,
    tools: list[Tool],
    scenarios: dict[str, Scenario],
    decode: DecodeSettings | None = None,
    settings: SessionSettings | None = None,
    student_levels: tuple[str, ...] = ("weak", "medium", "strong"),
    use_vllm: bool = False,
    tool_force_prob: float = 0.0,
    batched: bool = False,
):
    decode = decode if decode is not None else DecodeSettings()
    settings = settings if settings is not None else SessionSettings()

    # These reach build_session_tokens too, so the re-templated ids stay aligned
    tool_schemas = [t.tool_schema for t in tools] or None

    def rollout_func(prompts: list, trainer) -> dict:
        model = trainer.model
        tokenizer = trainer.processing_class
        tutor = decode.build_tutor(model, tokenizer, vllm_generation=trainer.vllm_generation if use_vllm else None)

        n_gen = max(1, getattr(trainer, "num_generations", 1) or 1)
        # The level is drawn once per GRPO group, weighted towards the weak student
        level_w = [STUDENT_LEVEL_WEIGHTS.get(l, 1.0) for l in student_levels]
        levels, cur = [], student_levels[0]
        for i in range(len(prompts)):
            if i % n_gen == 0:
                cur = random.choices(student_levels, weights=level_w, k=1)[0]
            levels.append(cur)

        out: dict[str, list] = {
            k: []
            for k in ("prompt_ids", "completion_ids", "logprobs", "env_mask", "answer", "problem", "transcript",
                      "student_level", "difficulty", "scenario", "meta")
        }
        rng = random.Random()

        # The per-prompt fields and the per-session forcing, which are the same either way
        probs = [p["problem"] if isinstance(p, dict) else p for p in prompts]
        anss = [p["answer"] if isinstance(p, dict) else "" for p in prompts]
        diffs = [p.get("difficulty", "medium") if isinstance(p, dict) else "medium" for p in prompts]
        scns = [p.get("scenario", "solve") if isinstance(p, dict) else "solve" for p in prompts]
        scenes = [scenarios.get(s, scenarios["solve"]) for s in scns]
        force_tools = [(sc.force_tool if sc.force_tool and random.random() < tool_force_prob else None)
                       for sc in scenes]

        def emit(session, meta, ans, prob, level, diff, scn):
            st = build_session_tokens(tokenizer, session.tutor_msgs, tools=tool_schemas)
            logps = completion_logprobs(model, st.prompt_ids, st.completion_ids, temperature=decode.temperature)
            out["prompt_ids"].append(st.prompt_ids)
            out["completion_ids"].append(st.completion_ids)
            out["logprobs"].append(logps)
            out["env_mask"].append(st.env_mask)
            out["answer"].append(ans)
            out["problem"].append(prob)
            out["transcript"].append(session.transcript)
            out["student_level"].append(level)
            out["difficulty"].append(diff)
            out["scenario"].append(scn)
            out["meta"].append(meta)

        was_training = model.training
        old_cache = getattr(model.config, "use_cache", True)
        model.eval()
        model.config.use_cache = True
        try:
            if batched:
                results = run_scenario_group(
                    scenes,
                    probs,
                    anss,
                    diffs,
                    levels,
                    force_tools,
                    tutor,
                    student,
                    tools,
                    rng,
                    settings=settings,
                )
                for i, (session, meta) in enumerate(results):
                    emit(session, meta, anss[i], probs[i], levels[i], diffs[i], scns[i])
            else:
                for i in range(len(prompts)):
                    session, meta = run_scenario_session(
                        scenes[i],
                        probs[i],
                        anss[i],
                        diffs[i],
                        tutor,
                        student,
                        tools,
                        rng,
                        settings=replace(settings, student_level=levels[i]),
                        force_tool=force_tools[i],
                    )
                    emit(session, meta, anss[i], probs[i], levels[i], diffs[i], scns[i])
        finally:
            model.config.use_cache = old_cache
            if was_training:
                model.train()
        return out

    return rollout_func


def make_scenario_reward_func(
    judge: Judge | None,
    scenarios: dict[str, Scenario],
    weights: dict | None = None,
    log_file=None,
):
    """
    Sends each session to its own scenario's `score()`, and logs one JSON line per session
    along with the per-scenario mean reward.
    """

    def reward_func(
        completions,
        answer=None,
        transcript=None,
        problem=None,
        student_level=None,
        difficulty=None,
        scenario=None,
        meta=None,
        trainer_state=None,
        **kwargs,
    ) -> list[float]:

        n = len(completions)

        answers = answer or [""] * n
        transcripts = transcript or [[] for _ in range(n)]
        problems = problem or [""] * n
        levels = student_level or [""] * n
        diffs = difficulty or [""] * n
        scens = scenario or ["solve"] * n
        metas = meta or [{} for _ in range(n)]
        step = getattr(trainer_state, "global_step", -1)

        rewards, breakdowns = [], []
        for tr, ans, prob, lvl, diff, scn, mta in zip(transcripts, answers, problems, levels, diffs, scens, metas):
            scenario_obj = scenarios.get(scn, scenarios["solve"])
            b = scenario_obj.score(Session([], tr, prob, ans),
                                   mta or {},
                                   judge,
                                   weights,
                                   student_level=lvl,
                                   difficulty=diff)
            # This is layered on here rather than inside each scenario, so it reaches solve too
            n_bad = bad_tool_use(tr)
            if n_bad:
                w = (weights or REWARD_WEIGHTS)["bad_tool_use"]
                b.reward -= w * n_bad
                b.components["bad_tool_use"] = -w * n_bad
            # A nudge towards the right move, looking only at the tag
            sig = scenario_obj.signature_action
            if sig:
                w = (weights or REWARD_WEIGHTS)["correct_tag"]
                bonus = w if any(t.get("role") == "tutor" and t.get("action") == sig for t in tr) else 0.0
                b.reward += bonus
                b.components["correct_tag"] = bonus
            # Cancels the tool-use credit a garbage session would otherwise keep
            if gibberish_score(tr) > 0.05:
                w = (weights or REWARD_WEIGHTS)["gibberish"]
                b.reward -= w
                b.components["gibberish"] = -w
            rewards.append(b.reward)
            breakdowns.append(b)

        if log_file is not None:
            for tr, ans, prob, lvl, diff, scn, b in zip(transcripts, answers, problems, levels, diffs, scens,
                                                        breakdowns):
                append_jsonl(
                    log_file, {
                        "type": "session",
                        "step": step,
                        "scenario": scn,
                        "problem": prob,
                        "answer": ans,
                        "student_level": lvl,
                        "difficulty": diff,
                        "reward": b.reward,
                        "leakage": b.leakage,
                        "tutor_turns": b.tutor_turns,
                        "components": {
                            k: round(v, 4)
                            for k, v in b.components.items()
                        },
                        "transcript": tr,
                    })

        if rewards:
            metrics = {"reward/mean": sum(rewards) / n}

            for s in set(scens):
                sr = [rw for rw, sc in zip(rewards, scens) if sc == s]
                metrics[f"reward/scenario_{s}"] = sum(sr) / len(sr)
            # Averaged over the sessions where each appears
            comp_sum: dict[str, float] = {}
            comp_cnt: dict[str, int] = {}
            for b in breakdowns:
                for k, v in b.components.items():
                    comp_sum[k] = comp_sum.get(k, 0.0) + v
                    comp_cnt[k] = comp_cnt.get(k, 0) + 1

            for k, tot in comp_sum.items():
                metrics[f"reward/comp_{k}"] = tot / comp_cnt[k]

            # Action distribution plus tool-call volume and failure rate
            act_cnt: dict[str, int] = {}
            tool_calls = tool_fail = n_tutor = 0
            for tr in transcripts:
                for t in tr:
                    if t.get("role") == "tutor":
                        a = str(t.get("action", "?"))
                        act_cnt[a] = act_cnt.get(a, 0) + 1
                        n_tutor += 1
                    elif t.get("role") == "tool":
                        tool_calls += 1
                        tool_fail += 0 if t.get("ok", True) else 1

            for a, c in act_cnt.items():
                metrics[f"actions/{a}"] = c / max(n_tutor, 1)

            metrics["tools/calls_per_session"] = tool_calls / n
            metrics["tools/fail_rate"] = tool_fail / max(tool_calls, 1)
            log_wandb(metrics)
        return rewards

    reward_func.__name__ = "scenario_reward"
    return reward_func
