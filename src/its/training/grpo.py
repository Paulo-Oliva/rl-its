"""
T3 GRPO, which RL fine-tunes the tutor on top of the T2 SFT checkpoint with a LoRA adapter.
One GRPO completion is a whole tutor and student-simulator session, a group is --group-size sessions on the same
problem, and the reward is the session score from reward.py.
With --tools the tutor also gets the three frozen tools and the scenario mix, which is T4.

Run `uv run grpo --help` for the options.
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import torch
from datasets import Dataset

from its.config import (MATH_KB_PATH, MAX_TURNS, PROBLEMS_PATH, RL_START_MODEL_ID, SIMULATOR_MODEL_ID, T3_DIR, T4_DIR,
                        TUTOR_ACTIONS)
from its.llm import HFChatModel
from its.training.reward import LLMJudge, make_reward_func
from its.training.rollout import make_rollout_func
from its.training.scenarios.kinds import build_scenarios
from its.training.scenarios.rollout import (make_scenario_reward_func, make_tool_rollout_func)
from its.training.settings import DecodeSettings, SessionSettings
from its.training.tools.kb import load_math_kb
from its.training.tools.planner import PlannerTool, load_problem_bank
from its.training.tools.retriever import RetrieverTool
from its.training.tools.verifier import VerifierTool

log = logging.getLogger(__name__)

# --- Data ------------------------------------------------------------------------


def load_problems(path: Path, max_examples: int = -1, scenarios: list[str] | None = None, seed: int = 0) -> Dataset:
    """
    Read the problem bank into the dataset that GRPO trains on.

    Every problem becomes one row whose prompt is a dict instead of text, so TRL hands it
    straight to the rollout_func rather than running a chat template over it.
    The scenario is drawn once per row, so all the sessions in a GRPO group share it.

    Args:
        path (Path): Problem bank JSONL, one problem per line.
        max_examples (int, optional): Stop after this many problems, for smoke tests.
            Defaults to -1, which reads the whole bank.
        scenarios (list[str] | None, optional): The T4 scenario mix to draw from.
            Defaults to None, which is T3 where every session is a solve.
        seed (int, optional): Seed for the scenario draw. Defaults to 0.

    Returns:
        Dataset: One row per problem, with a single prompt column.
    """
    rng = random.Random(seed)
    rows = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line:
            continue
        record = json.loads(line)
        # For injecting in the prompt
        prompt = {
            "problem": record["problem"],
            "answer": str(record["answer"]),
            "difficulty": record.get("difficulty", "medium"),
        }

        if scenarios:
            prompt["scenario"] = rng.choice(scenarios)

        rows.append({"prompt": prompt})

        if max_examples != -1 and len(rows) >= max_examples:
            break

    log.info("Loaded %d problems from %s%s", len(rows), path,
             f" (scenario mix: {sorted(set(scenarios))})" if scenarios else "")
    return Dataset.from_list(rows)


# --- Entry point ---------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training for the RL tutor (T3, or T4 with --tools)")
    p.add_argument("--model",
                   default=RL_START_MODEL_ID,
                   help=f"Policy start checkpoint, local dir or HF repo ID (default: {RL_START_MODEL_ID})")
    p.add_argument("--student-model", default=SIMULATOR_MODEL_ID, help="Student simulator model")
    p.add_argument("--student-device", default="auto", help='Device map for the student/judge model (e.g. "cuda:1")')
    p.add_argument("--problems", type=Path, default=PROBLEMS_PATH, help="Problem bank JSONL")
    p.add_argument("--output",
                   type=Path,
                   default=None,
                   help=f"Checkpoint dir (default: {T3_DIR}, or {T4_DIR} with --tools)")
    p.add_argument("--group-size", type=int, default=8, help="Sessions per problem (GRPO group size)")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.04, help="KL coefficient to the frozen T2 reference")
    p.add_argument("--max-turns", type=int, default=MAX_TURNS)
    p.add_argument("--max-new-tokens", type=int, default=256)

    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--action-temp",
                   type=float,
                   default=None,
                   help="Temperature for the action draw on its own (default: --temperature)")
    p.add_argument("--length-norm-actions",
                   action="store_true",
                   help="Score action tags by their mean per-token logprob instead of the sum, "
                   "which drops the length bias towards short tags")
    p.add_argument("--batch-size",
                   type=int,
                   default=None,
                   help="Sessions per device per step, a multiple of --group-size (default: group-size)")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--save-total-limit", type=int, default=5, help="Max checkpoints kept on disk (oldest deleted)")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-examples", type=int, default=-1)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--no-actions",
                   action="store_true",
                   help="Disable the discrete action space (free-form tutor turns, no tags)")
    p.add_argument("--tools",
                   action="store_true",
                   help="T4: give the tutor the frozen tools (Verifier, Planner, Retriever) and the "
                   "scenario mix. Needs the action space, so it ignores --no-actions.")
    p.add_argument("--tool-force-prob",
                   type=float,
                   default=0.0,
                   help="Probability that a non-solve session forces its scenario's tool call, to "
                   "get past the cold start where the tutor calls nothing and every tool-gated "
                   "reward is 0. Around 0.5 keeps a call/no-call contrast inside the group. "
                   "Needs --tools.")
    p.add_argument("--vllm",
                   action="store_true",
                   help="Generate the tutor turns with a vLLM engine colocated on the training GPU")
    p.add_argument("--vllm-gpu-mem", type=float, default=0.3, help="GPU memory fraction for the colocated vLLM engine")
    p.add_argument("--batched-rollout",
                   action="store_true",
                   help="Advance the whole rollout batch in lockstep, one batched call per turn, "
                   "which pairs well with --vllm")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Train into --output even if it already holds a run")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    if args.output is None:
        args.output = T4_DIR if args.tools else T3_DIR

    if args.output.exists() and any(args.output.iterdir()) and not (args.resume or args.overwrite):
        raise SystemExit(f"{args.output} is not empty. Pass --resume to continue that run, "
                         f"--overwrite to replace it, or --output to train somewhere else.")

    if args.tools and not MATH_KB_PATH.exists():
        raise SystemExit(f"{MATH_KB_PATH} is missing. Build the Retriever knowledge base with `uv run kb`.")
    return args


def _resolve_model_path(model: str) -> str:
    path = Path(model)
    return str(path.resolve()) if path.exists() else model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    args = _parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(args.output / "train.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))

    logging.getLogger().addHandler(file_handler)
    log.info("Policy start checkpoint: %s  ->  output: %s", args.model, args.output)
    session_log = args.output / "sessions.jsonl"
    log.info("Logging to %s and %s", args.output / "train.log", session_log)

    # To reduce import cost
    from peft import LoraConfig
    from peft.utils import save_and_load as _peft_sl
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.trainer.grpo_config import GRPOConfig
    from trl.trainer.grpo_trainer import GRPOTrainer

    # Patch because of vllm dependency constraints
    _orig_shard = _peft_sl._maybe_shard_state_dict_for_tp

    def _safe_shard(model, state_dict, adapter_name):
        try:
            return _orig_shard(model, state_dict, adapter_name)
        except ImportError:
            # Training runs on a single GPU, so there is no tensor parallelism to shard for
            return state_dict

    _peft_sl._maybe_shard_state_dict_for_tp = _safe_shard

    if not args.no_wandb:
        os.environ["WANDB_PROJECT"] = "ITS GRPO"

    model_id = _resolve_model_path(args.model)
    log.info("Loading policy (T2) from %s ...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

    log.info("Loading student simulator %s (device_map=%s) ...", args.student_model, args.student_device)
    student_tok = AutoTokenizer.from_pretrained(args.student_model)
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        dtype="auto",
        device_map=args.student_device,
    )
    student_model.eval()
    student = HFChatModel(model=student_model, tokenizer=student_tok, max_new_tokens=args.max_new_tokens)

    T4_SCENARIOS = ["solve", "check_my_work", "request_problem", "explain_concept"]
    train_ds = load_problems(args.problems, args.max_examples, scenarios=T4_SCENARIOS if args.tools else None)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Without --batch-size the batch is exactly one problem, so one whole group
    per_device = args.batch_size if args.batch_size is not None else args.group_size
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    vllm_kwargs: dict = {}
    if args.vllm:
        vllm_kwargs = {"use_vllm": True, "vllm_mode": "colocate", "vllm_gpu_memory_utilization": args.vllm_gpu_mem}

    cfg = GRPOConfig(
        output_dir=str(args.output),
        num_generations=args.group_size,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        epsilon=0.2,
        temperature=args.temperature,
        top_p=args.top_p,
        # Give enough space for the whole session
        max_completion_length=args.max_new_tokens * args.max_turns * 3,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps,
        num_train_epochs=1 if args.max_steps == -1 else 1,
        report_to="none" if args.no_wandb else "wandb",
        # Streams the tutor sessions and their rewards to wandb
        log_completions=not args.no_wandb,
        num_completions_to_print=args.group_size,
        **vllm_kwargs,
    )

    # The judge borrows the student instance with a judge system prompt, and only runs once the
    # session is over, so the two never generate at the same time
    judge = LLMJudge(student)

    decode = DecodeSettings(temperature=args.temperature,
                            top_p=args.top_p,
                            max_new_tokens=args.max_new_tokens,
                            action_temperature=args.action_temp,
                            length_norm_actions=args.length_norm_actions)

    if args.tools:
        kb = load_math_kb(MATH_KB_PATH)
        log.info("Retriever KB: %d concepts from %s", len(kb), MATH_KB_PATH)
        tools = [
            VerifierTool(student),
            PlannerTool(load_problem_bank(args.problems), rng=random.Random(0)),
            RetrieverTool(kb)
        ]
        scenarios = build_scenarios(kb)

        log.info("T4 tools enabled: %s", [t.name for t in tools])
        rollout_func = make_tool_rollout_func(
            student=student,
            tools=tools,
            scenarios=scenarios,
            decode=decode,
            settings=SessionSettings(max_turns=args.max_turns, ped_actions=TUTOR_ACTIONS),
            use_vllm=bool(args.vllm),
            tool_force_prob=args.tool_force_prob,
            batched=args.batched_rollout,
        )

        log.info("T4 tool-force probability: %.2f (batched=%s)", args.tool_force_prob, args.batched_rollout)
        reward_func = make_scenario_reward_func(judge=judge, scenarios=scenarios, log_file=session_log)
    else:
        rollout_func = make_rollout_func(
            student=student,
            decode=decode,
            max_turns=args.max_turns,
            actions=None if args.no_actions else TUTOR_ACTIONS,
            use_vllm=bool(args.vllm),
            batched=args.batched_rollout,
        )
        reward_func = make_reward_func(judge=judge, log_file=session_log)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_func],  # type: ignore
        args=cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=lora,
        rollout_func=rollout_func,
    )

    resume = args.resume and any(args.output.glob("checkpoint-*"))
    log.info(
        "Training T3 ... (group=%d, beta=%.3f, lr=%.0e, actions=%s, vllm=%s, batched=%s, resume=%s)",
        args.group_size,
        args.beta,
        args.lr,
        not args.no_actions,
        args.vllm or "off",
        args.batched_rollout,
        resume,
    )
    trainer.train(resume_from_checkpoint=resume)

    log.info("Saving LoRA adapter → %s", args.output)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    log.info("Done.")


if __name__ == "__main__":
    main()
