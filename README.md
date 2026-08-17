# SAGE - Socratic Actions with Group-relative policy optimisation for Education

Master thesis project. A Socratic maths tutor trained with reinforcement learning.

A tutor here is a language model that talks a student through a maths problem, asking questions
and offering hints along the way. It is told what the answer is, and it never gets to say it.
There are four tutors, from a plainly prompted model up to one that has been through
reinforcement learning and can call tools, and they are compared on whole tutoring sessions
rather than on a test loss. SAGE is the method behind the two reinforcement-learning stages,
T3 and T4.

| Tutor | Description        | Training  | Checkpoint                      |
| ----- | ------------------ | --------- | ------------------------------- |
| T1    | Prompt-only tutor  | none      | `Qwen/Qwen2.5-Math-7B-Instruct` |
| T2    | SFT tutor          | SFT       | `Paulo-Oliva/its-t2-sft`        |
| T3    | RL tutor, no tools | GRPO+LoRA | `Paulo-Oliva/its-t3-grpo`       |
| T4    | RL tutor + tools   | GRPO+LoRA | `Paulo-Oliva/its-t4-grpo`       |

Both T3 and T4 are trained on top of T2.

## Setup

Everything here runs through [uv](https://docs.astral.sh/uv/), so install that first and then
let it build the environment.

```bash
uv sync
```

## Data

To train the tutors, some preprocessing is needed.

```bash
uv run socrateach     # SocraTeach -> train/val/test dialogue splits (T2 only)
uv run preprocess     # BigMath -> data/preprocessed/problems.jsonl (T3/T4 only)
uv run kb             # Wikipedia -> the Retriever knowledge base (T4 only)
```

## Training


```bash
uv run sft                                        # T2: full fine-tune on SocraTeach
uv run grpo                                       # T3: GRPO + LoRA on multi-turn sessions
uv run grpo --tools                               # T4: the same, with tools and scenarios
```

To train a model to use socratic actions, run the `grpo` command with the `--model` argument set to a T2 checkpoint, e.g. `Paulo-Oliva/its-t2-sft`. You can also specify a different student model with `--student-model`. (Note that this was only tested with Qwen2.5-Math-7B-Instruct)

```bash
# Example: a smoke test of T3 with only two sessions, two turns each, and one GRPO step.
uv run grpo --model Qwen/Qwen2.5-0.5B-Instruct --student-model Qwen/Qwen2.5-0.5B-Instruct \
            --max-examples 2 --group-size 2 --max-turns 2 --max-steps 1 --no-wandb
```

## Evaluation

You can run the evaluation harness on your own checkpoints, or on the four tutors above. The
evaluation is a three-stage process: first the tutors are rolled out on a held-out set of problems, then the transcripts are scored by a panel of judges, and finally the scores are aggregated into a report with figures.

The external judges need API keys in your environment, e.g. `export OPENAI_API_KEY=...` or `export ANTHROPIC_API_KEY=...`.

```bash
# 0. Build the held-out set: every problem seen in training is subtracted from the bank,
#    then sampled stratified by difficulty. All four tutors share this exact set.
uv run eval-build-set --n-problems 200 --out data/eval/problems_eval.jsonl

# 1. Generate transcripts, 
#    --mode must match how the checkpoint was trained: plain (T1/T2), actions (T3), tools (T4).
uv run eval-sessions --label T2 --model Paulo-Oliva/its-t2-sft --mode plain
uv run eval-sessions --label T3 --model <t3-ckpt> --mode actions --base-model Paulo-Oliva/its-t2-sft

# 2. Score the same transcripts with each judge. Backends: hf: / openai: / anthropic:.
uv run eval-judge --judge openai:gpt-4o

# 3. Aggregate: rule metrics with bootstrap CIs, rubric means, rankings, inter-judge agreement.
uv run eval-report          # -> data/eval/session_report.md, summary.json, merged.csv
uv run eval-figures         # -> data/eval/figures/

uv run eval-chat --model <ckpt> --mode tools   # play the student yourself
```




## Layout

```
src/its/
├── config.py       # Model IDs, paths, shared constants
├── llm.py          # ChatModel interface + load_model (HFChatModel / VLLMChatModel)
├── prompts.py      # Tutor, student and tool system prompts
├── protocol.py     # Action tags, tool-call format, answer checking
├── training/
│   ├── sft.py      # T2
│   ├── grpo.py     # T3 and T4
│   ├── rollout.py  # Session rollout -> tokens + tutor-only loss mask
│   ├── reward.py   # Session reward: rule-based signals + LLM judge
│   ├── tools/      # Verifier / Planner / Retriever
│   └── scenarios/  # The four T4 scenarios and their rewards
├── evaluation/     # Session harness, judge panel, report, figures
└── scripts/        # Dataset preprocessing
```


## License

MIT - see [LICENSE](LICENSE).
