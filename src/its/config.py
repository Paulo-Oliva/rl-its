"""
Configuration stuff for the project.
"""

from pathlib import Path

# Default models to use for the tutor and simulator/judge
TUTOR_MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct"
SIMULATOR_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# Maximum number of turns in a session
MAX_TURNS = 10

# Output directories for the trained models
MODELS_DIR = Path("data/models")
T2_DIR = MODELS_DIR / "t2_sft"
T3_DIR = MODELS_DIR / "t3_grpo"
T4_DIR = MODELS_DIR / "t4_grpo"

# --- T2 SFT ------------------------------------------------------------------

# Where to store SocraTeach
SOCRATEACH_RAW_PATH = Path("data/raw/SocraTeach_multi.json")
# Where to store the preprocessed SocraTeach train/val/test splits
SOCRATEACH_TRAIN_PATH = Path("data/preprocessed/socrateach_train.jsonl")
SOCRATEACH_VAL_PATH = Path("data/preprocessed/socrateach_val.jsonl")
SOCRATEACH_TEST_PATH = Path("data/preprocessed/socrateach_test.jsonl")
# Custom eval samples for verifying during training
EVAL_SAMPLES_PATH = Path("data/eval_samples.jsonl")

# --- T3 GRPO -----------------------------------------------------------------

# RL start model for T3/T4 (the SFT checkpoint from T2)
RL_START_MODEL_ID = "Paulo-Oliva/its-t2-sft"
# Problem bank (generated from Big-Math using the `bigmath_preprocessing` script).
PROBLEMS_PATH = Path("data/preprocessed/problems.jsonl")
# The actions that the tutor can take
TUTOR_ACTIONS: dict[str, str] = {
    "ask_question": "ask a guiding Socratic question that prompts the student's own reasoning",
    "give_hint": "give a minimal concrete hint about the next step, only when the student is stuck",
    "check_understanding": "ask the student to explain or justify reasoning they have already given",
    "flag_mistake":
    "the student has made a mistake - point it out and probe it, and do NOT use this when their work is correct",
    "affirm": "the student's reasoning is correct or on track - affirm what they got right, then prompt the next step",
}

# Reward values for the various signals that the tutor receives from the environment
REWARD_WEIGHTS = {
    # Rule-based weights
    "solve_success": 3.0,
    "leakage": 2.0,
    "turn_cost": 0.2,
    # Judge-based weights
    "delta_mastery": 2.0,
    "progress": 0.8,
    "overhelp": 0.8,
    "pedagogical_match": 0.3,
    # T4-only weights
    "bad_tool_use": 0.5,
    "correct_tag": 0.5,
    "gibberish": 1.0,
    "verdict_contradiction": 1.5,
}

# Student ability reward scaling
SOLVE_LEVEL_FACTOR = {"strong": 0.8, "medium": 1.0, "weak": 1.25}
# Problem difficulty reward scaling
SOLVE_DIFFICULTY_FACTOR = {"easy": 0.8, "medium": 1.0, "hard": 1.25}

# Probabilities for each student ability level, used to sample a student for each session.
# Weak students are sampled more often to ensure that the tutor learns to help them, since they are the most challenging.
STUDENT_LEVEL_WEIGHTS = {"weak": 0.4, "medium": 0.3, "strong": 0.3}

# --- T4 GRPO -----------------------------------------------------------------

# Where the math knowledge base is stored (from Wikipedia using the
# `kb_preprocessing` script).
MATH_KB_PATH = Path("data/preprocessed/math_kb.jsonl")

# Scenario-specific action vocabularies
EXPLAIN_ACTIONS: dict[str, str] = {
    "explain": "explain the concept clearly and correctly, grounded in the reference material you looked up",
    "check_understanding": TUTOR_ACTIONS["check_understanding"],
    "ask_question": TUTOR_ACTIONS["ask_question"],
}
REQUEST_ACTIONS: dict[str, str] = {
    "present_problem": "state the practice problem you fetched, in full, for the student to attempt",
    "give_hint": TUTOR_ACTIONS["give_hint"],
    "ask_question": TUTOR_ACTIONS["ask_question"],
}
