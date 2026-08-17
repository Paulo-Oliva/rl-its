"""
Prepares Big-Math-RL-Verified for use as the training problem pool.

We load the dataset from the Hub, keep the problems with a 1 to 65 percent solve rate so that
they are neither trivial nor hopeless, drop any row missing a problem or an answer, tag each
one with a difficulty bucket, and write the result out as JSON Lines for the training loop to
sample from.

    uv run preprocess
"""

import json
import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# --- Config -------------------------------------------------------------------
HF_DATASET     = "SynthLabsAI/Big-Math-RL-Verified"
OUTPUT_DIR     = Path("data/preprocessed")
SOLVE_RATE_MIN = 0.01
SOLVE_RATE_MAX = 0.65

DIFFICULTY_BINS   = [0.0, 0.20, 0.45, 0.66]
DIFFICULTY_LABELS = ["hard", "medium", "easy"]


def assign_difficulty(solve_rate: float) -> str:
    idx = int(np.digitize(solve_rate, DIFFICULTY_BINS)) - 1
    idx = max(0, min(idx, len(DIFFICULTY_LABELS) - 1))
    return DIFFICULTY_LABELS[idx]


# --- Step 1 - Load ------------------------------------------------------------
def load_bigmath() -> pd.DataFrame:
    log.info("Loading %s ...", HF_DATASET)
    ds = load_dataset(HF_DATASET, split="train")
    # to_pandas can also return an iterator of frames, and masking one returns a union of
    # Series and DataFrame, so the casts below say what a split and a row mask actually give.
    df = cast(pd.DataFrame, ds.to_pandas())
    log.info("Loaded %d rows.", len(df))
    return df


# --- Step 2 - Filter solve rate -----------------------------------------------
def filter_solve_rate(df: pd.DataFrame) -> pd.DataFrame:
    solve_col = next(
        (c for c in ["llama8b_solve_rate", "model_solve_rate", "solve_rate"] if c in df.columns),
        None,
    )
    if solve_col is None:
        log.warning("No solve_rate column found - skipping solve rate filter.")
        return df
    before = len(df)
    df = cast(pd.DataFrame, df[df[solve_col].between(SOLVE_RATE_MIN, SOLVE_RATE_MAX)])
    log.info("Solve rate filter [%.0f%%-%.0f%%]: %d → %d problems",
             SOLVE_RATE_MIN * 100, SOLVE_RATE_MAX * 100, before, len(df))
    return df.reset_index(drop=True)


# --- Step 3 - Drop rows with missing problem/answer --------------------------
def filter_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = cast(pd.DataFrame, df[df["answer"].notna() & (df["answer"].astype(str).str.strip() != "")])
    df = cast(pd.DataFrame, df[df["problem"].notna() & (df["problem"].astype(str).str.strip() != "")])
    log.info("Removed %d problems with missing answer/problem.", before - len(df))
    return df.reset_index(drop=True)


# --- Step 4 - Tag difficulty --------------------------------------------------
def tag_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    solve_col = next(
        (c for c in ["llama8b_solve_rate", "model_solve_rate", "solve_rate"] if c in df.columns),
        None,
    )
    df["difficulty"] = df[solve_col].apply(assign_difficulty) if solve_col else "unknown"
    log.info("Difficulty: %s", df["difficulty"].value_counts().to_dict())
    return df


# --- Step 5 - Save ------------------------------------------------------------
def save(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "problems.jsonl"

    with open(out_path, "w") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Saving"):
            entry = {
                "id":         int(row.name),  # type: ignore[arg-type]
                "problem":    row["problem"],
                "answer":     str(row["answer"]),
                "difficulty": row["difficulty"],
            }
            f.write(json.dumps(entry) + "\n")

    log.info("Saved %d problems → %s", len(df), out_path)


def main() -> None:
    df = load_bigmath()
    df = filter_solve_rate(df)
    df = filter_missing(df)
    df = tag_difficulty(df)
    save(df)
    log.info("Done.")


if __name__ == "__main__":
    main()
