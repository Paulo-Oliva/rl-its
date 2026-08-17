"""
Downloads and converts SocraTeach_multi.json for the SFT stage.

The split follows the original SocraTeach authors, at
https://github.com/Ljyustc/SocraticLM/blob/main/codes/data_preprocess.py. We shuffle the
dialogue keys under a fixed seed, take the first 100 as test, indices 1000 to 2000 as
validation and everything past 2000 as train, dropping any train dialogue whose problem also
appears in test or validation so that no problem leaks across the splits.

Each dialogue becomes a chat-template conversation whose system message embeds the problem and
its answer. The structure is inspired by the SocraTeach paper, using a shortened form of our
own tutor instructions, and the analysis is left out to match the T1 inference setup, which
only injects the answer.

    uv run socrateach
"""

import json
import logging
import random
import urllib.request
from pathlib import Path

from tqdm import tqdm

from its.config import (
    SOCRATEACH_RAW_PATH,
    SOCRATEACH_TEST_PATH,
    SOCRATEACH_TRAIN_PATH,
    SOCRATEACH_VAL_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_RAW_URL = "https://raw.githubusercontent.com/Ljyustc/SocraticLM/main/data/SocraTeach_multi.json"
_SEED = 42
_N_TEST = 100
_VAL_START, _VAL_END = 1000, 2000

_TUTOR_SYSTEM = (
    "You are a Socratic math tutor. Guide the student to the answer through "
    "questions - never reveal it directly. Acknowledge their attempt, ask a "
    "reflective question, and only give a minimal hint if they're clearly stuck."
)


def _system_prompt(question: str, answer: str) -> str:
    return f"{_TUTOR_SYSTEM}\n\n[Problem] {question}\n[Answer] {answer}"


def download_raw() -> Path:
    if SOCRATEACH_RAW_PATH.exists():
        log.info("Raw file already cached at %s, skipping download.", SOCRATEACH_RAW_PATH)
        return SOCRATEACH_RAW_PATH
    SOCRATEACH_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading SocraTeach_multi.json from GitHub ...")
    try:
        urllib.request.urlretrieve(_RAW_URL, SOCRATEACH_RAW_PATH)
    except Exception as e:
        SOCRATEACH_RAW_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download failed: {e}\n"
            "If the file is stored in Git LFS, clone the repo manually:\n"
            "  git clone https://github.com/Ljyustc/SocraticLM /tmp/SocraticLM\n"
            f"  cp /tmp/SocraticLM/data/SocraTeach_multi.json {SOCRATEACH_RAW_PATH}"
        ) from e
    log.info("Downloaded → %s (%.1f MB)", SOCRATEACH_RAW_PATH, SOCRATEACH_RAW_PATH.stat().st_size / 1e6)
    return SOCRATEACH_RAW_PATH


def _build_messages(question: str, answer: str, turns: list[dict]) -> list[dict] | None:
    messages: list[dict] = [{"role": "system", "content": _system_prompt(question, answer)}]
    for turn in turns:
        teacher = turn.get("system", "").strip()
        student = turn.get("user", "").strip()
        if not teacher:
            continue
        messages.append({"role": "assistant", "content": teacher})
        if student:
            messages.append({"role": "user", "content": student})
    if not any(m["role"] == "assistant" for m in messages):
        return None
    return messages


def _problem_id(dialogue_key: str) -> str:
    """A dialogue key looks like 'GSM8K_train_0_2'; the problem ID is the prefix
    with the last underscore-segment stripped: 'GSM8K_train_0'."""
    return "_".join(dialogue_key.split("_")[:-1])


def parse_and_split(raw_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    log.info("Parsing %s ...", raw_path)
    data = json.loads(raw_path.read_text(encoding="utf-8"))

    # Flatten: dialogue_key -> {problem_id, messages}
    entries: dict[str, dict] = {}
    skipped = 0
    for prob_data in tqdm(data.values(), desc="Problems"):
        question = prob_data.get("question", "").strip()
        answer = str(prob_data.get("answer", "")).strip()
        if not question or not answer:
            continue
        for dialogue_key, turns in prob_data.get("dialogues", {}).items():
            messages = _build_messages(question, answer, turns)
            if messages is None:
                skipped += 1
                continue
            entries[dialogue_key] = {
                "problem_id": _problem_id(dialogue_key),
                "messages": messages,
            }
    log.info("Parsed %d dialogues, %d skipped malformed.", len(entries), skipped)

    # Shuffle and split - same logic as the SocraTeach authors.
    keys = list(entries.keys())
    random.Random(_SEED).shuffle(keys)
    test_keys = keys[:_N_TEST]
    val_keys = keys[_VAL_START:_VAL_END]
    held_out_problems = {entries[k]["problem_id"] for k in test_keys + val_keys}
    train_keys = [k for k in keys[_VAL_END:] if entries[k]["problem_id"] not in held_out_problems]

    log.info("Split: %d train / %d val / %d test (%d held-out problems)",
             len(train_keys), len(val_keys), len(test_keys), len(held_out_problems))

    def records(ks: list[str]) -> list[dict]:
        return [{"messages": entries[k]["messages"]} for k in ks]

    return records(train_keys), records(val_keys), records(test_keys)


def save(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    log.info("Saved %d records → %s", len(records), path)


def main() -> None:
    raw_path = download_raw()
    train, val, test = parse_and_split(raw_path)
    save(train, SOCRATEACH_TRAIN_PATH)
    save(val, SOCRATEACH_VAL_PATH)
    # Nothing in the project reads the test split. It is written for anyone who wants a
    # held-out perplexity number for the SFT stage, since our own comparison of the four
    # tutors is session-based rather than loss-based.
    save(test, SOCRATEACH_TEST_PATH)
    log.info("Done.")


if __name__ == "__main__":
    main()
