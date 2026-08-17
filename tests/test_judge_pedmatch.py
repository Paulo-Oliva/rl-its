import os
import statistics

import pytest
import torch

JUDGE_MODEL = os.environ.get("ITS_JUDGE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")

# Three problems of the same shape, so one tutor move fits all of them
PROBLEMS = [("3x = 30", "10"), ("5x = 45", "9"), ("4x = 20", "5")]

GOOD_Q = "What operation would undo the multiplication so you can isolate x?"
BAD_Q = "By the way, what did you have for breakfast this morning?"
GOOD_INSTRUCTION = "Divide both sides of the equation by the coefficient of x."

# The 2x2 probe: the tag either matches the move or not, and the move is either relevant or not
CELLS = {
    "clean+good": ("ask_question", GOOD_Q),
    "clean+bad": ("ask_question", BAD_Q),
    "mismatch+good": ("ask_question", GOOD_INSTRUCTION),
    "mismatch+bad": ("give_hint", BAD_Q),
}


def _transcript(action, content):
    return [
        {"role": "student", "content": "I'm stuck, I don't know how to solve for x."},
        {"role": "tutor", "action": action, "content": f"[{action}] {content}"},
        {"role": "student", "content": "Okay, let me think about that."},
    ]


@pytest.fixture(scope="session")
def judge():
    from its.llm import load_model
    from its.training.reward import LLMJudge

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return LLMJudge(load_model(JUDGE_MODEL, device, max_new_tokens=120))


@pytest.mark.judge
def test_pedmatch_tracks_plumbing_or_pedagogy(judge):
    pm = {}
    for label, (action, content) in CELLS.items():
        scores = [judge.score(_transcript(action, content), prob, ans).pedagogical_match for prob, ans in PROBLEMS]
        pm[label] = statistics.mean(scores)
        assert all(0.0 <= s <= 1.0 for s in scores), "pedMatch out of range"

    tag_effect = (pm["clean+good"] + pm["clean+bad"]) / 2 - (pm["mismatch+good"] + pm["mismatch+bad"]) / 2
    move_effect = (pm["clean+good"] + pm["mismatch+good"]) / 2 - (pm["clean+bad"] + pm["mismatch+bad"]) / 2

    if tag_effect > move_effect + 0.1:
        verdict = "tracks PLUMBING (tag-consistency) far more than pedagogy"
    elif move_effect > tag_effect + 0.1:
        verdict = "tracks PEDAGOGY more than tag-consistency"
    else:
        verdict = "responds to both / inconclusive"

    print(f"\n--- pedMatch probe (judge={JUDGE_MODEL}) ---")
    print(f"{'':14s} {'clean tag':>11s} {'mismatch tag':>13s}")
    print(f"{'good move':14s} {pm['clean+good']:>11.2f} {pm['mismatch+good']:>13.2f}")
    print(f"{'bad move':14s} {pm['clean+bad']:>11.2f} {pm['mismatch+bad']:>13.2f}")
    print(f"\ntag_effect  (clean - mismatch) = {tag_effect:+.3f}")
    print(f"move_effect (good  - bad)      = {move_effect:+.3f}")
    print(f"VERDICT: pedMatch {verdict}")
