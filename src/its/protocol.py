"""
The protocol for ITS tutoring turns, including action tags, tool calls, and answer checking.
"""

import re
from collections.abc import Mapping

from math_verify import parse, verify

from its._jsonparse import lenient_json
from its.config import EXPLAIN_ACTIONS, REQUEST_ACTIONS, TUTOR_ACTIONS

# --- Action tags ------------------------------------------------------------------

# Every tag any tutor can emit, across all four scenarios
ALL_TUTOR_ACTIONS: dict[str, str] = {**TUTOR_ACTIONS, **EXPLAIN_ACTIONS, **REQUEST_ACTIONS}
ALL_ACTION_NAMES: tuple[str, ...] = tuple(ALL_TUTOR_ACTIONS)


def action_instructions(actions: Mapping[str, str]) -> str:
    """The action menu appended to the tutor's system prompt, one tag per line."""
    lines = "\n".join(f"  [{name}] - {desc}" for name, desc in actions.items())
    primary = next(iter(actions))
    return ("\n\nStart your reply with exactly ONE action tag, then a short message (2-4 sentences), "
            "then STOP. Do NOT output more than one tag, and do NOT list several actions. Your tag must "
            "match what your message actually does. Allowed actions:\n"
            f"{lines}\n"
            f'Begin with the tag, e.g. "[{primary}] ...".')


def action_tag_regex(actions: Mapping[str, str]) -> re.Pattern:
    """A pattern matching the start of any of these action tags, mangled spellings included."""
    names = "|".join(re.escape(n).replace("_", "[_ ]?") for n in actions)
    return re.compile(rf"\[\s*(?:{names})", re.IGNORECASE)


def strip_extra_tags(message: str, actions: Mapping[str, str]) -> str:
    """Cut a message off at the first following action tag, leaving one action per turn."""
    return action_tag_regex(actions).split(message, maxsplit=1)[0].strip()


def action_stop_strings(actions: Mapping[str, str]) -> list[str]:
    """Stop strings that end generation at the next tag, spellings the tutor mangles included."""
    stops = []
    for n in actions:
        for body in {n, n.replace("_", " "), n.replace("_", "")}:
            stops += [f"[{body}", f"[ {body}"]
    return stops


_LEADING_TAG = re.compile(
    rf"^\s*\[\s*(?:{'|'.join(re.escape(n).replace('_', '[_ ]?') for n in ALL_ACTION_NAMES)})\s*\]\s*",
    re.IGNORECASE,
)


def strip_leading_tag(text: str) -> str:
    """Remove one leading "[action] " tag, which is what the student and the judge never see."""
    return _LEADING_TAG.sub("", text, count=1).strip()


# --- Tool-call protocol -----------------------------------------------------------

TOOL_CALL_STOP = "</tool_call>"

# Tool results go into the tutor's view under the "tool" role
TOOL_RESULT_ROLE = "tool"

# Appended to the system prompt of a tutor that has tools
TOOL_TURN_NOTE = ("\n\nYou also have tools available - their specifications and the required "
                  "<tool_call>...</tool_call> format are given in the # Tools section below. When a turn "
                  "genuinely needs one (to check the student's reasoning, fetch a practice problem, or look "
                  "up reference material), begin with your action tag and then emit the <tool_call> for it "
                  "INSTEAD of stopping. You will get the tool's result back and can then finish your message "
                  "to the student using it. Call a tool only when it is actually needed.")

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>|<tool_call>\s*(\{.*)", re.DOTALL)


def strip_tool_calls(text: str) -> str:
    """Remove every `<tool_call>` block, closed or truncated, from a tutor message."""
    return re.sub(r"<tool_call>.*?(?:</tool_call>|$)", "", text, flags=re.DOTALL).strip()


def ensure_closed(text: str) -> str:
    """Close a `<tool_call>` the tutor left open by stopping or running out of tokens."""
    if "<tool_call>" in text and "</tool_call>" not in text:
        return text + "\n</tool_call>"
    return text


# The arguments dict from the tutor's tool-call text, or None when there is no usable one
def _parse_tool_args(raw: str) -> dict | None:
    if "{" not in raw:
        return None
    # From the first brace to the last, or to the end when nothing closes it
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    blob = m.group(0) if m else raw[raw.index("{"):]

    # The model often under-closes the object, so a brace-balanced retry follows
    missing = blob.count("{") - blob.count("}")
    for candidate in (blob, blob + "}" * missing) if missing > 0 else (blob, ):
        d = lenient_json(candidate)
        if d is not None:
            return d
    return None


def parse_tool_call(text: str) -> tuple[str, dict | None] | None:
    """
    Read (name, args) out of a native `<tool_call>` block, where args is None when the block is
    malformed and the whole result is None when the message contains no tool call.
    """
    m = _TOOL_CALL_RE.search(text)
    if m is None:
        return None
    obj = _parse_tool_args(m.group(1) or m.group(2) or "")
    if not obj or "name" not in obj:
        # It opened a call, but the JSON was unusable
        return "call_invalid", None
    name = str(obj["name"]).strip()
    args = obj.get("arguments")
    if isinstance(args, dict):
        # The base model writes keys with stray spaces and mixed case
        args = {str(k).strip().lower(): v for k, v in args.items()}
        return name, args
    return name, None


# --- Answer checking --------------------------------------------------------------

_TRIVIAL_NUM = re.compile(r"^-?\d{1,2}$")
_PURE_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")
_ANSWER_CUE = r"(?:answer|result|solution|solves?|equals?|gets?|gives?|makes?|is|are|it'?s|that'?s|=|:|\\?boxed\{?)"


# True if `gold` shows up where an answer is being stated, after a cue word or an equals sign
def _stated_value(text: str, gold: str) -> bool:
    if text.strip(" .!?\n\t$") == gold:
        return True
    # Bounded so "10" does not match inside "3.10" or "1200"
    val = rf"(?<![\d.\w-]){re.escape(gold)}(?!\.?\d)"
    return re.search(_ANSWER_CUE + r"[^a-zA-Z0-9]{0,12}" + val, text) is not None


def verify_answer(text: str, answer: str) -> bool:
    """
    True if `text` states the gold answer.

    Tries math-verify first, so an answer written differently still matches, and falls back to
    a bounded substring search when the parser gets nothing usable.

    Args:
        text (str): The turn to check, usually a student reply.
        answer (str): The gold answer from the problem bank.

    Returns:
        bool: True if the answer is stated in the text.
    """
    if not answer or not text:
        return False
    gold = answer.strip().lower()
    text_l = text.strip().lower()

    if _TRIVIAL_NUM.match(gold):
        return _stated_value(text_l, gold)

    # Big-Math golds are bare LaTeX, which parse() drops or truncates unless it is wrapped
    try:
        gold_expr = parse(f"${answer}$") or parse(answer)
        if gold_expr and verify(gold_expr, parse(text)):
            return True
    # Malformed latex
    except Exception:
        pass
    # Don't count a 3 in 3.4 for example
    if _PURE_NUM.match(gold):
        return re.search(rf"(?<![\d.-]){re.escape(gold)}(?!\.?\d)", text_l) is not None
    return gold in text_l


# --- Rule-based session signals ---------------------------------------------------


def detect_leakage(transcript: list[dict], answer: str) -> int:
    """1 if a tutor turn states the answer before the student does, so an echo is exempt."""
    for turn in transcript:
        if turn["role"] == "student" and verify_answer(turn["content"], answer):
            return 0
        if turn["role"] == "tutor" and verify_answer(turn["content"], answer):
            return 1
    return 0


def detect_solve(transcript: list[dict], answer: str) -> int:
    """1 if any student turn states the right answer."""
    for turn in transcript:
        if turn["role"] == "student" and verify_answer(turn["content"], answer):
            return 1
    return 0


def count_tutor_turns(transcript: list[dict]) -> int:
    """How many pedagogical turns the tutor took, not counting the ones spent calling a tool."""
    return sum(1 for t in transcript if t["role"] == "tutor" and not str(t.get("action", "")).startswith("call_"))
