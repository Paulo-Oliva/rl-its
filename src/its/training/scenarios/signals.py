"""
Reward signals.
"""
import re

from its.protocol import verify_answer


# The turns the student actually sees
def _student_facing(transcript: list[dict]) -> list[dict]:
    out = []
    for t in transcript:
        if t["role"] == "tool":
            continue
        if t["role"] == "tutor" and str(t.get("action", "")).startswith("call_"):
            continue
        out.append(t)
    return out


def bad_tool_use(transcript: list[dict]) -> int:
    """
    Counts the bad tool calls, which are the malformed or failed ones plus any duplicate
    repeating a call already made with the same arguments.
    """
    bad = 0
    seen: set[tuple[str, str]] = set()
    last_call_content = ""
    for t in transcript:
        if t.get("role") == "tutor" and str(t.get("action", "")).startswith("call_"):
            last_call_content = str(t.get("content", ""))
            continue
        if t.get("role") != "tool":
            continue
        if t.get("forced"):
            continue
        # Malformed JSON, or a result the tool itself deemed useless
        if not t.get("ok", True):
            bad += 1
            continue
        # The same tool called with the same raw arguments
        key = (str(t.get("name", "")), last_call_content)
        if key in seen:
            bad += 1
        seen.add(key)
    return bad


# Check for random characters (the model sometimes outputs these)
_JUNK_SCRIPT = re.compile(r"[Ѐ-ӿ֐-׿؀-ۿ぀-鿿가-힯]")


def contradicts_verdict(transcript: list[dict]) -> bool:
    valid = None
    for t in transcript:
        if t.get("role") == "tool" and t.get("name") == "call_verifier" and t.get("ok", True):
            v = t.get("data", {}).get("valid")
            if v is not None:
                valid = bool(v)
        elif t.get("role") == "tutor" and valid is not None:
            a = t.get("action")
            if (a == "affirm" and not valid) or (a == "flag_mistake" and valid):
                return True
    return False


def gibberish_score(transcript: list[dict]) -> float:
    text = " ".join(
        str(t.get("content", "")) for t in transcript
        if t.get("role") == "tutor" and not str(t.get("action", "")).startswith("call_"))
    return len(_JUNK_SCRIPT.findall(text)) / len(text) if text else 0.0


def _tool_called(transcript: list[dict], name: str) -> bool:
    return any(t.get("role") == "tutor" and t.get("action") == name for t in transcript)


def _student_reached_answer(transcript: list[dict], answer: str) -> bool:
    return any(t.get("role") == "student" and verify_answer(t.get("content", ""), answer) for t in transcript)


# True if the tutor made a pedagogical move after the last result from tool `name`
def _tutor_move_after_last_tool(transcript: list[dict], name: str) -> bool:
    last = max((i for i, t in enumerate(transcript) if t.get("role") == "tool" and t.get("name") == name), default=-1)
    if last < 0:
        return False
    return any(
        t.get("role") == "tutor" and not str(t.get("action", "")).startswith("call_") for t in transcript[last + 1:])
