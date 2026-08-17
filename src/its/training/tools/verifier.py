"""
The Verifier, which checks whether the student's reasoning holds up rather than only whether
the final number matches.
"""

import logging
import re

from its._jsonparse import lenient_json
from its.prompts import VERIFIER_SYSTEM
from its.training.tools.base import Tool, ToolContext, ToolResult

log = logging.getLogger(__name__)


class VerifierTool(Tool):
    """
    Checks whether the student's reasoning is valid, rather than only whether the final answer
    matches the gold, with one model call over the worked steps.
    """

    name = "call_verifier"
    description = "Check whether the student's mathematical reasoning/steps are valid."
    parameters = {
        "type": "object",
        "properties": {
            "work": {
                "type": "string",
                "description": "the student's worked steps to check"
            }
        },
        "required": ["work"],
    }

    def __init__(self, llm) -> None:
        # Anything callable that takes a list of message dicts and returns a string
        self.llm = llm

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        work = str(args.get("work", "")).strip()
        if not work:
            # Falls back to the student's last message
            work = next(
                (str(t.get("content", "")).strip() for t in reversed(ctx.transcript) if t.get("role") == "student"), "")

        if not work:
            return ToolResult('error: provide the student\'s work to check in {"work": "..."}', ok=False)

        user = (f"[Problem] {ctx.problem}\n"
                f"[Correct answer - PRIVATE, do not reveal] {ctx.answer}\n\n"
                f"[Student's work]\n{work}")

        try:
            reply = self.llm([
                {
                    "role": "system",
                    "content": VERIFIER_SYSTEM
                },
                {
                    "role": "user",
                    "content": user
                },
            ])
            m = re.search(r"\{.*\}", reply, re.DOTALL)

            obj = lenient_json(m.group(0)) if m else None
            if obj is None:
                raise ValueError(f"no parseable JSON in verifier reply: {reply[:120]!r}")

            valid = bool(int(obj.get("valid", 0)))
            feedback = str(obj.get("feedback", "")).strip()

        except Exception as e:
            log.warning("Verifier LLM call failed (%s)", e)
            return ToolResult("error: could not verify the reasoning", ok=False)

        verdict = "valid" if valid else "flawed"

        return ToolResult(
            text=f"The student's reasoning appears {verdict}. {feedback}".strip(),
            data={
                "valid": valid,
                "feedback": feedback
            },
        )
