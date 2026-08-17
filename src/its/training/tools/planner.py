"""
The Planner, which fetches a practice problem from the bank matched to the student.
"""

import json
import logging
import random
from pathlib import Path

from its.training.tools.base import Tool, ToolContext, ToolResult

log = logging.getLogger(__name__)

_LEVEL_TO_DIFFICULTY = {"weak": "easy", "medium": "medium", "strong": "hard"}


class PlannerTool(Tool):
    """
    Returns a practice problem from the bank, matched to the student's level.

    It samples `problems.jsonl` by difficulty, taking the tutor's requested difficulty when
    that is valid and otherwise mapping from the level drawn for this session.
    """

    name = "call_planner"
    description = "Fetch a practice problem from the problem bank, matched to the student."
    parameters = {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "optional topic to focus on"},
            "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
        },
        "required": [],
    }

    def __init__(self, bank: list[dict], rng: random.Random | None = None) -> None:
        self.bank = bank
        self.rng = rng or random.Random()

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        diff = str(args.get("difficulty", "")).lower().strip()
        if diff not in ("easy", "medium", "hard"):
            diff = _LEVEL_TO_DIFFICULTY.get(ctx.student_level, "medium")
        candidates = [p for p in self.bank if p.get("difficulty") == diff] or self.bank
        if not candidates:
            return ToolResult("error: no problems available", ok=False)
        p = self.rng.choice(candidates)
        return ToolResult(
            # The answer is withheld here, so presenting this cannot leak it
            text=f"Practice problem ({diff}): {p['problem']}",
            data={"problem": p["problem"], "answer": str(p.get("answer", "")), "difficulty": diff},
        )


def load_problem_bank(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            rows.append({"problem": r["problem"], "answer": str(r.get("answer", "")),
                         "difficulty": r.get("difficulty", "medium")})
    return rows
