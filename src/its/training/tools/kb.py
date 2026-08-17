"""
Loader for the Retriever's knowledge base.
"""

import json
from pathlib import Path


def load_math_kb(path: str | Path) -> dict[str, str]:
    """Load a {concept: snippet} KB from a JSONL file (built by `uv run kb`)."""
    kb: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            kb[str(r["concept"]).lower()] = r["snippet"]
    return kb
