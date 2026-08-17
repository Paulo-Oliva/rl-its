"""
Lenient JSON parsing for LLM output.

A model writing JSON about maths breaks strict json.loads in several ways at once, with LaTeX
backslashes, bare keys, single quotes and trailing commas.
"""

import ast
import json
import re
import warnings
from collections.abc import Callable
from typing import Any


def _as_dict(parser: Callable[[str], Any], s: str) -> dict[str, Any] | None:
    try:
        obj = parser(s)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def lenient_json(blob: str) -> dict[str, Any] | None:
    """
    Parse a JSON-ish object string into a dict, or None if nothing yields one.

    Each candidate gets strict JSON first, then a relaxed pass that quotes bare keys, swaps
    single quotes for double and drops trailing commas, and finally a Python-literal eval.

    Args:
        blob (str): The candidate JSON object, as the model wrote it.

    Returns:
        dict[str, Any] | None: The parsed object, or None when nothing produced a dict.
    """
    candidates = [blob]
    if "\\" in blob:
        candidates.insert(0, re.sub(r'\\(?!["\\])', r'\\\\', blob))
    for cand in candidates:
        d = _as_dict(json.loads, cand)
        if d is not None:
            return d
        relaxed = re.sub(r'(?<=[{,])(\s*)([A-Za-z_]\w*)(\s*):', r'\1"\2":', cand)
        relaxed = relaxed.replace("'", '"')
        relaxed = re.sub(r',(\s*[}\]])', r'\1', relaxed)
        d = _as_dict(json.loads, relaxed)
        if d is not None:
            return d
    # A Python literal tolerates more backslashes than JSON, warning on the unknown escapes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return _as_dict(ast.literal_eval, blob)
