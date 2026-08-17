"""
Builds the T4 Retriever's knowledge base from Wikipedia.

For a curated list of school and early-college math concepts, we fetch the lead paragraph of
the matching Wikipedia article through the public REST summary API, trim it to a few
self-contained sentences, and save it as JSON Lines. The Retriever serves those snippets, so
the tutor can ground an explanation in reference text instead of reciting from memory.

Wikipedia content is CC BY-SA, and this is a derived dataset of short extracts.

    uv run kb              # build data/preprocessed/math_kb.jsonl
    uv run kb --max 20     # a smoke test over the first 20 concepts
"""

import argparse
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from its.config import MATH_KB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
# Wikipedia asks for a descriptive User-Agent; plain urllib UAs are often rejected.
_UA = "RL-ITS-thesis/1.0 (master-thesis Socratic tutor; educational use)"
# Snippets stay focused, since the faithfulness judge reads the whole thing.
_MAX_SENTENCES = 3
_MAX_CHARS = 600

# Curated Wikipedia article titles spanning the math a Socratic tutor would explain.
# The article TEXT is the external source; we only choose which concepts to include.
CONCEPTS: list[str] = [
    # arithmetic & number sense
    "Prime number", "Composite number", "Divisor", "Multiple (mathematics)",
    "Greatest common divisor", "Least common multiple", "Integer", "Rational number",
    "Irrational number", "Absolute value", "Order of operations", "Fraction",
    "Decimal", "Percentage", "Ratio", "Proportionality (mathematics)", "Exponentiation",
    "Square root", "Scientific notation", "Rounding",
    # number theory
    "Prime factorization", "Fundamental theorem of arithmetic", "Euclidean algorithm",
    "Modular arithmetic", "Parity (mathematics)", "Perfect number", "Divisibility rule",
    # algebra
    "Variable (mathematics)", "Algebraic expression", "Linear equation",
    "Quadratic equation", "Quadratic formula", "Completing the square",
    "Difference of two squares", "Factorization", "Polynomial", "Monomial",
    "System of linear equations", "Inequality (mathematics)", "Function (mathematics)",
    "Domain of a function", "Range of a function", "Slope", "Y-intercept", "Logarithm",
    "Exponential function", "Arithmetic progression", "Geometric progression",
    "Binomial theorem", "Distributive property", "Associative property",
    "Commutative property", "Identity element", "Inverse function",
    # geometry
    "Pythagorean theorem", "Triangle", "Right triangle", "Equilateral triangle",
    "Isosceles triangle", "Similarity (geometry)", "Congruence (geometry)",
    "Triangle inequality", "Area", "Perimeter", "Circle", "Circumference", "Radius",
    "Diameter", "Pi", "Angle", "Parallel (geometry)", "Perpendicular", "Polygon",
    "Quadrilateral", "Rectangle", "Square", "Parallelogram", "Trapezoid", "Rhombus",
    "Volume", "Surface area", "Sphere", "Cylinder", "Cone", "Prism (geometry)",
    "Cartesian coordinate system", "Euclidean distance", "Midpoint",
    "Angle bisector", "Vertical angles", "Complementary angles", "Supplementary angles",
    # trigonometry
    "Trigonometric functions", "Sine and cosine", "Tangent", "Law of sines",
    "Law of cosines", "Unit circle", "Radian", "Degree (angle)",
    # probability & statistics
    "Probability", "Mean", "Median", "Mode (statistics)", "Range (statistics)",
    "Standard deviation", "Variance", "Combination", "Permutation", "Factorial",
    "Expected value", "Histogram", "Probability distribution",
    # precalculus / calculus intro
    "Sequence", "Series (mathematics)", "Limit (mathematics)", "Derivative",
    "Integral", "Continuous function", "Slope field", "Asymptote",
    "Maxima and minima", "Rate of change",
    # sets & logic basics
    "Set (mathematics)", "Union (set theory)", "Intersection (set theory)",
    "Subset", "Venn diagram", "Mathematical induction",
]


def _fetch(title: str, retries: int = 4) -> str | None:
    """Return the plain-text lead extract for a Wikipedia title, or None on
    failure / disambiguation / empty article. Retries with exponential backoff
    on HTTP 429 (the summary endpoint rate-limits)."""
    url = _API + urllib.parse.quote(title.replace(" ", "_"), safe="")
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.load(resp)
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                # Backs off over 2, 4 and 8 seconds.
                wait = 2.0 * (2 ** attempt)
                log.info("  429 on %r - backing off %.0fs", title, wait)
                time.sleep(wait)
                continue
            log.warning("  fetch failed for %r: %s", title, e)
            return None
        except (urllib.error.URLError, TimeoutError) as e:
            log.warning("  fetch failed for %r: %s", title, e)
            return None
    else:
        return None
    if data.get("type") == "disambiguation":
        log.warning("  skipping disambiguation page %r", title)
        return None
    return (data.get("extract") or "").strip() or None


# Zero-width / invisible formatting chars Wikipedia leaves behind where inline math
# was stripped (ZWSP, ZWNJ/ZWJ, directional marks, word-joiner/invisible ops, BOM).
_INVISIBLE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")


def _clean(concept: str, extract: str) -> str:
    """Trim an extract to a few self-contained sentences. Strips the invisible chars
    Wikipedia leaves where inline math was removed, and tidies the resulting spacing
    (e.g. ' as where the variable  rep' -> ' as where the variable rep')."""
    text = _INVISIBLE.sub("", extract)
    # Removes the space before punctuation that stripping the math leaves behind.
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?]) ", text)
    snippet = " ".join(sentences[:_MAX_SENTENCES]).strip()
    if len(snippet) > _MAX_CHARS:
        snippet = snippet[:_MAX_CHARS].rsplit(" ", 1)[0] + "…"
    return snippet


def _concept_name(title: str) -> str:
    """Human-readable concept key: drop the disambiguation parenthetical, lowercase."""
    return re.sub(r"\s*\([^)]*\)", "", title).strip().lower()


def build(max_concepts: int = -1, delay: float = 0.5) -> list[dict]:
    titles = CONCEPTS if max_concepts < 0 else CONCEPTS[:max_concepts]
    rows, seen = [], set()
    for i, title in enumerate(titles, 1):
        extract = _fetch(title)
        if not extract:
            continue
        concept = _concept_name(title)
        if concept in seen:
            continue
        seen.add(concept)
        rows.append({"concept": concept, "snippet": _clean(concept, extract)})
        if i % 25 == 0:
            log.info("  fetched %d/%d ...", i, len(titles))
        # Polite to the API.
        time.sleep(delay)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the math knowledge base from Wikipedia.")
    ap.add_argument("--max", type=int, default=-1, help="limit to the first N concepts (smoke test)")
    ap.add_argument("--output", type=str, default=str(MATH_KB_PATH))
    args = ap.parse_args()

    attempted = len(CONCEPTS) if args.max < 0 else min(args.max, len(CONCEPTS))
    log.info("Building math KB from %d curated concepts ...", attempted)
    rows = build(args.max)
    from pathlib import Path
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Wrote %d concept snippets → %s (%d attempted, %d skipped/failed)",
             len(rows), out, attempted, attempted - len(rows))


if __name__ == "__main__":
    main()
