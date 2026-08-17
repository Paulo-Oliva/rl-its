"""
Aggregation, agreement and the written report for the cross-baseline evaluation.

    uv run eval-report

Run `uv run eval-report --help` for the options.
"""

import argparse
import csv
import json
import logging
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from its.evaluation.common import read_jsonl
from its.evaluation.judges import BINARY_DIMS, CONTINUOUS_DIMS

log = logging.getLogger(__name__)

DEFAULT_SESSIONS_DIR = Path("data/eval/sessions")
DEFAULT_SCORES_DIR = Path("data/eval/scores")
DEFAULT_OUT_DIR = Path("data/eval")

# Higher is better for every dimension except these (used for rankings)
LOWER_IS_BETTER = {"overhelp", "leakage_rate", "mean_turns"}


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


# Percentile bootstrap 95% CI of the mean
def bootstrap_ci(xs: Sequence[float], b: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(mean(rng.choices(xs, k=n)) for _ in range(b))
    return means[int(0.025 * b)], means[min(int(0.975 * b), b - 1)]


def cohen_kappa(a: list[int], b: list[int]) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0 - 1e-12:
        return float("nan")
    return (po - pe) / (1 - pe)


def _avg_ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        # The average rank for this block of ties, counting from 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        return float("nan")
    rx, ry = _avg_ranks(x), _avg_ranks(y)
    mx, my = mean(rx), mean(ry)
    sx = math.sqrt(sum((r - mx)**2 for r in rx))
    sy = math.sqrt(sum((r - my)**2 for r in ry))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy)


def krippendorff_alpha(ratings: dict[str, dict[str, float]]) -> float:
    units: dict[str, list[float]] = defaultdict(list)
    for by_item in ratings.values():
        for item, v in by_item.items():
            units[item].append(float(v))
    pairable = {u: vs for u, vs in units.items() if len(vs) >= 2}
    n = sum(len(vs) for vs in pairable.values())
    if n < 2:
        return float("nan")

    do = 0.0
    for vs in pairable.values():
        m = len(vs)
        do += sum((a - b)**2 for i, a in enumerate(vs) for j, b in enumerate(vs) if i != j) / (m - 1)
    do /= n

    all_vals = [v for vs in pairable.values() for v in vs]
    de = sum((a - b)**2 for i, a in enumerate(all_vals) for j, b in enumerate(all_vals) if i != j) / (n * (n - 1))
    if de == 0:
        # Everyone said the same thing everywhere
        return 1.0
    return 1.0 - do / de


def fmt(x: float, nd: int = 3) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


# --- Aggregation -----------------------------------------------------------------


def load_data(sessions_dir: Path, scores_dir: Path):
    sessions: dict[str, list[dict]] = {}
    seen: set[tuple] = set()
    for f in sorted(sessions_dir.glob("*.jsonl")):
        for r in read_jsonl(f):
            key = (r["label"], r["session_id"])
            if key in seen:
                continue
            seen.add(key)
            sessions.setdefault(r["label"], []).append(r)
    # scores[judge][(label, session_id)] = dims dict (parsed-ok rows only)
    scores: dict[str, dict[tuple, dict]] = {}
    for f in sorted(scores_dir.glob("*.jsonl")):
        for r in read_jsonl(f):
            if r.get("ok") and r.get("scores"):
                scores.setdefault(r["judge"], {})[(r["label"], r["session_id"])] = r["scores"]
    return sessions, scores


def rule_table(sessions: dict[str, list[dict]]) -> dict:
    out = {}
    for label, rows in sessions.items():
        solves = [r["solve"] for r in rows]
        leaks = [r["leakage"] for r in rows]
        turns = [r["tutor_turns"] for r in rows]
        out[label] = {
            "n": len(rows),
            "solve_rate": mean(solves),
            "solve_ci": bootstrap_ci(solves),
            "leakage_rate": mean(leaks),
            "leak_ci": bootstrap_ci(leaks),
            "mean_turns": mean(turns),
            "turns_ci": bootstrap_ci(turns),
            "by_difficulty": {
                d: mean([r["solve"] for r in rows if r["difficulty"] == d])
                for d in sorted({r["difficulty"]
                                 for r in rows})
            },
            "by_level": {
                l: mean([r["solve"] for r in rows if r["student_level"] == l])
                for l in sorted({r["student_level"]
                                 for r in rows})
            },
        }
    return out


# means[judge][label][dim]
def judge_table(sessions, scores) -> dict:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for judge, by_key in scores.items():
        out[judge] = {}
        for label in sessions:
            vals = [s for (lb, _), s in by_key.items() if lb == label]
            if vals:
                out[judge][label] = {d: mean(v[d] for v in vals) for d in (*CONTINUOUS_DIMS, *BINARY_DIMS)}
    return out


def rankings(judge_means: dict) -> dict:
    out: dict[str, dict[str, str]] = {}
    for judge, by_label in judge_means.items():
        out[judge] = {}
        for dim in CONTINUOUS_DIMS:
            labels = [l for l in by_label if dim in by_label[l]]
            best_first = sorted(labels, key=lambda l: by_label[l][dim], reverse=dim not in LOWER_IS_BETTER)
            out[judge][dim] = " > ".join(best_first)
    return out


def agreement(sessions, scores) -> dict:
    judges = sorted(scores)
    rule_by_key: dict[tuple, dict] = {(r["label"], r["session_id"]): r for rows in sessions.values() for r in rows}

    pairwise = {}
    for i, ja in enumerate(judges):
        for jb in judges[i + 1:]:
            common = sorted(set(scores[ja]) & set(scores[jb]))
            if not common:
                continue
            entry: dict[str, float] = {"n": len(common)}
            for d in BINARY_DIMS:
                a = [int(scores[ja][k][d] >= 0.5) for k in common]
                b = [int(scores[jb][k][d] >= 0.5) for k in common]
                entry[f"kappa_{d}"] = cohen_kappa(a, b)
            for d in CONTINUOUS_DIMS:
                entry[f"spearman_{d}"] = spearman([scores[ja][k][d] for k in common],
                                                  [scores[jb][k][d] for k in common])
            pairwise[f"{ja} vs {jb}"] = entry

    alpha = {}
    if len(judges) >= 2:
        for d in (*CONTINUOUS_DIMS, *BINARY_DIMS):
            alpha[d] = krippendorff_alpha({j: {f"{k[0]}/{k[1]}": scores[j][k][d] for k in scores[j]} for j in judges})

    vs_rule = {}
    for j in judges:
        keys = sorted(set(scores[j]) & set(rule_by_key))
        if not keys:
            continue
        vs_rule[j] = {
            "n":
            len(keys),
            "kappa_solve":
            cohen_kappa([rule_by_key[k]["solve"] for k in keys],
                        [int(scores[j][k]["student_solved"] >= 0.5) for k in keys]),
            "kappa_leakage":
            cohen_kappa([rule_by_key[k]["leakage"] for k in keys],
                        [int(scores[j][k]["tutor_revealed"] >= 0.5) for k in keys]),
        }
    return {"pairwise": pairwise, "krippendorff_alpha": alpha, "judge_vs_rule": vs_rule}


# --- Rendering -------------------------------------------------------------------


# A markdown table: the header, the separator under it, then one line per row
def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


# The solve/leakage/turns table, plus the same solve rate split by difficulty and level
def rule_outcomes(rules: dict) -> list[str]:
    labels = sorted(rules)
    rows = []
    for label in labels:
        r = rules[label]
        rows.append([
            label,
            str(r["n"]),
            f"{fmt(r['solve_rate'])} [{fmt(r['solve_ci'][0])}, {fmt(r['solve_ci'][1])}]",
            f"{fmt(r['leakage_rate'])} [{fmt(r['leak_ci'][0])}, {fmt(r['leak_ci'][1])}]",
            fmt(r["mean_turns"], 2),
        ])
    lines = ["## Rule-based outcomes (solve vs leakage - Pareto data)", ""]
    lines += table(["Baseline", "n", "Solve rate [95% CI]", "Leakage rate [95% CI]", "Mean tutor turns"], rows)

    # Every baseline saw the same problems, so one of them gives the column headers
    sample = next(iter(rules.values()))
    diffs = sorted(sample["by_difficulty"])
    levels = sorted(sample["by_level"])
    rows = []
    for label in labels:
        r = rules[label]
        rows.append(
            [label, *(fmt(r["by_difficulty"].get(d)) for d in diffs), *(fmt(r["by_level"].get(l)) for l in levels)])
    lines += ["", "### Solve rate by difficulty / student level", ""]
    lines += table(["Baseline", *(f"diff:{d}" for d in diffs), *(f"lvl:{l}" for l in levels)], rows)
    return lines


# One rubric table per judge, then how each judge ranks the baselines
def judge_rubrics(judge_means: dict, ranks: dict) -> list[str]:
    dims = (*CONTINUOUS_DIMS, *BINARY_DIMS)
    lines = ["", "## Judge-panel rubric means (per baseline × judge)", ""]
    for judge in sorted(judge_means):
        means = judge_means[judge]
        lines += [f"### {judge}", ""]
        lines += table(["Baseline", *dims], [[label, *(fmt(means[label][d]) for d in dims)] for label in sorted(means)])
        lines += [""]

    lines += ["## Baseline rankings per judge (best first; overhelp: lower is better)", ""]
    lines += table(["Judge", *CONTINUOUS_DIMS],
                   [[judge, *(ranks[judge][d] for d in CONTINUOUS_DIMS)] for judge in sorted(ranks)])
    return lines


# How far the judges agree with each other, pair by pair and across the panel
def judge_agreement(agree: dict) -> list[str]:
    lines = ["", "## Inter-judge agreement", ""]
    lines += table(
        ["Judge pair", "n", *(f"κ {d}" for d in BINARY_DIMS), *(f"ρ {d}" for d in CONTINUOUS_DIMS)],
        [[
            pair,
            str(e["n"]),
            *(fmt(e[f"kappa_{d}"]) for d in BINARY_DIMS),
            *(fmt(e[f"spearman_{d}"]) for d in CONTINUOUS_DIMS),
        ] for pair, e in agree["pairwise"].items()],
    )
    lines += [
        "", "Krippendorff's alpha across the panel (interval metric; "
        "α ≥ 0.8 reliable, 0.67-0.8 tentative):", ""
    ]
    lines += table(["Dimension", "alpha"], [[d, fmt(a)] for d, a in agree["krippendorff_alpha"].items()])
    return lines


# How far each judge agrees with the symbolic checks, which is what validates them
def rule_agreement(agree: dict) -> list[str]:
    lines = ["", "## Judge vs rule-based checks (validates the symbolic metrics)", ""]
    lines += table(
        ["Judge", "n", "κ solve (rule vs student_solved)", "κ leakage (rule vs tutor_revealed)"],
        [[judge, str(e["n"]), fmt(e["kappa_solve"]), fmt(e["kappa_leakage"])]
         for judge, e in agree["judge_vs_rule"].items()],
    )
    return lines


# The whole markdown report, section by section
def render_report(rules, judge_means, ranks, agree) -> str:
    lines = ["# Cross-baseline session evaluation", ""]
    lines += rule_outcomes(rules)
    if judge_means:
        lines += judge_rubrics(judge_means, ranks)
    if agree["pairwise"]:
        lines += judge_agreement(agree)
    if agree["judge_vs_rule"]:
        lines += rule_agreement(agree)
    lines += [""]
    return "\n".join(lines)


def write_merged_csv(sessions, scores, out: Path) -> None:
    judges = sorted(scores)
    cols = ["label", "session_id", "problem_id", "difficulty", "student_level", "solve", "leakage", "tutor_turns"]
    for j in judges:
        cols += [f"{j}::{d}" for d in (*CONTINUOUS_DIMS, *BINARY_DIMS)]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for label, rows in sorted(sessions.items()):
            for r in rows:
                row = [
                    label, r["session_id"], r["problem_id"], r["difficulty"], r["student_level"], r["solve"],
                    r["leakage"], r["tutor_turns"]
                ]
                for j in judges:
                    s = scores[j].get((label, r["session_id"]))
                    row += [s.get(d) if s else "" for d in (*CONTINUOUS_DIMS, *BINARY_DIMS)]
                w.writerow(row)


# --- Entry point -----------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    p = argparse.ArgumentParser(description="Aggregate eval sessions + judge scores into the comparison report")
    p.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS_DIR)
    p.add_argument("--scores", type=Path, default=DEFAULT_SCORES_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    sessions, scores = load_data(args.sessions, args.scores)
    if not sessions:
        raise SystemExit(f"No session files in {args.sessions} - run eval-sessions first")
    log.info("Baselines: %s | judges: %s", sorted(sessions), sorted(scores))

    rules = rule_table(sessions)
    judge_means = judge_table(sessions, scores)
    ranks = rankings(judge_means)
    agree = agreement(sessions, scores)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = render_report(rules, judge_means, ranks, agree)
    (args.out_dir / "session_report.md").write_text(report, encoding="utf-8")
    (args.out_dir / "summary.json").write_text(
        json.dumps({
            "rule_based": rules,
            "judge_means": judge_means,
            "rankings": ranks,
            "agreement": agree,
        },
                   indent=2,
                   default=str),
        encoding="utf-8",
    )
    write_merged_csv(sessions, scores, args.out_dir / "merged.csv")
    log.info("Wrote %s, summary.json, merged.csv", args.out_dir / "session_report.md")
    print(report)


if __name__ == "__main__":
    main()
