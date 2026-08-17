import argparse
import logging
from pathlib import Path

from its.evaluation.report import bootstrap_ci, load_data, mean

log = logging.getLogger(__name__)

# Okabe-Ito colour-blind-safe palette, one colour per baseline
COLORS = {"T1": "#E69F00", "T2": "#56B4E9", "T3": "#009E73", "T4": "#CC79A7"}
ORDER = ["T1", "T2", "T3", "T4"]


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
    })
    return plt


def _labels(sessions: dict) -> list[str]:
    return [l for l in ORDER if l in sessions] + sorted(set(sessions) - set(ORDER))


def _save(fig, out_dir: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    log.info("wrote %s/%s.{pdf,png}", out_dir, name)


def fig_pareto(plt, sessions, out_dir):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for label in _labels(sessions):
        rows = sessions[label]
        solves = [r["solve"] for r in rows]
        leaks = [r["leakage"] for r in rows]
        x, y = mean(leaks), mean(solves)
        xlo, xhi = bootstrap_ci(leaks)
        ylo, yhi = bootstrap_ci(solves)
        c = COLORS.get(label, "#555555")
        ax.errorbar(x,
                    y,
                    xerr=[[x - xlo], [xhi - x]],
                    yerr=[[y - ylo], [yhi - y]],
                    fmt="o",
                    ms=9,
                    color=c,
                    capsize=3,
                    lw=1.2,
                    zorder=3)
        dx, dy = (-30, -18) if label == "T4" else (8, 6)
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=12, fontweight="bold", color=c)
    ax.set_xlabel("Leakage rate  (tutor reveals the answer)  →  worse")
    ax.set_ylabel("Solve rate  →  better")
    ax.set_xlim(left=0)
    ax.annotate("ideal", xy=(0.02, 0.97), xycoords="axes fraction", fontsize=10, style="italic", color="#777777")
    ax.set_title("Solving success vs. answer leakage (95% CI)")
    _save(fig, out_dir, "pareto")
    plt.close(fig)


def _bar_with_ci(plt, labels, values, cis, colors, ylabel, title):
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    xs = range(len(labels))
    errs = [[v - lo for v, (lo, _) in zip(values, cis)], [hi - v for v, (_, hi) in zip(values, cis)]]
    ax.bar(xs, values, color=colors, yerr=errs, capsize=4, width=0.62)
    ax.set_xticks(list(xs), labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    for x, v, (_, hi) in zip(xs, values, cis):
        ax.text(x, hi + 0.025, f"{v:.2f}", ha="center", fontsize=10)
    return fig


def fig_clean_solve(plt, sessions, out_dir):
    labels = _labels(sessions)
    vals: list[float] = []
    cis: list[tuple[float, float]] = []
    for l in labels:
        clean = [int(r["solve"] and not r["leakage"]) for r in sessions[l]]
        vals.append(mean(clean))
        cis.append(bootstrap_ci(clean))
    fig = _bar_with_ci(
        plt,
        labels,
        vals,
        cis,
        [COLORS.get(l, "#555") for l in labels],
        "Clean solve rate",
        "Solved without leakage (95% CI)",
    )
    _save(fig, out_dir, "clean_solve")
    plt.close(fig)


# Grouped bars of the clean solve rate per baseline, within each `key` group
def _grouped(plt, sessions, out_dir, key, groups, name, title):
    labels = _labels(sessions)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    width = 0.8 / len(labels)
    for i, l in enumerate(labels):
        vals = []
        for g in groups:
            rows = [r for r in sessions[l] if r[key] == g]
            vals.append(mean([int(r["solve"] and not r["leakage"]) for r in rows]) if rows else 0.0)
        xs = [j + i * width for j in range(len(groups))]
        ax.bar(xs, vals, width=width * 0.92, color=COLORS.get(l, "#555"), label=l)
    ax.set_xticks([j + width * (len(labels) - 1) / 2 for j in range(len(groups))], groups)
    ax.set_ylabel("Clean solve rate")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(frameon=False, ncols=len(labels), loc="upper right", fontsize=9)
    _save(fig, out_dir, name)
    plt.close(fig)


def fig_judges(plt, sessions, scores, out_dir):
    from its.evaluation.judges import CONTINUOUS_DIMS
    for judge, by_key in scores.items():
        labels = _labels(sessions)
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        width = 0.8 / len(labels)
        for i, l in enumerate(labels):
            vals = []
            for d in CONTINUOUS_DIMS:
                v = [s[d] for (lb, _), s in by_key.items() if lb == l]
                vals.append(mean(v) if v else 0.0)
            xs = [j + i * width for j in range(len(CONTINUOUS_DIMS))]
            ax.bar(xs, vals, width=width * 0.92, color=COLORS.get(l, "#555"), label=l)
        ax.set_xticks([j + width * (len(labels) - 1) / 2 for j in range(len(CONTINUOUS_DIMS))], CONTINUOUS_DIMS)
        ax.set_ylabel("Judge mean score")
        ax.set_title(f"Judge rubric means - {judge}")
        ax.legend(frameon=False, ncols=len(labels), fontsize=9)
        slug = "".join(ch if ch.isalnum() else "-" for ch in judge)
        _save(fig, out_dir, f"judges_{slug}")
        plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    p = argparse.ArgumentParser(description="Generate evaluation figures (PDF + PNG)")
    p.add_argument("--sessions", type=Path, default=Path("data/eval/sessions"))
    p.add_argument("--scores", type=Path, default=Path("data/eval/scores"))
    p.add_argument("--out-dir", type=Path, default=Path("data/eval/figures"))
    p.add_argument("--labels",
                   nargs="*",
                   default=ORDER,
                   help="session labels to plot (default: the four main tutors; "
                   "pass explicitly to include ablations like T3-noact)")
    args = p.parse_args()

    sessions, scores = load_data(args.sessions, args.scores)
    sessions = {l: s for l, s in sessions.items() if l in args.labels}
    if not sessions:
        raise SystemExit(f"No session files in {args.sessions}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt = _style()

    fig_pareto(plt, sessions, args.out_dir)
    fig_clean_solve(plt, sessions, args.out_dir)
    _dorder = {"easy": 0, "medium": 1, "hard": 2}
    diffs = sorted({r["difficulty"] for rows in sessions.values() for r in rows}, key=lambda d: _dorder.get(d, 9))
    lvls = [
        l for l in ("weak", "medium", "strong")
        if any(r["student_level"] == l for rows in sessions.values() for r in rows)
    ]
    _grouped(plt, sessions, args.out_dir, "difficulty", diffs, "by_difficulty",
             "Clean solve rate by problem difficulty")
    _grouped(plt, sessions, args.out_dir, "student_level", lvls, "by_level", "Clean solve rate by student level")
    if scores:
        fig_judges(plt, sessions, scores, args.out_dir)
    else:
        log.info("No judge scores in %s yet - judge figures skipped (rerun after eval-judge)", args.scores)
    log.info("Done → %s", args.out_dir)


if __name__ == "__main__":
    main()
