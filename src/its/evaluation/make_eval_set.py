"""
Builds the held-out evaluation problem set. This runs locally, with no GPU and no models.

    uv run eval-build-set --n-problems 200 --out data/eval/problems_eval.jsonl

Run `uv run eval-build-set --help` for the options.
"""

import argparse
import logging
from pathlib import Path

from its.evaluation.harness import (DEFAULT_BANK, build_eval_set, discover_training_logs)

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    p = argparse.ArgumentParser(description="Build the held-out eval problem set from local training logs (no GPU)")
    p.add_argument("--bank", type=Path, default=DEFAULT_BANK, help="Problem bank JSONL")
    p.add_argument("--out",
                   type=Path,
                   default=Path("data/eval/problems_eval.jsonl"),
                   help="Output JSONL of held-out problems")
    p.add_argument("--n-problems",
                   type=int,
                   default=200,
                   help="How many to sample, stratified over difficulty (-1 = the whole held-out pool)")
    p.add_argument("--seed", type=int, default=42, help="Sampling seed (reproducible)")
    p.add_argument("--exclude-log",
                   type=Path,
                   action="append",
                   default=None,
                   help="Training sessions.jsonl to exclude (repeatable; "
                   "default: auto-discover every data/models/**/sessions.jsonl)")
    p.add_argument("--models-dir",
                   type=Path,
                   default=Path("data/models"),
                   help="Root scanned for sessions.jsonl when auto-discovering logs")
    args = p.parse_args()

    if args.out.exists():
        raise SystemExit(f"{args.out} already exists - delete it or choose another --out")

    logs = args.exclude_log if args.exclude_log else discover_training_logs(args.models_dir)
    if not logs:
        log.warning(
            "No training logs found under %s - the eval set will exclude nothing. "
            "Are the sessions.jsonl files on this machine?", args.models_dir)
    else:
        log.info("Excluding problems from %d training log(s):", len(logs))
        for lg in logs:
            log.info("  %s", lg)

    build_eval_set(args.bank, logs, args.n_problems, args.seed, args.out)
    log.info("Done. Copy %s to the pod and run: eval-sessions --eval-set %s ...", args.out, args.out.name)


if __name__ == "__main__":
    main()
