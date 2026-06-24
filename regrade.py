"""Post-mortem re-grading of saved Pet-Bench trajectories.

Every episode is saved as a self-contained Inspect `.eval` log (the solver persists
the full ground-truth world into the sample store), so we can re-run the scorers
against a saved run WITHOUT replaying the agent or standing the server back up.

Usage:
    python regrade.py                       # re-grade every log in ./logs (append scores)
    python regrade.py logs/<run>.eval       # re-grade one log
    python regrade.py --scorer behaviour    # only the model-graded behaviour judge
    python regrade.py --action overwrite    # replace prior scores instead of appending
    python regrade.py --repeat 3            # run the (non-deterministic) judge 3x for variance

The grader model comes from the log's saved `model_roles["grader"]`; override with
the PETBENCH_GRADER env var. Equivalent single-log CLI: `inspect score <log> --scorer ...`.
"""

from __future__ import annotations

import argparse
import os
import sys

from inspect_ai import score
from inspect_ai.log import list_eval_logs, read_eval_log, write_eval_log

from petbench import behaviour_judge, trajectory_scorer

SCORERS = {
    "behaviour": lambda: [behaviour_judge()],
    "trajectory": lambda: [trajectory_scorer()],
    "both": lambda: [trajectory_scorer(), behaviour_judge()],
}


def _logs(target: str | None) -> list[str]:
    if target and target.endswith(".eval"):
        return [target]
    log_dir = target or os.getenv("INSPECT_LOG_DIR", "logs")
    return [info.name for info in list_eval_logs(log_dir)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Re-grade saved Pet-Bench .eval logs.")
    ap.add_argument("target", nargs="?", default=None,
                    help="a .eval file, or a log dir (default: ./logs)")
    ap.add_argument("--scorer", choices=SCORERS, default="behaviour")
    ap.add_argument("--action", choices=["append", "overwrite"], default="append")
    ap.add_argument("--repeat", type=int, default=1,
                    help="re-score N times (variance estimate for the LLM judge)")
    args = ap.parse_args(argv)

    logs = _logs(args.target)
    if not logs:
        print("no .eval logs found.", file=sys.stderr)
        return 1

    print(f"re-grading {len(logs)} log(s) with scorer='{args.scorer}' "
          f"action='{args.action}' repeat={args.repeat}")
    for name in logs:
        log = read_eval_log(name)
        for i in range(args.repeat):
            log = score(log, SCORERS[args.scorer](), action=args.action)
        write_eval_log(log, name)
        # surface the freshly-aggregated metrics
        scores = (log.results.scores if log.results else []) or []
        summary = ", ".join(
            f"{s.name}:{ {k: round(m.value, 3) for k, m in s.metrics.items()} }"
            for s in scores
        )
        print(f"  ✓ {os.path.basename(name)} → {summary or '(no metrics)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
