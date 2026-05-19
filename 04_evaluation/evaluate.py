"""CLI: evaluate MC Medical VQA predictions from an inference JSON file."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.mc_eval import run_mc_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MC Medical VQA predictions")
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional metrics JSON path (default: <results_stem>_eval.json next to results)",
    )
    parser.add_argument(
        "--eval_runs_csv",
        type=str,
        default=None,
        help=(
            "Append aggregate metrics row to this CSV (default: eval_runs_mc.csv in the same "
            "directory as --results_file)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_mc_evaluation(
        Path(args.results_file),
        Path(args.output_file) if args.output_file else None,
        eval_runs_csv=Path(args.eval_runs_csv) if args.eval_runs_csv else None,
    )


if __name__ == "__main__":
    main()
