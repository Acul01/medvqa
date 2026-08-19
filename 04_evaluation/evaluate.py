"""CLI: evaluate MC Medical VQA predictions from an inference JSON file."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.mc_eval import run_mc_evaluation, add_aurc_cli_arguments, resolve_eval_runs_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MC Medical VQA predictions")
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional metrics JSON path (default: <results_stem>_eval.json next to results)",
    )
    add_aurc_cli_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_mc_evaluation(
        Path(args.results_file),
        Path(args.output_file) if args.output_file else None,
        eval_runs_csv=resolve_eval_runs_csv(args),
        compute_aurc=not args.skip_aurc,
        combined_alpha=args.combined_alpha,
    )


if __name__ == "__main__":
    main()
