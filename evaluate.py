"""Evaluation script for MC Medical VQA predictions."""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from utils.mc_vqa import correct_index_to_letter, parse_predicted_letter

EVAL_RUN_COLUMNS = [
    "start_time",
    "dataset",
    "model_path",
    "n_samples",
    "closed_accuracy",
    "open_accuracy",
    "mc_overall_accuracy",
    "token_based_entropy",
    "mean_agreement",
    "mean_confidence",
]


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _extract_meta_from_filename(results_path: Path) -> Tuple[str, str, str]:
    stem = results_path.stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return parts[-2], "_".join(parts[:-2]), parts[-1]
    return "unknown", "unknown", "unknown"


def _resolve_gt_letter(row: Dict) -> str | None:
    gt_letter = row.get("ground_truth_letter")
    if isinstance(gt_letter, str) and gt_letter.strip():
        return gt_letter.strip().upper()
    correct_index = row.get("correct_index")
    if isinstance(correct_index, int) and correct_index >= 0:
        return correct_index_to_letter(correct_index)
    return None


def _resolve_pred_letter(row: Dict) -> str | None:
    pred_letter = row.get("predicted_letter")
    if isinstance(pred_letter, str) and pred_letter.strip():
        return pred_letter.strip().upper()
    options = row.get("options")
    num_options = len(options) if isinstance(options, list) else 5
    return parse_predicted_letter(str(row.get("predicted_answer", "")), num_options=num_options)


def _group_samples(results: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    valid, closed, open_rows = [], [], []
    for row in results:
        gt = _resolve_gt_letter(row)
        if gt is None:
            continue
        pred = _resolve_pred_letter(row)
        qtype = str(row.get("question_type", "")).strip().lower()
        packed = {"row": row, "gt_letter": gt, "pred_letter": pred, "question_type": qtype}
        valid.append(packed)
        if qtype == "closed":
            closed.append(packed)
        elif qtype == "open":
            open_rows.append(packed)
    return valid, closed, open_rows


def _accuracy(samples: List[Dict]) -> float:
    if not samples:
        return 0.0
    correct = sum(1 for s in samples if s["pred_letter"] == s["gt_letter"])
    return _safe_div(correct, len(samples))


def _compute_token_based_entropy(results: List[Dict]) -> float:
    values = []
    for row in results:
        value = row.get("entropy")
        if isinstance(value, (int, float)) and not math.isnan(value):
            values.append(float(value))
    return _safe_div(sum(values), len(values)) if values else float("nan")


def _compute_mean_metric(results: List[Dict], field: str) -> float:
    values = []
    for row in results:
        value = row.get(field)
        if isinstance(value, (int, float)) and not math.isnan(value):
            values.append(float(value))
    return _safe_div(sum(values), len(values)) if values else float("nan")


def _append_eval_runs_csv(eval_runs_path: Path, row: Dict) -> None:
    eval_runs_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not eval_runs_path.exists()
    with open(eval_runs_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_RUN_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MC Medical VQA predictions")
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--eval_runs_file", type=str, default="results/eval_runs.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results_path = Path(args.results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    dataset, model_path, n_samples = _extract_meta_from_filename(results_path)
    valid_rows, closed_rows, open_rows = _group_samples(results)

    closed_acc = _accuracy(closed_rows)
    open_acc = _accuracy(open_rows)
    overall_acc = _accuracy(valid_rows)
    token_entropy = _compute_token_based_entropy(results)
    mean_agreement = _compute_mean_metric(results, "agreement")
    mean_confidence = _compute_mean_metric(results, "confidence")

    metrics = {
        "start_time": start_time,
        "dataset": dataset,
        "model_path": model_path,
        "n_samples": n_samples,
        "counts": {
            "total_rows": len(results),
            "valid_rows": len(valid_rows),
            "closed_rows": len(closed_rows),
            "open_rows": len(open_rows),
        },
        "closed": {"accuracy": closed_acc},
        "open": {"accuracy": open_acc},
        "overall": {"accuracy": overall_acc},
        "token_based_entropy": token_entropy,
        "mean_agreement": mean_agreement,
        "mean_confidence": mean_confidence,
    }

    run_row = {
        "start_time": start_time,
        "dataset": dataset,
        "model_path": model_path,
        "n_samples": n_samples,
        "closed_accuracy": f"{closed_acc:.4f}",
        "open_accuracy": f"{open_acc:.4f}",
        "mc_overall_accuracy": f"{overall_acc:.4f}",
        "token_based_entropy": "" if math.isnan(token_entropy) else f"{token_entropy:.6f}",
        "mean_agreement": "" if math.isnan(mean_agreement) else f"{mean_agreement:.6f}",
        "mean_confidence": "" if math.isnan(mean_confidence) else f"{mean_confidence:.6f}",
    }
    _append_eval_runs_csv(Path(args.eval_runs_file), run_row)

    print("\n" + "=" * 72)
    print("MC Evaluation Metrics")
    print("=" * 72)
    print(f"Start time: {start_time}")
    print(f"Dataset: {dataset}")
    print(f"Model path: {model_path}")
    print(f"n_samples: {n_samples}")
    print(f"Rows: total={len(results)}, valid={len(valid_rows)}, closed={len(closed_rows)}, open={len(open_rows)}")
    print("-" * 72)
    print(f"Closed  -> accuracy={closed_acc:.4f}")
    print(f"Open    -> accuracy={open_acc:.4f}")
    print(f"Overall -> accuracy={overall_acc:.4f}")
    print(f"Token-based entropy={token_entropy:.6f}" if not math.isnan(token_entropy) else "Token-based entropy=N/A")
    print(f"Mean agreement={mean_agreement:.6f}" if not math.isnan(mean_agreement) else "Mean agreement=N/A")
    print(f"Mean confidence={mean_confidence:.6f}" if not math.isnan(mean_confidence) else "Mean confidence=N/A")
    print(f"eval_runs.csv updated: {args.eval_runs_file}")
    print("=" * 72)

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Detailed metrics saved: {output_path}")


if __name__ == "__main__":
    main()
