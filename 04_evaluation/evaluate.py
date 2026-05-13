"""Evaluation script for MC Medical VQA predictions."""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from utils.mc_vqa import correct_index_to_letter, parse_predicted_letter
from utils.paths import get_eval_runs_csv

EVAL_RUN_COLUMNS = [
    "input_path",
    "yesno_accuracy",
    "yesno_entropy",
    "yesno_confidence",
    "yesno_agreement",
    "open_accuracy",
    "open_entropy",
    "open_confidence",
    "open_agreement",
    "overall_accuracy",
    "overall_entropy",
    "overall_confidence",
    "overall_agreement",
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


def _compute_mean_metric_from_group(samples: List[Dict], field: str) -> float:
    values = []
    for sample in samples:
        value = sample.get("row", {}).get(field)
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
    parser.add_argument("--eval_runs_file", type=str, default=get_eval_runs_csv())
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
    overall_entropy = _compute_mean_metric_from_group(valid_rows, "entropy")
    overall_agreement = _compute_mean_metric_from_group(valid_rows, "agreement")
    overall_confidence = _compute_mean_metric_from_group(valid_rows, "confidence")
    closed_entropy = _compute_mean_metric_from_group(closed_rows, "entropy")
    closed_agreement = _compute_mean_metric_from_group(closed_rows, "agreement")
    closed_confidence = _compute_mean_metric_from_group(closed_rows, "confidence")
    open_entropy = _compute_mean_metric_from_group(open_rows, "entropy")
    open_agreement = _compute_mean_metric_from_group(open_rows, "agreement")
    open_confidence = _compute_mean_metric_from_group(open_rows, "confidence")

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
        "closed": {
            "accuracy": closed_acc,
            "entropy": closed_entropy,
            "confidence": closed_confidence,
            "agreement": closed_agreement,
        },
        "open": {
            "accuracy": open_acc,
            "entropy": open_entropy,
            "confidence": open_confidence,
            "agreement": open_agreement,
        },
        "overall": {
            "accuracy": overall_acc,
            "entropy": overall_entropy,
            "confidence": overall_confidence,
            "agreement": overall_agreement,
        },
    }

    run_row = {
        "input_path": str(results_path),
        "yesno_accuracy": f"{closed_acc:.4f}",
        "yesno_entropy": "" if math.isnan(closed_entropy) else f"{closed_entropy:.6f}",
        "yesno_confidence": "" if math.isnan(closed_confidence) else f"{closed_confidence:.6f}",
        "yesno_agreement": "" if math.isnan(closed_agreement) else f"{closed_agreement:.6f}",
        "open_accuracy": f"{open_acc:.4f}",
        "open_entropy": "" if math.isnan(open_entropy) else f"{open_entropy:.6f}",
        "open_confidence": "" if math.isnan(open_confidence) else f"{open_confidence:.6f}",
        "open_agreement": "" if math.isnan(open_agreement) else f"{open_agreement:.6f}",
        "overall_accuracy": f"{overall_acc:.4f}",
        "overall_entropy": "" if math.isnan(overall_entropy) else f"{overall_entropy:.6f}",
        "overall_confidence": "" if math.isnan(overall_confidence) else f"{overall_confidence:.6f}",
        "overall_agreement": "" if math.isnan(overall_agreement) else f"{overall_agreement:.6f}",
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
    print(
        f"          entropy={closed_entropy:.6f}, confidence={closed_confidence:.6f}, agreement={closed_agreement:.6f}"
        if not any(math.isnan(v) for v in [closed_entropy, closed_confidence, closed_agreement])
        else "          entropy/confidence/agreement=N/A"
    )
    print(f"Open    -> accuracy={open_acc:.4f}")
    print(
        f"          entropy={open_entropy:.6f}, confidence={open_confidence:.6f}, agreement={open_agreement:.6f}"
        if not any(math.isnan(v) for v in [open_entropy, open_confidence, open_agreement])
        else "          entropy/confidence/agreement=N/A"
    )
    print(f"Overall -> accuracy={overall_acc:.4f}")
    print(
        f"          entropy={overall_entropy:.6f}, confidence={overall_confidence:.6f}, agreement={overall_agreement:.6f}"
        if not any(math.isnan(v) for v in [overall_entropy, overall_confidence, overall_agreement])
        else "          entropy/confidence/agreement=N/A"
    )
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
