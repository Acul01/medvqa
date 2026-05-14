"""MC VQA metrics from inference JSON (shared by evaluate CLI and post-inference hook)."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils.mc_vqa import correct_index_to_letter, parse_predicted_letter


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _extract_meta_from_filename(results_path: Path) -> Tuple[str, str, str]:
    stem = results_path.stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return parts[-2], "_".join(parts[:-2]), parts[-1]
    return "unknown", "unknown", "unknown"


def _resolve_gt_letter(row: Dict[str, Any]) -> str | None:
    gt_letter = row.get("ground_truth_letter")
    if isinstance(gt_letter, str) and gt_letter.strip():
        return gt_letter.strip().upper()
    correct_index = row.get("correct_index")
    if isinstance(correct_index, int) and correct_index >= 0:
        return correct_index_to_letter(correct_index)
    return None


def _resolve_pred_letter(row: Dict[str, Any]) -> str | None:
    pred_letter = row.get("predicted_letter")
    if isinstance(pred_letter, str) and pred_letter.strip():
        return pred_letter.strip().upper()
    options = row.get("options")
    num_options = len(options) if isinstance(options, list) else 5
    return parse_predicted_letter(str(row.get("predicted_answer", "")), num_options=num_options)


def _group_samples(results: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
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


def run_mc_evaluation(results_path: Path, metrics_output_path: Path | None = None) -> dict[str, Any]:
    """Load inference JSON, compute MC metrics, write ``{stem}_eval.json`` next to it unless path is given."""
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, encoding="utf-8") as f:
        results: List[Dict[str, Any]] = json.load(f)

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset, model_path_meta, n_samples = _extract_meta_from_filename(results_path)
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

    metrics: dict[str, Any] = {
        "start_time": start_time,
        "results_file": str(results_path.resolve()),
        "dataset": dataset,
        "model_path": model_path_meta,
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

    out = metrics_output_path if metrics_output_path is not None else results_path.parent / f"{results_path.stem}_eval.json"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("MC Evaluation Metrics")
    print("=" * 72)
    print(f"Start time: {start_time}")
    print(f"Dataset: {dataset}")
    print(f"Model path (from filename): {model_path_meta}")
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
    print(f"Metrics saved: {out}")
    print("=" * 72)

    return metrics
