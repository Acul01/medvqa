"""MC VQA metrics from inference JSON (shared by evaluate CLI and post-inference hook)."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils.mc_vqa import correct_index_to_letter, parse_predicted_letter

EVAL_RUN_MC_COLUMNS = [
    "input_path",
    "yesno_accuracy",
    "yesno_entropy",
    "yesno_confidence",
    "yesno_agreement",
    "yesno_aurc_entropy",
    "yesno_aurc_combined",
    "yesno_aurc_confidence",
    "open_accuracy",
    "open_entropy",
    "open_confidence",
    "open_agreement",
    "open_aurc_entropy",
    "open_aurc_combined",
    "open_aurc_confidence",
    "overall_accuracy",
    "overall_entropy",
    "overall_confidence",
    "overall_agreement",
    "overall_aurc_entropy",
    "overall_aurc_combined",
    "overall_aurc_confidence",
    "combined_alpha",
]


def _load_risk_coverage_module():
    """Lazy-load 04_evaluation/risk_coverage_entropy.py without requiring package install."""
    script_path = Path(__file__).resolve().parents[1] / "04_evaluation" / "risk_coverage_entropy.py"
    spec = importlib.util.spec_from_file_location("risk_coverage_entropy", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load risk_coverage_entropy from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _format_optional_float(value: float | None, digits: int = 6) -> str:
    if value is None or not isinstance(value, (int, float)) or math.isnan(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def add_aurc_cli_arguments(parser) -> None:
    """Shared CLI flags for post-inference MC evaluation / dual AURC."""
    parser.add_argument(
        "--combined_alpha",
        type=float,
        default=0.5,
        help="Weight for token entropy in combined AURC (default: 0.5); 1-alpha = vote entropy",
    )
    parser.add_argument(
        "--skip_aurc",
        action="store_true",
        help="Skip dual AURC after inference (accuracy metrics only)",
    )
    parser.add_argument(
        "--eval_runs_csv",
        type=str,
        default=None,
        help=(
            "Append aggregate metrics row to this CSV path (independent of results_dir). "
            "Default: <results_json_dir>/eval_runs_mc.csv"
        ),
    )


def resolve_eval_runs_csv(args) -> Path | None:
    """Return Path for --eval_runs_csv if set, else None (use default next to results JSON)."""
    path = getattr(args, "eval_runs_csv", None)
    return Path(path) if path else None


def apply_aurc_to_wandb_summary(summary: Any, metrics: dict[str, Any]) -> None:
    """Write dual AURC fields from run_mc_evaluation into a wandb summary object."""
    if summary is None or not metrics:
        return
    for split in ("closed", "open", "overall"):
        payload = metrics.get(split) or {}
        for key in ("aurc_entropy", "aurc_combined", "aurc_confidence"):
            value = payload.get(key)
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                summary[f"{split}_{key}"] = float(value)
    alpha = metrics.get("combined_alpha")
    if isinstance(alpha, (int, float)) and not math.isnan(float(alpha)):
        summary["combined_alpha"] = float(alpha)


def _append_eval_runs_mc_csv(csv_path: Path, row: Dict[str, str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EVAL_RUN_MC_COLUMNS)

    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)

        # Migrate older overview CSVs that predate AURC columns.
        if existing_fields != fieldnames:
            migrated = []
            for old_row in existing_rows:
                migrated.append({col: old_row.get(col, "") for col in fieldnames})
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(migrated)

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in fieldnames})


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


def run_mc_evaluation(
    results_path: Path,
    metrics_output_path: Path | None = None,
    *,
    eval_runs_csv: Path | None = None,
    compute_aurc: bool = True,
    combined_alpha: float = 0.5,
    aurc_num_thresholds: int = 200,
) -> dict[str, Any]:
    """Load inference JSON, compute MC metrics, write ``{stem}_eval.json`` next to it unless path is given.

    Appends one row to ``eval_runs_csv`` when passed; otherwise to ``eval_runs_mc.csv`` next to the results JSON.

    When ``compute_aurc`` is True (default), also computes dual AURC (entropy-only and combined)
    via ``risk_coverage_entropy.compute_dual_aurc_summary`` and records it in the metrics JSON,
    the overview CSV, and the printed summary. Full risk-coverage curve artifacts remain available
    via the standalone ``04_evaluation/risk_coverage_entropy.py`` CLI.
    """
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

    aurc_summary: dict[str, Any] | None = None
    if compute_aurc:
        try:
            risk_mod = _load_risk_coverage_module()
            aurc_summary = risk_mod.compute_dual_aurc_summary(
                results,
                combined_alpha=combined_alpha,
                risk_metric="match",
                max_entropy=None,
                num_thresholds=aurc_num_thresholds,
            )
        except Exception as exc:  # noqa: BLE001 - keep inference/eval usable if AURC fails
            print(f"Warning: dual AURC computation failed ({exc}); continuing without AURC.")
            aurc_summary = None

    def _aurc_value(split: str, key: str) -> float:
        if not aurc_summary:
            return float("nan")
        split_payload = aurc_summary.get(split) or {}
        value = split_payload.get(key, float("nan"))
        return float(value) if isinstance(value, (int, float)) else float("nan")

    closed_aurc_entropy = _aurc_value("closed", "aurc_entropy")
    closed_aurc_combined = _aurc_value("closed", "aurc_combined")
    closed_aurc_confidence = _aurc_value("closed", "aurc_confidence")
    open_aurc_entropy = _aurc_value("open", "aurc_entropy")
    open_aurc_combined = _aurc_value("open", "aurc_combined")
    open_aurc_confidence = _aurc_value("open", "aurc_confidence")
    overall_aurc_entropy = _aurc_value("overall", "aurc_entropy")
    overall_aurc_combined = _aurc_value("overall", "aurc_combined")
    overall_aurc_confidence = _aurc_value("overall", "aurc_confidence")

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
            "aurc_entropy": closed_aurc_entropy,
            "aurc_combined": closed_aurc_combined,
            "aurc_confidence": closed_aurc_confidence,
        },
        "open": {
            "accuracy": open_acc,
            "entropy": open_entropy,
            "confidence": open_confidence,
            "agreement": open_agreement,
            "aurc_entropy": open_aurc_entropy,
            "aurc_combined": open_aurc_combined,
            "aurc_confidence": open_aurc_confidence,
        },
        "overall": {
            "accuracy": overall_acc,
            "entropy": overall_entropy,
            "confidence": overall_confidence,
            "agreement": overall_agreement,
            "aurc_entropy": overall_aurc_entropy,
            "aurc_combined": overall_aurc_combined,
            "aurc_confidence": overall_aurc_confidence,
        },
        "combined_alpha": combined_alpha if compute_aurc else None,
        "aurc": aurc_summary,
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
    print(
        f"          aurc_entropy={closed_aurc_entropy:.6f}, aurc_combined={closed_aurc_combined:.6f}, "
        f"aurc_confidence={closed_aurc_confidence:.6f}"
        if not any(
            math.isnan(v)
            for v in [closed_aurc_entropy, closed_aurc_combined, closed_aurc_confidence]
        )
        else "          aurc_entropy/aurc_combined/aurc_confidence=N/A"
    )
    print(f"Open    -> accuracy={open_acc:.4f}")
    print(
        f"          entropy={open_entropy:.6f}, confidence={open_confidence:.6f}, agreement={open_agreement:.6f}"
        if not any(math.isnan(v) for v in [open_entropy, open_confidence, open_agreement])
        else "          entropy/confidence/agreement=N/A"
    )
    print(
        f"          aurc_entropy={open_aurc_entropy:.6f}, aurc_combined={open_aurc_combined:.6f}, "
        f"aurc_confidence={open_aurc_confidence:.6f}"
        if not any(
            math.isnan(v) for v in [open_aurc_entropy, open_aurc_combined, open_aurc_confidence]
        )
        else "          aurc_entropy/aurc_combined/aurc_confidence=N/A"
    )
    print(f"Overall -> accuracy={overall_acc:.4f}")
    print(
        f"          entropy={overall_entropy:.6f}, confidence={overall_confidence:.6f}, agreement={overall_agreement:.6f}"
        if not any(math.isnan(v) for v in [overall_entropy, overall_confidence, overall_agreement])
        else "          entropy/confidence/agreement=N/A"
    )
    print(
        f"          aurc_entropy={overall_aurc_entropy:.6f}, aurc_combined={overall_aurc_combined:.6f}, "
        f"aurc_confidence={overall_aurc_confidence:.6f}"
        if not any(
            math.isnan(v)
            for v in [overall_aurc_entropy, overall_aurc_combined, overall_aurc_confidence]
        )
        else "          aurc_entropy/aurc_combined/aurc_confidence=N/A"
    )
    if compute_aurc:
        print(f"combined_alpha={combined_alpha}")
    print(f"Metrics saved: {out}")
    print("=" * 72)

    runs_row = {
        "input_path": str(results_path.resolve()),
        "yesno_accuracy": f"{closed_acc:.4f}",
        "yesno_entropy": "" if math.isnan(closed_entropy) else f"{closed_entropy:.6f}",
        "yesno_confidence": "" if math.isnan(closed_confidence) else f"{closed_confidence:.6f}",
        "yesno_agreement": "" if math.isnan(closed_agreement) else f"{closed_agreement:.6f}",
        "yesno_aurc_entropy": _format_optional_float(closed_aurc_entropy),
        "yesno_aurc_combined": _format_optional_float(closed_aurc_combined),
        "yesno_aurc_confidence": _format_optional_float(closed_aurc_confidence),
        "open_accuracy": f"{open_acc:.4f}",
        "open_entropy": "" if math.isnan(open_entropy) else f"{open_entropy:.6f}",
        "open_confidence": "" if math.isnan(open_confidence) else f"{open_confidence:.6f}",
        "open_agreement": "" if math.isnan(open_agreement) else f"{open_agreement:.6f}",
        "open_aurc_entropy": _format_optional_float(open_aurc_entropy),
        "open_aurc_combined": _format_optional_float(open_aurc_combined),
        "open_aurc_confidence": _format_optional_float(open_aurc_confidence),
        "overall_accuracy": f"{overall_acc:.4f}",
        "overall_entropy": "" if math.isnan(overall_entropy) else f"{overall_entropy:.6f}",
        "overall_confidence": "" if math.isnan(overall_confidence) else f"{overall_confidence:.6f}",
        "overall_agreement": "" if math.isnan(overall_agreement) else f"{overall_agreement:.6f}",
        "overall_aurc_entropy": _format_optional_float(overall_aurc_entropy),
        "overall_aurc_combined": _format_optional_float(overall_aurc_combined),
        "overall_aurc_confidence": _format_optional_float(overall_aurc_confidence),
        "combined_alpha": f"{combined_alpha:.4f}" if compute_aurc else "",
    }
    runs_csv = Path(eval_runs_csv) if eval_runs_csv is not None else results_path.parent / "eval_runs_mc.csv"
    _append_eval_runs_mc_csv(runs_csv, runs_row)
    print(f"Appended run to {runs_csv}")

    return metrics
