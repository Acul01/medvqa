"""
Entropy-threshold baseline for reliable VQA.

Computes risk/coverage sweep, plots risk-coverage curve, and reports AURC
(Area Under the Risk-Coverage Curve).
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


from utils.mc_vqa import correct_index_to_letter, parse_predicted_letter
from utils.paths import get_results_dir


def exact_match_letter(row: Dict) -> bool:
    gt = row.get("ground_truth_letter")
    if not isinstance(gt, str) or not gt.strip():
        idx = row.get("correct_index")
        if isinstance(idx, int) and idx >= 0:
            gt = correct_index_to_letter(idx)
    if not isinstance(gt, str) or not gt.strip():
        return False
    gt = gt.strip().upper()

    pred = row.get("predicted_letter")
    if isinstance(pred, str) and pred.strip():
        pred = pred.strip().upper()
    else:
        options = row.get("options")
        n = len(options) if isinstance(options, list) else 5
        pred = parse_predicted_letter(str(row.get("predicted_answer", "")), num_options=n)

    return pred == gt


def is_valid_number(value) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(value)


def load_results(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("results_file must contain a JSON list.")
    return data


def metric_value_from_row(row: Dict, risk_metric: str) -> float | None:
    """Extract per-sample metric value in [0,1] for risk computation."""
    if risk_metric == "match":
        return float(exact_match_letter(row))

    key_map = {
        "bleu1": "bleu_1",
        "bleu4": "bleu_4",
        "rouge": "rougeL",
        "bertscore": "bert_f1",
    }
    metric_key = key_map[risk_metric]
    val = row.get(metric_key)
    if not is_valid_number(val):
        return None
    val = float(val)
    # Clamp softly for robustness against tiny numeric drifts.
    return max(0.0, min(1.0, val))


def prepare_samples(
    results: List[Dict],
    risk_metric: str,
    score_field: str = "entropy",
    max_entropy: float | None = None,
) -> List[Dict]:
    samples: List[Dict] = []
    for r in results:
        score_value = r.get(score_field)
        if not is_valid_number(score_value):
            continue
        score_value = float(score_value)
        if score_field == "entropy" and max_entropy is not None and score_value > max_entropy:
            continue
        metric_val = metric_value_from_row(r, risk_metric=risk_metric)
        if metric_val is None:
            continue

        samples.append(
            {
                "score_value": score_value,
                "metric_value": metric_val,
            }
        )
    return samples


def unique_thresholds(values: List[float], num_thresholds: int) -> List[float]:
    uniq = sorted(set(values))
    if len(uniq) <= num_thresholds:
        return uniq
    # evenly sample thresholds over sorted support
    idxs = [round(i * (len(uniq) - 1) / (num_thresholds - 1)) for i in range(num_thresholds)]
    return [uniq[i] for i in sorted(set(idxs))]


def compute_risk_coverage_at_threshold(samples: List[Dict], threshold: float, answer_when_higher: bool) -> Dict:
    total = len(samples)
    if answer_when_higher:
        answered = [s for s in samples if s["score_value"] >= threshold]
    else:
        answered = [s for s in samples if s["score_value"] <= threshold]
    n_answered = len(answered)
    if n_answered == 0:
        return {
            "threshold": threshold,
            "coverage": 0.0,
            "risk": 0.0,
            "answered": 0,
            "errors": 0,
            "correct": 0,
            "metric_mean": 0.0,
        }

    metric_mean = sum(float(s["metric_value"]) for s in answered) / n_answered
    risk = 1.0 - metric_mean
    correct = sum(1 for s in answered if math.isclose(float(s["metric_value"]), 1.0))
    errors = n_answered - correct
    coverage = n_answered / total
    return {
        "threshold": threshold,
        "coverage": coverage,
        "risk": risk,
        "answered": n_answered,
        "errors": errors,
        "correct": correct,
        "metric_mean": metric_mean,
    }


def compute_curve(samples: List[Dict], num_thresholds: int, answer_when_higher: bool) -> List[Dict]:
    thresholds = unique_thresholds([s["score_value"] for s in samples], max(2, num_thresholds))
    rows = [compute_risk_coverage_at_threshold(samples, t, answer_when_higher=answer_when_higher) for t in thresholds]
    rows = sorted(rows, key=lambda x: x["coverage"])
    return rows


def aurc_trapezoid(curve_rows: List[Dict]) -> float:
    if len(curve_rows) < 2:
        return 0.0
    area = 0.0
    for left, right in zip(curve_rows[:-1], curve_rows[1:]):
        dx = right["coverage"] - left["coverage"]
        y_avg = 0.5 * (left["risk"] + right["risk"])
        area += dx * y_avg
    return area


def coverage_at_risk(curve_rows: List[Dict], risk_targets: List[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for target in risk_targets:
        valid = [r["coverage"] for r in curve_rows if r["risk"] <= target]
        out[f"coverage_at_risk_{int(target * 100)}"] = max(valid) if valid else 0.0
    return out


def risk_at_coverage(curve_rows: List[Dict], cov_targets: List[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for target in cov_targets:
        valid = [r["risk"] for r in curve_rows if r["coverage"] >= target]
        out[f"risk_at_coverage_{int(target * 100)}"] = min(valid) if valid else 1.0
    return out


def save_curve_csv(curve_rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["threshold", "coverage", "risk", "metric_mean", "answered", "correct", "errors"],
        )
        writer.writeheader()
        for row in curve_rows:
            writer.writerow(row)


def plot_curve(curve_rows: List[Dict], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = [r["coverage"] for r in curve_rows]
    y = [r["risk"] for r in curve_rows]
    t = [r["threshold"] for r in curve_rows]

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    ax2 = ax1.twinx()

    # Main risk-coverage curve (left y-axis)
    line_risk = ax1.plot(x, y, linewidth=2, color="tab:blue", label="Risk")[0]
    ax1.set_xlabel("Coverage")
    ax1.set_ylabel("Risk", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title(title)
    ax1.grid(alpha=0.25)

    # Threshold trace (right y-axis)
    line_thr = ax2.plot(x, t, linewidth=2, color="tab:orange", label="Entropy threshold")[0]
    ax2.set_ylabel("Entropy threshold", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Entropy thresholding baseline for risk/coverage + AURC")
    parser.add_argument("--results_file", type=str, required=True, help="Path to results JSON")
    parser.add_argument("--output_dir", type=str, default=get_results_dir(), help="Output directory")
    parser.add_argument("--num_thresholds", type=int, default=200, help="Number of thresholds for sweep")
    parser.add_argument(
        "--max_entropy",
        type=float,
        default=None,
        help="Optional max entropy filter (useful for focusing low-entropy regime)",
    )
    parser.add_argument(
        "--risk_metric",
        type=str,
        default="match",
        choices=["match", "bleu1", "bleu4", "rouge", "bertscore"],
        help="Metric used for risk: risk = 1 - metric_value",
    )
    parser.add_argument(
        "--score_field",
        type=str,
        default="entropy",
        choices=["entropy", "agreement", "confidence"],
        help="Field used for thresholding (entropy lower=more confident, agreement/confidence higher=more confident)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_path = Path(args.results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    results = load_results(results_path)
    samples = prepare_samples(
        results,
        risk_metric=args.risk_metric,
        score_field=args.score_field,
        max_entropy=args.max_entropy,
    )
    if not samples:
        raise ValueError("No valid samples found with letter labels and numeric entropy.")

    answer_when_higher = args.score_field in {"agreement", "confidence"}
    curve = compute_curve(samples, num_thresholds=args.num_thresholds, answer_when_higher=answer_when_higher)
    area = aurc_trapezoid(curve)
    c_at_r = coverage_at_risk(curve, risk_targets=[0.01, 0.05, 0.10])
    r_at_c = risk_at_coverage(curve, cov_targets=[0.20, 0.50, 0.80])

    out_dir = Path(args.output_dir)
    base = results_path.stem
    csv_path = out_dir / f"{base}_risk_coverage.csv"
    plot_path = out_dir / f"{base}_risk_coverage_curve.png"
    summary_path = out_dir / f"{base}_risk_coverage_summary.json"

    save_curve_csv(curve, csv_path)
    plot_curve(curve, plot_path, title=f"Risk-Coverage Curve ({base})")

    summary = {
        "results_file": str(results_path),
        "total_input_rows": len(results),
        "valid_samples": len(samples),
        "num_thresholds": len(curve),
        "max_entropy_filter": args.max_entropy,
        "risk_metric": args.risk_metric,
        "score_field": args.score_field,
        "aurc": area,
        **c_at_r,
        **r_at_c,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Valid samples: {len(samples)}")
    print(f"AURC: {area:.6f}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
