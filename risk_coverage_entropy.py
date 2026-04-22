"""
Entropy-threshold baseline for reliable VQA.

Computes risk/coverage sweep, plots risk-coverage curve, and reports AURC
(Area Under the Risk-Coverage Curve).
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def normalize_answer(answer: str) -> str:
    answer = (answer or "").lower()
    answer = re.sub(r"[^\w\s]", "", answer)
    return " ".join(answer.split())


def exact_match(pred: str, gt: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gt)


def is_valid_number(value) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(value)


def load_results(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("results_file must contain a JSON list.")
    return data


def prepare_samples(results: List[Dict], max_entropy: float | None = None) -> List[Dict]:
    samples: List[Dict] = []
    for r in results:
        gt = r.get("ground_truth_answer")
        entropy = r.get("entropy")
        if gt is None or str(gt).strip() == "":
            continue
        if not is_valid_number(entropy):
            continue
        entropy = float(entropy)
        if max_entropy is not None and entropy > max_entropy:
            continue

        samples.append(
            {
                "entropy": entropy,
                "correct": exact_match(str(r.get("predicted_answer", "")), str(gt)),
            }
        )
    return samples


def unique_thresholds(entropies: List[float], num_thresholds: int) -> List[float]:
    uniq = sorted(set(entropies))
    if len(uniq) <= num_thresholds:
        return uniq
    # evenly sample thresholds over sorted support
    idxs = [round(i * (len(uniq) - 1) / (num_thresholds - 1)) for i in range(num_thresholds)]
    return [uniq[i] for i in sorted(set(idxs))]


def compute_risk_coverage_at_threshold(samples: List[Dict], threshold: float) -> Dict:
    total = len(samples)
    answered = [s for s in samples if s["entropy"] <= threshold]
    n_answered = len(answered)
    if n_answered == 0:
        return {
            "threshold": threshold,
            "coverage": 0.0,
            "risk": 0.0,
            "answered": 0,
            "errors": 0,
            "correct": 0,
        }

    correct = sum(1 for s in answered if s["correct"])
    errors = n_answered - correct
    coverage = n_answered / total
    risk = errors / n_answered
    return {
        "threshold": threshold,
        "coverage": coverage,
        "risk": risk,
        "answered": n_answered,
        "errors": errors,
        "correct": correct,
    }


def compute_curve(samples: List[Dict], num_thresholds: int) -> List[Dict]:
    thresholds = unique_thresholds([s["entropy"] for s in samples], max(2, num_thresholds))
    rows = [compute_risk_coverage_at_threshold(samples, t) for t in thresholds]
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
            fieldnames=["threshold", "coverage", "risk", "answered", "correct", "errors"],
        )
        writer.writeheader()
        for row in curve_rows:
            writer.writerow(row)


def plot_curve(curve_rows: List[Dict], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = [r["coverage"] for r in curve_rows]
    y = [r["risk"] for r in curve_rows]
    t = [r["threshold"] for r in curve_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Main risk-coverage curve (line only; no markers)
    axes[0].plot(x, y, linewidth=2)
    axes[0].set_xlabel("Coverage")
    axes[0].set_ylabel("Risk")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)

    # Threshold trace for easy operating-point lookup
    axes[1].plot(x, t, linewidth=2, color="tab:orange")
    axes[1].set_xlabel("Coverage")
    axes[1].set_ylabel("Entropy threshold")
    axes[1].set_title("Threshold vs Coverage")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Entropy thresholding baseline for risk/coverage + AURC")
    parser.add_argument("--results_file", type=str, required=True, help="Path to results JSON")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--num_thresholds", type=int, default=200, help="Number of thresholds for sweep")
    parser.add_argument(
        "--max_entropy",
        type=float,
        default=None,
        help="Optional max entropy filter (useful for focusing low-entropy regime)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_path = Path(args.results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    results = load_results(results_path)
    samples = prepare_samples(results, max_entropy=args.max_entropy)
    if not samples:
        raise ValueError("No valid samples found with ground truth and numeric entropy.")

    curve = compute_curve(samples, num_thresholds=args.num_thresholds)
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
