"""
Plot entropy vs. performance metrics for a single results JSON file.

This script helps visualize how predictive entropy relates to model performance.
Expected input: results JSON produced by inference.py / inference_med.py.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def normalize_answer(answer: str) -> str:
    """Normalize answer for exact-match comparison."""
    answer = (answer or "").lower()
    answer = re.sub(r"[^\w\s]", "", answer)
    return " ".join(answer.split())


def is_valid_number(value) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(value)


def compute_exact_match(predicted: str, ground_truth: str) -> float:
    return float(normalize_answer(predicted) == normalize_answer(ground_truth))


def load_results(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Results file must contain a JSON list of samples.")
    return data


def build_plot_rows(results: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for r in results:
        entropy = r.get("entropy")
        gt = r.get("ground_truth_answer")
        pred = r.get("predicted_answer", "")
        if not is_valid_number(entropy):
            continue
        if gt is None or str(gt).strip() == "":
            continue

        row = {
            "entropy": float(entropy),
            "exact_match": compute_exact_match(str(pred), str(gt)),
            "bleu_1": r.get("bleu_1"),
            "bleu_4": r.get("bleu_4"),
            "rougeL": r.get("rougeL"),
            "bert_f1": r.get("bert_f1"),
        }
        rows.append(row)
    return rows


def pearson_correlation(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def bin_means(rows: List[Dict], key: str, num_bins: int) -> tuple[List[float], List[float]]:
    valid = [r for r in rows if is_valid_number(r.get(key))]
    if not valid:
        return [], []

    entropies = [r["entropy"] for r in valid]
    min_e, max_e = min(entropies), max(entropies)
    if min_e == max_e:
        return [min_e], [sum(r[key] for r in valid) / len(valid)]

    bin_width = (max_e - min_e) / num_bins
    bins: List[List[float]] = [[] for _ in range(num_bins)]
    centers = [min_e + (i + 0.5) * bin_width for i in range(num_bins)]

    for r in valid:
        idx = int((r["entropy"] - min_e) / bin_width)
        idx = min(idx, num_bins - 1)
        bins[idx].append(float(r[key]))

    plot_x, plot_y = [], []
    for center, vals in zip(centers, bins):
        if vals:
            plot_x.append(center)
            plot_y.append(sum(vals) / len(vals))
    return plot_x, plot_y


def plot_entropy_relationships(rows: List[Dict], output_path: Path, num_bins: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metric_specs = [
        ("exact_match", "Exact Match"),
        ("bleu_4", "BLEU-4"),
        ("rougeL", "ROUGE-L"),
        ("bert_f1", "BERTScore-F1"),
    ]

    for ax, (metric_key, metric_name) in zip(axes, metric_specs):
        valid = [r for r in rows if is_valid_number(r.get(metric_key))]
        xs = [r["entropy"] for r in valid]
        ys = [float(r[metric_key]) for r in valid]

        if not valid:
            ax.set_title(f"{metric_name} (no valid samples)")
            ax.set_xlabel("Entropy")
            ax.set_ylabel(metric_name)
            continue

        ax.scatter(xs, ys, alpha=0.35, s=18, label="samples")
        bx, by = bin_means(valid, metric_key, num_bins=num_bins)
        if bx and by:
            ax.plot(bx, by, color="crimson", linewidth=2, marker="o", label="bin mean")

        corr = pearson_correlation(xs, ys)
        corr_txt = f"{corr:.3f}" if is_valid_number(corr) else "n/a"
        ax.set_title(f"{metric_name} vs Entropy (r={corr_txt})")
        ax.set_xlabel("Entropy")
        ax.set_ylabel(metric_name)
        ax.grid(alpha=0.2)
        ax.legend(loc="best")

    fig.suptitle("Entropy vs Performance", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot entropy-performance relationships from results JSON")
    parser.add_argument("--results_file", type=str, required=True, help="Path to results JSON file")
    parser.add_argument("--output_file", type=str, default=None, help="Path for saved figure (png)")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of entropy bins for trend line")
    return parser.parse_args()


def main():
    args = parse_args()
    results_path = Path(args.results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    results = load_results(results_path)
    rows = build_plot_rows(results)
    if not rows:
        raise ValueError("No plottable rows found. Need valid ground_truth_answer + numeric entropy.")

    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = Path("results") / f"{results_path.stem}_entropy_vs_metrics.png"

    plot_entropy_relationships(rows, output_path=output_path, num_bins=max(2, args.num_bins))
    print(f"Loaded rows: {len(rows)}")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
