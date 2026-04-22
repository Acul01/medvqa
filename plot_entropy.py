"""Create one combined normalized metric-vs-entropy plot."""

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


def build_bin_edges(entropies: List[float], num_bins: int) -> List[float]:
    """Build equal-width bin edges from entropy values."""
    if not entropies:
        return []
    min_e, max_e = min(entropies), max(entropies)
    if math.isclose(min_e, max_e):
        return [min_e, max_e]
    width = (max_e - min_e) / num_bins
    return [min_e + i * width for i in range(num_bins + 1)]


def bin_means(rows: List[Dict], key: str, bin_edges: List[float]) -> tuple[List[float], List[float]]:
    valid = [r for r in rows if is_valid_number(r.get(key))]
    if not valid:
        return [], []

    entropies = [r["entropy"] for r in valid]
    min_e, max_e = min(entropies), max(entropies)
    if math.isclose(min_e, max_e):
        return [min_e], [sum(r[key] for r in valid) / len(valid)]

    if len(bin_edges) < 2:
        return [], []

    num_bins = len(bin_edges) - 1
    bins: List[List[float]] = [[] for _ in range(num_bins)]
    centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(num_bins)]
    min_edge = bin_edges[0]
    max_edge = bin_edges[-1]
    bin_width = (max_edge - min_edge) / num_bins if num_bins else 0.0
    if math.isclose(bin_width, 0.0):
        return [min_edge], [sum(r[key] for r in valid) / len(valid)]

    for r in valid:
        idx = int((r["entropy"] - min_edge) / bin_width)
        idx = min(idx, num_bins - 1)
        bins[idx].append(float(r[key]))

    plot_x, plot_y = [], []
    for center, vals in zip(centers, bins):
        if vals:
            plot_x.append(center)
            plot_y.append(sum(vals) / len(vals))
    return plot_x, plot_y


def minmax_normalize(values: List[float]) -> List[float]:
    """Min-max normalize to [0,1]. Constant series mapped to 1.0."""
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def plot_entropy_relationships(rows: List[Dict], output_path: Path, num_bins: int) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax_dist = ax.twinx()

    metric_specs = [
        ("exact_match", "Exact Match", "#1f77b4"),
        ("bleu_4", "BLEU-4", "#ff7f0e"),
        ("rougeL", "ROUGE-L", "#2ca02c"),
        ("bert_f1", "BERTScore-F1", "#d62728"),
    ]
    all_entropies = [r["entropy"] for r in rows if is_valid_number(r.get("entropy"))]
    bin_edges = build_bin_edges(all_entropies, num_bins)
    if len(bin_edges) < 2:
        raise ValueError("Not enough entropy variation for binning.")

    # Distribution overlay: histogram of entropy counts (secondary axis)
    counts, _, _ = ax_dist.hist(
        all_entropies,
        bins=bin_edges,
        color="#b0b0b0",
        alpha=0.28,
        edgecolor="white",
        linewidth=0.6,
        label="Entropy distribution",
    )
    ax_dist.set_ylabel("Sample count per entropy bin", color="#6e6e6e")
    ax_dist.tick_params(axis="y", labelcolor="#6e6e6e")
    max_count = max(counts) if len(counts) else 0
    if max_count > 0:
        ax_dist.set_ylim(0, max_count * 1.25)

    any_plotted = False
    for metric_key, metric_name, color in metric_specs:
        valid = [r for r in rows if is_valid_number(r.get(metric_key))]
        if not valid:
            continue

        bx, by = bin_means(valid, metric_key, bin_edges=bin_edges)
        if not bx or not by:
            continue
        by_norm = minmax_normalize(by)
        ax.plot(
            bx,
            by_norm,
            marker="o",
            linewidth=2,
            color=color,
            label=metric_name,
        )
        any_plotted = True

    if not any_plotted:
        raise ValueError("No valid metric series available for plotting.")

    ax.set_xlabel("Entropy (higher -> lower, left to right)")
    ax.set_ylabel("Normalized metric value [0, 1]")
    ax.set_title("Normalized Evaluation Metrics vs Entropy (with entropy distribution)")
    ax.grid(alpha=0.25)
    line_handles, line_labels = ax.get_legend_handles_labels()
    dist_handles, dist_labels = ax_dist.get_legend_handles_labels()
    ax.legend(line_handles + dist_handles, line_labels + dist_labels, loc="best")
    # Lower entropy is better, so show descending entropy left→right.
    ax.invert_xaxis()

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
