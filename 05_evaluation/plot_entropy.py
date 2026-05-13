"""Plot MC correctness vs entropy."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from utils.mc_vqa import correct_index_to_letter, parse_predicted_letter


def is_valid_number(value) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(value)


def load_results(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Results file must contain a JSON list of samples.")
    return data


def _resolve_gt_letter(row: Dict) -> str | None:
    gt = row.get("ground_truth_letter")
    if isinstance(gt, str) and gt.strip():
        return gt.strip().upper()
    idx = row.get("correct_index")
    if isinstance(idx, int) and idx >= 0:
        return correct_index_to_letter(idx)
    return None


def _resolve_pred_letter(row: Dict) -> str | None:
    pred = row.get("predicted_letter")
    if isinstance(pred, str) and pred.strip():
        return pred.strip().upper()
    options = row.get("options")
    n = len(options) if isinstance(options, list) else 5
    return parse_predicted_letter(str(row.get("predicted_answer", "")), num_options=n)


def build_plot_rows(results: List[Dict], max_entropy: float | None = None) -> List[Dict]:
    rows = []
    for row in results:
        entropy = row.get("entropy")
        if not is_valid_number(entropy):
            continue
        entropy = float(entropy)
        if max_entropy is not None and entropy > max_entropy:
            continue

        gt = _resolve_gt_letter(row)
        pred = _resolve_pred_letter(row)
        if gt is None:
            continue

        rows.append({
            "entropy": entropy,
            "exact_match": float(pred == gt),
        })
    return rows


def build_bin_edges(entropies: List[float], num_bins: int) -> List[float]:
    if not entropies:
        return []
    min_e, max_e = min(entropies), max(entropies)
    if math.isclose(min_e, max_e):
        return [min_e, max_e]
    width = (max_e - min_e) / num_bins
    return [min_e + i * width for i in range(num_bins + 1)]


def bin_means(rows: List[Dict], bin_edges: List[float]) -> tuple[List[float], List[float]]:
    if len(bin_edges) < 2:
        return [], []
    num_bins = len(bin_edges) - 1
    bins: List[List[float]] = [[] for _ in range(num_bins)]
    centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(num_bins)]
    min_edge = bin_edges[0]
    max_edge = bin_edges[-1]
    if math.isclose(min_edge, max_edge):
        return [min_edge], [sum(r["exact_match"] for r in rows) / len(rows)]

    width = (max_edge - min_edge) / num_bins
    for row in rows:
        idx = int((row["entropy"] - min_edge) / width)
        idx = min(max(0, idx), num_bins - 1)
        bins[idx].append(row["exact_match"])

    x, y = [], []
    for center, vals in zip(centers, bins):
        if vals:
            x.append(center)
            y.append(sum(vals) / len(vals))
    return x, y


def plot_entropy_relationship(rows: List[Dict], output_path: Path, num_bins: int) -> None:
    all_entropies = [r["entropy"] for r in rows]
    bin_edges = build_bin_edges(all_entropies, num_bins)
    if len(bin_edges) < 2:
        raise ValueError("Not enough entropy variation for plotting.")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax_dist = ax.twinx()

    ax_dist.hist(all_entropies, bins=bin_edges, color="#b0b0b0", alpha=0.28, edgecolor="white", linewidth=0.6, label="Entropy distribution")
    ax_dist.set_ylabel("Sample count", color="#6e6e6e")
    ax_dist.tick_params(axis="y", labelcolor="#6e6e6e")

    bx, by = bin_means(rows, bin_edges=bin_edges)
    ax.plot(bx, by, marker="o", linewidth=2, color="#1f77b4", label="MC accuracy")

    ax.set_xlabel("Entropy (higher -> lower, left to right)")
    ax.set_ylabel("Accuracy")
    ax.set_title("MC Accuracy vs Entropy")
    ax.grid(alpha=0.25)
    ax.invert_xaxis()

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_dist.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot MC accuracy vs entropy")
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--max_entropy", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    results_path = Path(args.results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    results = load_results(results_path)
    rows = build_plot_rows(results, max_entropy=args.max_entropy)
    if not rows:
        raise ValueError("No plottable rows found with letter labels and numeric entropy.")

    output_path = Path(args.output_file) if args.output_file else Path("07_results") / f"{results_path.stem}_entropy_vs_mc_accuracy.png"
    plot_entropy_relationship(rows, output_path=output_path, num_bins=max(2, args.num_bins))

    print(f"Loaded rows: {len(rows)}")
    print(f"Entropy filter: <= {args.max_entropy}")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
