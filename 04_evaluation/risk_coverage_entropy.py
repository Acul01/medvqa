"""
Entropy-threshold baseline for reliable VQA.

Computes risk/coverage sweep, plots risk-coverage curve, and reports AURC
(Area Under the Risk-Coverage Curve) for entropy-only and combined uncertainty.
"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt


from utils.mc_vqa import correct_index_to_letter, parse_predicted_letter
from utils.paths import get_results_dir

QUESTION_TYPES = ["closed", "open", "overall"]
ANSWER_TYPE_LABEL = {"closed": "closed", "open": "open", "overall": "all"}
UNCERTAINTY_MEASURES = ["entropy", "combined", "confidence"]
RISK_TARGETS = [0.01, 0.05, 0.10]
COVERAGE_TARGETS = [0.20, 0.50, 0.80]


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


def filter_by_question_type(results: List[Dict], question_type: str) -> List[Dict]:
    if question_type == "overall":
        return results
    return [
        r
        for r in results
        if str(r.get("question_type", "")).strip().lower() == question_type
    ]


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


def agreement_letters(row: Dict) -> List[str]:
    """Extract non-empty answer letters from agreement_samples."""
    raw = row.get("agreement_samples")
    if not isinstance(raw, list):
        return []
    letters: List[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            letters.append(item.strip().upper())
    return letters


def compute_vote_entropy(samples: Sequence[str]) -> float:
    """
    Shannon entropy of the answer distribution across generations.
    Measures between-generation inconsistency.
    """
    if not samples:
        return 0.0
    counts = Counter(samples)
    total = float(sum(counts.values()))
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0.0:
            entropy -= p * math.log(p)
    return float(entropy)


def compute_combined_uncertainties(
    token_H: Sequence[float],
    vote_H: Sequence[float],
    alpha: float,
) -> List[float]:
    """
    Symmetric Min-Max normalize token and vote entropy, then combine.

    Higher values indicate higher uncertainty.
    """
    if len(token_H) != len(vote_H):
        raise ValueError("token_H and vote_H must have the same length.")
    if not token_H:
        return []

    token_min = min(token_H)
    token_max = max(token_H)
    vote_min = min(vote_H)
    vote_max = max(vote_H)
    token_span = token_max - token_min + 1e-12
    vote_span = vote_max - vote_min + 1e-12

    combined: List[float] = []
    for tok, vote in zip(token_H, vote_H):
        token_norm = (tok - token_min) / token_span
        vote_norm = (vote - vote_min) / vote_span
        combined.append(alpha * token_norm + (1.0 - alpha) * vote_norm)
    return combined


def prepare_dual_samples(
    results: List[Dict],
    risk_metric: str,
    max_entropy: float | None,
    alpha: float,
) -> List[Dict]:
    """
    Build samples usable for entropy, combined, and confidence AURC.

    Requires valid entropy, confidence, non-empty agreement_samples, and a risk metric.
    Combined uncertainty is Min-Max normalized over this subset only.
    """
    candidates: List[Dict] = []
    token_H: List[float] = []
    vote_H: List[float] = []

    for r in results:
        entropy = r.get("entropy")
        if not is_valid_number(entropy):
            continue
        entropy = float(entropy)
        if max_entropy is not None and entropy > max_entropy:
            continue

        confidence = r.get("confidence")
        if not is_valid_number(confidence):
            continue
        confidence = float(confidence)

        letters = agreement_letters(r)
        if not letters:
            continue

        metric_val = metric_value_from_row(r, risk_metric=risk_metric)
        if metric_val is None:
            continue

        candidates.append(
            {
                "entropy": entropy,
                "confidence": confidence,
                "metric_value": metric_val,
            }
        )
        token_H.append(entropy)
        vote_H.append(compute_vote_entropy(letters))

    combined = compute_combined_uncertainties(token_H, vote_H, alpha=alpha)
    samples: List[Dict] = []
    for cand, comb in zip(candidates, combined):
        samples.append(
            {
                "entropy": cand["entropy"],
                "confidence": cand["confidence"],
                "combined_uncertainty": float(comb),
                "metric_value": cand["metric_value"],
            }
        )
    return samples


def samples_with_score(dual_samples: List[Dict], score_key: str) -> List[Dict]:
    return [
        {"score_value": float(s[score_key]), "metric_value": float(s["metric_value"])}
        for s in dual_samples
    ]


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


def plot_curve(
    curve_rows: List[Dict],
    out_path: Path,
    title: str,
    threshold_label: str = "Entropy threshold",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = [r["coverage"] for r in curve_rows]
    y = [r["risk"] for r in curve_rows]
    t = [r["threshold"] for r in curve_rows]

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    ax2 = ax1.twinx()

    # Main risk-coverage curve (left y-axis)
    ax1.plot(x, y, linewidth=2, color="tab:blue", label="Risk")
    ax1.set_xlabel("Coverage")
    ax1.set_ylabel("Risk", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title(title)
    ax1.grid(alpha=0.25)

    # Threshold trace (right y-axis)
    ax2.plot(x, t, linewidth=2, color="tab:orange", label=threshold_label)
    ax2.set_ylabel(threshold_label, color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_comparison_curves(
    curves_by_measure: Dict[str, List[Dict]],
    aurc_by_measure: Dict[str, float],
    out_path: Path,
    title: str,
) -> None:
    """Overlay entropy / combined / confidence risk-coverage curves in one figure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "entropy": "tab:blue",
        "combined": "tab:orange",
        "confidence": "tab:green",
    }
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for measure in UNCERTAINTY_MEASURES:
        curve = curves_by_measure.get(measure) or []
        if not curve:
            continue
        x = [r["coverage"] for r in curve]
        y = [r["risk"] for r in curve]
        aurc = float(aurc_by_measure.get(measure, float("nan")))
        label = f"{measure} (AURC={aurc:.4f})" if not math.isnan(aurc) else measure
        ax.plot(x, y, linewidth=2, color=colors.get(measure, None), label=label)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def artifact_answer_type(question_type: str) -> str:
    return ANSWER_TYPE_LABEL.get(question_type, question_type)


def artifact_base(stem: str, question_type: str, uncertainty: str) -> str:
    """Naming: {model_dataset_stem}_{all|open|closed}_{entropy|combined|confidence}."""
    return f"{stem}_{artifact_answer_type(question_type)}_{uncertainty}"


def artifact_paths(
    out_dir: Path,
    stem: str,
    question_type: str,
    uncertainty: str,
) -> tuple[Path, Path]:
    base = artifact_base(stem, question_type, uncertainty)
    csv_path = out_dir / f"{base}.csv"
    plot_path = out_dir / f"{base}.png"
    return csv_path, plot_path


def comparison_artifact_path(out_dir: Path, stem: str, question_type: str) -> Path:
    return out_dir / f"{stem}_{artifact_answer_type(question_type)}_comparison.png"


def empty_type_metrics(filtered_rows: int) -> Dict:
    return {
        "filtered_rows": filtered_rows,
        "valid_samples": 0,
        "num_thresholds": 0,
        "aurc": 0.0,
        **coverage_at_risk([], risk_targets=RISK_TARGETS),
        **risk_at_coverage([], cov_targets=COVERAGE_TARGETS),
    }


def metrics_from_curve(
    curve: List[Dict],
    *,
    filtered_rows: int,
    valid_samples: int,
) -> Dict:
    return {
        "filtered_rows": filtered_rows,
        "valid_samples": valid_samples,
        "num_thresholds": len(curve),
        "aurc": aurc_trapezoid(curve),
        **coverage_at_risk(curve, risk_targets=RISK_TARGETS),
        **risk_at_coverage(curve, cov_targets=COVERAGE_TARGETS),
        "curve": curve,
    }


def evaluate_question_type(
    results: List[Dict],
    question_type: str,
    *,
    risk_metric: str,
    score_field: str,
    max_entropy: float | None,
    num_thresholds: int,
) -> Dict:
    filtered_results = filter_by_question_type(results, question_type=question_type)
    if not filtered_results:
        return {**empty_type_metrics(0), "curve": []}

    samples = prepare_samples(
        filtered_results,
        risk_metric=risk_metric,
        score_field=score_field,
        max_entropy=max_entropy,
    )
    if not samples:
        return {**empty_type_metrics(len(filtered_results)), "curve": []}

    answer_when_higher = score_field in {"agreement", "confidence"}
    curve = compute_curve(samples, num_thresholds=num_thresholds, answer_when_higher=answer_when_higher)
    area = aurc_trapezoid(curve)
    c_at_r = coverage_at_risk(curve, risk_targets=RISK_TARGETS)
    r_at_c = risk_at_coverage(curve, cov_targets=COVERAGE_TARGETS)

    return {
        "filtered_rows": len(filtered_results),
        "valid_samples": len(samples),
        "num_thresholds": len(curve),
        "aurc": area,
        **c_at_r,
        **r_at_c,
        "curve": curve,
    }


def evaluate_dual_question_type(
    results: List[Dict],
    question_type: str,
    *,
    risk_metric: str,
    max_entropy: float | None,
    num_thresholds: int,
    combined_alpha: float,
) -> Dict:
    """
    Evaluate entropy, combined, and confidence AURC on the same sample set.
    """
    filtered_results = filter_by_question_type(results, question_type=question_type)
    empty = {**empty_type_metrics(0 if not filtered_results else len(filtered_results)), "curve": []}
    if not filtered_results:
        return {
            "entropy": empty,
            "combined": {**empty, "alpha": combined_alpha},
            "confidence": empty,
        }

    dual_samples = prepare_dual_samples(
        filtered_results,
        risk_metric=risk_metric,
        max_entropy=max_entropy,
        alpha=combined_alpha,
    )
    if not dual_samples:
        empty = {**empty_type_metrics(len(filtered_results)), "curve": []}
        return {
            "entropy": empty,
            "combined": {**empty, "alpha": combined_alpha},
            "confidence": empty,
        }

    n = len(dual_samples)
    filtered_n = len(filtered_results)
    entropy_curve = compute_curve(
        samples_with_score(dual_samples, "entropy"),
        num_thresholds=num_thresholds,
        answer_when_higher=False,
    )
    combined_curve = compute_curve(
        samples_with_score(dual_samples, "combined_uncertainty"),
        num_thresholds=num_thresholds,
        answer_when_higher=False,
    )
    confidence_curve = compute_curve(
        samples_with_score(dual_samples, "confidence"),
        num_thresholds=num_thresholds,
        answer_when_higher=True,
    )
    entropy_metrics = metrics_from_curve(
        entropy_curve,
        filtered_rows=filtered_n,
        valid_samples=n,
    )
    combined_metrics = metrics_from_curve(
        combined_curve,
        filtered_rows=filtered_n,
        valid_samples=n,
    )
    combined_metrics["alpha"] = combined_alpha
    confidence_metrics = metrics_from_curve(
        confidence_curve,
        filtered_rows=filtered_n,
        valid_samples=n,
    )
    return {
        "entropy": entropy_metrics,
        "combined": combined_metrics,
        "confidence": confidence_metrics,
    }


def print_aurc_table(aurc_by_type: Dict[str, float], title: str = "AURC") -> None:
    col_width = 12
    headers = QUESTION_TYPES
    values = [aurc_by_type[qtype] for qtype in headers]
    print(f"\n{title}")
    print("".join(h.ljust(col_width) for h in headers))
    print("".join(f"{v:.6f}".ljust(col_width) for v in values))


def compute_dual_aurc_summary(
    results: List[Dict],
    *,
    combined_alpha: float = 0.5,
    risk_metric: str = "match",
    max_entropy: float | None = None,
    num_thresholds: int = 200,
) -> Dict:
    """
    Compact dual-AURC summary for post-inference / MC-eval hooks.

    Does not write curve artifacts. Returns NaN AURC values when a split has
    no valid dual samples (missing entropy/confidence and/or agreement_samples).
    """
    summary: Dict = {
        "combined_alpha": combined_alpha,
        "risk_metric": risk_metric,
        "num_thresholds": num_thresholds,
    }
    for question_type in QUESTION_TYPES:
        dual = evaluate_dual_question_type(
            results,
            question_type,
            risk_metric=risk_metric,
            max_entropy=max_entropy,
            num_thresholds=num_thresholds,
            combined_alpha=combined_alpha,
        )
        entropy_eval = dual["entropy"]
        combined_eval = dual["combined"]
        confidence_eval = dual["confidence"]
        valid = int(entropy_eval.get("valid_samples", 0))
        if valid <= 0:
            aurc_entropy = float("nan")
            aurc_combined = float("nan")
            aurc_confidence = float("nan")
        else:
            aurc_entropy = float(entropy_eval["aurc"])
            aurc_combined = float(combined_eval["aurc"])
            aurc_confidence = float(confidence_eval["aurc"])
        summary[question_type] = {
            "valid_samples": valid,
            "filtered_rows": int(entropy_eval.get("filtered_rows", 0)),
            "aurc_entropy": aurc_entropy,
            "aurc_combined": aurc_combined,
            "aurc_confidence": aurc_confidence,
        }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entropy + combined-uncertainty risk/coverage + AURC"
    )
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
        help="Kept for evaluate_question_type API; main() always runs entropy + combined",
    )
    parser.add_argument(
        "--combined_alpha",
        type=float,
        default=0.5,
        help="Weight for token entropy in combined uncertainty (1-alpha = vote entropy)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.combined_alpha <= 1.0:
        raise ValueError(f"--combined_alpha must be in [0, 1], got {args.combined_alpha}")

    results_path = Path(args.results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    results = load_results(results_path)
    out_dir = Path(args.output_dir)
    stem = results_path.stem

    type_results: Dict[str, Dict] = {}
    aurc_by_measure: Dict[str, Dict[str, float]] = {
        measure: {} for measure in UNCERTAINTY_MEASURES
    }
    threshold_labels = {
        "entropy": "Entropy threshold",
        "combined": "Combined uncertainty threshold",
        "confidence": "Confidence threshold",
    }

    for question_type in QUESTION_TYPES:
        dual = evaluate_dual_question_type(
            results,
            question_type,
            risk_metric=args.risk_metric,
            max_entropy=args.max_entropy,
            num_thresholds=args.num_thresholds,
            combined_alpha=args.combined_alpha,
        )
        entropy_eval = dual["entropy"]
        combined_eval = dual["combined"]
        confidence_eval = dual["confidence"]
        curves = {
            "entropy": entropy_eval.pop("curve"),
            "combined": combined_eval.pop("curve"),
            "confidence": confidence_eval.pop("curve"),
        }

        type_results[question_type] = {
            **entropy_eval,
            "combined": combined_eval,
            "confidence": confidence_eval,
        }
        aurc_by_measure["entropy"][question_type] = entropy_eval["aurc"]
        aurc_by_measure["combined"][question_type] = combined_eval["aurc"]
        aurc_by_measure["confidence"][question_type] = confidence_eval["aurc"]

        if curves["entropy"]:
            answer_label = artifact_answer_type(question_type)
            for measure in UNCERTAINTY_MEASURES:
                csv_path, plot_path = artifact_paths(out_dir, stem, question_type, measure)
                base = artifact_base(stem, question_type, measure)
                save_curve_csv(curves[measure], csv_path)
                plot_curve(
                    curves[measure],
                    plot_path,
                    title=f"Risk-Coverage ({base})",
                    threshold_label=threshold_labels[measure],
                )
                print(f"  Saved: {csv_path}")
                print(f"  Saved: {plot_path}")

            comparison_path = comparison_artifact_path(out_dir, stem, question_type)
            plot_comparison_curves(
                curves_by_measure=curves,
                aurc_by_measure={
                    "entropy": entropy_eval["aurc"],
                    "combined": combined_eval["aurc"],
                    "confidence": confidence_eval["aurc"],
                },
                out_path=comparison_path,
                title=f"Risk-Coverage comparison ({stem}_{answer_label})",
            )
            print(f"  Saved: {comparison_path}")
            print(
                f"{answer_label}: valid_samples={entropy_eval['valid_samples']}, "
                f"AURC_entropy={entropy_eval['aurc']:.6f}, "
                f"AURC_combined={combined_eval['aurc']:.6f}, "
                f"AURC_confidence={confidence_eval['aurc']:.6f}"
            )
        else:
            print(
                f"{artifact_answer_type(question_type)}: no valid samples "
                f"(filtered_rows={entropy_eval['filtered_rows']})"
            )

    if type_results["overall"]["valid_samples"] == 0:
        raise ValueError(
            "No valid samples found for question_type='overall'. "
            "AURC requires numeric entropy, confidence, non-empty agreement_samples, "
            "and a valid risk metric per row."
        )

    summary_path = out_dir / f"{stem}_risk_coverage_summary.json"
    summary = {
        "results_file": str(results_path),
        "total_input_rows": len(results),
        "max_entropy_filter": args.max_entropy,
        "risk_metric": args.risk_metric,
        "score_field": "entropy",
        "combined_alpha": args.combined_alpha,
        "artifact_naming": (
            "{stem}_{all|open|closed}_{entropy|combined|confidence} and "
            "{stem}_{all|open|closed}_comparison.png"
        ),
        "counts": {
            "closed_rows": type_results["closed"]["filtered_rows"],
            "open_rows": type_results["open"]["filtered_rows"],
            "overall_rows": type_results["overall"]["filtered_rows"],
        },
        "closed": type_results["closed"],
        "open": type_results["open"],
        "overall": type_results["overall"],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {summary_path}")
    for measure in UNCERTAINTY_MEASURES:
        print_aurc_table(aurc_by_measure[measure], title=f"AURC {measure}")


if __name__ == "__main__":
    main()
