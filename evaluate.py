"""
Evaluation script for Medical VQA predictions.

Computes grouped and overall exact-match-based metrics plus
BLEU-1/BLEU-4, ROUGE-L, BERTScore-F1 and token-based entropy.
Appends one row per run to eval_runs.csv.
"""

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

EVAL_RUN_COLUMNS = [
    "start_time",
    "dataset",
    "model_path",
    "n_samples",
    "yesno_accuracy",
    "open_accuracy",
    "overall_accuracy",
    "bleu_1",
    "bleu_4",
    "rougeL",
    "bert_f1",
    "token_based_entropy",
]


BLEU_WEIGHTS = {
    1: (1, 0, 0, 0),
    4: (0.25, 0.25, 0.25, 0.25),
}


def normalize_answer(answer: str) -> str:
    """Normalize answer for robust exact-match comparison."""
    answer = (answer or "").lower()
    answer = re.sub(r"[^\w\s]", "", answer)
    answer = " ".join(answer.split())
    return answer


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _compute_bleu_1_4(prediction: str, reference: str) -> Dict[str, float]:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoother = SmoothingFunction().method1
    except ImportError:
        return {"bleu_1": 0.0, "bleu_4": 0.0}
    pred_tokens = prediction.lower().strip().split()
    ref_tokens = reference.lower().strip().split()
    if not pred_tokens or not ref_tokens:
        return {"bleu_1": 0.0, "bleu_4": 0.0}
    return {
        f"bleu_{n}": sentence_bleu(
            [ref_tokens],
            pred_tokens,
            weights=weights,
            smoothing_function=smoother,
        )
        for n, weights in BLEU_WEIGHTS.items()
    }


def _compute_rouge_l(prediction: str, reference: str) -> float:
    try:
        from rouge_score import rouge_scorer
        rouge_scorer_inst = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    except ImportError:
        return 0.0
    pred = prediction.lower().strip()
    ref = reference.lower().strip()
    if not pred or not ref:
        return 0.0
    return rouge_scorer_inst.score(ref, pred)["rougeL"].fmeasure


def _compute_bert_f1(predictions: List[str], references: List[str]) -> float:
    if not predictions or not references:
        return 0.0
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        return 0.0
    _, _, f1 = bert_score_fn(predictions, references, lang="en", verbose=False)
    return f1.mean().item()


def _extract_meta_from_filename(results_path: Path) -> Tuple[str, str, str]:
    """
    Parse {model}_{dataset}_{n_samples}.json style filenames.
    Returns (dataset, model_path, n_samples) with robust fallbacks.
    """
    stem = results_path.stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return parts[-2], "_".join(parts[:-2]), parts[-1]
    return "unknown", "unknown", "unknown"


def _group_samples(results: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Return (valid, yesno, open_ended) where valid samples have GT present."""
    valid = []
    yesno = []
    open_ended = []

    for r in results:
        gt = r.get("ground_truth_answer")
        if gt is None or str(gt).strip() == "":
            continue
        valid.append(r)
        gt_norm = normalize_answer(str(gt))
        if gt_norm in ("yes", "no"):
            yesno.append(r)
        else:
            open_ended.append(r)

    return valid, yesno, open_ended


def _exact_match_binary_metrics(samples: List[Dict]) -> Dict[str, float]:
    """
    Exact-match-based binary metrics (correct vs incorrect).

    In this setup precision/recall/f1 are defined over exact-match correctness
    and reported consistently with the selected exact-match binary formulation.
    """
    total = len(samples)
    if total == 0:
        return {"accuracy": 0.0}

    tp = 0
    for s in samples:
        pred = normalize_answer(str(s.get("predicted_answer", "")))
        gt = normalize_answer(str(s.get("ground_truth_answer", "")))
        if pred == gt:
            tp += 1

    accuracy = _safe_div(tp, total)
    return {"accuracy": accuracy}


def _compute_text_metrics(samples: List[Dict]) -> Dict[str, float]:
    """Compute BLEU-1, BLEU-4, ROUGE-L and BERTScore-F1 over valid samples."""
    if not samples:
        return {"bleu_1": 0.0, "bleu_4": 0.0, "rougeL": 0.0, "bert_f1": 0.0}

    bleu1_sum = 0.0
    bleu4_sum = 0.0
    rougeL_sum = 0.0
    preds = []
    refs = []

    for s in samples:
        pred = str(s.get("predicted_answer", ""))
        gt = str(s.get("ground_truth_answer", ""))
        b = _compute_bleu_1_4(pred, gt)
        r_rougeL = _compute_rouge_l(pred, gt)
        bleu1_sum += b.get("bleu_1", 0.0)
        bleu4_sum += b.get("bleu_4", 0.0)
        rougeL_sum += r_rougeL
        preds.append(pred)
        refs.append(gt)

    bert = _compute_bert_f1(preds, refs)
    n = len(samples)
    return {
        "bleu_1": _safe_div(bleu1_sum, n),
        "bleu_4": _safe_div(bleu4_sum, n),
        "rougeL": _safe_div(rougeL_sum, n),
        "bert_f1": bert,
    }


def _compute_token_based_entropy(results: List[Dict]) -> float:
    """Mean over valid per-sample entropy values."""
    values = []
    for r in results:
        value = r.get("entropy")
        if isinstance(value, (int, float)) and not math.isnan(value):
            values.append(float(value))
    return _safe_div(sum(values), len(values)) if values else float("nan")


def _append_eval_runs_csv(eval_runs_path: Path, row: Dict) -> None:
    """Append one evaluation run row, writing header if file is new."""
    eval_runs_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not eval_runs_path.exists()
    with open(eval_runs_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_RUN_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Medical VQA predictions")
    parser.add_argument("--results_file", type=str, required=True, help="JSON file with inference results")
    parser.add_argument("--output_file", type=str, default=None, help="Optional JSON file for metrics output")
    parser.add_argument(
        "--eval_runs_file",
        type=str,
        default="results/eval_runs.csv",
        help="CSV file that stores one aggregated row per evaluation run",
    )
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
    valid_samples, yesno_samples, open_samples = _group_samples(results)

    yesno_metrics = _exact_match_binary_metrics(yesno_samples)
    open_metrics = _exact_match_binary_metrics(open_samples)
    overall_metrics = _exact_match_binary_metrics(valid_samples)
    text_metrics = _compute_text_metrics(valid_samples)
    token_entropy = _compute_token_based_entropy(results)

    metrics = {
        "start_time": start_time,
        "dataset": dataset,
        "model_path": model_path,
        "n_samples": n_samples,
        "counts": {
            "total_rows": len(results),
            "valid_rows": len(valid_samples),
            "yesno_rows": len(yesno_samples),
            "open_rows": len(open_samples),
        },
        "yesno": yesno_metrics,
        "open": open_metrics,
        "overall": overall_metrics,
        "text": text_metrics,
        "token_based_entropy": token_entropy,
    }

    run_row = {
        "start_time": start_time,
        "dataset": dataset,
        "model_path": model_path,
        "n_samples": n_samples,
        "yesno_accuracy": f"{yesno_metrics['accuracy']:.4f}",
        "open_accuracy": f"{open_metrics['accuracy']:.4f}",
        "overall_accuracy": f"{overall_metrics['accuracy']:.4f}",
        "bleu_1": f"{text_metrics['bleu_1']:.4f}",
        "bleu_4": f"{text_metrics['bleu_4']:.4f}",
        "rougeL": f"{text_metrics['rougeL']:.4f}",
        "bert_f1": f"{text_metrics['bert_f1']:.4f}",
        "token_based_entropy": "" if math.isnan(token_entropy) else f"{token_entropy:.6f}",
    }

    _append_eval_runs_csv(Path(args.eval_runs_file), run_row)

    print("\n" + "=" * 72)
    print("Evaluation Metrics")
    print("=" * 72)
    print(f"Start time: {start_time}")
    print(f"Dataset: {dataset}")
    print(f"Model path: {model_path}")
    print(f"n_samples: {n_samples}")
    print(f"Rows: total={len(results)}, valid={len(valid_samples)}, yes/no={len(yesno_samples)}, open={len(open_samples)}")
    print("-" * 72)
    print(f"Yes/No  -> accuracy={yesno_metrics['accuracy']:.4f}")
    print(f"Open    -> accuracy={open_metrics['accuracy']:.4f}")
    print(f"Overall -> accuracy={overall_metrics['accuracy']:.4f}")
    print("-" * 72)
    print(
        f"BLEU-1={text_metrics['bleu_1']:.4f}, BLEU-4={text_metrics['bleu_4']:.4f}, "
        f"ROUGE-L={text_metrics['rougeL']:.4f}, BERTScore-F1={text_metrics['bert_f1']:.4f}"
    )
    print(f"Token-based entropy={token_entropy:.6f}" if not math.isnan(token_entropy) else "Token-based entropy=N/A")
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
