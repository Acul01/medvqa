"""
Shared utilities for saving inference results in multiple formats
and maintaining a results history log.
"""

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer

_smoother = SmoothingFunction().method1

BLEU_WEIGHTS = {
    1: (1, 0, 0, 0),
    2: (0.5, 0.5, 0, 0),
    3: (1/3, 1/3, 1/3, 0),
    4: (0.25, 0.25, 0.25, 0.25),
}


# ── Answer cleaning ─────────────────────────────────────────────────────────────

def clean_answer(text: str) -> str:
    """Clean a generated answer by removing noise tokens.

    Applied identically for both LLaVA and LLaVA-Med outputs.

    Steps:
        1. Keep first line only (discard multi-line garbage)
        2. Truncate yes/no answers that are followed by explanations
           (e.g. "No, there is no leakage" → "no")
        3. Remove emoji / non-ASCII symbols
        4. Collapse multiple whitespace
        5. Strip trailing isolated digits (e.g. "yes 4" → "yes"),
           but keep standalone numbers ("4" stays "4")
        6. Strip trailing punctuation
    """
    text = text.split("\n")[0].strip()
    m = re.match(r'^(yes|no)\b', text, re.IGNORECASE)
    if m and len(text) > len(m.group(1)):
        text = m.group(1).lower()
    text = re.sub(r'[^\x00-\x7F\u00C0-\u00FF]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+\d+$', '', text)
    text = re.sub(r'[.,;:!?]+$', '', text)
    return text.strip()


_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on",
    "to", "and", "or", "it", "its", "this", "that", "for", "with",
    "by", "at", "from", "as", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "but", "not", "so", "if", "than",
})


def normalize_for_comparison(text: str) -> str:
    """Normalize an answer string for exact-match comparison.

    Applied symmetrically to both prediction and ground truth so that
    superficial differences (articles, punctuation, hyphens) don't cause
    false mismatches.

    Steps:
        1. Lowercase
        2. Replace hyphens and slashes with spaces
        3. Remove all punctuation except alphanumeric and whitespace
        4. Remove stop words (articles, prepositions, etc.)
        5. Collapse whitespace
    """
    text = text.lower().strip()
    text = re.sub(r'[-/]', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in _STOP_WORDS]
    return " ".join(tokens)


def classify_answer_type(ground_truth: str) -> str:
    """Classify a sample as 'yes_no' or 'open' based on its ground truth answer."""
    gt = ground_truth.lower().strip()
    return "yes_no" if gt in ("yes", "no") else "open"


def compute_bleu_scores(prediction: str, reference: str) -> dict:
    """Compute BLEU-1 through BLEU-4 for a single prediction/reference pair."""
    pred_tokens = prediction.lower().strip().split()
    ref_tokens = reference.lower().strip().split()
    if not pred_tokens or not ref_tokens:
        return {f"bleu_{n}": 0.0 for n in range(1, 5)}
    scores = {}
    for n, weights in BLEU_WEIGHTS.items():
        scores[f"bleu_{n}"] = sentence_bleu(
            [ref_tokens], pred_tokens,
            weights=weights, smoothing_function=_smoother,
        )
    return scores


_rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]


def compute_rouge_scores(prediction: str, reference: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 for a single prediction/reference pair."""
    pred = prediction.lower().strip()
    ref = reference.lower().strip()
    if not pred or not ref:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = _rouge_scorer.score(ref, pred)
    return {k: scores[k].fmeasure for k in ROUGE_KEYS}


def compute_bert_scores(predictions: list[str], references: list[str]) -> dict:
    """Compute BERTScore (Precision, Recall, F1) for a batch of predictions.

    Returns dict with 'per_sample' (list of per-sample F1) and
    'avg' (dict with mean precision, recall, f1).
    """
    if not predictions or not references:
        return {"per_sample": [], "avg": {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}}

    P, R, F1 = bert_score_fn(predictions, references, lang="en", verbose=False)
    per_sample = F1.tolist()
    return {
        "per_sample": per_sample,
        "avg": {
            "bert_precision": P.mean().item(),
            "bert_recall": R.mean().item(),
            "bert_f1": F1.mean().item(),
        },
    }


RESULTS_DIR = Path("results")
HISTORY_FILE = RESULTS_DIR / "results_history.csv"

HISTORY_COLUMNS = [
    "timestamp",
    "model",
    "finetuning",
    "dataset",
    "n_samples",
    "exact_match_accuracy",
    "yesno_accuracy",
    "open_accuracy",
    "bleu_1",
    "bleu_2",
    "bleu_3",
    "bleu_4",
    "rouge1",
    "rouge2",
    "rougeL",
    "bert_f1",
]


def _derive_model_short_name(model_path: str) -> str:
    """Derive a short, filename-safe model name from the full path/ID."""
    name = Path(model_path).name
    name_lower = name.lower()
    if "llava-med" in name_lower:
        return "llava-med"
    if "llava" in name_lower:
        return "llava"
    return name.replace("/", "_").replace(" ", "_")


def save_results(
    results: list[dict],
    model_path: str,
    dataset: str,
    n_samples: int | None,
    finetuned: bool,
    exact_acc: float,
    bleu_scores: dict[str, float],
    rouge_scores: dict[str, float] | None = None,
    bert_scores: dict[str, float] | None = None,
    yesno_acc: float | None = None,
    open_acc: float | None = None,
    results_dir: str | Path = RESULTS_DIR,
) -> None:
    """Save inference results as JSON, CSV, XLSX and append to history.

    Files are written to ``results_dir`` with the naming scheme
    ``{model}_{dataset}_{n_samples}.{ext}``.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_name = _derive_model_short_name(model_path)
    n_tag = str(n_samples) if n_samples is not None else "all"
    base_name = f"{model_name}_{dataset}_{n_tag}"

    # ── JSON ────────────────────────────────────────────────────────────────
    json_path = results_dir / f"{base_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path}")

    # ── CSV ─────────────────────────────────────────────────────────────────
    csv_path = results_dir / f"{base_name}.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  Saved: {csv_path}")

    # ── XLSX ────────────────────────────────────────────────────────────────
    xlsx_path = results_dir / f"{base_name}.xlsx"
    try:
        df.to_excel(xlsx_path, index=False)
        print(f"  Saved: {xlsx_path}")
    except ModuleNotFoundError:
        print(f"  Skipped XLSX (install openpyxl: pip install openpyxl)")

    # ── Results history ─────────────────────────────────────────────────────
    _append_history(
        model=model_name,
        finetuned=finetuned,
        dataset=dataset,
        n_samples=n_tag,
        exact_acc=exact_acc,
        bleu_scores=bleu_scores,
        rouge_scores=rouge_scores,
        bert_scores=bert_scores,
        yesno_acc=yesno_acc,
        open_acc=open_acc,
        results_dir=results_dir,
    )


def _append_history(
    model: str,
    finetuned: bool,
    dataset: str,
    n_samples: str,
    exact_acc: float,
    bleu_scores: dict[str, float],
    rouge_scores: dict[str, float] | None = None,
    bert_scores: dict[str, float] | None = None,
    yesno_acc: float | None = None,
    open_acc: float | None = None,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Append one row to the results history CSV."""
    history_path = results_dir / "results_history.csv"
    write_header = not history_path.exists()

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "finetuning": "yes" if finetuned else "no",
        "dataset": dataset,
        "n_samples": n_samples,
        "exact_match_accuracy": f"{exact_acc:.4f}",
        "yesno_accuracy": f"{yesno_acc:.4f}" if yesno_acc is not None else "",
        "open_accuracy": f"{open_acc:.4f}" if open_acc is not None else "",
        "bleu_1": f"{bleu_scores.get('bleu_1', 0):.4f}",
        "bleu_2": f"{bleu_scores.get('bleu_2', 0):.4f}",
        "bleu_3": f"{bleu_scores.get('bleu_3', 0):.4f}",
        "bleu_4": f"{bleu_scores.get('bleu_4', 0):.4f}",
        "rouge1": f"{rouge_scores.get('rouge1', 0):.4f}" if rouge_scores else "0.0000",
        "rouge2": f"{rouge_scores.get('rouge2', 0):.4f}" if rouge_scores else "0.0000",
        "rougeL": f"{rouge_scores.get('rougeL', 0):.4f}" if rouge_scores else "0.0000",
        "bert_f1": f"{bert_scores.get('bert_f1', 0):.4f}" if bert_scores else "0.0000",
    }

    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"  History updated: {history_path}")
