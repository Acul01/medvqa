"""Utilities for saving inference outputs and history logs."""

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.paths import get_results_dir

RESULTS_DIR = Path(get_results_dir())
HISTORY_FILE = RESULTS_DIR / "results_history.csv"

HISTORY_COLUMNS = [
    "timestamp",
    "model",
    "finetuning",
    "dataset",
    "n_samples",
    "exact_match_accuracy",
    "mc_overall_accuracy",
    "mc_closed_accuracy",
    "mc_open_accuracy",
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


def clean_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\x00-\x7F\u00C0-\u00FF]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+\d+$", "", text)
    text = re.sub(r"[.,;:!?]+$", "", text)
    return text.strip()


def clean_yesno_answer(text: str) -> str:
    text = clean_answer(text).lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return text


def _derive_model_short_name(model_path: str) -> str:
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
    mc_overall_acc: float | None = None,
    mc_closed_acc: float | None = None,
    mc_open_acc: float | None = None,
) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_name = _derive_model_short_name(model_path)
    n_tag = str(n_samples) if n_samples is not None else "all"
    base_name = f"{model_name}_{dataset}_{n_tag}"

    json_path = results_dir / f"{base_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path}")

    df = pd.DataFrame(results)
    df_export = df.copy()
    for col in ("entropy_tokens", "options"):
        if col in df_export.columns:
            df_export[col] = df_export[col].apply(
                lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
            )

    csv_path = results_dir / f"{base_name}.csv"
    df_export.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  Saved: {csv_path}")

    xlsx_path = results_dir / f"{base_name}.xlsx"
    try:
        df_export.to_excel(xlsx_path, index=False)
        print(f"  Saved: {xlsx_path}")
    except ModuleNotFoundError:
        print("  Skipped XLSX (install openpyxl: pip install openpyxl)")

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
        mc_overall_acc=mc_overall_acc,
        mc_closed_acc=mc_closed_acc,
        mc_open_acc=mc_open_acc,
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
    mc_overall_acc: float | None = None,
    mc_closed_acc: float | None = None,
    mc_open_acc: float | None = None,
) -> None:
    history_path = results_dir / "results_history.csv"
    write_header = not history_path.exists()

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "finetuning": "yes" if finetuned else "no",
        "dataset": dataset,
        "n_samples": n_samples,
        "exact_match_accuracy": f"{exact_acc:.4f}",
        "mc_overall_accuracy": f"{mc_overall_acc:.4f}" if mc_overall_acc is not None else "",
        "mc_closed_accuracy": f"{mc_closed_acc:.4f}" if mc_closed_acc is not None else "",
        "mc_open_accuracy": f"{mc_open_acc:.4f}" if mc_open_acc is not None else "",
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
