"""
Shared utilities for saving inference results in multiple formats
and maintaining a results history log.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
HISTORY_FILE = RESULTS_DIR / "results_history.csv"

HISTORY_COLUMNS = [
    "timestamp",
    "model",
    "finetuning",
    "dataset",
    "n_samples",
    "exact_match_accuracy",
    "contains_match_accuracy",
]


def _derive_model_short_name(model_path: str) -> str:
    """Derive a short, filename-safe model name from the full path/ID."""
    name = Path(model_path).name  # e.g. "llava-med-v1.5-mistral-7b"
    # Common mappings
    name_lower = name.lower()
    if "llava-med" in name_lower:
        return "llava-med"
    if "llava" in name_lower:
        return "llava"
    # Fallback: sanitise the raw name
    return name.replace("/", "_").replace(" ", "_")


def save_results(
    results: list[dict],
    model_path: str,
    dataset: str,
    n_samples: int | None,
    finetuned: bool,
    exact_acc: float,
    contains_acc: float,
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
        contains_acc=contains_acc,
        results_dir=results_dir,
    )


def _append_history(
    model: str,
    finetuned: bool,
    dataset: str,
    n_samples: str,
    exact_acc: float,
    contains_acc: float,
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
        "contains_match_accuracy": f"{contains_acc:.4f}",
    }

    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"  History updated: {history_path}")
