"""Utilities for saving inference outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from utils.paths import get_results_dir
from utils.qwen_vl import model_short_name as qwen_model_short_name

RESULTS_DIR = Path(get_results_dir())


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


def derive_model_short_name(model_path: str) -> str:
    path_lower = model_path.lower()
    if "llava-med" in path_lower:
        return "llava-med"
    if "llava" in path_lower:
        return "llava"
    if "medvlm" in path_lower:
        return "medvlm-r1"
    if "qwen" in path_lower:
        return qwen_model_short_name(model_path)
    name = Path(model_path).name
    return name.replace("/", "_").replace(" ", "_")


def build_results_base_name(
    model_path: str,
    dataset: str,
    n_samples: int | None,
    name_suffix: str | None = None,
) -> str:
    model_name = derive_model_short_name(model_path)
    n_tag = str(n_samples) if n_samples is not None else "all"
    base_name = f"{model_name}_{dataset}_{n_tag}"
    if name_suffix:
        base_name = f"{base_name}_{name_suffix}"
    return base_name


def save_results(
    results: list[dict],
    model_path: str,
    dataset: str,
    n_samples: int | None,
    results_dir: str | Path = RESULTS_DIR,
    name_suffix: str | None = None,
) -> Path:
    """Write inference JSON/CSV/XLSX under ``results_dir``; return path to the JSON file."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    base_name = build_results_base_name(model_path, dataset, n_samples, name_suffix=name_suffix)

    json_path = results_dir / f"{base_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path}")

    df = pd.DataFrame(results)
    df_export = df.copy()
    for col in ("entropy_tokens", "entropy_tokens_all", "options"):
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

    return json_path
