#!/usr/bin/env python3
"""Merge RF + AF inference JSONs from a dual-order (--both_json_orders) checkpoint.

Joins samples on (source_image_path, question), concatenates the stochastic
agreement draws (default 5+5=10), and recomputes agreement / entropy / confidence.
``predicted_letter`` is the majority vote over the pooled letters.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from utils.mc_eval import add_aurc_cli_arguments, resolve_eval_runs_csv, run_mc_evaluation


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        rows = data["results"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unexpected JSON shape in {path}")
    if not isinstance(rows, list) or (rows and not isinstance(rows[0], dict)):
        raise ValueError(f"Expected a list of sample dicts in {path}")
    return rows


def sample_key(row: dict[str, Any]) -> tuple[str, str]:
    image = str(row.get("source_image_path") or row.get("image_path") or "").strip()
    question = str(row.get("question") or "").strip()
    if not image or not question:
        raise ValueError(f"Row missing source_image_path/image_path or question: {row!r}")
    return image, question


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _mean_finite(values: list[Any]) -> float:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not nums:
        return float("nan")
    return sum(nums) / len(nums)


def _compute_agreement(letters: list[Any]) -> float:
    valid = [str(letter).strip() for letter in letters if letter]
    if not valid:
        return 0.0
    counts = Counter(valid)
    return max(counts.values()) / len(valid)


def _majority_letter(letters: list[Any]) -> str | None:
    valid = [str(letter).strip() for letter in letters if letter and str(letter).strip()]
    if not valid:
        return None
    counts = Counter(valid)
    top_count = max(counts.values())
    # Tie: first letter among those with top_count (RF draws come first).
    for letter in valid:
        if counts[letter] == top_count:
            return letter
    return None


def _index_rows(rows: list[dict[str, Any]], *, label: str) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = sample_key(row)
        if key in indexed:
            raise ValueError(f"Duplicate sample key in {label} file: {key}")
        indexed[key] = row
    return indexed


def merge_rows(
    rf_row: dict[str, Any],
    af_row: dict[str, Any],
) -> dict[str, Any]:
    rf_letters = _as_list(rf_row.get("agreement_samples"))
    af_letters = _as_list(af_row.get("agreement_samples"))
    pooled_letters = rf_letters + af_letters

    rf_ents = _as_list(rf_row.get("entropy_samples"))
    af_ents = _as_list(af_row.get("entropy_samples"))
    pooled_ents = rf_ents + af_ents

    letter = _majority_letter(pooled_letters)
    gt = rf_row.get("ground_truth_letter") or af_row.get("ground_truth_letter")
    entropy = _mean_finite(pooled_ents)
    confidence = _mean_finite([rf_row.get("confidence"), af_row.get("confidence")])

    merged = dict(rf_row)
    merged.update(
        {
            "agreement_samples": pooled_letters,
            "agreement_samples_rf": rf_letters,
            "agreement_samples_af": af_letters,
            "entropy_samples": pooled_ents,
            "entropy_samples_rf": rf_ents,
            "entropy_samples_af": af_ents,
            "agreement": _compute_agreement(pooled_letters),
            "entropy": entropy,
            "confidence": confidence,
            "predicted_letter": letter,
            "is_correct": bool(letter) and letter == gt,
            "predicted_letter_rf": rf_row.get("predicted_letter"),
            "predicted_letter_af": af_row.get("predicted_letter"),
            "is_correct_rf": rf_row.get("is_correct"),
            "is_correct_af": af_row.get("is_correct"),
            "predicted_answer_rf": rf_row.get("predicted_answer"),
            "predicted_answer_af": af_row.get("predicted_answer"),
            "predicted_reasoning_text_rf": rf_row.get("predicted_reasoning_text"),
            "predicted_reasoning_text_af": af_row.get("predicted_reasoning_text"),
            "predicted_json_parse_ok_rf": rf_row.get("predicted_json_parse_ok"),
            "predicted_json_parse_ok_af": af_row.get("predicted_json_parse_ok"),
            "entropy_sequence_mean_rf": rf_row.get("entropy_sequence_mean"),
            "entropy_sequence_mean_af": af_row.get("entropy_sequence_mean"),
            "json_order_merge": "rf+af",
            "n_agreement_samples": len(pooled_letters),
        }
    )
    if math.isfinite(entropy) and letter:
        merged["entropy_tokens"] = {letter: entropy}
    else:
        merged["entropy_tokens"] = {}
    # Fused parse_ok is not well-defined; keep True if either main run parsed.
    rf_ok = bool(rf_row.get("predicted_json_parse_ok"))
    af_ok = bool(af_row.get("predicted_json_parse_ok"))
    if "predicted_json_parse_ok" in rf_row or "predicted_json_parse_ok" in af_row:
        merged["predicted_json_parse_ok"] = rf_ok or af_ok
    merged.pop("predicted_reasoning_text", None)
    merged.pop("predicted_answer", None)
    return merged


def merge_result_files(
    rf_rows: list[dict[str, Any]],
    af_rows: list[dict[str, Any]],
    *,
    strict: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rf_index = _index_rows(rf_rows, label="RF")
    af_index = _index_rows(af_rows, label="AF")
    rf_keys = set(rf_index)
    af_keys = set(af_index)
    shared = [key for key in rf_index if key in af_index]
    stats = {
        "rf": len(rf_index),
        "af": len(af_index),
        "merged": len(shared),
        "rf_only": len(rf_keys - af_keys),
        "af_only": len(af_keys - rf_keys),
    }
    if strict and (stats["rf_only"] or stats["af_only"]):
        raise ValueError(
            f"RF/AF sample sets differ: rf_only={stats['rf_only']} af_only={stats['af_only']}"
        )
    merged = [merge_rows(rf_index[key], af_index[key]) for key in shared]
    return merged, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge RF and AF inference JSONs (pool agreement samples, recompute metrics)."
    )
    parser.add_argument("--rf", required=True, type=str, help="RF inference JSON")
    parser.add_argument("--af", required=True, type=str, help="AF inference JSON")
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Merged JSON path (CSV/XLSX written beside it)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if RF and AF do not contain the exact same samples",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run run_mc_evaluation on the merged JSON",
    )
    add_aurc_cli_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rf_path = Path(args.rf)
    af_path = Path(args.af)
    out_path = Path(args.output)
    if not rf_path.exists():
        raise FileNotFoundError(rf_path)
    if not af_path.exists():
        raise FileNotFoundError(af_path)

    merged, stats = merge_result_files(
        _load_rows(rf_path),
        _load_rows(af_path),
        strict=args.strict,
    )
    print(
        f"Merged {stats['merged']} samples "
        f"(rf={stats['rf']}, af={stats['af']}, "
        f"rf_only={stats['rf_only']}, af_only={stats['af_only']})"
    )
    if merged:
        n_draw = merged[0].get("n_agreement_samples")
        print(f"  pooled agreement draws per sample: {n_draw}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    try:
        import pandas as pd

        df = pd.DataFrame(merged)
        for col in ("entropy_tokens", "entropy_tokens_all", "options"):
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                )
        csv_path = out_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"  Saved: {csv_path}")
        try:
            xlsx_path = out_path.with_suffix(".xlsx")
            df.to_excel(xlsx_path, index=False)
            print(f"  Saved: {xlsx_path}")
        except ModuleNotFoundError:
            print("  Skipped XLSX (install openpyxl)")
    except ImportError:
        print("  Skipped CSV (pandas not installed)")

    if args.evaluate:
        run_mc_evaluation(
            out_path,
            eval_runs_csv=resolve_eval_runs_csv(args),
            compute_aurc=not args.skip_aurc,
            combined_alpha=args.combined_alpha,
        )


if __name__ == "__main__":
    main()
