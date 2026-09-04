#!/usr/bin/env python3
"""Aggregate format-parse success rates over FINAL eval prediction JSONs.

For JSON reasoning / zeroshot runs (rows with ``predicted_json_parse_ok``):
  format_ok = predicted_json_parse_ok is True

For answer-only / letter-only runs (key absent on all rows):
  format_ok = predicted_letter is not None
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


DEFAULT_ROOTS = [
    "06_results/FINAL/noX",
    "06_results/FINAL/withX/zeroshot",
    "06_results/FINAL/withX/pseudo_qwen3",
    "06_results/FINAL/withX/pseudo_som",
]

PRED_MARKERS = ("predicted_answer", "predicted_letter")


def _infer_category(path: Path, roots: list[Path]) -> str:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for root in roots:
        try:
            rel = resolved.relative_to(root.resolve())
        except ValueError:
            continue
        parts = root.parts
        if "noX" in parts:
            return "noX"
        if "zeroshot" in parts:
            return "zeroshot"
        if "pseudo_som" in parts:
            return "pseudo_som"
        if "pseudo_qwen3" in parts or "pseudo" in parts:
            return "pseudo_sft"
        return root.name
    # Fallback from path string
    s = str(path).replace("\\", "/")
    if "/noX/" in s:
        return "noX"
    if "/zeroshot/" in s:
        return "zeroshot"
    if "/pseudo_som/" in s:
        return "pseudo_som"
    if "/pseudo" in s:
        return "pseudo_sft"
    return "other"


def _is_prediction_json(path: Path, data: object) -> bool:
    if path.name.endswith("_eval.json"):
        return False
    if not isinstance(data, list) or not data:
        return False
    first = data[0]
    if not isinstance(first, dict):
        return False
    return any(k in first for k in PRED_MARKERS)


def analyze_rows(rows: list[dict]) -> tuple[str, int, int]:
    """Return (metric, n, format_ok_count)."""
    has_json_flag = any("predicted_json_parse_ok" in r for r in rows)
    n = len(rows)
    if has_json_flag:
        ok = sum(1 for r in rows if r.get("predicted_json_parse_ok") is True)
        return "json_parse", n, ok
    ok = sum(1 for r in rows if r.get("predicted_letter") is not None)
    return "letter_extract", n, ok


def collect_json_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            print(f"warning: root missing, skip: {root}", file=sys.stderr)
            continue
        for path in sorted(root.rglob("*.json")):
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            files.append(path)
    return files


def analyze_file(path: Path, roots: list[Path]) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"warning: cannot load {path}: {e}", file=sys.stderr)
        return None
    if not _is_prediction_json(path, data):
        return None
    rows = [r for r in data if isinstance(r, dict)]
    if not rows:
        return None
    metric, n, ok = analyze_rows(rows)
    pct = (100.0 * ok / n) if n else 0.0
    return {
        "file": str(path),
        "category": _infer_category(path, roots),
        "n": n,
        "format_ok": ok,
        "format_ok_pct": round(pct, 2),
        "metric": metric,
    }


def _print_markdown(rows: list[dict]) -> None:
    headers = ["file", "category", "n", "format_ok", "format_ok_pct", "metric"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(r["file"]),
                    str(r["category"]),
                    str(r["n"]),
                    str(r["format_ok"]),
                    f"{r['format_ok_pct']:.2f}",
                    str(r["metric"]),
                ]
            )
            + " |"
        )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file", "category", "n", "format_ok", "format_ok_pct", "metric"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute format-parse success rates for FINAL eval JSONs."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=DEFAULT_ROOTS,
        help="Directories to scan recursively (default: FINAL noX / zeroshot / pseudo*).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to write a CSV summary.",
    )
    args = parser.parse_args()

    roots = [Path(r) for r in args.roots]
    files = collect_json_files(roots)
    results: list[dict] = []
    for path in files:
        row = analyze_file(path, roots)
        if row is not None:
            results.append(row)

    if not results:
        print("No prediction JSONs found under the given roots.", file=sys.stderr)
        return 1

    results.sort(key=lambda r: (r["category"], r["file"]))
    _print_markdown(results)

    if args.csv:
        out = Path(args.csv)
        _write_csv(out, results)
        print(f"\nWrote CSV: {out}", file=sys.stderr)

    # Category totals
    print("\n### Totals by category", file=sys.stderr)
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        n = sum(r["n"] for r in rows)
        ok = sum(r["format_ok"] for r in rows)
        pct = (100.0 * ok / n) if n else 0.0
        metrics = sorted({r["metric"] for r in rows})
        print(
            f"  {cat}: {ok}/{n} = {pct:.2f}%  (files={len(rows)}, metrics={','.join(metrics)})",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
