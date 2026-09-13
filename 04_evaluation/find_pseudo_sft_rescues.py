#!/usr/bin/env python3
"""Find SLAKE (or other) samples that only Pseudo-SFT gets right.

Default filter (``--mode families``):
  every loaded answer-only variant is wrong,
  every loaded zeroshot variant is wrong,
  at least one Pseudo-SFT variant is correct.

``--mode rcc`` uses the three representative files in
``configs/rcc_model_compare.json`` instead of all thesis specs.

Writes JSON, CSV, and XLSX (openpyxl).

Example:
  python 04_evaluation/find_pseudo_sft_rescues.py --dataset slake
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.mc_eval import _resolve_gt_letter, _resolve_pred_letter

DEFAULT_SPECS = Path("configs/rcc_experiment_specs.json")
DEFAULT_RCC = Path("configs/rcc_model_compare.json")
FAMILY_ORDER = ("answer_only", "zeroshot", "pseudo_sft")
FAMILY_PREFIX = {
    "answer_only": "ao",
    "zeroshot": "zs",
    "pseudo_sft": "ps",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(raw: str, repo_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def candidate_result_paths(run: dict[str, Any], dataset: str, repo_root: Path) -> list[Path]:
    raws: list[str] = []
    results = run.get("results")
    if isinstance(results, dict) and dataset in results and results[dataset]:
        value = results[dataset]
        if isinstance(value, list):
            raws.extend(str(item) for item in value if item)
        else:
            raws.append(str(value))
    elif isinstance(results, str) and results.strip():
        raws.append(results)
    if run.get("results_template"):
        raws.append(str(run["results_template"]))
    for template in run.get("results_templates") or []:
        if template:
            raws.append(str(template))

    paths: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        resolved = str(path)
        if resolved not in seen:
            seen.add(resolved)
            paths.append(path)

    for raw in raws:
        filled = raw.replace("{dataset}", dataset)
        if any(ch in filled for ch in "*?["):
            pattern = filled if Path(filled).is_absolute() else str(repo_root / filled)
            for match in sorted(glob.glob(pattern, recursive=True)):
                match_path = Path(match)
                if match_path.is_file():
                    _add(match_path.resolve())
            continue
        _add(resolve_path(filled, repo_root))
    return paths


def first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_result_rows(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        rows = data["results"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unexpected JSON shape in {path}")
    return [row for row in rows if isinstance(row, dict)]


def sample_key(row: dict[str, Any]) -> tuple[str, str]:
    image = str(row.get("source_image_path") or row.get("image_path") or "").strip()
    question = " ".join(str(row.get("question") or "").split())
    return Path(image).name, question


def option_for_letter(row: dict[str, Any], letter: str | None) -> str:
    if not letter:
        return ""
    options = row.get("options") or []
    if isinstance(options, list):
        idx = ord(letter) - 65
        if 0 <= idx < len(options):
            return str(options[idx])
    return ""


def reasoning_text(row: dict[str, Any]) -> str:
    for key in (
        "predicted_reasoning_text",
        "predicted_rationale",
        "rationale",
        "predicted_raw",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def pack_prediction(row: dict[str, Any]) -> dict[str, Any]:
    gt = _resolve_gt_letter(row)
    pred = _resolve_pred_letter(row)
    return {
        "gt_letter": gt,
        "pred_letter": pred,
        "pred_text": option_for_letter(row, pred),
        "is_correct": bool(gt) and bool(pred) and pred == gt,
        "reasoning": reasoning_text(row),
        "image_path": str(row.get("source_image_path") or row.get("image_path") or ""),
        "question": str(row.get("question") or ""),
        "question_type": str(row.get("question_type") or ""),
        "options": row.get("options") if isinstance(row.get("options"), list) else [],
        "gt_answer": str(row.get("ground_truth_answer") or option_for_letter(row, gt) or ""),
    }


def collect_spec_runs(cfg: dict[str, Any], dataset: str, repo_root: Path) -> list[dict[str, Any]]:
    runs_out: list[dict[str, Any]] = []
    experiments = cfg.get("experiments") or {}
    for family in FAMILY_ORDER:
        exp = experiments.get(family) or {}
        for run in exp.get("runs") or []:
            candidates = candidate_result_paths(run, dataset, repo_root)
            path = first_existing(candidates)
            record = {
                "family": family,
                "id": run.get("id") or run.get("label"),
                "label": run.get("label") or run.get("id"),
                "col": f"{FAMILY_PREFIX[family]}_{run.get('id') or run.get('label')}",
                "results_file": str(path) if path else None,
                "missing": path is None,
                "n_rows": 0,
            }
            if path is not None:
                rows = load_result_rows(path)
                record["n_rows"] = len(rows)
                record["by_key"] = {sample_key(row): pack_prediction(row) for row in rows}
            runs_out.append(record)
    return runs_out


def collect_rcc_runs(cfg: dict[str, Any], dataset: str, repo_root: Path) -> list[dict[str, Any]]:
    slot_to_family = {"noX": "answer_only", "zeroshot": "zeroshot", "pseudo_sft": "pseudo_sft"}
    dataset_cfg = (cfg.get("datasets") or {}).get(dataset) or {}
    runs_out: list[dict[str, Any]] = []
    for slot, family in slot_to_family.items():
        slot_cfg = dataset_cfg.get(slot) or {}
        raw = slot_cfg.get("results")
        path = resolve_path(str(raw), repo_root) if raw else None
        if path is not None and not path.is_file():
            path = None
        record = {
            "family": family,
            "id": slot,
            "label": slot_cfg.get("label") or slot,
            "col": f"{FAMILY_PREFIX[family]}_{slot}",
            "results_file": str(path) if path else None,
            "missing": path is None,
            "n_rows": 0,
        }
        if path is not None:
            rows = load_result_rows(path)
            record["n_rows"] = len(rows)
            record["by_key"] = {sample_key(row): pack_prediction(row) for row in rows}
        runs_out.append(record)
    return runs_out


def union_keys(runs: list[dict[str, Any]]) -> list[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for run in runs:
        keys.update(run.get("by_key") or {})
    return sorted(keys)


def pick_base_row(runs: list[dict[str, Any]], key: tuple[str, str]) -> dict[str, Any]:
    for family in ("pseudo_sft", "answer_only", "zeroshot"):
        for run in runs:
            if run["family"] != family:
                continue
            packed = (run.get("by_key") or {}).get(key)
            if packed:
                return packed
    return {}


def family_correctness(runs: list[dict[str, Any]], key: tuple[str, str], family: str) -> list[bool]:
    flags: list[bool] = []
    for run in runs:
        if run["family"] != family or run.get("missing"):
            continue
        packed = (run.get("by_key") or {}).get(key)
        if packed is None:
            continue
        flags.append(bool(packed["is_correct"]))
    return flags


def is_rescue(runs: list[dict[str, Any]], key: tuple[str, str]) -> bool:
    ao = family_correctness(runs, key, "answer_only")
    zs = family_correctness(runs, key, "zeroshot")
    ps = family_correctness(runs, key, "pseudo_sft")
    if not ao or not zs or not ps:
        return False
    return (not any(ao)) and (not any(zs)) and any(ps)


def flatten_sample(base: dict[str, Any], runs: list[dict[str, Any]], key: tuple[str, str]) -> dict[str, Any]:
    image_name, question = key
    options = base.get("options") or []
    options_str = "  ".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(options))
    row: dict[str, Any] = {
        "image_name": image_name,
        "image_path": base.get("image_path", ""),
        "question": question or base.get("question", ""),
        "question_type": base.get("question_type", ""),
        "options": options_str,
        "gt_letter": base.get("gt_letter") or "",
        "gt_answer": base.get("gt_answer") or "",
    }
    ao_ok = family_correctness(runs, key, "answer_only")
    zs_ok = family_correctness(runs, key, "zeroshot")
    ps_ok = family_correctness(runs, key, "pseudo_sft")
    row["n_answer_only_correct"] = sum(ao_ok)
    row["n_zeroshot_correct"] = sum(zs_ok)
    row["n_pseudo_sft_correct"] = sum(ps_ok)
    winners: list[str] = []
    for run in runs:
        col = run["col"]
        packed = (run.get("by_key") or {}).get(key)
        if packed is None:
            row[f"{col}_pred"] = ""
            row[f"{col}_text"] = ""
            row[f"{col}_correct"] = ""
            row[f"{col}_reasoning"] = ""
            continue
        row[f"{col}_pred"] = packed.get("pred_letter") or ""
        row[f"{col}_text"] = packed.get("pred_text") or ""
        row[f"{col}_correct"] = packed["is_correct"]
        row[f"{col}_reasoning"] = packed.get("reasoning") or ""
        if run["family"] == "pseudo_sft" and packed["is_correct"]:
            winners.append(run["label"])
    row["pseudo_sft_correct_variants"] = ", ".join(winners)
    return row


def write_outputs(payload: dict[str, Any], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    xlsx_path = out_dir / f"{stem}.xlsx"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")

    table_rows = payload["samples"]
    run_rows = [
        {k: v for k, v in run.items() if k != "by_key"}
        for run in payload["runs"]
    ]
    if table_rows:
        fieldnames = list(table_rows[0].keys())
    else:
        fieldnames = ["image_name", "question", "gt_letter"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"Wrote {csv_path}")

    try:
        import pandas as pd

        with pd.ExcelWriter(xlsx_path) as writer:
            pd.DataFrame(table_rows, columns=fieldnames).to_excel(
                writer, sheet_name="rescues", index=False
            )
            pd.DataFrame(run_rows).to_excel(writer, sheet_name="runs", index=False)
            pd.DataFrame([payload["summary"]]).to_excel(writer, sheet_name="summary", index=False)
        print(f"Wrote {xlsx_path}")
    except ModuleNotFoundError:
        print("Skipped XLSX (install openpyxl: pip install openpyxl)", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Samples only Pseudo-SFT answers correctly")
    parser.add_argument("--dataset", default="slake", choices=["slake", "vqarad", "medvqa2019", "pathvqa"])
    parser.add_argument(
        "--mode",
        default="families",
        choices=["families", "rcc"],
        help="families = all thesis specs; rcc = three representative RCC files",
    )
    parser.add_argument("--specs_config", type=Path, default=DEFAULT_SPECS)
    parser.add_argument("--rcc_config", type=Path, default=DEFAULT_RCC)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--allow_incomplete",
        action="store_true",
        help="Keep samples that are missing from a whole experiment family",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    if args.mode == "rcc":
        cfg = load_json(args.rcc_config)
        runs = collect_rcc_runs(cfg, args.dataset, repo_root)
        filter_text = (
            "RCC representatives: answer-only wrong, zeroshot wrong, Pseudo-SFT correct"
        )
        stem = f"{args.dataset}_rcc_pseudo_sft_rescues"
    else:
        cfg = load_json(args.specs_config)
        runs = collect_spec_runs(cfg, args.dataset, repo_root)
        filter_text = (
            "all loaded answer-only variants wrong, all loaded zeroshot variants wrong, "
            "at least one Pseudo-SFT variant correct"
        )
        stem = f"{args.dataset}_pseudo_sft_rescues"

    present = {run["family"] for run in runs if not run["missing"]}
    missing_families = [fam for fam in FAMILY_ORDER if fam not in present]
    if missing_families:
        raise SystemExit(
            "No results loaded for: "
            + ", ".join(missing_families)
            + ". Check the config paths / --skip is not used here."
        )

    loaded = [run for run in runs if not run["missing"]]
    keys = union_keys(loaded)
    samples: list[dict[str, Any]] = []
    n_incomplete = 0
    for key in keys:
        if not args.allow_incomplete:
            if any(not family_correctness(loaded, key, fam) for fam in FAMILY_ORDER):
                n_incomplete += 1
                continue
        if not is_rescue(loaded, key):
            continue
        base = pick_base_row(loaded, key)
        samples.append(flatten_sample(base, loaded, key))

    out_dir = args.output_dir or Path("06_results/FINAL") / f"{args.dataset}_pseudo_sft_rescues"
    summary = {
        "dataset": args.dataset,
        "mode": args.mode,
        "filter": filter_text,
        "n_runs_configured": len(runs),
        "n_runs_loaded": len(loaded),
        "n_keys_union": len(keys),
        "n_keys_incomplete": n_incomplete,
        "n_rescues": len(samples),
        "n_closed": sum(1 for row in samples if str(row.get("question_type", "")).lower() == "closed"),
        "n_open": sum(1 for row in samples if str(row.get("question_type", "")).lower() == "open"),
    }
    payload = {
        "summary": summary,
        "filter": filter_text,
        "runs": [{k: v for k, v in run.items() if k != "by_key"} for run in runs],
        "samples": samples,
    }
    write_outputs(payload, out_dir, stem)
    print(
        f"{args.dataset}: {len(samples)} rescues "
        f"(closed={summary['n_closed']}, open={summary['n_open']}) "
        f"from {len(loaded)}/{len(runs)} result files"
    )
    for run in runs:
        status = "ok" if not run["missing"] else "MISSING"
        print(f"  [{status}] {run['family']}/{run['id']}  {run.get('results_file') or '-'}")


if __name__ == "__main__":
    main()
