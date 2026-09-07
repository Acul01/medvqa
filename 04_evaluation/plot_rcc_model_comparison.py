"""
Compare risk-coverage curves across experiment slots for each dataset.

For every dataset and question type, writes one PDF (no title/legend) with six curves:
  color = experiment slot (noX / zeroshot / pseudo_sft)
  solid = entropy ranking, dashed = combined uncertainty ranking

Caption/legend details go to rcc_model_compare_meta.json.

Reuses curve / AURC helpers from risk_coverage_entropy.py.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from utils.paths import get_results_dir

DEFAULT_SLOTS = ["noX", "zeroshot", "pseudo_sft"]
DEFAULT_QUESTION_TYPES = ["closed", "open", "overall"]
ANSWER_TYPE_LABEL = {"closed": "closed", "open": "open", "overall": "all"}

# Color = experiment slot; linestyle = uncertainty measure.
SLOT_COLORS = {
    "noX": "tab:blue",
    "zeroshot": "tab:orange",
    "pseudo_sft": "tab:green",
}
FALLBACK_COLORS = ["tab:purple", "tab:red", "tab:brown", "tab:pink", "tab:gray"]
MEASURE_STYLE = {
    "entropy": {
        "linestyle": "-",
        "linestyle_name": "solid",
        "curve_key": "curve_entropy",
        "aurc_key": "aurc_entropy",
    },
    "combined": {
        "linestyle": "--",
        "linestyle_name": "dashed",
        "curve_key": "curve_combined",
        "aurc_key": "aurc_combined",
    },
}


def _load_risk_coverage_module():
    script_path = Path(__file__).resolve().parent / "risk_coverage_entropy.py"
    spec = importlib.util.spec_from_file_location("risk_coverage_entropy", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load risk_coverage_entropy from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object.")
    if "datasets" not in data or not isinstance(data["datasets"], dict):
        raise ValueError("Config must contain a 'datasets' object.")
    return data


def resolve_path(raw: str, repo_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def slot_color(slot: str, index: int) -> str:
    if slot in SLOT_COLORS:
        return SLOT_COLORS[slot]
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


def evaluate_slot_curves(
    rce: Any,
    results: List[Dict],
    question_type: str,
    *,
    risk_metric: str,
    max_entropy: float | None,
    num_thresholds: int,
    combined_alpha: float,
) -> Dict[str, Any]:
    dual = rce.evaluate_dual_question_type(
        results,
        question_type,
        risk_metric=risk_metric,
        max_entropy=max_entropy,
        num_thresholds=num_thresholds,
        combined_alpha=combined_alpha,
    )
    entropy_eval = dual["entropy"]
    combined_eval = dual["combined"]
    return {
        "valid_samples": int(entropy_eval.get("valid_samples", 0)),
        "filtered_rows": int(entropy_eval.get("filtered_rows", 0)),
        "aurc_entropy": float(entropy_eval.get("aurc", 0.0)),
        "aurc_combined": float(combined_eval.get("aurc", 0.0)),
        "curve_entropy": entropy_eval.get("curve") or [],
        "curve_combined": combined_eval.get("curve") or [],
    }


def plot_dataset_question_type(
    slot_payloads: List[Dict[str, Any]],
    *,
    dataset: str,
    question_type: str,
    out_path: Path,
    combined_alpha: float,
) -> Dict[str, Any]:
    """Single-axis RCC without title/legend: solid=entropy, dashed=combined."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    answer_label = ANSWER_TYPE_LABEL.get(question_type, question_type)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    curves_meta: List[Dict[str, Any]] = []

    for idx, payload in enumerate(slot_payloads):
        color = slot_color(payload["slot"], idx)
        for measure, style in MEASURE_STYLE.items():
            curve = payload.get(style["curve_key"]) or []
            if not curve:
                continue
            x = [row["coverage"] for row in curve]
            y = [row["risk"] for row in curve]
            aurc = float(payload.get(style["aurc_key"], float("nan")))
            ax.plot(
                x,
                y,
                linewidth=2.2,
                color=color,
                linestyle=style["linestyle"],
            )
            curves_meta.append(
                {
                    "slot": payload["slot"],
                    "label": payload["label"],
                    "measure": measure,
                    "linestyle": style["linestyle_name"],
                    "color": color,
                    "aurc": None if math.isnan(aurc) else aurc,
                    "valid_samples": payload.get("valid_samples"),
                    "filtered_rows": payload.get("filtered_rows"),
                    "results_file": payload.get("results_file"),
                }
            )

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)

    return {
        "file": out_path.name,
        "path": str(out_path),
        "dataset": dataset,
        "question_type": question_type,
        "answer_type_label": answer_label,
        "title_suggested": (
            f"Risk–Coverage comparison — {dataset} / {answer_label} "
            f"(α={combined_alpha:g})"
        ),
        "axis_labels": {"x": "Coverage", "y": "Risk"},
        "encoding": {
            "color": "experiment slot (model)",
            "linestyle_solid": "entropy ranking",
            "linestyle_dashed": "combined uncertainty ranking",
        },
        "curves": curves_meta,
    }


def write_aurc_table_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "question_type",
        "slot",
        "label",
        "results_file",
        "valid_samples",
        "filtered_rows",
        "aurc_entropy",
        "aurc_combined",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_dataset_aurc_table(dataset: str, slot_rows: List[Dict[str, Any]], question_types: List[str]) -> None:
    print(f"\n=== {dataset} AURC ===")
    header = f"{'slot':<14}" + "".join(f"{qt + '_ent':>12}{qt + '_cmb':>12}" for qt in question_types)
    print(header)
    by_slot: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in slot_rows:
        by_slot.setdefault(row["slot"], {})[row["question_type"]] = {
            "entropy": float(row["aurc_entropy"]),
            "combined": float(row["aurc_combined"]),
        }
    for slot, qt_map in by_slot.items():
        line = f"{slot:<14}"
        for qt in question_types:
            vals = qt_map.get(qt, {})
            ent = vals.get("entropy", float("nan"))
            cmb = vals.get("combined", float("nan"))
            line += f"{ent:12.6f}{cmb:12.6f}"
        print(line)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay RCC curves for noX / zeroshot / pseudo-SFT per dataset × question type"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rcc_model_compare.json",
        help="JSON config with datasets → slot → results path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override config output_dir (default: config or results dir)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of dataset keys from the config",
    )
    parser.add_argument(
        "--combined_alpha",
        type=float,
        default=None,
        help="Override config combined_alpha",
    )
    parser.add_argument(
        "--num_thresholds",
        type=int,
        default=None,
        help="Override config num_thresholds",
    )
    parser.add_argument(
        "--risk_metric",
        type=str,
        default=None,
        choices=["match", "bleu1", "bleu4", "rouge", "bertscore"],
        help="Override config risk_metric",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(args.config, repo_root)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    slots: List[str] = list(config.get("slots") or DEFAULT_SLOTS)
    question_types: List[str] = list(config.get("question_types") or DEFAULT_QUESTION_TYPES)
    combined_alpha = float(
        args.combined_alpha if args.combined_alpha is not None else config.get("combined_alpha", 0.5)
    )
    num_thresholds = int(
        args.num_thresholds if args.num_thresholds is not None else config.get("num_thresholds", 200)
    )
    risk_metric = str(args.risk_metric or config.get("risk_metric", "match"))
    max_entropy = config.get("max_entropy", None)
    if max_entropy is not None:
        max_entropy = float(max_entropy)

    if not 0.0 <= combined_alpha <= 1.0:
        raise ValueError(f"combined_alpha must be in [0, 1], got {combined_alpha}")

    out_dir_raw = args.output_dir or config.get("output_dir") or get_results_dir()
    out_dir = resolve_path(str(out_dir_raw), repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_keys = list(config["datasets"].keys())
    if args.datasets:
        missing = [d for d in args.datasets if d not in config["datasets"]]
        if missing:
            raise KeyError(f"Unknown dataset keys in --datasets: {missing}")
        dataset_keys = list(args.datasets)

    rce = _load_risk_coverage_module()
    all_table_rows: List[Dict[str, Any]] = []
    figure_metas: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "config": str(config_path),
        "combined_alpha": combined_alpha,
        "num_thresholds": num_thresholds,
        "risk_metric": risk_metric,
        "max_entropy": max_entropy,
        "slots": slots,
        "question_types": question_types,
        "output_dir": str(out_dir),
        "datasets": {},
    }

    for dataset in dataset_keys:
        ds_cfg = config["datasets"][dataset]
        if not isinstance(ds_cfg, dict):
            raise ValueError(f"datasets['{dataset}'] must be an object of slots.")

        # Preload results once per slot.
        loaded: Dict[str, Dict[str, Any]] = {}
        for slot in slots:
            if slot not in ds_cfg:
                raise KeyError(f"Dataset '{dataset}' missing slot '{slot}' in config.")
            entry = ds_cfg[slot]
            if not isinstance(entry, dict) or "results" not in entry:
                raise ValueError(f"datasets['{dataset}']['{slot}'] must have a 'results' path.")
            results_path = resolve_path(str(entry["results"]), repo_root)
            if not results_path.exists():
                raise FileNotFoundError(
                    f"Results file not found for {dataset}/{slot}: {results_path}\n"
                    "Update configs/rcc_model_compare.json with your chosen best-run paths."
                )
            loaded[slot] = {
                "slot": slot,
                "label": str(entry.get("label") or slot),
                "results_path": results_path,
                "results": rce.load_results(results_path),
            }

        ds_summary: Dict[str, Any] = {"slots": {}, "figures": {}}
        dataset_table_rows: List[Dict[str, Any]] = []

        for question_type in question_types:
            slot_payloads: List[Dict[str, Any]] = []
            for slot in slots:
                meta = loaded[slot]
                eval_out = evaluate_slot_curves(
                    rce,
                    meta["results"],
                    question_type,
                    risk_metric=risk_metric,
                    max_entropy=max_entropy,
                    num_thresholds=num_thresholds,
                    combined_alpha=combined_alpha,
                )
                payload = {
                    "slot": slot,
                    "label": meta["label"],
                    "results_file": str(meta["results_path"]),
                    **eval_out,
                }
                slot_payloads.append(payload)

                row = {
                    "dataset": dataset,
                    "question_type": question_type,
                    "slot": slot,
                    "label": meta["label"],
                    "results_file": str(meta["results_path"]),
                    "valid_samples": eval_out["valid_samples"],
                    "filtered_rows": eval_out["filtered_rows"],
                    "aurc_entropy": f"{eval_out['aurc_entropy']:.6f}",
                    "aurc_combined": f"{eval_out['aurc_combined']:.6f}",
                }
                all_table_rows.append(row)
                dataset_table_rows.append(
                    {
                        **row,
                        "aurc_entropy": eval_out["aurc_entropy"],
                        "aurc_combined": eval_out["aurc_combined"],
                    }
                )

                ds_summary["slots"].setdefault(slot, {})
                ds_summary["slots"][slot][question_type] = {
                    "label": meta["label"],
                    "results_file": str(meta["results_path"]),
                    "valid_samples": eval_out["valid_samples"],
                    "filtered_rows": eval_out["filtered_rows"],
                    "aurc_entropy": eval_out["aurc_entropy"],
                    "aurc_combined": eval_out["aurc_combined"],
                }

            answer_label = ANSWER_TYPE_LABEL.get(question_type, question_type)
            fig_path = out_dir / f"{dataset}_{answer_label}.pdf"
            fig_meta = plot_dataset_question_type(
                slot_payloads,
                dataset=dataset,
                question_type=question_type,
                out_path=fig_path,
                combined_alpha=combined_alpha,
            )
            figure_metas.append(fig_meta)
            ds_summary["figures"][question_type] = str(fig_path)
            print(f"Saved: {fig_path}")

        summary["datasets"][dataset] = ds_summary
        print_dataset_aurc_table(dataset, dataset_table_rows, question_types)

    table_path = out_dir / "rcc_model_compare_aurc.csv"
    summary_path = out_dir / "rcc_model_compare_summary.json"
    meta_path = out_dir / "rcc_model_compare_meta.json"
    write_aurc_table_csv(all_table_rows, table_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    meta = {
        "config": str(config_path),
        "combined_alpha": combined_alpha,
        "num_thresholds": num_thresholds,
        "risk_metric": risk_metric,
        "max_entropy": max_entropy,
        "slots": slots,
        "question_types": question_types,
        "output_dir": str(out_dir),
        "legend_encoding": {
            "color": "experiment slot / model (see curves[].color)",
            "solid": "entropy ranking",
            "dashed": "combined uncertainty ranking",
        },
        "figures": figure_metas,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved: {table_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
