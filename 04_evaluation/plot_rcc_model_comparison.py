"""
Compare risk-coverage curves.

Two config modes:

1. Best-of-experiment (configs/rcc_model_compare.json):
   one figure per dataset × question type with noX / zeroshot / pseudo_sft.

2. All specs per experiment (configs/rcc_experiment_specs.json):
   one figure per experiment × dataset × question type with every listed
   model-specification combination (missing result files skipped by default).

No title/legend on PDFs; caption details go to rcc_model_compare_meta.json.
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

# Color = experiment slot (best-of mode) or tab20 index (all-specs mode).
SLOT_COLORS = {
    "noX": "tab:blue",
    "zeroshot": "tab:orange",
    "pseudo_sft": "tab:green",
}
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
    if "experiments" in data:
        if not isinstance(data["experiments"], dict) or not data["experiments"]:
            raise ValueError("Config 'experiments' must be a non-empty object.")
        return data
    if "datasets" not in data or not isinstance(data["datasets"], dict):
        raise ValueError("Config must contain a 'datasets' object or an 'experiments' object.")
    return data


def resolve_path(raw: str, repo_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def slot_color(slot: str, index: int, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if slot in SLOT_COLORS:
        return SLOT_COLORS[slot]
    cmap = plt.get_cmap("tab20")
    r, g, b, _a = cmap(index % 20)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


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
    experiment: str | None = None,
    experiment_label: str | None = None,
) -> Dict[str, Any]:
    """Single-axis RCC without title/legend: solid=entropy, dashed=combined."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    answer_label = ANSWER_TYPE_LABEL.get(question_type, question_type)
    group_name = experiment_label or experiment or "best-of-experiment"
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    curves_meta: List[Dict[str, Any]] = []

    for idx, payload in enumerate(slot_payloads):
        color = slot_color(payload["slot"], idx, payload.get("color"))
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
        "experiment": experiment,
        "experiment_label": group_name,
        "dataset": dataset,
        "question_type": question_type,
        "answer_type_label": answer_label,
        "title_suggested": (
            f"Risk–Coverage — {group_name} — {dataset} / {answer_label} "
            f"(α={combined_alpha:g})"
        ),
        "axis_labels": {"x": "Coverage", "y": "Risk"},
        "encoding": {
            "color": "model specification (see curves[].label / color)",
            "linestyle_solid": "entropy ranking",
            "linestyle_dashed": "combined uncertainty ranking",
        },
        "curves": curves_meta,
    }


def write_aurc_table_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
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


def candidate_result_paths(run: Dict[str, Any], dataset: str, repo_root: Path) -> List[Path]:
    """Collect candidate results JSON paths (overrides, then templates)."""
    raws: List[str] = []
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

    paths: List[Path] = []
    seen: set[str] = set()
    for raw in raws:
        resolved = str(resolve_path(raw.replace("{dataset}", dataset), repo_root))
        if resolved not in seen:
            seen.add(resolved)
            paths.append(Path(resolved))
    return paths


def load_first_existing_results(
    rce: Any,
    candidates: List[Path],
    *,
    skip_missing: bool,
    context: str,
) -> tuple[List[Dict] | None, Path | None]:
    tried: List[Path] = []
    for path in candidates:
        tried.append(path)
        if path.exists():
            return rce.load_results(path), path
    detail = "; ".join(str(p) for p in tried) if tried else "(no path configured)"
    message = f"Results file not found for {context}: {detail}"
    if skip_missing:
        print(f"skip: {message}")
        return None, None
    raise FileNotFoundError(message)


def plot_group_for_dataset(
    *,
    rce: Any,
    dataset: str,
    loaded: Dict[str, Dict[str, Any]],
    question_types: List[str],
    out_dir: Path,
    combined_alpha: float,
    risk_metric: str,
    max_entropy: float | None,
    num_thresholds: int,
    experiment: str,
    experiment_label: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Plot all question types for one dataset × experiment group."""
    all_table_rows: List[Dict[str, Any]] = []
    figure_metas: List[Dict[str, Any]] = []
    ds_summary: Dict[str, Any] = {"slots": {}, "figures": {}}
    dataset_table_rows: List[Dict[str, Any]] = []

    for question_type in question_types:
        slot_payloads: List[Dict[str, Any]] = []
        for slot, meta in loaded.items():
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
                "color": meta.get("color"),
                "results_file": str(meta["results_path"]),
                **eval_out,
            }
            slot_payloads.append(payload)

            row = {
                "experiment": experiment,
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
            experiment=experiment,
            experiment_label=experiment_label,
        )
        figure_metas.append(fig_meta)
        ds_summary["figures"][question_type] = str(fig_path)
        print(f"Saved: {fig_path}")

    print_dataset_aurc_table(f"{experiment}/{dataset}", dataset_table_rows, question_types)
    return all_table_rows, figure_metas, ds_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay RCC curves (best-of-experiment or all specs per experiment)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rcc_model_compare.json",
        help="JSON config: datasets→slots (best-of) or experiments→runs (all specs)",
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
        help="Optional subset of dataset keys",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of experiment keys (experiment-spec config only)",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        default=None,
        help="Skip missing result JSONs instead of aborting (default: true for experiment-spec configs)",
    )
    parser.add_argument(
        "--require_all",
        action="store_true",
        help="Fail if any configured results file is missing",
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


def _plot_settings(args, config: Dict[str, Any]) -> Dict[str, Any]:
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
    return {
        "question_types": question_types,
        "combined_alpha": combined_alpha,
        "num_thresholds": num_thresholds,
        "risk_metric": risk_metric,
        "max_entropy": max_entropy,
    }


def _write_outputs(
    *,
    out_dir: Path,
    all_table_rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    figure_metas: List[Dict[str, Any]],
    extra_meta: Dict[str, Any],
) -> None:
    table_path = out_dir / "rcc_model_compare_aurc.csv"
    summary_path = out_dir / "rcc_model_compare_summary.json"
    meta_path = out_dir / "rcc_model_compare_meta.json"
    write_aurc_table_csv(all_table_rows, table_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    meta = {
        **extra_meta,
        "legend_encoding": {
            "color": "model specification (see curves[].color / label)",
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


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_combined_from_experiment_dirs(out_dir: Path, extra_meta: Dict[str, Any]) -> None:
    """Rebuild top-level JSON/CSV from per-experiment artifacts so subset reruns do not wipe others."""
    table_rows: List[Dict[str, Any]] = []
    figures: List[Dict[str, Any]] = []
    experiments: Dict[str, Any] = {}
    experiment_ids: List[str] = []
    for child in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        meta_path = child / "rcc_model_compare_meta.json"
        summary_path = child / "rcc_model_compare_summary.json"
        csv_path = child / "rcc_model_compare_aurc.csv"
        if not meta_path.exists():
            continue
        experiment_ids.append(child.name)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        figures.extend(meta.get("figures") or [])
        table_rows.extend(_read_csv_rows(csv_path))
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
            exp_block = (summary.get("experiments") or {}).get(child.name)
            if exp_block:
                experiments[child.name] = exp_block
    combined_summary = {
        **{k: extra_meta.get(k) for k in (
            "config", "mode", "combined_alpha", "num_thresholds",
            "risk_metric", "max_entropy", "question_types", "output_dir",
        )},
        "experiments": experiments,
        "merged_from": experiment_ids,
    }
    extra = {
        **extra_meta,
        "experiments": experiment_ids,
        "merged_from": experiment_ids,
        "note": (
            "Combined caption/AURC files merge per-experiment artifacts under "
            "output_dir/<experiment>/. Re-running one experiment updates only that "
            "subdir and then rebuilds these combined files."
        ),
    }
    _write_outputs(
        out_dir=out_dir,
        all_table_rows=table_rows,
        summary=combined_summary,
        figure_metas=figures,
        extra_meta=extra,
    )


def run_best_of_mode(
    *,
    args,
    config: Dict[str, Any],
    config_path: Path,
    repo_root: Path,
    rce: Any,
    settings: Dict[str, Any],
    out_dir: Path,
    skip_missing: bool,
) -> None:
    slots: List[str] = list(config.get("slots") or DEFAULT_SLOTS)
    dataset_keys = list(config["datasets"].keys())
    if args.datasets:
        missing = [d for d in args.datasets if d not in config["datasets"]]
        if missing:
            raise KeyError(f"Unknown dataset keys in --datasets: {missing}")
        dataset_keys = list(args.datasets)

    all_table_rows: List[Dict[str, Any]] = []
    figure_metas: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "config": str(config_path),
        "mode": "best_of_experiment",
        "combined_alpha": settings["combined_alpha"],
        "num_thresholds": settings["num_thresholds"],
        "risk_metric": settings["risk_metric"],
        "max_entropy": settings["max_entropy"],
        "slots": slots,
        "question_types": settings["question_types"],
        "output_dir": str(out_dir),
        "datasets": {},
    }

    for dataset in dataset_keys:
        ds_cfg = config["datasets"][dataset]
        if not isinstance(ds_cfg, dict):
            raise ValueError(f"datasets['{dataset}'] must be an object of slots.")
        loaded: Dict[str, Dict[str, Any]] = {}
        for slot in slots:
            if slot not in ds_cfg:
                raise KeyError(f"Dataset '{dataset}' missing slot '{slot}' in config.")
            entry = ds_cfg[slot]
            if not isinstance(entry, dict) or "results" not in entry:
                raise ValueError(f"datasets['{dataset}']['{slot}'] must have a 'results' path.")
            results_path = resolve_path(str(entry["results"]), repo_root)
            rows, used_path = load_first_existing_results(
                rce,
                [results_path],
                skip_missing=skip_missing,
                context=f"{dataset}/{slot}",
            )
            if rows is None or used_path is None:
                continue
            loaded[slot] = {
                "slot": slot,
                "label": str(entry.get("label") or slot),
                "color": entry.get("color"),
                "results_path": used_path,
                "results": rows,
            }
        if not loaded:
            print(f"skip: no result files for dataset '{dataset}'")
            continue
        table_rows, figs, ds_summary = plot_group_for_dataset(
            rce=rce,
            dataset=dataset,
            loaded=loaded,
            question_types=settings["question_types"],
            out_dir=out_dir,
            combined_alpha=settings["combined_alpha"],
            risk_metric=settings["risk_metric"],
            max_entropy=settings["max_entropy"],
            num_thresholds=settings["num_thresholds"],
            experiment="best_of",
            experiment_label="best-of-experiment",
        )
        all_table_rows.extend(table_rows)
        figure_metas.extend(figs)
        summary["datasets"][dataset] = ds_summary

    _write_outputs(
        out_dir=out_dir,
        all_table_rows=all_table_rows,
        summary=summary,
        figure_metas=figure_metas,
        extra_meta={
            "config": str(config_path),
            "mode": "best_of_experiment",
            "combined_alpha": settings["combined_alpha"],
            "num_thresholds": settings["num_thresholds"],
            "risk_metric": settings["risk_metric"],
            "max_entropy": settings["max_entropy"],
            "slots": slots,
            "question_types": settings["question_types"],
            "output_dir": str(out_dir),
        },
    )


def run_experiment_spec_mode(
    *,
    args,
    config: Dict[str, Any],
    config_path: Path,
    repo_root: Path,
    rce: Any,
    settings: Dict[str, Any],
    out_dir: Path,
    skip_missing: bool,
) -> None:
    experiment_keys = list(config["experiments"].keys())
    if args.experiments:
        missing = [e for e in args.experiments if e not in config["experiments"]]
        if missing:
            raise KeyError(f"Unknown experiment keys in --experiments: {missing}")
        experiment_keys = list(args.experiments)

    default_datasets = list(config.get("dataset_list") or ["slake", "vqarad", "medvqa2019"])
    dataset_keys = list(args.datasets) if args.datasets else default_datasets

    all_table_rows: List[Dict[str, Any]] = []
    figure_metas: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "config": str(config_path),
        "mode": "experiment_specs",
        "combined_alpha": settings["combined_alpha"],
        "num_thresholds": settings["num_thresholds"],
        "risk_metric": settings["risk_metric"],
        "max_entropy": settings["max_entropy"],
        "question_types": settings["question_types"],
        "output_dir": str(out_dir),
        "experiments": {},
    }

    for experiment in experiment_keys:
        exp_cfg = config["experiments"][experiment]
        if not isinstance(exp_cfg, dict) or "runs" not in exp_cfg:
            raise ValueError(f"experiments['{experiment}'] must contain a 'runs' list.")
        runs = exp_cfg["runs"]
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"experiments['{experiment}'].runs must be a non-empty list.")
        experiment_label = str(exp_cfg.get("label") or experiment)
        exp_out = out_dir / experiment
        exp_out.mkdir(parents=True, exist_ok=True)
        summary["experiments"][experiment] = {"label": experiment_label, "datasets": {}}
        exp_table_rows: List[Dict[str, Any]] = []
        exp_figure_metas: List[Dict[str, Any]] = []

        for dataset in dataset_keys:
            loaded: Dict[str, Dict[str, Any]] = {}
            for run in runs:
                if not isinstance(run, dict) or "id" not in run:
                    raise ValueError(f"Each run in {experiment} needs an 'id'.")
                slot = str(run["id"])
                candidates = candidate_result_paths(run, dataset, repo_root)
                if not candidates:
                    print(f"skip: no path for {experiment}/{dataset}/{slot}")
                    continue
                rows, used_path = load_first_existing_results(
                    rce,
                    candidates,
                    skip_missing=skip_missing,
                    context=f"{experiment}/{dataset}/{slot}",
                )
                if rows is None or used_path is None:
                    continue
                loaded[slot] = {
                    "slot": slot,
                    "label": str(run.get("label") or slot),
                    "color": run.get("color"),
                    "results_path": used_path,
                    "results": rows,
                }
            if not loaded:
                print(f"skip: no result files for {experiment}/{dataset}")
                continue
            table_rows, figs, ds_summary = plot_group_for_dataset(
                rce=rce,
                dataset=dataset,
                loaded=loaded,
                question_types=settings["question_types"],
                out_dir=exp_out,
                combined_alpha=settings["combined_alpha"],
                risk_metric=settings["risk_metric"],
                max_entropy=settings["max_entropy"],
                num_thresholds=settings["num_thresholds"],
                experiment=experiment,
                experiment_label=experiment_label,
            )
            all_table_rows.extend(table_rows)
            figure_metas.extend(figs)
            exp_table_rows.extend(table_rows)
            exp_figure_metas.extend(figs)
            summary["experiments"][experiment]["datasets"][dataset] = ds_summary

        _write_outputs(
            out_dir=exp_out,
            all_table_rows=exp_table_rows,
            summary={
                "config": str(config_path),
                "mode": "experiment_specs",
                "experiment": experiment,
                "experiment_label": experiment_label,
                "combined_alpha": settings["combined_alpha"],
                "num_thresholds": settings["num_thresholds"],
                "risk_metric": settings["risk_metric"],
                "max_entropy": settings["max_entropy"],
                "question_types": settings["question_types"],
                "output_dir": str(exp_out),
                "experiments": {experiment: summary["experiments"][experiment]},
            },
            figure_metas=exp_figure_metas,
            extra_meta={
                "config": str(config_path),
                "mode": "experiment_specs",
                "experiment": experiment,
                "experiment_label": experiment_label,
                "combined_alpha": settings["combined_alpha"],
                "num_thresholds": settings["num_thresholds"],
                "risk_metric": settings["risk_metric"],
                "max_entropy": settings["max_entropy"],
                "question_types": settings["question_types"],
                "output_dir": str(exp_out),
                "experiments": [experiment],
            },
        )

    _write_combined_from_experiment_dirs(
        out_dir,
        extra_meta={
            "config": str(config_path),
            "mode": "experiment_specs",
            "combined_alpha": settings["combined_alpha"],
            "num_thresholds": settings["num_thresholds"],
            "risk_metric": settings["risk_metric"],
            "max_entropy": settings["max_entropy"],
            "question_types": settings["question_types"],
            "output_dir": str(out_dir),
        },
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(args.config, repo_root)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    settings = _plot_settings(args, config)
    out_dir_raw = args.output_dir or config.get("output_dir") or get_results_dir()
    out_dir = resolve_path(str(out_dir_raw), repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    rce = _load_risk_coverage_module()

    spec_mode = "experiments" in config
    if args.require_all:
        skip_missing = False
    elif args.skip_missing is True:
        skip_missing = True
    elif spec_mode:
        skip_missing = bool(config.get("skip_missing", True))
    else:
        skip_missing = bool(config.get("skip_missing", False))

    common = dict(
        args=args,
        config=config,
        config_path=config_path,
        repo_root=repo_root,
        rce=rce,
        settings=settings,
        out_dir=out_dir,
        skip_missing=skip_missing,
    )
    if spec_mode:
        run_experiment_spec_mode(**common)
    else:
        run_best_of_mode(**common)


if __name__ == "__main__":
    main()
