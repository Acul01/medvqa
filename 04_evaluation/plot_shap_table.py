#!/usr/bin/env python3
"""Compose a SHAP comparison table from shap_visualize.py heatmaps.

Works for:
  - answer-only (same variants on every model: base vs full)
  - zeroshot (same variants: af / rf / 2s / 2s-L)
  - per-model row configs (e.g. pseudo-SFT: Images/rf/af vs rf/af/rf+af)

Example:
  python 04_evaluation/plot_shap_table.py \\
      --config configs/shap_vqarad_zeroshot_table.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

DEFAULT_CONFIG = Path("configs/shap_vqarad_answer_only_table.json")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def variants_from_config(cfg: dict) -> list[dict]:
    if cfg.get("variants"):
        return list(cfg["variants"])
    experiment = cfg.get("experiment", "answer_only")
    if experiment == "zeroshot":
        return [
            {"tag": "af", "label": "af", "json_order": "answer_first"},
            {"tag": "rf", "label": "rf", "json_order": "rationale_first"},
            {"tag": "2s", "label": "2s", "json_order": "rationale_first_2stage"},
            {"tag": "2s-L", "label": "2s-L", "json_order": "rationale_first_2stage_norat"},
        ]
    return [
        {"tag": "base", "label": "base", "json_order": "answer_only"},
        {"tag": "full", "label": "full", "json_order": "answer_only"},
    ]


def uses_per_model_rows(cfg: dict) -> bool:
    return any("rows" in model_cfg for model_cfg in cfg.get("models", []))


def rows_for_model(model_cfg: dict, cfg: dict) -> list[dict]:
    if model_cfg.get("rows"):
        return list(model_cfg["rows"])
    return [{"type": "shap", **variant} for variant in variants_from_config(cfg)]


def heatmap_path(
    shap_root: Path,
    shap_model: str,
    dataset: str,
    experiment: str,
    json_order: str,
    tag: str,
    sample_idx: int,
) -> Path:
    return (
        shap_root
        / shap_model
        / dataset
        / experiment
        / json_order
        / tag
        / f"sample_{sample_idx:04d}_heatmap.png"
    )


def qa_text(record: dict) -> str:
    options = record.get("options_str") or "  ".join(
        f"{chr(65 + i)}) {o}" for i, o in enumerate(record.get("options") or [])
    )
    qtype = record.get("question_type") or ""
    prefix = f"[{qtype}] " if qtype else ""
    return f"{prefix}Q: {record.get('question', '')}\n{options}\nGT: {record.get('answer', '?')}"


def hide_ax(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a SHAP comparison table")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selected", type=Path, default=None,
                        help="Override selected_samples.json (default: <shap_root>/selected_samples.json)")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_json(args.config)
    shap_root = Path(cfg["shap_root"])
    selected_path = args.selected or (shap_root / "selected_samples.json")
    if not selected_path.exists():
        raise SystemExit(
            f"Missing {selected_path}. Run select_shap_table_samples.py first."
        )
    selected = load_json(selected_path)
    records = selected["samples"]
    dataset = cfg["dataset"]
    experiment = cfg.get("experiment", "answer_only")
    n_samples = len(records)
    models = cfg["models"]
    per_model_rows = uses_per_model_rows(cfg)
    show_qa_row = bool(cfg.get("show_qa_row", True))
    show_global_originals = bool(cfg.get("show_originals_row", not per_model_rows))
    model_rows = [rows_for_model(model_cfg, cfg) for model_cfg in models]
    n_model_rows = sum(len(rows) for rows in model_rows)
    header_rows = int(show_qa_row) + int(show_global_originals)
    n_gs_rows = header_rows + n_model_rows

    height_ratios: list[float] = []
    if show_qa_row:
        height_ratios.append(1.15)
    if show_global_originals:
        height_ratios.append(1.0)
    height_ratios.extend(1.0 for _ in range(n_model_rows))

    fig_w = 3.1 * n_samples + 2.4
    fig_h = 1.25 * n_gs_rows + 0.6
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=160)
    gs = GridSpec(
        n_gs_rows,
        2 + n_samples,
        figure=fig,
        width_ratios=[0.55, 1.15] + [1.0] * n_samples,
        height_ratios=height_ratios,
        wspace=0.08,
        hspace=0.10,
        left=0.04,
        right=0.99,
        top=0.97,
        bottom=0.03,
    )

    missing: list[str] = []
    gs_row = 0

    if show_qa_row:
        ax = fig.add_subplot(gs[gs_row, 1])
        ax.text(0.0, 0.5, "Question + Answer", va="center", ha="left", fontsize=9, fontweight="bold")
        hide_ax(ax)
        for col, rec in enumerate(records):
            ax = fig.add_subplot(gs[gs_row, 2 + col])
            ax.text(0.02, 0.5, qa_text(rec), va="center", ha="left", fontsize=6.5, wrap=True)
            if not show_global_originals and not per_model_rows:
                ax.set_title(f"sample{col + 1}", fontsize=8)
            hide_ax(ax)
        gs_row += 1

    if show_global_originals:
        ax = fig.add_subplot(gs[gs_row, 1])
        ax.text(0.0, 0.5, "Original_Images", va="center", ha="left", fontsize=9, fontweight="bold")
        hide_ax(ax)
        for col, rec in enumerate(records):
            ax = fig.add_subplot(gs[gs_row, 2 + col])
            img_path = Path(rec["image_path"])
            if img_path.exists():
                ax.imshow(Image.open(img_path).convert("RGB"))
            else:
                ax.text(0.5, 0.5, f"missing\n{img_path.name}", ha="center", va="center", fontsize=7)
                missing.append(str(img_path))
            ax.set_title(f"sample{col + 1}", fontsize=8)
            hide_ax(ax)
        gs_row += 1

    for model_cfg, rows in zip(models, model_rows):
        n_var = len(rows)
        group_ax = fig.add_subplot(gs[gs_row:gs_row + n_var, 0])
        group_ax.text(
            0.5, 0.5, model_cfg["group_label"],
            va="center", ha="center", fontsize=10, fontweight="bold", rotation=90,
        )
        hide_ax(group_ax)

        for t_i, row in enumerate(rows):
            r = gs_row + t_i
            lab = fig.add_subplot(gs[r, 1])
            lab.text(0.0, 0.5, row.get("label", row.get("tag", "")), va="center", ha="left", fontsize=9)
            hide_ax(lab)
            row_type = row.get("type", "shap")
            for col, rec in enumerate(records):
                ax = fig.add_subplot(gs[r, 2 + col])
                if row_type == "original":
                    img_path = Path(rec["image_path"])
                    if img_path.exists():
                        ax.imshow(Image.open(img_path).convert("RGB"))
                    else:
                        ax.text(0.5, 0.5, f"missing\n{img_path.name}", ha="center", va="center", fontsize=7)
                        missing.append(str(img_path))
                    if t_i == 0:
                        ax.set_title(f"sample{col + 1}", fontsize=8)
                else:
                    path = heatmap_path(
                        shap_root,
                        model_cfg["shap_model"],
                        dataset,
                        experiment,
                        row["json_order"],
                        row["tag"],
                        rec["index"],
                    )
                    if path.exists():
                        ax.imshow(Image.open(path).convert("RGB"))
                    else:
                        ax.text(0.5, 0.5, "missing\nheatmap", ha="center", va="center", fontsize=7, color="0.4")
                        missing.append(str(path))
                hide_ax(ax)
        gs_row += n_var

    figure_name = cfg.get("figure_name", f"{dataset}_{experiment}_shap_table")
    out_path = args.output or (shap_root / f"{figure_name}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"Saved {pdf_path}")
    if missing:
        print(f"Missing {len(missing)} assets:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
