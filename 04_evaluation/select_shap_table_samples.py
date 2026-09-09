#!/usr/bin/env python3
"""Pick reproducible VQA-Rad MC samples and print SHAP commands.

Writes <shap_root>/selected_samples.json so every model/variant uses the same indices.

Example:
  python 04_evaluation/select_shap_table_samples.py \\
      --config configs/shap_vqarad_zeroshot_table.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.load_data import MedicalVQALoader
from utils.paths import get_data_root, get_mc_data_root

DEFAULT_CONFIG = Path("configs/shap_vqarad_answer_only_table.json")


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def is_closed(question_type: str | None) -> bool:
    qt = (question_type or "").strip().lower()
    return qt in {"closed", "yes/no", "yesno", "y/n"}


def is_open(question_type: str | None) -> bool:
    qt = (question_type or "").strip().lower()
    return qt in {"open", "open-ended", "openended"}


def pick_indices(samples, *, seed: int, n_closed: int, n_open: int) -> list[int]:
    rng = random.Random(seed)
    closed = [i for i, s in enumerate(samples) if is_closed(s.question_type)]
    opened = [i for i, s in enumerate(samples) if is_open(s.question_type)]
    rng.shuffle(closed)
    rng.shuffle(opened)
    if len(closed) < n_closed:
        raise SystemExit(f"Need {n_closed} closed samples, found {len(closed)}")
    if len(opened) < n_open:
        raise SystemExit(f"Need {n_open} open samples, found {len(opened)}")
    return closed[:n_closed] + opened[:n_open]


def sample_record(idx: int, sample, *, column: int) -> dict:
    opts = sample.mc_options or []
    image_path = Path(sample.image_path)
    stem = f"sample_{idx:04d}"
    return {
        "table_column": f"sample{column}",
        "index": idx,
        "image_path": str(image_path),
        "image_name": image_path.name,
        "question": sample.question,
        "options": opts,
        "answer": sample.answer,
        "question_type": sample.question_type,
        "options_str": "  ".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(opts)),
        "files": {
            "plot": f"{stem}.png",
            "heatmap": f"{stem}_heatmap.png",
            "meta": f"{stem}.json",
        },
    }


def selected_payload(cfg: dict, idxs: list[int], samples) -> dict:
    records = [sample_record(i, samples[i], column=k + 1) for k, i in enumerate(idxs)]
    return {
        "dataset": cfg["dataset"],
        "split": cfg.get("split", "test"),
        "seed": cfg.get("seed", 42),
        "n_closed": int(cfg.get("n_closed", 2)),
        "n_open": int(cfg.get("n_open", 2)),
        "experiment": cfg.get("experiment", "answer_only"),
        "shap_root": cfg["shap_root"],
        "heatmap_layout": (
            "{shap_root}/{shap_model}/{dataset}/{experiment}/{json_order}/{run_tag}/{heatmap}"
        ),
        "indices": idxs,
        "samples": records,
    }


def write_selected_samples(cfg: dict, idxs: list[int], samples) -> Path:
    out_root = Path(cfg["shap_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "selected_samples.json"
    payload = selected_payload(cfg, idxs, samples)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return out_path


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
        {"tag": "base", "label": "base", "json_order": "answer_only", "path_key": "base"},
        {"tag": "full", "label": "full", "json_order": "answer_only", "path_key": "full"},
    ]


def model_path_for(model_cfg: dict, variant: dict) -> str:
    if "model_path" in model_cfg:
        return model_cfg["model_path"]
    path_key = variant.get("path_key")
    if path_key and path_key in model_cfg and isinstance(model_cfg[path_key], dict):
        return model_cfg[path_key]["model_path"]
    raise KeyError(f"No model_path for variant {variant.get('tag')!r} in {model_cfg.get('key')}")


def shap_rows(model_cfg: dict, cfg: dict) -> list[dict]:
    """SHAP command rows for one model (skips original-image placeholder rows)."""
    if model_cfg.get("rows"):
        return [row for row in model_cfg["rows"] if row.get("type", "shap") == "shap"]
    return list(variants_from_config(cfg))


def shap_command(cfg: dict, model_cfg: dict, variant: dict, indices: list[int]) -> str:
    idx_str = ",".join(str(i) for i in indices)
    shap_model = model_cfg["shap_model"]
    model_path = model_path_for(model_cfg, variant)
    root = cfg["shap_root"]
    max_evals = cfg.get("max_evals", 300)
    experiment = cfg.get("experiment", "answer_only")
    json_order = variant["json_order"]
    tag = variant["tag"]
    return (
        "python 04_evaluation/shap_visualize.py \\\n"
        f"    --model {shap_model} \\\n"
        f"    --model_path {model_path} \\\n"
        f"    --dataset {cfg['dataset']} \\\n"
        f"    --split {cfg.get('split', 'test')} \\\n"
        f"    --experiment {experiment} \\\n"
        f"    --json_order {json_order} \\\n"
        f"    --sample_indices {idx_str} \\\n"
        f"    --run_tag {tag} \\\n"
        f"    --max_evals {max_evals} \\\n"
        f"    --output_dir {root}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Select VQA-Rad samples for a SHAP table")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data_root", type=str, default=get_data_root())
    parser.add_argument("--mc_data_root", type=str, default=get_mc_data_root())
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    loader = MedicalVQALoader(data_root=args.data_root, mc_data_root=args.mc_data_root)
    samples = loader.load_dataset(cfg["dataset"], split=cfg.get("split", "test"))
    samples = [s for s in samples if s.mc_options and s.mc_correct_index is not None]
    if not samples:
        raise SystemExit("No MC samples loaded — check data_root / mc_data_root.")

    idxs = pick_indices(
        samples,
        seed=int(cfg.get("seed", 42)),
        n_closed=int(cfg.get("n_closed", 2)),
        n_open=int(cfg.get("n_open", 2)),
    )
    out_path = write_selected_samples(cfg, idxs, samples)
    records = json.loads(out_path.read_text())["samples"]

    print(f"Wrote {out_path}")
    print(f"Indices: {idxs}")
    for rec in records:
        print(
            f"  {rec.get('table_column', '?')} [{rec['index']}] "
            f"{rec.get('question_type') or '?'} | "
            f"{rec.get('image_name', '')} | "
            f"Q: {rec['question'][:80]} | GT: {rec['answer']}"
        )
    print("\n# SHAP runs (same samples for every model)\n")
    n_cmds = 0
    for model_cfg in cfg["models"]:
        for variant in shap_rows(model_cfg, cfg):
            print(f"# {model_cfg['group_label']} {variant.get('label', variant['tag'])}")
            print(shap_command(cfg, model_cfg, variant, idxs))
            print()
            n_cmds += 1
    print(f"# After all {n_cmds} runs:")
    print(
        "python 04_evaluation/plot_shap_table.py "
        f"--config {args.config}"
    )


if __name__ == "__main__":
    main()
