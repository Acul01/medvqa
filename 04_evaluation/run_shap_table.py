#!/usr/bin/env python3
"""Run every shap_visualize.py job for one thesis table, sequentially.

Pick samples once, then load each model/variant in a fresh subprocess so GPU
memory is released between checkpoints. Pin a GPU with CUDA_VISIBLE_DEVICES
or --gpu.

Example:
  CUDA_VISIBLE_DEVICES=0 python -u 04_evaluation/run_shap_table.py \\
      --config configs/shap_vqarad_answer_only_table.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.load_data import MedicalVQALoader
from utils.paths import get_data_root, get_mc_data_root, prepend_llava_med_checkout

from select_shap_table_samples import (  # noqa: E402
    load_config,
    model_path_for,
    pick_indices,
    shap_rows,
    write_selected_samples,
)

SHAP_SCRIPT = Path(__file__).with_name("shap_visualize.py")


def heatmap_path(cfg: dict, model_cfg: dict, variant: dict, sample_idx: int) -> Path:
    return (
        Path(cfg["shap_root"])
        / model_cfg["shap_model"]
        / cfg["dataset"]
        / cfg.get("experiment", "answer_only")
        / variant["json_order"]
        / variant["tag"]
        / f"sample_{sample_idx:04d}_heatmap.png"
    )


def job_complete(cfg: dict, model_cfg: dict, variant: dict, indices: list[int]) -> bool:
    return all(
        heatmap_path(cfg, model_cfg, variant, idx).exists()
        for idx in indices
    )


def shap_argv(
    cfg: dict,
    model_cfg: dict,
    variant: dict,
    indices: list[int],
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable, "-u", str(SHAP_SCRIPT),
        "--model", model_cfg["shap_model"],
        "--model_path", model_path_for(model_cfg, variant),
        "--dataset", cfg["dataset"],
        "--split", cfg.get("split", "test"),
        "--experiment", cfg.get("experiment", "answer_only"),
        "--json_order", variant["json_order"],
        "--sample_indices", ",".join(str(i) for i in indices),
        "--run_tag", variant["tag"],
        "--max_evals", str(cfg.get("max_evals", 300)),
        "--output_dir", cfg["shap_root"],
        "--dtype", args.dtype,
        "--data_root", args.data_root,
        "--mc_data_root", args.mc_data_root,
    ]
    if args.mask_type:
        cmd.extend(["--mask_type", args.mask_type])
    return cmd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all SHAP jobs for one table config on a single GPU",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Set CUDA_VISIBLE_DEVICES for this process and its children",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--mask_type", type=str, default=None)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a model/variant if all sample heatmaps already exist",
    )
    parser.add_argument(
        "--keep_going",
        action="store_true",
        help="Continue with the next job if one shap_visualize.py call fails",
    )
    parser.add_argument("--data_root", type=str, default=get_data_root())
    parser.add_argument("--mc_data_root", type=str, default=get_mc_data_root())
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"CUDA_VISIBLE_DEVICES={args.gpu}", flush=True)
    checkout = prepend_llava_med_checkout()
    if checkout is not None:
        print(f"LLaVA-Med checkout on PYTHONPATH: {checkout}", flush=True)

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
    selected_path = write_selected_samples(cfg, idxs, samples)
    print(f"Wrote {selected_path}", flush=True)
    print(f"Indices: {idxs}", flush=True)

    jobs = [
        (model_cfg, variant)
        for model_cfg in cfg["models"]
        for variant in shap_rows(model_cfg, cfg)
    ]
    failed: list[str] = []
    for i, (model_cfg, variant) in enumerate(jobs, start=1):
        label = f"{model_cfg['group_label']} {variant.get('label', variant['tag'])}"
        print(f"\n=== [{i}/{len(jobs)}] {label} ===", flush=True)
        if args.skip_existing and job_complete(cfg, model_cfg, variant, idxs):
            print(f"skip existing heatmaps for {label}", flush=True)
            continue
        cmd = shap_argv(cfg, model_cfg, variant, idxs, args)
        print(" ".join(cmd), flush=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            msg = f"{label} failed with exit code {result.returncode}"
            print(msg, file=sys.stderr, flush=True)
            if args.keep_going:
                failed.append(msg)
                continue
            raise SystemExit(result.returncode)

    print(f"\nDone. {len(jobs)} jobs for {args.config}", flush=True)
    if failed:
        print(f"{len(failed)} failed jobs:", file=sys.stderr, flush=True)
        for item in failed:
            print(f"  {item}", file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
