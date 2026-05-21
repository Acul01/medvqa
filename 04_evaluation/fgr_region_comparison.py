"""Region-level performance comparison for FGR perturbation.

For each sample, computes the region_score for:
    (1) full image   (baseline)
    (2) best Level-1 quadrant
    (3) best Level-2 sub-quadrant

and reports how many samples each granularity level wins.

Runs directly on the dataset + Qwen model — no prior inference results needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import wandb
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from utils.fgr import run_hierarchical_perturbation
from utils.load_data import MedicalVQALoader
from utils.paths import get_data_root, get_mc_data_root, get_results_dir
from utils.qwen_vl import DEFAULT_MODEL_PATH, resolve_processor_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare region_score: full image vs. best L1 vs. best L2 quadrant"
    )
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--data_root", type=str, default=get_data_root())
    p.add_argument("--mc_data_root", type=str, default=get_mc_data_root())
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["pathvqa", "slake", "vqarad", "medvqa2019"],
    )
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--results_dir", type=str, default=get_results_dir())
    p.add_argument("--n_samples", type=int, default=None,
                   help="Limit number of samples (useful for quick tests).")
    p.add_argument("--n_levels", type=int, default=2, choices=[1, 2, 3],
                   help="Depth of hierarchical perturbation (1=4, 2=8, 3=12 crops per sample).")
    p.add_argument("--perturb_samples", type=int, default=5,
                   help="Stochastic samples per crop.")
    p.add_argument("--perturb_temperature", type=float, default=0.7)
    p.add_argument("--perturb_top_p", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--wandb_project", type=str, default="medvqa")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--disable_wandb", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return (a / b) if b else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Load dataset ─────────────────────────────────────────────────────────
    loader = MedicalVQALoader(data_root=args.data_root, mc_data_root=args.mc_data_root)
    samples = loader.load_dataset(args.dataset, split=args.split)
    if args.n_samples is not None:
        samples = samples[: args.n_samples]

    # ── Load Qwen2.5-VL ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_lora = (
        Path(args.model_path).exists()
        and (Path(args.model_path) / "adapter_config.json").exists()
    )
    if is_lora:
        from peft import PeftModel
        base_path = args.base_model or DEFAULT_MODEL_PATH
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_path, torch_dtype=torch.float32, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, args.model_path).merge_and_unload()
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float32, device_map="auto"
        )
    model.eval()

    processor_path = resolve_processor_path(args.base_model or args.model_path)
    processor = AutoProcessor.from_pretrained(processor_path)

    # ── W&B ─────────────────────────────────────────────────────────────────
    run_name = args.wandb_run_name or f"fgr_region_cmp_{args.dataset}"
    if not args.disable_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            group=args.dataset,
            tags=["fgr", "region_comparison", "qwen", args.dataset],
            config=vars(args),
        )

    # ── Per-sample loop ──────────────────────────────────────────────────────
    results: list[dict] = []

    for sample in tqdm(samples, desc="region comparison"):
        if not sample.mc_options or sample.mc_correct_index is None:
            continue

        try:
            image = Image.open(sample.image_path).convert("RGB")
        except Exception as exc:
            tqdm.write(f"  [WARN] Cannot open {sample.image_path}: {exc}")
            continue

        gt_letter = sample.answer

        # Run baseline + L1 + L2.
        # global_threshold=0.0 disables the evidence gate so L2 always runs,
        # ensuring every sample contributes to all three score columns.
        patches, _, baseline_score = run_hierarchical_perturbation(
            model,
            processor,
            image,
            sample.question,
            sample.mc_options,
            gt_letter,
            device,
            n_levels=args.n_levels,
            global_threshold=0.0,
            n_samples=args.perturb_samples,
            temperature=args.perturb_temperature,
            top_p=args.perturb_top_p,
            max_new_tokens=args.max_new_tokens,
        )

        l1_patches = [p for p in patches if p.level == 1]
        l2_patches = [p for p in patches if p.level == 2]
        l3_patches = [p for p in patches if p.level == 3]

        best_l1 = max(l1_patches, key=lambda p: p.importance) if l1_patches else None
        best_l2 = max(l2_patches, key=lambda p: p.importance) if l2_patches else None
        best_l3 = max(l3_patches, key=lambda p: p.importance) if l3_patches else None

        best_l1_score = best_l1.importance if best_l1 else 0.0
        best_l2_score = best_l2.importance if best_l2 else 0.0
        best_l3_score = best_l3.importance if best_l3 else 0.0

        scores = {"baseline": baseline_score, "l1": best_l1_score, "l2": best_l2_score}
        if best_l3 is not None:
            scores["l3"] = best_l3_score
        winner = max(scores, key=lambda k: scores[k])

        results.append({
            "image_path":    sample.image_path,
            "question":      sample.question,
            "gt_letter":     gt_letter,
            "question_type": sample.question_type or "unknown",
            # Scores
            "baseline_score": round(baseline_score, 6),
            "best_l1_score":  round(best_l1_score, 6),
            "best_l2_score":  round(best_l2_score, 6),
            "best_l3_score":  round(best_l3_score, 6) if best_l3 else None,
            "winner":         winner,
            # Best-patch details
            "best_l1_bbox":              list(best_l1.bbox) if best_l1 else None,
            "best_l1_correct_agreement": round(best_l1.correct_agreement, 4) if best_l1 else None,
            "best_l1_agreement":         round(best_l1.agreement, 4) if best_l1 else None,
            "best_l1_mean_confidence":   round(best_l1.mean_confidence, 4) if best_l1 else None,
            "best_l1_mean_entropy":      round(best_l1.mean_entropy, 4) if best_l1 else None,
            "best_l2_bbox":              list(best_l2.bbox) if best_l2 else None,
            "best_l2_correct_agreement": round(best_l2.correct_agreement, 4) if best_l2 else None,
            "best_l2_agreement":         round(best_l2.agreement, 4) if best_l2 else None,
            "best_l2_mean_confidence":   round(best_l2.mean_confidence, 4) if best_l2 else None,
            "best_l2_mean_entropy":      round(best_l2.mean_entropy, 4) if best_l2 else None,
            "best_l3_bbox":              list(best_l3.bbox) if best_l3 else None,
            "best_l3_correct_agreement": round(best_l3.correct_agreement, 4) if best_l3 else None,
            "best_l3_agreement":         round(best_l3.agreement, 4) if best_l3 else None,
            "best_l3_mean_confidence":   round(best_l3.mean_confidence, 4) if best_l3 else None,
            "best_l3_mean_entropy":      round(best_l3.mean_entropy, 4) if best_l3 else None,
            # All patches (for deeper inspection)
            "all_patches": [
                {
                    "level":             p.level,
                    "bbox":              list(p.bbox),
                    "region_score":      round(p.importance, 6),
                    "correct_agreement": round(p.correct_agreement, 4),
                    "agreement":         round(p.agreement, 4),
                    "mean_confidence":   round(p.mean_confidence, 4),
                    "mean_entropy":      round(p.mean_entropy, 4),
                }
                for p in patches
            ],
        })

    # ── Aggregate ────────────────────────────────────────────────────────────
    total = len(results)
    has_l3 = any(r.get("best_l3_bbox") is not None for r in results)

    counts = {"baseline": 0, "l1": 0, "l2": 0}
    if has_l3:
        counts["l3"] = 0
    for r in results:
        w = r["winner"]
        counts[w] = counts.get(w, 0) + 1

    mean_baseline = _safe_div(sum(r["baseline_score"] for r in results), total)
    mean_l1       = _safe_div(sum(r["best_l1_score"]  for r in results), total)
    mean_l2       = _safe_div(sum(r["best_l2_score"]  for r in results), total)
    mean_l3       = _safe_div(
        sum(r.get("best_l3_score") or 0.0 for r in results), total
    ) if has_l3 else None

    rows = [
        ("baseline", "Full image (baseline)",    mean_baseline),
        ("l1",       "Best L1 quadrant",         mean_l1),
        ("l2",       "Best L2 sub-quadrant",     mean_l2),
    ]
    if has_l3:
        rows.append(("l3", "Best L3 sub-quadrant", mean_l3))

    print("\n── Region Score Comparison ────────────────────────────────────────")
    print(f"  Total samples evaluated : {total}")
    print()
    print(f"  {'Region':<30} {'Wins':>6}  {'%':>6}  {'Mean score':>10}")
    print(f"  {'-'*56}")
    for key, label, mean in rows:
        n   = counts.get(key, 0)
        pct = 100 * n / total if total else 0
        print(f"  {label:<30} {n:>6}  {pct:>5.1f}%  {mean:>10.4f}")
    print(f"  {'-'*56}")
    print(f"\n  Mean baseline score : {mean_baseline:.4f}")
    print(f"  Mean best-L1 score  : {mean_l1:.4f}")
    print(f"  Mean best-L2 score  : {mean_l2:.4f}")
    if has_l3:
        print(f"  Mean best-L3 score  : {mean_l3:.4f}")
    print("───────────────────────────────────────────────────────────────────\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir  = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_tag    = str(args.n_samples) if args.n_samples is not None else "all"
    out_stem = f"fgr_region_comparison_{args.dataset}_{n_tag}"

    agg = {
        "total_samples":       total,
        "winner_counts":       counts,
        "winner_fractions":    {k: _safe_div(v, total) for k, v in counts.items()},
        "mean_baseline_score": mean_baseline,
        "mean_best_l1_score":  mean_l1,
        "mean_best_l2_score":  mean_l2,
        "mean_best_l3_score":  mean_l3,
    }

    json_path = out_dir / f"{out_stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "samples": results}, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path}")

    flat = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in results]
    csv_path = out_dir / f"{out_stem}.csv"
    pd.DataFrame(flat).to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  Saved: {csv_path}")

    # ── W&B ─────────────────────────────────────────────────────────────────
    if not args.disable_wandb:
        for k, v in agg["winner_counts"].items():
            wandb.summary[f"winner_count_{k}"] = v
        for k, v in agg["winner_fractions"].items():
            wandb.summary[f"winner_frac_{k}"] = v
        wandb.summary["mean_baseline_score"] = mean_baseline
        wandb.summary["mean_best_l1_score"]  = mean_l1
        wandb.summary["mean_best_l2_score"]  = mean_l2

        table = wandb.Table(columns=[
            "image_path", "question", "gt_letter", "question_type",
            "baseline_score", "best_l1_score", "best_l2_score", "winner",
        ])
        for r in results:
            table.add_data(
                r["image_path"], r["question"], r["gt_letter"], r["question_type"],
                r["baseline_score"], r["best_l1_score"], r["best_l2_score"], r["winner"],
            )
        wandb.log({"region_comparison": table})
        wandb.finish()


if __name__ == "__main__":
    main()
