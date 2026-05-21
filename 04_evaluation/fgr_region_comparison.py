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
            base_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, args.model_path).merge_and_unload()
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
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
        patches, _, baseline_score, baseline_score_ag, baseline_ca = run_hierarchical_perturbation(
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

        # Supervised scores (correct_agreement-based)
        best_l1_score = best_l1.importance if best_l1 else 0.0
        best_l2_score = best_l2.importance if best_l2 else 0.0
        best_l3_score = best_l3.importance if best_l3 else 0.0

        scores_sup = {"baseline": baseline_score, "l1": best_l1_score}
        if best_l2:
            scores_sup["l2"] = best_l2_score
        if best_l3:
            scores_sup["l3"] = best_l3_score
        winner_supervised = max(scores_sup, key=lambda k: scores_sup[k])

        # Unsupervised scores — delta from baseline (importance_agreement is already
        # delta-transformed in run_hierarchical_perturbation).
        # Baseline delta is 0 by definition (it IS the reference point).
        best_l1_score_ag = best_l1.importance_agreement if best_l1 else 0.0
        best_l2_score_ag = best_l2.importance_agreement if best_l2 else 0.0
        best_l3_score_ag = best_l3.importance_agreement if best_l3 else 0.0

        scores_ag = {"baseline": 0.0, "l1": best_l1_score_ag}
        if best_l2:
            scores_ag["l2"] = best_l2_score_ag
        if best_l3:
            scores_ag["l3"] = best_l3_score_ag
        winner_agreement = max(scores_ag, key=lambda k: scores_ag[k])

        winners_match = winner_supervised == winner_agreement

        # ── Correct-agreement of the supervised winner ───────────────────────
        _winner_ca_map = {"baseline": baseline_ca}
        if best_l1: _winner_ca_map["l1"] = best_l1.correct_agreement
        if best_l2: _winner_ca_map["l2"] = best_l2.correct_agreement
        if best_l3: _winner_ca_map["l3"] = best_l3.correct_agreement
        winner_sup_ca = _winner_ca_map.get(winner_supervised, 0.0)
        # majority-vote correctness: ≥50% of stochastic samples got the right answer
        winner_sup_is_correct = winner_sup_ca >= 0.5

        # ── BBox of each winner (None = full image) ──────────────────────────
        _bbox_map = {
            "baseline": None,
            "l1": list(best_l1.bbox) if best_l1 else None,
            "l2": list(best_l2.bbox) if best_l2 else None,
            "l3": list(best_l3.bbox) if best_l3 else None,
        }
        winner_sup_bbox = _bbox_map.get(winner_supervised)
        winner_ag_bbox  = _bbox_map.get(winner_agreement)

        results.append({
            "image_path":    sample.image_path,
            "question":      sample.question,
            "mc_options":    sample.mc_options,
            "gt_letter":     gt_letter,
            "question_type": sample.question_type or "unknown",
            # ── Supervised scores (correct_agreement × conf × (1−H)) ──────────
            "baseline_score": round(baseline_score, 6),
            "best_l1_score":  round(best_l1_score, 6),
            "best_l2_score":  round(best_l2_score, 6),
            "best_l3_score":  round(best_l3_score, 6) if best_l3 else None,
            "winner_supervised":          winner_supervised,
            "winner_supervised_bbox":     winner_sup_bbox,
            "winner_supervised_score":    round(scores_sup[winner_supervised], 6),
            "winner_supervised_ca":       round(winner_sup_ca, 4),
            "winner_supervised_correct":  winner_sup_is_correct,
            # ── Unsupervised scores — Δ from baseline (> 0 = better than full image) ─
            "baseline_score_ag":          round(baseline_score_ag, 6),   # absolute ref value
            "best_l1_score_ag_delta":     round(best_l1_score_ag, 6),
            "best_l2_score_ag_delta":     round(best_l2_score_ag, 6),
            "best_l3_score_ag_delta":     round(best_l3_score_ag, 6) if best_l3 else None,
            "winner_agreement":           winner_agreement,
            "winner_agreement_bbox":      winner_ag_bbox,
            "winner_agreement_score_delta": round(scores_ag[winner_agreement], 6),
            # ── Concordance ───────────────────────────────────────────────────
            "winners_match": winners_match,
            # ── Best-patch details ────────────────────────────────────────────
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
            # ── All patches ───────────────────────────────────────────────────
            "all_patches": [
                {
                    "level":                p.level,
                    "bbox":                 list(p.bbox),
                    "region_score":         round(p.importance, 6),
                    "region_score_ag":      round(p.importance_agreement, 6),
                    "correct_agreement":    round(p.correct_agreement, 4),
                    "agreement":            round(p.agreement, 4),
                    "mean_confidence":      round(p.mean_confidence, 4),
                    "mean_entropy":         round(p.mean_entropy, 4),
                }
                for p in patches
            ],
        })

    # ── Aggregate ────────────────────────────────────────────────────────────
    total = len(results)
    has_l2 = any(r.get("best_l2_bbox") is not None for r in results)
    has_l3 = any(r.get("best_l3_bbox") is not None for r in results)

    counts_sup = {"baseline": 0, "l1": 0}
    counts_ag  = {"baseline": 0, "l1": 0}
    if has_l2:
        counts_sup["l2"] = 0
        counts_ag["l2"]  = 0
    if has_l3:
        counts_sup["l3"] = 0
        counts_ag["l3"]  = 0
    concordant = 0
    for r in results:
        counts_sup[r["winner_supervised"]] = counts_sup.get(r["winner_supervised"], 0) + 1
        counts_ag[r["winner_agreement"]]   = counts_ag.get(r["winner_agreement"],   0) + 1
        if r["winners_match"]:
            concordant += 1

    concordance_rate = _safe_div(concordant, total)

    # ── Mean scores: winning samples only (per level) ────────────────────────
    # For each level, mean score is computed only over samples where that level won.
    def _mean_winning(key: str, score_key: str) -> float:
        vals = [r[score_key] for r in results if r["winner_supervised"] == key]
        return _safe_div(sum(vals), len(vals))

    mean_w_baseline = _mean_winning("baseline", "baseline_score")
    mean_w_l1       = _mean_winning("l1",       "best_l1_score")
    mean_w_l2       = _mean_winning("l2",       "best_l2_score") if has_l2 else None
    mean_w_l3       = _mean_winning("l3",       "best_l3_score") if has_l3 else None

    def _mean_winning_ag(key: str, delta_key: str) -> float:
        if key == "baseline":
            return 0.0
        vals = [r[delta_key] for r in results if r["winner_agreement"] == key]
        return _safe_div(sum(vals), len(vals))

    mean_w_baseline_ag = 0.0
    mean_w_l1_ag       = _mean_winning_ag("l1", "best_l1_score_ag_delta")
    mean_w_l2_ag       = _mean_winning_ag("l2", "best_l2_score_ag_delta") if has_l2 else None
    mean_w_l3_ag       = _mean_winning_ag("l3", "best_l3_score_ag_delta") if has_l3 else None

    # ── Winner correctness (supervised winner only) ──────────────────────────
    winner_correct_count = sum(1 for r in results if r["winner_supervised_correct"])
    winner_correct_rate  = _safe_div(winner_correct_count, total)
    # softer: at least one correct sample (ca > 0)
    winner_any_correct_count = sum(1 for r in results if r["winner_supervised_ca"] > 0)
    winner_any_correct_rate  = _safe_div(winner_any_correct_count, total)

    # Absolute baseline agreement score (reference, shown separately)
    mean_baseline_ag_abs = _safe_div(sum(r["baseline_score_ag"] for r in results), total)
    # Unsupervised deltas (all samples, for reference)
    mean_l1_ag_delta = _safe_div(sum(r["best_l1_score_ag_delta"] for r in results), total)
    mean_l2_ag_delta = _safe_div(
        sum(r["best_l2_score_ag_delta"] for r in results), total
    ) if has_l2 else None
    mean_l3_ag_delta = _safe_div(
        sum(r.get("best_l3_score_ag_delta") or 0.0 for r in results), total
    ) if has_l3 else None

    level_keys = ["baseline", "l1"] + (["l2"] if has_l2 else []) + (["l3"] if has_l3 else [])
    level_labels = {
        "baseline": "Full image (baseline)",
        "l1":       "Best L1 quadrant",
        "l2":       "Best L2 sub-quadrant",
        "l3":       "Best L3 sub-quadrant",
    }
    mean_w_sup = {"baseline": mean_w_baseline, "l1": mean_w_l1, "l2": mean_w_l2, "l3": mean_w_l3}
    mean_w_ag  = {"baseline": mean_w_baseline_ag, "l1": mean_w_l1_ag, "l2": mean_w_l2_ag, "l3": mean_w_l3_ag}

    print("\n── Region Score Comparison ─────────────────────────────────────────────────────────────")
    print(f"  Total samples evaluated : {total}")
    print(f"  Unsupervised baseline score (absolute reference) : {mean_baseline_ag_abs:.4f}")
    print()
    print(f"  {'Region':<28}  {'Supervised (GT)':>26}  {'Unsupervised conf×(1−H) Δ':>30}")
    print(f"  {'':28}  {'Wins':>5} {'%':>5} {'Mean(W)':>9}  {'Wins':>5} {'%':>5} {'Mean Δ(W)':>11}")
    print(f"  {'-'*90}")
    for key in level_keys:
        label = level_labels[key]
        ns = counts_sup.get(key, 0)
        na = counts_ag.get(key, 0)
        ps = 100 * ns / total if total else 0
        pa = 100 * na / total if total else 0
        mws = mean_w_sup[key]
        mwa = mean_w_ag[key]
        mws_str = f"{mws:.4f}"  if (mws is not None and not (isinstance(mws, float) and mws != mws)) else "     N/A"
        mwa_str = f"{mwa:+.4f}" if (mwa is not None and key != "baseline" and not (isinstance(mwa, float) and mwa != mwa)) else "  0.0 (ref)" if key == "baseline" else "     N/A"
        print(f"  {label:<28}  {ns:>5} {ps:>4.1f}% {mws_str:>9}  {na:>5} {pa:>4.1f}% {mwa_str:>11}")
    print(f"  {'-'*90}")
    print(f"\n  Supervised winner correct (majority ≥50%) : {winner_correct_count}/{total}  ({100*winner_correct_rate:.1f}%)")
    print(f"  Supervised winner correct (any sample >0%) : {winner_any_correct_count}/{total}  ({100*winner_any_correct_rate:.1f}%)")
    print(f"  Concordance (same winner both methods)     : {concordant}/{total}  ({100*concordance_rate:.1f}%)")
    print("────────────────────────────────────────────────────────────────────────────────────────\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir  = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_tag    = str(args.n_samples) if args.n_samples is not None else "all"
    out_stem = f"fgr_region_comparison_{args.dataset}_{n_tag}"

    agg = {
        "total_samples":            total,
        "concordant_samples":       concordant,
        "concordance_rate":         concordance_rate,
        # supervised (correct_agreement-based)
        "winner_counts_supervised":        counts_sup,
        "winner_fractions_supervised":     {k: _safe_div(v, total) for k, v in counts_sup.items()},
        "mean_winning_score_baseline":     mean_w_baseline,
        "mean_winning_score_l1":           mean_w_l1,
        "mean_winning_score_l2":           mean_w_l2,
        "mean_winning_score_l3":           mean_w_l3,
        "winner_correct_majority":         winner_correct_count,
        "winner_correct_majority_rate":    winner_correct_rate,
        "winner_correct_any":              winner_any_correct_count,
        "winner_correct_any_rate":         winner_any_correct_rate,
        # unsupervised (delta scoring: crop_score_ag − baseline_score_ag)
        "winner_counts_agreement":         counts_ag,
        "winner_fractions_agreement":      {k: _safe_div(v, total) for k, v in counts_ag.items()},
        "mean_baseline_score_ag_abs":      mean_baseline_ag_abs,
        "mean_winning_delta_l1":           mean_w_l1_ag,
        "mean_winning_delta_l2":           mean_w_l2_ag,
        "mean_winning_delta_l3":           mean_w_l3_ag,
        "mean_all_l1_ag_delta":            mean_l1_ag_delta,
        "mean_all_l2_ag_delta":            mean_l2_ag_delta,
    }

    json_path = out_dir / f"{out_stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "samples": results}, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path}")

    # ── Winning-crops JSON (for later inference comparison) ───────────────────
    winning_crops = [
        {
            "image_path":              r["image_path"],
            "question":                r["question"],
            "mc_options":              r["mc_options"],
            "gt_letter":               r["gt_letter"],
            "question_type":           r["question_type"],
            # supervised selection (use as inference input)
            "winner_supervised":       r["winner_supervised"],
            "winner_supervised_bbox":  r["winner_supervised_bbox"],   # None = full image
            # unsupervised selection (for comparison)
            "winner_agreement":        r["winner_agreement"],
            "winner_agreement_bbox":   r["winner_agreement_bbox"],    # None = full image
        }
        for r in results
    ]
    crops_path = out_dir / f"{out_stem}_winning_crops.json"
    with open(crops_path, "w", encoding="utf-8") as f:
        json.dump(winning_crops, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {crops_path}")

    flat = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in results]
    csv_path = out_dir / f"{out_stem}.csv"
    pd.DataFrame(flat).to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  Saved: {csv_path}")

    # ── W&B ─────────────────────────────────────────────────────────────────
    if not args.disable_wandb:
        wandb.summary["concordance_rate"]    = concordance_rate
        wandb.summary["concordant_samples"]  = concordant
        for k, v in counts_sup.items():
            wandb.summary[f"winner_sup_count_{k}"] = v
            wandb.summary[f"winner_sup_frac_{k}"]  = _safe_div(v, total)
        for k, v in counts_ag.items():
            wandb.summary[f"winner_ag_count_{k}"] = v
            wandb.summary[f"winner_ag_frac_{k}"]  = _safe_div(v, total)
        wandb.summary["mean_winning_score_baseline"]   = mean_w_baseline
        wandb.summary["mean_winning_score_l1"]         = mean_w_l1
        wandb.summary["mean_winning_score_l2"]         = mean_w_l2
        wandb.summary["winner_correct_majority_rate"]  = winner_correct_rate
        wandb.summary["winner_correct_any_rate"]       = winner_any_correct_rate
        wandb.summary["mean_baseline_score_ag_abs"]    = mean_baseline_ag_abs
        wandb.summary["mean_winning_delta_l1"]         = mean_w_l1_ag
        wandb.summary["mean_winning_delta_l2"]         = mean_w_l2_ag

        table = wandb.Table(columns=[
            "image_path", "question", "gt_letter", "question_type",
            "winner_supervised", "winner_supervised_score", "winner_supervised_correct",
            "winner_agreement", "winner_agreement_score_delta", "winners_match",
        ])
        for r in results:
            table.add_data(
                r["image_path"], r["question"], r["gt_letter"], r["question_type"],
                r["winner_supervised"], r["winner_supervised_score"], r["winner_supervised_correct"],
                r["winner_agreement"], r["winner_agreement_score_delta"], r["winners_match"],
            )
        wandb.log({"region_comparison": table})
        wandb.finish()


if __name__ == "__main__":
    main()
