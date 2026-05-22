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

    # ── Checkpoint setup ─────────────────────────────────────────────────────
    out_dir  = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_tag      = str(args.n_samples) if args.n_samples is not None else "all"
    out_stem   = f"fgr_region_comparison_{args.dataset}_{n_tag}"
    ckpt_path  = out_dir / f"{out_stem}_checkpoint.jsonl"

    # Resume: load already-completed samples so we can skip them
    completed_paths: set[str] = set()
    results: list[dict] = []
    if ckpt_path.exists():
        with open(ckpt_path, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    _r = json.loads(_line)
                    results.append(_r)
                    completed_paths.add(_r["image_path"])
        tqdm.write(f"  [RESUME] Loaded {len(results)} completed samples from {ckpt_path}")

    _ckpt_fh = open(ckpt_path, "a", encoding="utf-8")

    # ── Per-sample loop ──────────────────────────────────────────────────────
    for sample in tqdm(samples, desc="region comparison"):
        if not sample.mc_options or sample.mc_correct_index is None:
            continue

        # Skip samples already completed in a previous (interrupted) run
        if sample.image_path in completed_paths:
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

        QUADS = ["TL", "TR", "BL", "BR"]

        l1_patches = [p for p in patches if p.level == 1]
        l2_patches = [p for p in patches if p.level == 2]

        # ── Per-quadrant score dicts ─────────────────────────────────────────
        l1_scores_sup = {
            QUADS[i]: {
                "correct_agreement": round(p.correct_agreement, 4),
                "mean_entropy":      round(p.mean_entropy, 4),
                "mean_confidence":   round(p.mean_confidence, 4),
                "score":             round(p.importance, 6),
            }
            for i, p in enumerate(l1_patches)
        }
        l1_scores_ag = {
            QUADS[i]: {
                "mean_entropy":    round(p.mean_entropy, 4),
                "mean_confidence": round(p.mean_confidence, 4),
                "delta_score":     round(p.importance_agreement, 6),
            }
            for i, p in enumerate(l1_patches)
        }
        l2_scores_sup = {
            QUADS[i]: {
                "correct_agreement": round(p.correct_agreement, 4),
                "mean_entropy":      round(p.mean_entropy, 4),
                "mean_confidence":   round(p.mean_confidence, 4),
                "score":             round(p.importance, 6),
            }
            for i, p in enumerate(l2_patches)
        } if l2_patches else None
        l2_scores_ag = {
            QUADS[i]: {
                "mean_entropy":    round(p.mean_entropy, 4),
                "mean_confidence": round(p.mean_confidence, 4),
                "delta_score":     round(p.importance_agreement, 6),
            }
            for i, p in enumerate(l2_patches)
        } if l2_patches else None

        # ── Winner selection ─────────────────────────────────────────────────
        best_l1_score = max(v["score"] for v in l1_scores_sup.values()) if l1_scores_sup else 0.0
        best_l2_score = max(v["score"] for v in l2_scores_sup.values()) if l2_scores_sup else 0.0
        best_l1_delta = max(v["delta_score"] for v in l1_scores_ag.values()) if l1_scores_ag else 0.0
        best_l2_delta = max(v["delta_score"] for v in l2_scores_ag.values()) if l2_scores_ag else 0.0

        scores_sup = {"baseline": baseline_score, "l1": best_l1_score}
        if l2_scores_sup:
            scores_sup["l2"] = best_l2_score
        winner_supervised = max(scores_sup, key=lambda k: scores_sup[k])

        scores_ag = {"baseline": 0.0, "l1": best_l1_delta}
        if l2_scores_ag:
            scores_ag["l2"] = best_l2_delta
        winner_unsupervised = max(scores_ag, key=lambda k: scores_ag[k])

        concordance = winner_supervised == winner_unsupervised

        # ── Winner bbox ──────────────────────────────────────────────────────
        def _winner_bbox(winner: str, sup_dict, ag_dict) -> list | None:
            if winner == "baseline":
                return None
            src = sup_dict if ag_dict is None else (sup_dict if winner == "l1" else ag_dict)
            # find the quadrant with the highest score
            if winner == "l1":
                best_q = max(l1_scores_sup, key=lambda q: l1_scores_sup[q]["score"])
                return list(l1_patches[QUADS.index(best_q)].bbox)
            if winner == "l2":
                best_q = max(l2_scores_sup, key=lambda q: l2_scores_sup[q]["score"])
                return list(l2_patches[QUADS.index(best_q)].bbox)
            return None

        sup_bbox = _winner_bbox(winner_supervised, l1_scores_sup, l2_scores_sup)
        ag_bbox  = None
        if winner_unsupervised == "l1":
            best_q = max(l1_scores_ag, key=lambda q: l1_scores_ag[q]["delta_score"])
            ag_bbox = list(l1_patches[QUADS.index(best_q)].bbox)
        elif winner_unsupervised == "l2":
            best_q = max(l2_scores_ag, key=lambda q: l2_scores_ag[q]["delta_score"])
            ag_bbox = list(l2_patches[QUADS.index(best_q)].bbox)

        # ── Save winning crop images ─────────────────────────────────────────
        crops_dir = out_dir / f"{out_stem}_crops"
        crops_dir.mkdir(exist_ok=True)
        img_stem = Path(sample.image_path).stem

        def _save_crop(bbox, suffix: str) -> str:
            crop_img = image.crop(bbox) if bbox else image
            fname = f"{img_stem}_{suffix}.jpg"
            crop_img.convert("RGB").save(crops_dir / fname, quality=95)
            return str(crops_dir / fname)

        sup_crop_path = _save_crop(sup_bbox, f"sup_{winner_supervised}")
        ag_crop_path  = _save_crop(ag_bbox,  f"ag_{winner_unsupervised}")

        # ── Internal fields for aggregate (not in CSV columns) ───────────────
        # correct_agreement of each winner (for winner_correct stats)
        if winner_supervised == "baseline":
            _sup_ca = baseline_ca
        elif winner_supervised == "l1":
            best_q = max(l1_scores_sup, key=lambda q: l1_scores_sup[q]["score"])
            _sup_ca = l1_scores_sup[best_q]["correct_agreement"]
        else:
            best_q = max(l2_scores_sup, key=lambda q: l2_scores_sup[q]["score"])
            _sup_ca = l2_scores_sup[best_q]["correct_agreement"]

        results.append({
            # ── Identification ────────────────────────────────────────────────
            "image_path":   sample.image_path,
            "question":     sample.question,
            "options":      sample.mc_options,
            "gt_letter":    gt_letter,
            "question_type": sample.question_type or "unknown",
            # ── Supervised ────────────────────────────────────────────────────
            "baseline_score_supervised": round(baseline_score, 6),
            "l1_scores_supervised":      l1_scores_sup,
            "l2_scores_supervised":      l2_scores_sup,
            "winner_supervised":         winner_supervised,
            "winner_supervised_crop":    sup_crop_path,
            # ── Unsupervised ──────────────────────────────────────────────────
            "baseline_score_unsupervised": round(baseline_score_ag, 6),
            "l1_scores_unsupervised":      l1_scores_ag,
            "l2_scores_unsupervised":      l2_scores_ag,
            "winner_unsupervised":         winner_unsupervised,
            "winner_unsupervised_crop":    ag_crop_path,
            # ── Concordance ───────────────────────────────────────────────────
            "concordance": concordance,
            # ── Internal (aggregate use only, excluded from CSV) ──────────────
            "_baseline_ca":        round(baseline_ca, 4),
            "_winner_sup_ca":      round(_sup_ca, 4),
            "_winner_sup_correct": _sup_ca >= 0.5,
        })
        # Persist immediately so progress survives a crash or kill
        _ckpt_fh.write(json.dumps(results[-1], ensure_ascii=False) + "\n")
        _ckpt_fh.flush()

    _ckpt_fh.close()

    # ── Aggregate ────────────────────────────────────────────────────────────
    total  = len(results)
    has_l2 = any(r.get("l2_scores_supervised") is not None for r in results)

    counts_sup = {"baseline": 0, "l1": 0, **({"l2": 0} if has_l2 else {})}
    counts_ag  = {"baseline": 0, "l1": 0, **({"l2": 0} if has_l2 else {})}
    concordant = 0
    for r in results:
        counts_sup[r["winner_supervised"]]  = counts_sup.get(r["winner_supervised"],  0) + 1
        counts_ag[r["winner_unsupervised"]] = counts_ag.get(r["winner_unsupervised"], 0) + 1
        if r["concordance"]:
            concordant += 1

    concordance_rate = _safe_div(concordant, total)

    # Mean winning score per level (winners-only)
    def _mean_w_sup(key: str) -> float | None:
        if key == "baseline":
            vals = [r["baseline_score_supervised"] for r in results if r["winner_supervised"] == "baseline"]
        elif key == "l1":
            vals = [max(v["score"] for v in r["l1_scores_supervised"].values())
                    for r in results if r["winner_supervised"] == "l1"]
        elif key == "l2":
            vals = [max(v["score"] for v in r["l2_scores_supervised"].values())
                    for r in results if r["winner_supervised"] == "l2" and r["l2_scores_supervised"]]
        else:
            return None
        return _safe_div(sum(vals), len(vals)) if vals else float("nan")

    def _mean_w_ag(key: str) -> float | None:
        if key == "baseline":
            return 0.0
        elif key == "l1":
            vals = [max(v["delta_score"] for v in r["l1_scores_unsupervised"].values())
                    for r in results if r["winner_unsupervised"] == "l1"]
        elif key == "l2":
            vals = [max(v["delta_score"] for v in r["l2_scores_unsupervised"].values())
                    for r in results if r["winner_unsupervised"] == "l2" and r["l2_scores_unsupervised"]]
        else:
            return None
        return _safe_div(sum(vals), len(vals)) if vals else float("nan")

    mean_w_sup = {k: _mean_w_sup(k) for k in counts_sup}
    mean_w_ag  = {k: _mean_w_ag(k)  for k in counts_ag}

    # Winner correctness
    winner_correct_count     = sum(1 for r in results if r["_winner_sup_correct"])
    winner_correct_rate      = _safe_div(winner_correct_count, total)
    winner_any_correct_count = sum(1 for r in results if r["_winner_sup_ca"] > 0)
    winner_any_correct_rate  = _safe_div(winner_any_correct_count, total)

    mean_baseline_ag_abs = _safe_div(
        sum(r["baseline_score_unsupervised"] for r in results), total
    )

    level_keys   = ["baseline", "l1"] + (["l2"] if has_l2 else [])
    level_labels = {"baseline": "Full image (baseline)",
                    "l1": "Best L1 quadrant", "l2": "Best L2 sub-quadrant"}

    print("\n── Region Score Comparison ─────────────────────────────────────────────────────────────")
    print(f"  Total samples evaluated : {total}")
    print(f"  Unsupervised baseline score (absolute reference) : {mean_baseline_ag_abs:.4f}")
    print()
    print(f"  {'Region':<28}  {'Supervised (GT)':>26}  {'Unsupervised conf×(1−H) Δ':>30}")
    print(f"  {'':28}  {'Wins':>5} {'%':>5} {'Mean(W)':>9}  {'Wins':>5} {'%':>5} {'Mean Δ(W)':>11}")
    print(f"  {'-'*90}")
    for key in level_keys:
        label = level_labels[key]
        ns, na = counts_sup.get(key, 0), counts_ag.get(key, 0)
        ps, pa = 100 * ns / total if total else 0, 100 * na / total if total else 0
        mws, mwa = mean_w_sup.get(key), mean_w_ag.get(key)
        mws_s = f"{mws:.4f}" if mws is not None and mws == mws else "    N/A"
        mwa_s = ("  0.0 (ref)" if key == "baseline"
                 else f"{mwa:+.4f}" if mwa is not None and mwa == mwa else "    N/A")
        print(f"  {label:<28}  {ns:>5} {ps:>4.1f}% {mws_s:>9}  {na:>5} {pa:>4.1f}% {mwa_s:>11}")
    print(f"  {'-'*90}")
    print(f"\n  Supervised winner correct (majority ≥50%) : {winner_correct_count}/{total}  ({100*winner_correct_rate:.1f}%)")
    print(f"  Supervised winner correct (any >0%)        : {winner_any_correct_count}/{total}  ({100*winner_any_correct_rate:.1f}%)")
    print(f"  Concordance (same winner both methods)     : {concordant}/{total}  ({100*concordance_rate:.1f}%)")
    print("────────────────────────────────────────────────────────────────────────────────────────\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    agg = {
        "total_samples":                total,
        "concordant_samples":           concordant,
        "concordance_rate":             concordance_rate,
        "winner_counts_supervised":     counts_sup,
        "winner_fractions_supervised":  {k: _safe_div(v, total) for k, v in counts_sup.items()},
        "mean_winning_score":           {k: mean_w_sup[k] for k in level_keys},
        "winner_correct_majority":      winner_correct_count,
        "winner_correct_majority_rate": winner_correct_rate,
        "winner_correct_any":           winner_any_correct_count,
        "winner_correct_any_rate":      winner_any_correct_rate,
        "winner_counts_unsupervised":   counts_ag,
        "winner_fractions_unsupervised":{k: _safe_div(v, total) for k, v in counts_ag.items()},
        "mean_winning_delta":           {k: mean_w_ag[k] for k in level_keys},
        "mean_baseline_score_ag_abs":   mean_baseline_ag_abs,
    }

    # ── JSON (full data) ──────────────────────────────────────────────────────
    # Strip internal _ fields from the JSON samples output
    samples_out = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    json_path = out_dir / f"{out_stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "samples": samples_out}, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path}")

    # ── Winning-crops JSON (for later inference) ──────────────────────────────
    winning_crops = [
        {
            "image_path":               r["image_path"],
            "question":                 r["question"],
            "options":                  r["options"],
            "gt_letter":                r["gt_letter"],
            "question_type":            r["question_type"],
            "winner_supervised":        r["winner_supervised"],
            "winner_supervised_crop":   r["winner_supervised_crop"],
            "winner_unsupervised":      r["winner_unsupervised"],
            "winner_unsupervised_crop": r["winner_unsupervised_crop"],
            "concordance":              r["concordance"],
        }
        for r in results
    ]
    crops_path = out_dir / f"{out_stem}_winning_crops.json"
    with open(crops_path, "w", encoding="utf-8") as f:
        json.dump(winning_crops, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {crops_path}")

    # ── CSV (flat, exact column order, dict cols as JSON strings) ────────────
    CSV_COLS = [
        "image_path", "question", "options", "gt_letter", "question_type",
        "baseline_score_supervised",
        "l1_scores_supervised",
        *( ["l2_scores_supervised"] if has_l2 else []),
        "winner_supervised", "winner_supervised_crop",
        "baseline_score_unsupervised",
        "l1_scores_unsupervised",
        *( ["l2_scores_unsupervised"] if has_l2 else []),
        "winner_unsupervised", "winner_unsupervised_crop",
        "concordance",
    ]
    flat = []
    for r in results:
        row = {}
        for col in CSV_COLS:
            val = r.get(col)
            row[col] = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else val
        flat.append(row)
    csv_path = out_dir / f"{out_stem}.csv"
    pd.DataFrame(flat, columns=CSV_COLS).to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  Saved: {csv_path}")

    # Remove checkpoint now that final outputs are written
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"  Removed checkpoint: {ckpt_path}")

    # ── W&B ─────────────────────────────────────────────────────────────────
    if not args.disable_wandb:
        wandb.summary["concordance_rate"]           = concordance_rate
        wandb.summary["concordant_samples"]         = concordant
        wandb.summary["winner_correct_majority_rate"] = winner_correct_rate
        wandb.summary["winner_correct_any_rate"]    = winner_any_correct_rate
        wandb.summary["mean_baseline_score_ag_abs"] = mean_baseline_ag_abs
        for k, v in counts_sup.items():
            wandb.summary[f"winner_sup_count_{k}"] = v
            wandb.summary[f"winner_sup_frac_{k}"]  = _safe_div(v, total)
        for k, v in counts_ag.items():
            wandb.summary[f"winner_ag_count_{k}"]  = v
            wandb.summary[f"winner_ag_frac_{k}"]   = _safe_div(v, total)
        for k, v in mean_w_sup.items():
            if v is not None:
                wandb.summary[f"mean_winning_score_{k}"] = v
        for k, v in mean_w_ag.items():
            if v is not None:
                wandb.summary[f"mean_winning_delta_{k}"] = v

        table = wandb.Table(columns=[
            "image_path", "question", "gt_letter", "question_type",
            "winner_supervised", "winner_supervised_correct",
            "winner_unsupervised", "concordance",
        ])
        for r in results:
            table.add_data(
                r["image_path"], r["question"], r["gt_letter"], r["question_type"],
                r["winner_supervised"], r["_winner_sup_correct"],
                r["winner_unsupervised"], r["concordance"],
            )
        wandb.log({"region_comparison": table})
        wandb.finish()


if __name__ == "__main__":
    main()
