"""FGR (Faithful Grounding Rationale) evaluation for Qwen2.5-VL.

Pipeline
--------
1. Load MedVLM-R1 results JSON  →  extract (image_path, question, options,
   GT, reasoning trace *r*).
2. For each sample run hierarchical 2×2 perturbation (3 levels, 12 passes)
   to locate the visually relevant region.
3. Build image variants for Settings C / D / E.
4. Evaluate Qwen2.5-VL under all five settings with *r* as context.
5. Compute per-sample and aggregate FGR metrics.

Settings
--------
    A  text-only (q + r)               → S_lang
    B  full image + r                  → reference baseline
    C  relevant-region crop + r        → S_suff
    D  irrelevant-region crop + r      → S_shortcut  (size-matched to C)
    E  full image (relevant mean-filled) + r  → S_nec = correct(B) – correct(E)

Global-evidence samples (max L1 importance < threshold) are excluded from
C / D / E and only receive Settings A and B.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
import wandb
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from utils.fgr import (
    PerturbPatch,
    blur_fill_region,
    crop_irrelevant_size_matched,
    crop_relevant,
    mean_fill_region,
    patches_to_bbox,
    run_hierarchical_perturbation,
    select_irrelevant_patches,
    select_relevant_patches,
)
from utils.fgr_conversation import build_fgr_conversation
from utils.mc_vqa import parse_predicted_letter
from utils.paths import get_results_dir
from utils.qwen_vl import DEFAULT_MODEL_PATH, prepare_qwen_inputs, resolve_processor_path
from utils.save_results import save_results

# ---------------------------------------------------------------------------
# Reasoning extraction
# ---------------------------------------------------------------------------

_THINKING_RE = re.compile(r"<redacted_thinking>(.*?)</redacted_thinking>", re.DOTALL)


def extract_reasoning(predicted_answer: str) -> str:
    """Pull the chain-of-thought text out of a MedVLM-R1 predicted_answer string."""
    m = _THINKING_RE.search(predicted_answer)
    if m:
        return m.group(1).strip()
    # Fallback: strip <answer>…</answer> tags and return whatever remains.
    stripped = re.sub(r"<answer>.*?</answer>", "", predicted_answer, flags=re.DOTALL)
    return stripped.strip()


# ---------------------------------------------------------------------------
# Per-setting inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_setting(
    model: torch.nn.Module,
    processor,
    conversation: list[dict],
    device: torch.device,
    *,
    ground_truth_letter: str,
    n_options: int,
    max_new_tokens: int = 8,
) -> dict:
    """Generate one MC answer under a given FGR setting.

    Returns a dict with keys:
        predicted_letter, predicted_text, is_correct, confidence, entropy
    """
    inputs = prepare_qwen_inputs(processor, conversation, add_generation_prompt=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    gen_out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

    generated_ids = gen_out.sequences[0][input_length:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
    predicted_letter = parse_predicted_letter(generated_text, num_options=n_options)

    # Confidence + entropy from the first generated token's logits
    confidence, entropy = float("nan"), float("nan")
    if gen_out.scores:
        first_logits = gen_out.scores[0][0].double()
        probs = torch.softmax(first_logits, dim=-1)
        confidence = float(probs.max().item())
        entropy = float(
            -torch.where(probs > 0, probs * torch.log(probs), torch.zeros_like(probs))
            .sum()
            .item()
        )

    return {
        "predicted_text": generated_text,
        "predicted_letter": predicted_letter,
        "is_correct": bool(predicted_letter == ground_truth_letter),
        "confidence": confidence,
        "entropy": entropy,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run FGR faithfulness evaluation for Qwen2.5-VL on MedVLM-R1 outputs"
    )
    p.add_argument(
        "--medvlm_results",
        type=str,
        required=True,
        help="Path to MedVLM-R1 inference JSON (contains predicted_answer with reasoning).",
    )
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["pathvqa", "slake", "vqarad", "medvqa2019"],
    )
    p.add_argument("--results_dir", type=str, default=get_results_dir())
    p.add_argument("--n_samples", type=int, default=None,
                   help="Limit total number of samples to evaluate.")
    # Perturbation sampling
    p.add_argument(
        "--perturb_samples",
        type=int,
        default=5,
        help="Stochastic samples per crop during hierarchical perturbation.",
    )
    p.add_argument("--perturb_temperature", type=float, default=0.7)
    p.add_argument("--perturb_top_p", type=float, default=0.9)
    # Global-evidence gate
    p.add_argument(
        "--global_threshold",
        type=float,
        default=0.6,
        help=(
            "A sample is classified as global-evidence if "
            "best_L1_score < global_threshold × baseline_score."
        ),
    )
    p.add_argument(
        "--cumulative_threshold",
        type=float,
        default=0.80,
        help="Cumulative importance fraction for relevant patch selection (default 0.80).",
    )
    p.add_argument(
        "--fill_mode",
        type=str,
        default="mean",
        choices=["mean", "blur"],
        help="How to obscure the relevant region in Setting E.",
    )
    p.add_argument("--n_levels", type=int, default=3, choices=[1, 2, 3],
                   help="Depth of hierarchical perturbation (1=4, 2=8, 3=12 crops per sample).")
    p.add_argument("--blur_radius", type=int, default=15)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--wandb_project", type=str, default="medvqa")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--disable_wandb", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _move_to_device(inputs: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def _safe_div(a: float, b: float) -> float:
    return (a / b) if b else float("nan")


def _binary(setting_result: dict | None) -> int | None:
    if setting_result is None:
        return None
    return 1 if setting_result.get("is_correct") else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Load MedVLM-R1 results ───────────────────────────────────────────────
    medvlm_path = Path(args.medvlm_results)
    if not medvlm_path.exists():
        raise FileNotFoundError(f"MedVLM-R1 results not found: {medvlm_path}")
    with open(medvlm_path, encoding="utf-8") as f:
        medvlm_data: list[dict] = json.load(f)

    if args.n_samples is not None:
        medvlm_data = medvlm_data[: args.n_samples]

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
    run_name = args.wandb_run_name or f"fgr_qwen_{args.dataset}"
    if not args.disable_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            group=args.dataset,
            tags=["fgr", "faithfulness", "qwen", args.dataset],
            config=vars(args),
        )

    # ── Per-sample loop ──────────────────────────────────────────────────────
    results: list[dict] = []

    for entry in tqdm(medvlm_data, desc="FGR eval"):
        image_path = entry.get("image_path") or entry.get("source_image_path")
        question = entry.get("question", "")
        options: list[str] = entry.get("options") or []
        gt_letter: str = entry.get("ground_truth_letter") or entry.get("answer", "")
        predicted_answer_raw: str = entry.get("predicted_answer", "")
        qtype = entry.get("question_type", "unknown")

        # Skip samples that can't be evaluated
        if not options or not gt_letter or not image_path:
            continue

        reasoning = extract_reasoning(predicted_answer_raw)
        if not reasoning:
            # No reasoning trace → skip (FGR requires r)
            continue

        n_options = len(options)

        # ── Load image ───────────────────────────────────────────────────────
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            tqdm.write(f"  [WARN] Cannot open image {image_path}: {exc}")
            continue

        # ── Hierarchical perturbation ────────────────────────────────────────
        patches, evidence_type, baseline_score, baseline_score_agreement, _ = run_hierarchical_perturbation(
            model,
            processor,
            image,
            question,
            options,
            gt_letter,
            device,
            n_levels=args.n_levels,
            global_threshold=args.global_threshold,
            n_samples=args.perturb_samples,
            temperature=args.perturb_temperature,
            top_p=args.perturb_top_p,
            max_new_tokens=args.max_new_tokens,
        )

        # ── Region selection ─────────────────────────────────────────────────
        rel_patches = select_relevant_patches(patches, args.cumulative_threshold)
        rel_bbox = patches_to_bbox(rel_patches)
        irrel_patches = select_irrelevant_patches(patches, k=len(rel_patches))

        # ── Image variants for Settings C / D / E ───────────────────────────
        img_c: Image.Image | None = None
        img_d: Image.Image | None = None
        img_e: Image.Image | None = None
        ref_size: tuple[int, int] | None = None

        if evidence_type == "local":
            img_c, _ = crop_relevant(image, rel_patches)
            ref_size = img_c.size  # (width, height)
            img_d = crop_irrelevant_size_matched(image, irrel_patches, ref_size)
            if args.fill_mode == "blur":
                img_e = blur_fill_region(image, rel_bbox, radius=args.blur_radius)
            else:
                img_e = mean_fill_region(image, rel_bbox)

        # ── Run Settings A – E ───────────────────────────────────────────────
        def _run(img: Image.Image | None) -> dict:
            conv = build_fgr_conversation(question, options, reasoning, image=img)
            return run_setting(
                model, processor, conv, device,
                ground_truth_letter=gt_letter,
                n_options=n_options,
                max_new_tokens=args.max_new_tokens,
            )

        setting_a = _run(None)
        setting_b = _run(image)
        setting_c = _run(img_c) if img_c is not None else None
        setting_d = _run(img_d) if img_d is not None else None
        setting_e = _run(img_e) if img_e is not None else None

        # ── Per-sample FGR metrics ───────────────────────────────────────────
        b_bin = _binary(setting_b)
        s_lang     = _binary(setting_a)
        s_suff     = _binary(setting_c)
        s_nec      = (b_bin - _binary(setting_e)) if (b_bin is not None and setting_e) else None
        s_shortcut = _binary(setting_d)
        s_faithful = (
            (s_suff + s_nec - s_shortcut)
            if (s_suff is not None and s_nec is not None and s_shortcut is not None)
            else None
        )

        # Confidence-weighted variant of S_nec (continuous)
        s_nec_conf = (
            (setting_b["confidence"] - setting_e["confidence"])
            if setting_e is not None
            else None
        )

        results.append(
            {
                # ── Provenance ────────────────────────────────────────────────
                "image_path": image_path,
                "question": question,
                "options": options,
                "ground_truth_letter": gt_letter,
                "question_type": qtype,
                # ── MedVLM-R1 source fields ───────────────────────────────────
                "medvlm_predicted_letter": entry.get("predicted_letter"),
                "medvlm_is_correct": entry.get("is_correct"),
                "reasoning": reasoning,
                # ── Perturbation output ───────────────────────────────────────
                "baseline_score": round(baseline_score, 6),
                "baseline_score_agreement": round(baseline_score_agreement, 6),
                "evidence_type": evidence_type,
                "relevant_bbox": list(rel_bbox),
                "relevant_region_fraction": round(
                    ((rel_bbox[2] - rel_bbox[0]) * (rel_bbox[3] - rel_bbox[1]))
                    / (image.width * image.height),
                    4,
                ),
                "n_relevant_patches": len(rel_patches),
                "perturbation_patches": [
                    {
                        "level": p.level,
                        "bbox": list(p.bbox),
                        "importance": round(p.importance, 6),
                        "importance_agreement": round(p.importance_agreement, 6),
                        "correct_agreement": round(p.correct_agreement, 4),
                        "agreement": round(p.agreement, 4),
                        "mean_confidence": round(p.mean_confidence, 4),
                        "mean_entropy": round(p.mean_entropy, 4),
                    }
                    for p in patches
                ],
                # ── Settings A–E ──────────────────────────────────────────────
                "fgr_setting_a": setting_a,
                "fgr_setting_b": setting_b,
                "fgr_setting_c": setting_c,
                "fgr_setting_d": setting_d,
                "fgr_setting_e": setting_e,
                # ── Per-sample metrics ────────────────────────────────────────
                "s_lang": s_lang,
                "s_suff": s_suff,
                "s_nec": s_nec,
                "s_nec_conf": s_nec_conf,
                "s_shortcut": s_shortcut,
                "s_faithful": s_faithful,
            }
        )

    # ── Aggregate metrics ────────────────────────────────────────────────────
    total = len(results)
    local_rows = [r for r in results if r["evidence_type"] == "local"]
    global_rows = [r for r in results if r["evidence_type"] == "global"]

    def _mean(key: str, rows: list[dict]) -> float:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return _safe_div(sum(vals), len(vals))

    agg = {
        "total_samples": total,
        "local_samples": len(local_rows),
        "global_samples": len(global_rows),
        # Setting-level accuracy
        "acc_setting_a": _mean("s_lang", results),
        "acc_setting_b": sum(
            1 for r in results if r["fgr_setting_b"] and r["fgr_setting_b"].get("is_correct")
        ) / max(total, 1),
        # FGR metrics (local-evidence samples only)
        "s_lang_mean": _mean("s_lang", results),
        "s_suff_mean": _mean("s_suff", local_rows),
        "s_nec_mean": _mean("s_nec", local_rows),
        "s_nec_conf_mean": _mean("s_nec_conf", local_rows),
        "s_shortcut_mean": _mean("s_shortcut", local_rows),
        "s_faithful_mean": _mean("s_faithful", local_rows),
    }

    print("\n── FGR Aggregate Results ──────────────────────────────────────────")
    print(f"  Samples total / local / global : {total} / {len(local_rows)} / {len(global_rows)}")
    print(f"  S_lang  (Setting A accuracy)   : {agg['s_lang_mean']:.4f}")
    print(f"  Setting B accuracy (reference) : {agg['acc_setting_b']:.4f}")
    print(f"  S_suff  (Setting C accuracy)   : {agg['s_suff_mean']:.4f}")
    print(f"  S_nec   (B – E, binary)        : {agg['s_nec_mean']:.4f}")
    print(f"  S_nec   (B – E, confidence)    : {agg['s_nec_conf_mean']:.4f}")
    print(f"  S_shortcut (Setting D accuracy): {agg['s_shortcut_mean']:.4f}")
    print(f"  S_faithful (suff+nec–shortcut) : {agg['s_faithful_mean']:.4f}")
    print("───────────────────────────────────────────────────────────────────\n")

    # ── Save results ─────────────────────────────────────────────────────────
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    n_tag = str(args.n_samples) if args.n_samples is not None else "all"
    out_stem = f"fgr_qwen_{args.dataset}_{n_tag}"

    json_path = results_dir / f"{out_stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "samples": results}, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path}")

    # Flat CSV for downstream analysis
    import pandas as pd

    flat_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
        for setting_key in ("fgr_setting_a", "fgr_setting_b", "fgr_setting_c",
                            "fgr_setting_d", "fgr_setting_e"):
            s = r.get(setting_key) or {}
            prefix = setting_key.replace("fgr_", "")
            for field in ("predicted_letter", "is_correct", "confidence", "entropy"):
                row[f"{prefix}_{field}"] = s.get(field)
        flat_rows.append(row)

    df = pd.DataFrame(flat_rows)
    csv_path = results_dir / f"{out_stem}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  Saved: {csv_path}")

    # ── W&B logging ──────────────────────────────────────────────────────────
    if not args.disable_wandb:
        for k, v in agg.items():
            wandb.summary[k] = v

        table = wandb.Table(
            columns=[
                "image_path", "question", "ground_truth_letter", "evidence_type",
                "s_lang", "s_suff", "s_nec", "s_shortcut", "s_faithful",
                "a_correct", "b_correct", "c_correct", "d_correct", "e_correct",
            ]
        )
        for r in results:
            table.add_data(
                r["image_path"], r["question"], r["ground_truth_letter"],
                r["evidence_type"],
                r.get("s_lang"), r.get("s_suff"), r.get("s_nec"),
                r.get("s_shortcut"), r.get("s_faithful"),
                r["fgr_setting_a"].get("is_correct"),
                r["fgr_setting_b"].get("is_correct"),
                r["fgr_setting_c"].get("is_correct") if r["fgr_setting_c"] else None,
                r["fgr_setting_d"].get("is_correct") if r["fgr_setting_d"] else None,
                r["fgr_setting_e"].get("is_correct") if r["fgr_setting_e"] else None,
            )
        wandb.log({"fgr_predictions": table})
        wandb.finish()


if __name__ == "__main__":
    main()
