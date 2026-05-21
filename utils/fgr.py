"""Faithful Grounding Rationale (FGR) — perturbation & region utilities.

Perturbation logic
------------------
Each quadrant is shown to the model *in isolation* (crop-only, never mask-out).
The model is sampled ``n_samples`` times per crop to produce three uncertainty
signals: agreement, mean confidence, mean entropy.  These are combined into a
single region score:

    region_score      = correct_agreement × mean_confidence × (1 − normalised_entropy)  [supervised]
    region_score_ag   = mean_confidence × (1 − normalised_entropy)                       [unsupervised, no GT]

The quadrant with the highest region score advances to the next level.
After three levels (4 + 4 + 4 = 12 generate-calls per sample) the 80 %-
cumulative-importance rule identifies the visually relevant region.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageFilter

from utils.mc_vqa import parse_predicted_letter
from utils.qwen_vl import build_mc_conversation, prepare_qwen_inputs


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class PerturbPatch:
    """Scored crop from the hierarchical perturbation pass."""

    level: int                           # 1 | 2 | 3
    bbox: tuple[int, int, int, int]      # (x1, y1, x2, y2) in original image pixels
    importance: float                    # correct_agreement × mean_conf × (1−norm_H)  ← ranking key
    importance_agreement: float          # mean_conf × (1−norm_H) − baseline  ← unsupervised proxy (no GT, no agreement)
    correct_agreement: float             # P(predicted == gt_letter) across n_samples
    agreement: float                     # plurality agreement (any letter)
    mean_confidence: float               # mean max-softmax confidence across samples
    mean_entropy: float                  # mean first-token entropy across samples (nats)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_2x2(bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    """Divide *bbox* into four equal quadrants (TL, TR, BL, BR)."""
    x1, y1, x2, y2 = bbox
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    return [
        (x1, y1, mx, my),
        (mx, y1, x2, my),
        (x1, my, mx, y2),
        (mx, my, x2, y2),
    ]


@torch.no_grad()
def _sample_region(
    model: torch.nn.Module,
    processor,
    image_crop: Image.Image,
    question: str,
    options: list[str],
    gt_letter: str,
    device: torch.device,
    *,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    log_vocab_size: float,
) -> tuple[float, float, float, float, float]:
    """Show *image_crop* in isolation, sample *n_samples* times.

    Returns
    -------
    correct_agreement : float
        Fraction of samples that predict *gt_letter* (0–1).  Used as the
        primary ranking signal so that crops guiding the model to the right
        answer rank above crops that produce confident but *wrong* answers.
    agreement : float
        Plurality agreement (any letter); retained for diagnostic logging.
    mean_confidence : float
        Mean max-softmax probability of the first generated token (0–1).
    mean_entropy : float
        Mean first-token entropy in nats.
    region_score : float
        correct_agreement × mean_confidence × (1 − normalised_entropy).
    region_score_agreement : float
        mean_confidence × (1 − normalised_entropy) — uncertainty-only signal usable
        without GT.  Agreement is intentionally omitted because it rewards
        consistent-but-wrong outputs; confidence and entropy are better-calibrated
        proxies for local visual evidence.
    """
    n_options = len(options)
    conv = build_mc_conversation(question, options, image=image_crop)
    inputs = prepare_qwen_inputs(processor, conv, add_generation_prompt=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    letters: list[str | None] = []
    confidences: list[float] = []
    entropies: list[float] = []

    for _ in range(n_samples):
        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=True,
        )
        generated_ids = gen_out.sequences[0][input_length:]
        text = processor.decode(generated_ids, skip_special_tokens=True).strip()
        letters.append(parse_predicted_letter(text, num_options=n_options))

        if gen_out.scores:
            first_logits = gen_out.scores[0][0].double()
            probs = torch.softmax(first_logits, dim=-1)
            confidences.append(float(probs.max().item()))
            entropies.append(float(
                -torch.where(probs > 0, probs * torch.log(probs), torch.zeros_like(probs))
                .sum().item()
            ))

    valid = [l for l in letters if l is not None]

    # correct_agreement: fraction that hit the ground-truth letter (ranking key)
    correct_agreement = (
        sum(1 for l in valid if l == gt_letter) / len(valid)
    ) if valid else 0.0

    # agreement: plurality fraction (diagnostic; may differ when model is consistently wrong)
    agreement = (max(Counter(valid).values()) / len(valid)) if valid else 0.0

    mean_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
    mean_ent  = (sum(entropies)   / len(entropies))   if entropies  else 0.0

    norm_entropy = min(mean_ent / log_vocab_size, 1.0) if log_vocab_size > 0 else 0.0

    region_score           = correct_agreement * mean_conf * (1.0 - norm_entropy)
    region_score_agreement =                    mean_conf * (1.0 - norm_entropy)
    return correct_agreement, agreement, mean_conf, mean_ent, region_score, region_score_agreement


# ---------------------------------------------------------------------------
# Hierarchical perturbation
# ---------------------------------------------------------------------------

def run_hierarchical_perturbation(
    model: torch.nn.Module,
    processor,
    image: Image.Image,
    question: str,
    options: list[str],
    gt_letter: str,
    device: torch.device,
    *,
    n_levels: int = 2,
    global_threshold: float = 0.6,
    n_samples: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 8,
) -> tuple[list[PerturbPatch], str, float]:
    """Run hierarchical 2×2 crop-and-score perturbation.

    At every level each quadrant is cropped from the **original image** and
    shown to Qwen in isolation.  The model is sampled *n_samples* times per
    crop; the quadrant with the highest region score advances to the next
    level (beam-width = 1).

    Global-evidence gate
    --------------------
    After Level 1 the best quadrant score is compared to the full-image
    baseline score.  If ``best_l1_score < global_threshold × baseline_score``
    no single quadrant carries the answer → sample is ``"global"``.

    Parameters
    ----------
    global_threshold:
        Fraction of baseline below which L1 triggers the global gate (0.6).
    n_samples:
        Stochastic samples per crop for agreement / confidence / entropy.

    Returns
    -------
    patches:
        All PerturbPatch objects across all levels.
    evidence_type:
        ``"local"`` or ``"global"``.
    baseline_score:
        Full-image score using correct_agreement (supervised).
    baseline_score_agreement:
        Full-image score using plurality agreement (unsupervised proxy).
    """
    w, h = image.size
    full_bbox = (0, 0, w, h)

    # Vocabulary size for entropy normalisation
    vocab_size = getattr(processor.tokenizer, "vocab_size", None) or len(processor.tokenizer)
    log_vocab_size = math.log(max(vocab_size, 2))

    sample_kw = dict(
        n_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        log_vocab_size=log_vocab_size,
    )

    def _score(crop: Image.Image):
        ca, ag, mc, me, score, score_ag = _sample_region(
            model, processor, crop, question, options, gt_letter, device, **sample_kw
        )
        return ca, ag, mc, me, score, score_ag

    def _patch(level, bbox, ca, ag, mc, me, score, score_ag) -> PerturbPatch:
        return PerturbPatch(
            level=level, bbox=bbox,
            importance=score, importance_agreement=score_ag,
            correct_agreement=ca, agreement=ag,
            mean_confidence=mc, mean_entropy=me,
        )

    # ── Baseline: full image ─────────────────────────────────────────────────
    baseline_ca, _, _, _, baseline_score, baseline_score_agreement = _score(image)

    all_patches: list[PerturbPatch] = []

    def _delta_ag(patches: list[PerturbPatch]) -> None:
        """Subtract baseline agreement score in-place → importance_agreement becomes a delta.

        After this transform importance_agreement > 0 means the crop improves
        over the full-image unsupervised signal; < 0 means it hurts.  The
        baseline's implicit delta is therefore always 0 (the reference point),
        which eliminates the baseline-consistency bias present in the absolute
        agreement score.
        """
        for p in patches:
            p.importance_agreement = p.importance_agreement - baseline_score_agreement

    # ── Level 1: four quadrants of the full image ────────────────────────────
    l1_patches: list[PerturbPatch] = []
    for bbox in _split_2x2(full_bbox):
        ca, ag, mc, me, score, score_ag = _score(image.crop(bbox))
        l1_patches.append(_patch(1, bbox, ca, ag, mc, me, score, score_ag))
    _delta_ag(l1_patches)
    all_patches.extend(l1_patches)

    # ── Global-evidence gate (uses supervised score for ranking) ─────────────
    best_l1_score = max(p.importance for p in l1_patches)
    if baseline_score > 0 and best_l1_score < global_threshold * baseline_score:
        return all_patches, "global", baseline_score, baseline_score_agreement, baseline_ca

    if n_levels < 2:
        return all_patches, "local", baseline_score, baseline_score_agreement, baseline_ca, baseline_ca

    # ── Level 2: four sub-quadrants of the best L1 patch ────────────────────
    best_l1 = max(l1_patches, key=lambda p: p.importance)
    l2_patches: list[PerturbPatch] = []
    for bbox in _split_2x2(best_l1.bbox):
        ca, ag, mc, me, score, score_ag = _score(image.crop(bbox))
        l2_patches.append(_patch(2, bbox, ca, ag, mc, me, score, score_ag))
    _delta_ag(l2_patches)
    all_patches.extend(l2_patches)

    if n_levels < 3:
        return all_patches, "local", baseline_score, baseline_score_agreement, baseline_ca

    # ── Level 3: four sub-quadrants of the best L2 patch ────────────────────
    best_l2 = max(l2_patches, key=lambda p: p.importance)
    l3_patches: list[PerturbPatch] = []
    for bbox in _split_2x2(best_l2.bbox):
        ca, ag, mc, me, score, score_ag = _score(image.crop(bbox))
        l3_patches.append(_patch(3, bbox, ca, ag, mc, me, score, score_ag))
    _delta_ag(l3_patches)
    all_patches.extend(l3_patches)

    return all_patches, "local", baseline_score, baseline_score_agreement


# ---------------------------------------------------------------------------
# Region selection
# ---------------------------------------------------------------------------

def select_relevant_patches(
    patches: list[PerturbPatch],
    cumulative_threshold: float = 0.80,
) -> list[PerturbPatch]:
    """Smallest top-score set whose cumulative importance ≥ *cumulative_threshold* × total.

    Only patches with positive importance are considered.  If all scores are
    zero or negative the single highest-scoring patch is returned as fallback.
    """
    positive = [p for p in patches if p.importance > 0]
    if not positive:
        return [max(patches, key=lambda p: p.importance)]

    sorted_p = sorted(positive, key=lambda p: p.importance, reverse=True)
    total = sum(p.importance for p in sorted_p)
    cumulative, relevant = 0.0, []
    for p in sorted_p:
        relevant.append(p)
        cumulative += p.importance
        if cumulative >= cumulative_threshold * total:
            break
    return relevant


def select_irrelevant_patches(patches: list[PerturbPatch], k: int) -> list[PerturbPatch]:
    """Bottom-*k* patches by importance score (least informative), k ≥ 1."""
    return sorted(patches, key=lambda p: p.importance)[: max(k, 1)]


def patches_to_bbox(patches: list[PerturbPatch]) -> tuple[int, int, int, int]:
    """Axis-aligned union bounding box of all patches."""
    return (
        min(p.bbox[0] for p in patches),
        min(p.bbox[1] for p in patches),
        max(p.bbox[2] for p in patches),
        max(p.bbox[3] for p in patches),
    )


# ---------------------------------------------------------------------------
# Image manipulations (Settings C / D / E)
# ---------------------------------------------------------------------------

def crop_relevant(
    image: Image.Image,
    relevant_patches: list[PerturbPatch],
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """BBox-crop to the union of *relevant_patches* (Setting C)."""
    bbox = patches_to_bbox(relevant_patches)
    return image.crop(bbox), bbox


def crop_irrelevant_size_matched(
    image: Image.Image,
    irrelevant_patches: list[PerturbPatch],
    ref_size: tuple[int, int],
) -> Image.Image:
    """BBox-crop to irrelevant region, resized to *ref_size* = (width, height) (Setting D).

    Size-matching ensures Settings C and D enter the model at identical
    resolution, removing spatial scale as a confound.
    """
    bbox = patches_to_bbox(irrelevant_patches)
    return image.crop(bbox).resize(ref_size, Image.LANCZOS)


def mean_fill_region(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
) -> Image.Image:
    """Replace *bbox* with its per-channel spatial mean (Setting E, fill_mode='mean').

    Mean-fill removes local spatial structure without the OOD black-rectangle
    artefact that the model might interpret as an explicit masking signal.
    """
    arr = np.array(image, dtype=np.uint8).copy()
    x1, y1, x2, y2 = bbox
    region = arr[y1:y2, x1:x2]
    if region.size == 0:
        return Image.fromarray(arr)
    arr[y1:y2, x1:x2] = region.mean(axis=(0, 1)).round().astype(np.uint8)
    return Image.fromarray(arr)


def blur_fill_region(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    radius: int = 15,
) -> Image.Image:
    """Replace *bbox* with a Gaussian-blurred copy of that region (Setting E, fill_mode='blur')."""
    img = image.copy()
    blurred = img.crop(bbox).filter(ImageFilter.GaussianBlur(radius=radius))
    img.paste(blurred, (bbox[0], bbox[1]))
    return img
