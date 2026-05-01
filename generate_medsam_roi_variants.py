"""
Generate MedSAM-based ROI image variants for Medical VQA datasets.

Supported variants:
- overlay: original image with semi-transparent ROI overlay + ROI box
- crop: ROI bounding-box crop with configurable margin
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from utils.load_data import MedicalVQALoader


SUPPORTED_DATASETS = ("pathvqa", "vqarad", "slake")
SUPPORTED_VARIANTS = ("overlay", "crop")


@dataclass
class RoiInfo:
    bbox_xyxy: list[int]
    area_pixels: int
    area_ratio: float
    score: float | None
    rank_score: float = 0.0
    ranking_components: dict[str, float] | None = None
    guidance_mode: str = "neutral"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MedSAM ROI variants for selected Medical VQA datasets/splits."
    )
    parser.add_argument("--data_root", type=str, default="data", help="Dataset root directory")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["pathvqa", "vqarad"],
        choices=SUPPORTED_DATASETS,
        help="Datasets to process",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Dataset splits to process (e.g., test, train, validation)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["overlay", "crop"],
        choices=SUPPORTED_VARIANTS,
        help="ROI variants to generate",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data_roi",
        help="Output root for generated ROI images",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        required=True,
        help="Path to MedSAM/SAM checkpoint (.pth)",
    )
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_b",
        help="SAM model type (e.g., vit_b, vit_l, vit_h)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cuda/cpu). Default: cuda if available else cpu",
    )
    parser.add_argument(
        "--crop_margin_ratio",
        type=float,
        default=0.10,
        help="Relative margin around ROI box for crop variant",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.35,
        help="Overlay alpha in [0,1] for ROI mask rendering",
    )
    parser.add_argument(
        "--max_rois",
        type=int,
        default=1,
        help="Maximum ROI masks considered; first ROI is used for variants",
    )
    parser.add_argument(
        "--save_index_json",
        action="store_true",
        default=False,
        help="Write per-dataset per-split index JSON files",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Only process first N samples per dataset/split (smoke testing)",
    )
    parser.add_argument(
        "--min_mask_area_ratio",
        type=float,
        default=0.0005,
        help="Reject tiny masks below this relative area",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_predictor(model_type: str, checkpoint_path: str, device: str):
    """Load MedSAM/SAM model as SamPredictor (supports bounding-box prompts)."""
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "Could not import segment_anything. Install SAM/MedSAM dependencies first."
        ) from exc

    model = sam_model_registry[model_type](checkpoint=checkpoint_path)
    model.to(device=device)
    model.eval()
    return SamPredictor(model)


def _generate_masks(predictor, image_np: np.ndarray) -> list[dict]:
    """
    Run MedSAM inference with the full-image bounding box as prompt.

    MedSAM was fine-tuned for prompted (bbox) segmentation, not automatic
    mask generation. Passing the full image as the bbox lets MedSAM locate
    the most salient medical structure. multimask_output=True returns three
    candidate masks; we sort by predicted_iou so _select_primary_roi picks
    the best one.
    """
    h, w = image_np.shape[:2]
    predictor.set_image(image_np)
    bbox = np.array([0, 0, w, h], dtype=float)
    masks, scores, _ = predictor.predict(box=bbox, multimask_output=True)
    result = [
        {"segmentation": mask, "predicted_iou": float(score)}
        for mask, score in zip(masks, scores)
    ]
    result.sort(key=lambda x: x["predicted_iou"], reverse=True)
    return result


def _mask_to_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Mask is empty; cannot derive bounding box.")
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return [x_min, y_min, x_max, y_max]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _question_guidance(question: str) -> dict[str, Any]:
    text = (question or "").strip().lower()
    if not text:
        return {
            "mode": "neutral",
            "reliable": False,
            "weights": {"iou": 0.45, "area": 0.30, "center": 0.25},
        }

    # Lightweight rule parser for VQA-Rad style prompts.
    yn_starts = (
        "is ",
        "are ",
        "does ",
        "do ",
        "can ",
        "could ",
        "was ",
        "were ",
        "has ",
        "have ",
    )
    is_yesno = text.startswith(yn_starts)
    is_global = bool(
        re.search(
            r"(overall|normal|abnormal|study|scan|image|x-?ray|ct|mri|view|quality|status)",
            text,
        )
    )
    has_location = bool(
        re.search(
            r"(left|right|upper|lower|lobe|lung|chest|heart|spine|brain|skull|rib|kidney|liver|abdomen)",
            text,
        )
    )
    has_measurement = bool(
        re.search(
            r"(size|diameter|length|width|measure|cm|mm|enlarged|small|large|severe|mild)",
            text,
        )
    )

    if is_global and not has_location:
        mode = "global"
        reliable = True
        weights = {"iou": 0.50, "area": 0.40, "center": 0.10}
    elif has_location or has_measurement:
        mode = "local"
        reliable = True
        weights = {"iou": 0.40, "area": 0.20, "center": 0.40}
    elif is_yesno:
        mode = "binary"
        reliable = True
        weights = {"iou": 0.50, "area": 0.30, "center": 0.20}
    else:
        mode = "neutral"
        reliable = False
        weights = {"iou": 0.45, "area": 0.30, "center": 0.25}

    return {"mode": mode, "reliable": reliable, "weights": weights}


def _bbox_center_distance_norm(bbox_xyxy: list[int], image_w: int, image_h: int) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    mx = (image_w - 1) / 2.0
    my = (image_h - 1) / 2.0
    dx = cx - mx
    dy = cy - my
    dist = math.sqrt(dx * dx + dy * dy)
    max_dist = math.sqrt(mx * mx + my * my) if image_w > 1 and image_h > 1 else 1.0
    return _clamp01(dist / max_dist)


def _area_preference_score(area_ratio: float) -> float:
    # Prefers medium-size ROIs, discourages tiny/noisy and near-full-image masks.
    if area_ratio <= 0.0:
        return 0.0
    ideal = 0.18
    spread = 0.18
    score = 1.0 - abs(area_ratio - ideal) / spread
    return _clamp01(score)


def _select_primary_roi(
    masks: list[dict[str, Any]],
    question: str,
    image_h: int,
    image_w: int,
    min_mask_area_ratio: float,
    max_rois: int,
) -> tuple[RoiInfo, np.ndarray] | None:
    if not masks:
        return None

    usable = masks[: max(1, max_rois)]
    image_area = float(image_h * image_w)
    guidance = _question_guidance(question)
    weights = guidance["weights"]
    best: tuple[RoiInfo, np.ndarray] | None = None
    best_score = -1.0

    for item in usable:
        seg = item.get("segmentation")
        if seg is None:
            continue
        seg_arr = np.asarray(seg, dtype=bool)
        area_pixels = int(seg_arr.sum())
        area_ratio = area_pixels / image_area if image_area else 0.0
        if area_ratio < min_mask_area_ratio:
            continue
        bbox = _mask_to_bbox(seg_arr)
        pred_iou = float(item.get("predicted_iou", 0.0))
        center_dist_norm = _bbox_center_distance_norm(bbox, image_w=image_w, image_h=image_h)
        center_score = 1.0 - center_dist_norm
        area_score = _area_preference_score(area_ratio)

        rank_score = (
            weights["iou"] * _clamp01(pred_iou)
            + weights["area"] * area_score
            + weights["center"] * center_score
        )
        components = {
            "predicted_iou": _clamp01(pred_iou),
            "area_score": area_score,
            "center_score": center_score,
            "center_dist_norm": center_dist_norm,
            "w_iou": weights["iou"],
            "w_area": weights["area"],
            "w_center": weights["center"],
        }
        roi = RoiInfo(
            bbox_xyxy=bbox,
            area_pixels=area_pixels,
            area_ratio=area_ratio,
            score=pred_iou,
            rank_score=rank_score,
            ranking_components=components,
            guidance_mode=guidance["mode"] if guidance["reliable"] else "neutral_fallback",
        )
        if rank_score > best_score:
            best_score = rank_score
            best = (roi, seg_arr)
    return best


def _overlay_variant(image: Image.Image, mask: np.ndarray, bbox_xyxy: list[int], alpha: float) -> Image.Image:
    image_np = np.array(image).astype(np.float32)
    mask_bool = mask.astype(bool)
    color = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    image_np[mask_bool] = (1.0 - alpha) * image_np[mask_bool] + alpha * color
    out = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8))

    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = bbox_xyxy
    draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=2)
    return out


def _crop_variant(image: Image.Image, bbox_xyxy: list[int], margin_ratio: float) -> Image.Image:
    w, h = image.size
    x1, y1, x2, y2 = bbox_xyxy
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)

    mx = int(round(bw * margin_ratio))
    my = int(round(bh * margin_ratio))
    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w - 1, x2 + mx)
    cy2 = min(h - 1, y2 + my)
    return image.crop((cx1, cy1, cx2 + 1, cy2 + 1))


def _stable_name(image_path: str) -> str:
    stem = Path(image_path).stem
    digest = hashlib.md5(image_path.encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{digest}.png"


def _save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _to_jsonable_path(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    output_root = Path(args.output_root)
    index_dir = output_root / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing MedSAM/SAM predictor (bbox-prompt mode)...")
    predictor = _load_predictor(
        model_type=args.sam_model_type,
        checkpoint_path=args.sam_checkpoint,
        device=device,
    )
    print(f"Device: {device}")
    print(f"Output root: {output_root}")

    loader = MedicalVQALoader(data_root=args.data_root)

    for dataset in args.datasets:
        for split in args.splits:
            print(f"\n=== Processing {dataset}:{split} ===")
            samples = loader.load_dataset(dataset, split=split)
            if args.n_samples is not None:
                samples = samples[: args.n_samples]
                print(f"Subset mode: n_samples={len(samples)}")

            records: list[dict[str, Any]] = []
            generated_counts = {variant: 0 for variant in args.variants}
            failed_count = 0
            no_roi_count = 0

            for sample in tqdm(samples, desc=f"{dataset}/{split}", unit="sample"):
                image_path = sample.image_path
                out_name = _stable_name(image_path)
                base_record: dict[str, Any] = {
                    "dataset": dataset,
                    "split": split,
                    "question": sample.question,
                    "answer": sample.answer,
                    "original_image_path": image_path,
                    "medsam_ok": False,
                    "error": None,
                    "bbox_xyxy": None,
                    "mask_area_pixels": None,
                    "mask_area_ratio": None,
                    "predicted_iou": None,
                    "roi_rank_score": None,
                    "roi_ranking_components": None,
                    "guidance_mode": None,
                    "variant_paths": {variant: None for variant in args.variants},
                }

                try:
                    image = Image.open(image_path).convert("RGB")
                    image_np = np.array(image)
                    h, w = image_np.shape[:2]

                    masks = _generate_masks(predictor, image_np)
                    selected = _select_primary_roi(
                        masks=masks,
                        question=sample.question,
                        image_h=h,
                        image_w=w,
                        min_mask_area_ratio=args.min_mask_area_ratio,
                        max_rois=args.max_rois,
                    )

                    if selected is None:
                        no_roi_count += 1
                        base_record["error"] = "no_valid_roi"
                        records.append(base_record)
                        continue

                    roi, mask_arr = selected
                    base_record["medsam_ok"] = True
                    base_record["bbox_xyxy"] = roi.bbox_xyxy
                    base_record["mask_area_pixels"] = roi.area_pixels
                    base_record["mask_area_ratio"] = roi.area_ratio
                    base_record["predicted_iou"] = roi.score
                    base_record["roi_rank_score"] = roi.rank_score
                    base_record["roi_ranking_components"] = roi.ranking_components
                    base_record["guidance_mode"] = roi.guidance_mode

                    for variant in args.variants:
                        out_path = output_root / dataset / split / variant / out_name
                        if variant == "overlay":
                            out_image = _overlay_variant(
                                image=image,
                                mask=mask_arr,
                                bbox_xyxy=roi.bbox_xyxy,
                                alpha=args.overlay_alpha,
                            )
                        elif variant == "crop":
                            out_image = _crop_variant(
                                image=image,
                                bbox_xyxy=roi.bbox_xyxy,
                                margin_ratio=args.crop_margin_ratio,
                            )
                        else:
                            raise ValueError(f"Unsupported variant: {variant}")

                        _save_image(out_image, out_path)
                        base_record["variant_paths"][variant] = _to_jsonable_path(out_path)
                        generated_counts[variant] += 1

                    records.append(base_record)
                except Exception as exc:  # pragma: no cover (robust batch handling)
                    failed_count += 1
                    base_record["error"] = str(exc)
                    records.append(base_record)

            total = len(samples)
            successful = sum(1 for r in records if r["medsam_ok"])
            print(
                f"[{dataset}/{split}] total={total}, successful={successful}, "
                f"no_roi={no_roi_count}, failed={failed_count}"
            )
            for variant in args.variants:
                print(f"[{dataset}/{split}] generated_{variant}={generated_counts[variant]}")

            # Simple consistency checks for smoke validation.
            for variant in args.variants:
                expected = successful
                actual = generated_counts[variant]
                if actual != expected:
                    print(
                        f"Warning: variant '{variant}' count mismatch: "
                        f"expected={expected}, actual={actual}"
                    )

            if args.save_index_json:
                index_path = index_dir / f"{dataset}_{split}_roi_index.json"
                with open(index_path, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2, ensure_ascii=False)
                print(f"Saved index: {index_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
