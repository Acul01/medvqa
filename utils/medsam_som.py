"""MedSAM-backed automatic SoM mask generation via a multi-scale box grid.

MedSAM is promptable (boxes/points). For offline SoM annotation we synthesise
prompts: a grid of boxes over the *body ROI* (non-black pixels), skip boxes
whose mean intensity is near background black, decode with MedSAM, then
drop empty / near-black masks, area-filter, and IoU-NMS.

Output of :meth:`MedSamBoxGridGenerator.generate` matches SAM Automatic's
mask dicts (``segmentation``, ``area``, ``bbox``) so
``annotate_image_with_som`` can stay backend-agnostic.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from PIL import Image

DEFAULT_MEDSAM_MODEL_ID = "wanglab/medsam-vit-base"
DEFAULT_IOU_NMS_THRESH = 0.7
DEFAULT_MAX_MASK_FRAC = 0.85
# Pixel mean (0–255 gray) below which a region counts as CT/X-ray background.
DEFAULT_BG_MEAN = 18.0
# Fraction of box pixels that must be above bg to keep the prompt.
DEFAULT_MIN_FG_FRAC = 0.08


def to_gray_u8(image_rgb: np.ndarray) -> np.ndarray:
    """HxW uint8 luminance from HxWx3 RGB."""
    if image_rgb.ndim == 2:
        return image_rgb.astype(np.uint8, copy=False)
    return np.round(
        0.299 * image_rgb[..., 0] + 0.587 * image_rgb[..., 1] + 0.114 * image_rgb[..., 2]
    ).astype(np.uint8)


def body_roi(
    gray: np.ndarray,
    *,
    bg_mean: float = DEFAULT_BG_MEAN,
    pad_frac: float = 0.02,
) -> tuple[int, int, int, int]:
    """Tight bbox ``(x0, y0, x1, y1)`` around non-background pixels.

    Falls back to the full image if almost nothing is above *bg_mean*
    (avoids empty grids on unusual images).
    """
    h, w = gray.shape[:2]
    fg = gray > bg_mean
    if fg.sum() < max(64, int(0.001 * h * w)):
        return 0, 0, w, h
    ys, xs = np.where(fg)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad_x = int(pad_frac * (x1 - x0))
    pad_y = int(pad_frac * (y1 - y0))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0, 0, w, h
    return x0, y0, x1, y1


def build_box_grid(
    height: int,
    width: int,
    *,
    points_per_side: int = 4,
    overlap: float = 0.2,
    roi: Optional[tuple[int, int, int, int]] = None,
) -> list[list[float]]:
    """Return boxes ``[x0, y0, x1, y1]`` for two grid densities inside *roi*.

    Primary grid uses *points_per_side*; a second denser grid uses
    ``points_per_side + 2``. Boxes tile the ROI with the given *overlap*.
    *roi* is ``(x0, y0, x1, y1)`` in image coordinates; default = full image.
    """
    if roi is None:
        rx0, ry0, rx1, ry1 = 0, 0, width, height
    else:
        rx0, ry0, rx1, ry1 = roi
        rx0 = max(0, int(rx0))
        ry0 = max(0, int(ry0))
        rx1 = min(width, int(rx1))
        ry1 = min(height, int(ry1))
    roi_w = max(1, rx1 - rx0)
    roi_h = max(1, ry1 - ry0)

    sides = sorted({max(1, int(points_per_side)), max(1, int(points_per_side) + 2)})
    boxes: list[list[float]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for n in sides:
        cell_w = roi_w / n
        cell_h = roi_h / n
        stride_x = cell_w * (1.0 - overlap)
        stride_y = cell_h * (1.0 - overlap)
        nx = max(1, int(np.ceil((roi_w - cell_w) / stride_x)) + 1) if stride_x > 0 else 1
        ny = max(1, int(np.ceil((roi_h - cell_h) / stride_y)) + 1) if stride_y > 0 else 1
        for iy in range(ny):
            for ix in range(nx):
                x0 = rx0 + ix * stride_x
                y0 = ry0 + iy * stride_y
                x1 = min(rx1, x0 + cell_w)
                y1 = min(ry1, y0 + cell_h)
                x0 = max(float(rx0), x0)
                y0 = max(float(ry0), y0)
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                key = (int(x0), int(y0), int(x1), int(y1))
                if key in seen:
                    continue
                seen.add(key)
                boxes.append([float(x0), float(y0), float(x1), float(y1)])
    return boxes


def box_has_foreground(
    gray: np.ndarray,
    box: list[float],
    *,
    bg_mean: float = DEFAULT_BG_MEAN,
    min_fg_frac: float = DEFAULT_MIN_FG_FRAC,
) -> bool:
    """True if enough pixels inside *box* are brighter than background."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return False
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return False
    fg = crop > bg_mean
    return float(fg.mean()) >= min_fg_frac and float(crop.mean()) > bg_mean


def mask_mean_intensity(gray: np.ndarray, seg: np.ndarray) -> float:
    if not seg.any():
        return 0.0
    return float(gray[seg].mean())


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def bbox_xywh_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def filter_and_nms_masks(
    masks: list[dict],
    *,
    image_h: int,
    image_w: int,
    min_mask_area: int = 400,
    max_mask_frac: float = DEFAULT_MAX_MASK_FRAC,
    iou_thresh: float = DEFAULT_IOU_NMS_THRESH,
    max_marks: int = 24,
    gray: Optional[np.ndarray] = None,
    bg_mean: float = DEFAULT_BG_MEAN,
) -> list[dict]:
    """Drop tiny / near-full / near-black masks, IoU-NMS, keep up to *max_marks*."""
    img_area = max(1, image_h * image_w)
    max_area = int(max_mask_frac * img_area)

    candidates = []
    for m in masks:
        seg = m["segmentation"]
        area = int(m.get("area", int(seg.sum())))
        if area < min_mask_area or area > max_area:
            continue
        if gray is not None and mask_mean_intensity(gray, seg) <= bg_mean:
            continue
        candidates.append({**m, "area": area, "bbox": m.get("bbox") or bbox_xywh_from_mask(seg)})

    candidates.sort(key=lambda m: m["area"], reverse=True)
    kept: list[dict] = []
    for cand in candidates:
        if len(kept) >= max_marks:
            break
        if any(mask_iou(cand["segmentation"], k["segmentation"]) >= iou_thresh for k in kept):
            continue
        kept.append(cand)
    return kept


class MedSamBoxGridGenerator:
    """SAM-compatible ``.generate(image_rgb_np) -> list[mask_dict]`` using MedSAM."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MEDSAM_MODEL_ID,
        device: str = "cuda",
        points_per_side: int = 4,
        min_mask_region_area: int = 400,
        max_mask_frac: float = DEFAULT_MAX_MASK_FRAC,
        iou_nms_thresh: float = DEFAULT_IOU_NMS_THRESH,
        max_marks: int = 24,
        overlap: float = 0.2,
        bg_mean: float = DEFAULT_BG_MEAN,
        min_fg_frac: float = DEFAULT_MIN_FG_FRAC,
    ):
        import torch
        from transformers import SamModel, SamProcessor

        self.device = device
        self.points_per_side = points_per_side
        self.min_mask_region_area = min_mask_region_area
        self.max_mask_frac = max_mask_frac
        self.iou_nms_thresh = iou_nms_thresh
        self.max_marks = max_marks
        self.overlap = overlap
        self.bg_mean = bg_mean
        self.min_fg_frac = min_fg_frac
        self.model_id = model_id
        self._torch = torch

        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(device)
        self.model.eval()

    def generate(self, image_rgb: np.ndarray) -> list[dict]:
        """Segment *image_rgb* (H, W, 3 uint8) with a MedSAM box grid on the body ROI."""
        torch = self._torch
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB array, got shape {image_rgb.shape}")

        h, w = image_rgb.shape[:2]
        gray = to_gray_u8(image_rgb)
        roi = body_roi(gray, bg_mean=self.bg_mean)
        pil = Image.fromarray(image_rgb).convert("RGB")
        boxes = build_box_grid(
            h,
            w,
            points_per_side=self.points_per_side,
            overlap=self.overlap,
            roi=roi,
        )
        boxes = [
            b
            for b in boxes
            if box_has_foreground(
                gray, b, bg_mean=self.bg_mean, min_fg_frac=self.min_fg_frac
            )
        ]
        if not boxes:
            return []

        raw_masks: list[dict] = []
        chunk = 8
        for start in range(0, len(boxes), chunk):
            chunk_boxes = boxes[start : start + chunk]
            inputs = self.processor(
                pil,
                input_boxes=[chunk_boxes],
                return_tensors="pt",
            )
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            with torch.no_grad():
                outputs = self.model(**inputs, multimask_output=False)

            post = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
                binarize=True,
            )
            masks_t = post[0]
            for i in range(masks_t.shape[0]):
                seg = masks_t[i, 0].numpy().astype(bool)
                area = int(seg.sum())
                if area == 0:
                    continue
                raw_masks.append(
                    {
                        "segmentation": seg,
                        "area": area,
                        "bbox": bbox_xywh_from_mask(seg),
                    }
                )

        return filter_and_nms_masks(
            raw_masks,
            image_h=h,
            image_w=w,
            min_mask_area=self.min_mask_region_area,
            max_mask_frac=self.max_mask_frac,
            iou_thresh=self.iou_nms_thresh,
            max_marks=self.max_marks,
            gray=gray,
            bg_mean=self.bg_mean,
        )


def build_medsam_generator(
    *,
    model_id: str = DEFAULT_MEDSAM_MODEL_ID,
    device: str = "cuda",
    points_per_side: int = 4,
    min_mask_region_area: int = 400,
    max_marks: int = 24,
    bg_mean: float = DEFAULT_BG_MEAN,
    min_fg_frac: float = DEFAULT_MIN_FG_FRAC,
    **_: Any,
) -> MedSamBoxGridGenerator:
    return MedSamBoxGridGenerator(
        model_id=model_id,
        device=device,
        points_per_side=points_per_side,
        min_mask_region_area=min_mask_region_area,
        max_marks=max_marks,
        bg_mean=bg_mean,
        min_fg_frac=min_fg_frac,
    )
