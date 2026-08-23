"""Set-of-Mark (SoM) image annotation and resolution utilities.

Workflow:
1. Run scripts/precompute_som.py to annotate all images of a dataset once.
   This persists, per region: its bounding box and its pixel-level mask
   (as an individual PNG) so that a later step can render a *subset* of
   marks (e.g. only the regions relevant to a given question) without
   re-running SAM.
2. Pass --use_som to any training/inference script to swap original images
   for their precomputed SoM versions at runtime via SomResolver.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw


# ── Colour palette for region marks ─────────────────────────────────────────

_PALETTE = [
    (220, 50, 50),
    (50, 150, 220),
    (50, 200, 100),
    (240, 160, 30),
    (160, 80, 220),
    (220, 120, 180),
    (80, 220, 200),
    (200, 200, 50),
    (100, 100, 220),
    (220, 140, 60),
    (60, 180, 180),
    (180, 60, 120),
]


def color_for_label(label: str) -> tuple[int, int, int]:
    """Colour of a region mark, derived from its label rather than its position.

    Rendering only a subset of an image's regions must not recolour them: a
    reasoning that cites "region 3" has to point at the same coloured outline
    the annotator saw when all regions were drawn.
    """
    text = str(label)
    try:
        idx = int(text) - 1
    except ValueError:
        idx = sum(ord(c) for c in text)
    return _PALETTE[idx % len(_PALETTE)]


def _mask_border(seg: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Return border pixels of a boolean mask via erosion."""
    eroded = seg.copy()
    for _ in range(thickness):
        eroded[1:] &= eroded[:-1]
        eroded[:-1] &= eroded[1:]
        eroded[:, 1:] &= eroded[:, :-1]
        eroded[:, :-1] &= eroded[:, 1:]
    return seg & ~eroded


# ── Per-region mask persistence ──────────────────────────────────────────────

def save_region_mask(mask: np.ndarray, path: Path) -> None:
    """Persist a boolean (H, W) mask as a binary PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).convert("L").save(path, format="PNG")


def load_region_mask(path: Path) -> np.ndarray:
    """Load a boolean (H, W) mask previously saved with save_region_mask."""
    return np.array(Image.open(path).convert("L")) > 127


# ── Drawing ──────────────────────────────────────────────────────────────────

def _adaptive_circle_radius(seg: np.ndarray, override: Optional[int] = None) -> int:
    """Label-dot radius scaled to the segment so small masks stay readable."""
    if override is not None:
        return max(1, int(override))
    ys, xs = np.where(seg)
    if len(xs) == 0:
        return 6
    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)
    return int(np.clip(0.12 * min(w, h), 4, 9))


def draw_region_marks(
    image: Image.Image,
    regions: list[dict],
    *,
    border_thickness: int = 2,
    circle_radius: Optional[int] = None,
) -> Image.Image:
    """Draw numbered free-form mask borders for *regions* onto *image*.

    Each region dict must have "label" (str) and "mask" (np.ndarray bool, H==image.height).
    Color follows the label, so a region keeps its color whether the full set or
    only a filtered subset is drawn.

    *circle_radius* defaults to an adaptive size from each segment's bbox so a
    fixed 11px disc no longer swallows small structures. Pass an int to force
    a constant radius.
    """
    base = image.convert("RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for region in regions:
        label = region["label"]
        seg: np.ndarray = region["mask"]
        r, g, b = color_for_label(label)

        border = _mask_border(seg, thickness=border_thickness)
        mask_rgba = np.zeros((*seg.shape, 4), dtype=np.uint8)
        mask_rgba[border] = [r, g, b, 230]
        overlay = Image.alpha_composite(overlay, Image.fromarray(mask_rgba, mode="RGBA"))
        draw = ImageDraw.Draw(overlay)

        ys, xs = np.where(seg)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        radius = _adaptive_circle_radius(seg, circle_radius)

        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(r, g, b, 230),
        )
        # Keep digits roughly centred; smaller radius → slightly tighter text offset.
        tx = cx - max(2, radius // 2) * len(label)
        draw.text((tx, cy - max(4, radius // 2)), label, fill=(255, 255, 255, 255))

    return Image.alpha_composite(base, overlay).convert("RGB")


# ── Core annotation function ─────────────────────────────────────────────────

def build_som_generator(
    *,
    backend: str = "sam",
    sam_checkpoint: str = "checkpoints/sam_vit_b_01ec64.pth",
    sam_model_type: str = "vit_b",
    device: str = "cuda",
    points_per_side: int = 16,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    min_mask_region_area: int = 400,
    medsam_model_id: str = "wanglab/medsam-vit-base",
    max_marks: int = 24,
    bg_mean: float = 18.0,
    min_fg_frac: float = 0.08,
):
    """Load a mask generator for SoM annotation (SAM or MedSAM box-grid).

    Anyone annotating more than one image must build this once and reuse it:
    re-creating it per image re-reads the checkpoint from disk and re-uploads
    the weights to the GPU, which dominates the runtime.

    Both backends expose ``.generate(rgb_hwc_uint8) -> list[mask_dict]`` with
    keys ``segmentation``, ``area``, ``bbox``.
    """
    backend = (backend or "sam").lower().strip()
    if backend == "medsam":
        from utils.medsam_som import build_medsam_generator

        return build_medsam_generator(
            model_id=medsam_model_id,
            device=device,
            points_per_side=points_per_side,
            min_mask_region_area=min_mask_region_area,
            max_marks=max_marks,
            bg_mean=bg_mean,
            min_fg_frac=min_fg_frac,
        )

    if backend != "sam":
        raise ValueError(f"Unknown SoM backend {backend!r}; expected 'sam' or 'medsam'.")

    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError as e:
        raise ImportError(
            "segment-anything is required for SoM annotation. "
            "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from e

    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
        # One crop layer: run the point grid on image crops as well so smaller
        # interior structures (e.g. lung subregions) are not missed.
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )


def annotate_image_with_som(
    image: Image.Image,
    sam_checkpoint: str = "checkpoints/sam_vit_b_01ec64.pth",
    sam_model_type: str = "vit_b",
    device: str = "cuda",
    max_marks: int = 12,
    points_per_side: int = 16,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    min_mask_region_area: int = 400,
    generator=None,
    backend: str = "sam",
    medsam_model_id: str = "wanglab/medsam-vit-base",
    bg_mean: float = 18.0,
    max_mask_frac: float = 0.85,
    min_roi_frac: float = 0.5,
) -> tuple[Image.Image, list[dict]]:
    """Segment *image* and overlay numbered free-form mask borders.

    Pass *generator* (from build_som_generator) when annotating many images;
    the sam_*/medsam_* arguments are then already baked into it.

    Post-filters (all backends):
    - drop near-full-image masks (area ``> max_mask_frac``)
    - drop masks whose pixels mostly lie *outside* the body ROI (padding /
      table), using ``bg_mean`` only to *find* the ROI — not to reject dark
      lung air, which would otherwise be wiped on CT/X-ray

    Returns:
        annotated_image: PIL.Image with all regions marked (preview/fallback image).
        regions: list of dicts with keys "label" (str), "bbox" ([x,y,w,h]),
                 "mask" (np.ndarray bool, H x W).
    """
    if generator is None:
        generator = build_som_generator(
            backend=backend,
            sam_checkpoint=sam_checkpoint,
            sam_model_type=sam_model_type,
            device=device,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
            medsam_model_id=medsam_model_id,
            max_marks=max_marks,
            bg_mean=bg_mean,
        )

    img_rgb = image.convert("RGB")
    img_np = np.array(img_rgb)
    masks = generator.generate(img_np)

    from utils.medsam_som import body_roi, mask_inside_roi_frac, to_gray_u8

    gray = to_gray_u8(img_np)
    h, w = gray.shape[:2]
    roi = body_roi(gray, bg_mean=bg_mean)
    max_area = int(max_mask_frac * h * w)
    cleaned = []
    for m in masks:
        seg = m["segmentation"]
        area = int(m.get("area", int(seg.sum())))
        if area < min_mask_region_area or area > max_area:
            continue
        # Keep dark anatomy (lungs); only reject masks stuck in black padding.
        if mask_inside_roi_frac(seg, roi) < min_roi_frac:
            continue
        cleaned.append({**m, "area": area})

    # Prefer SAM's own confidence when present, else area — covers small but
    # confident interior regions better than "largest only".
    def _rank(m: dict) -> tuple:
        iou = float(m.get("predicted_iou", 0.0) or 0.0)
        stab = float(m.get("stability_score", 0.0) or 0.0)
        return (iou + stab, m["area"])

    masks = sorted(cleaned, key=_rank, reverse=True)[:max_marks]

    regions: list[dict] = []
    for idx, mask in enumerate(masks):
        label = str(idx + 1)
        bbox = mask.get("bbox")
        if bbox is None:
            from utils.medsam_som import bbox_xywh_from_mask

            bbox = bbox_xywh_from_mask(mask["segmentation"])
        x, y, w_box, h_box = (int(v) for v in bbox)
        regions.append(
            {"label": label, "bbox": [x, y, w_box, h_box], "mask": mask["segmentation"]}
        )

    annotated = draw_region_marks(img_rgb, regions)
    return annotated, regions


# ── SomResolver ──────────────────────────────────────────────────────────────

MANIFEST_FILENAME = "manifest.json"


class SomResolver:
    """Resolves original image paths to pre-annotated SoM image paths.

    The manifest (written by precompute_som.py) maps the original image
    filename stem+suffix → relative path of the SoM image inside som_dir,
    plus per-region bbox + mask_path metadata.

    Usage:
        resolver = SomResolver(som_dir="02_data/som")
        resolver.load_dataset("slake")
        som_path = resolver.resolve("/abs/path/to/slake_images/img001.jpg")
        # returns Path to SoM image, or None if not found
    """

    def __init__(self, som_dir: str | Path):
        self.som_dir = Path(som_dir)
        # dataset_slug -> {filename -> relative som path}
        self._manifests: dict[str, dict[str, str]] = {}

    def load_dataset(self, dataset: str) -> None:
        """Load the manifest for *dataset* (e.g. 'slake', 'pathvqa')."""
        slug = dataset.lower()
        manifest_path = self.som_dir / slug / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"SoM manifest not found: {manifest_path}\n"
                f"Run scripts/precompute_som.py --dataset {slug} first."
            )
        with open(manifest_path, "r", encoding="utf-8") as f:
            self._manifests[slug] = json.load(f)

    @staticmethod
    def _manifest_key(image_path: str) -> str:
        """Derive the manifest key from an image path (last 2 path components)."""
        p = Path(image_path)
        return "/".join(p.parts[-2:])

    def _entry(self, image_path: str) -> Optional[dict]:
        key = self._manifest_key(image_path)
        for manifest in self._manifests.values():
            if key in manifest:
                return manifest[key]
        return None

    def resolve(self, image_path: str) -> Optional[Path]:
        """Return the SoM preview image Path for *image_path*, or None if unavailable."""
        entry = self._entry(image_path)
        return self.som_dir / entry["som_path"] if entry else None

    def resolve_or_original(self, image_path: str) -> str:
        """Return SoM path as string if found, otherwise return *image_path* unchanged."""
        resolved = self.resolve(image_path)
        return str(resolved) if resolved is not None else image_path

    def get_regions(self, image_path: str) -> list[dict]:
        """Return region metadata (label, bbox, mask_path) for *image_path*, or [] if not found."""
        entry = self._entry(image_path)
        return entry.get("regions", []) if entry else []

    def load_region_masks(
        self, image_path: str, labels: Optional[set[str]] = None
    ) -> list[dict]:
        """Return regions with their boolean masks loaded into memory (key "mask").

        If *labels* is given, only those regions are loaded from disk (avoids
        reading masks that will be discarded, e.g. after question filtering).
        """
        loaded = []
        for region in self.get_regions(image_path):
            if labels is not None and region["label"] not in labels:
                continue
            mask = load_region_mask(self.som_dir / region["mask_path"])
            loaded.append({**region, "mask": mask})
        return loaded
