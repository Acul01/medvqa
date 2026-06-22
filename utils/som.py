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


def _label_color(idx: int) -> tuple[int, int, int]:
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

def draw_region_marks(
    image: Image.Image,
    regions: list[dict],
    *,
    border_thickness: int = 2,
    circle_radius: int = 11,
) -> Image.Image:
    """Draw numbered free-form mask borders for *regions* onto *image*.

    Each region dict must have "label" (str) and "mask" (np.ndarray bool, H==image.height).
    Color is assigned by position in the list, so callers should pass regions
    in their original (precompute-time) order to keep colors stable across
    full vs. filtered renders.
    """
    base = image.convert("RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for idx, region in enumerate(regions):
        label = region["label"]
        seg: np.ndarray = region["mask"]
        r, g, b = _label_color(idx)

        border = _mask_border(seg, thickness=border_thickness)
        mask_rgba = np.zeros((*seg.shape, 4), dtype=np.uint8)
        mask_rgba[border] = [r, g, b, 230]
        overlay = Image.alpha_composite(overlay, Image.fromarray(mask_rgba, mode="RGBA"))
        draw = ImageDraw.Draw(overlay)

        ys, xs = np.where(seg)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())

        draw.ellipse(
            [cx - circle_radius, cy - circle_radius, cx + circle_radius, cy + circle_radius],
            fill=(r, g, b, 230),
        )
        tx = cx - 3 * len(label)
        draw.text((tx, cy - 7), label, fill=(255, 255, 255, 255))

    return Image.alpha_composite(base, overlay).convert("RGB")


# ── Core annotation function ─────────────────────────────────────────────────

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
) -> tuple[Image.Image, list[dict]]:
    """Segment *image* with SAM and overlay numbered free-form mask borders.

    Returns:
        annotated_image: PIL.Image with all regions marked (preview/fallback image).
        regions: list of dicts with keys "label" (str), "bbox" ([x,y,w,h]),
                 "mask" (np.ndarray bool, H x W) — the raw per-region mask,
                 for the caller to persist individually if desired.
    """
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError as e:
        raise ImportError(
            "segment-anything is required for SoM annotation. "
            "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from e

    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )

    img_rgb = image.convert("RGB")
    img_np = np.array(img_rgb)
    masks = generator.generate(img_np)

    # Keep largest regions, up to max_marks
    masks = sorted(masks, key=lambda m: m["area"], reverse=True)[:max_marks]

    regions: list[dict] = []
    for idx, mask in enumerate(masks):
        label = str(idx + 1)
        x, y, w, h = (int(v) for v in mask["bbox"])
        regions.append({"label": label, "bbox": [x, y, w, h], "mask": mask["segmentation"]})

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

    def load_region_masks(self, image_path: str) -> list[dict]:
        """Return regions with their boolean masks loaded into memory (key "mask")."""
        loaded = []
        for region in self.get_regions(image_path):
            mask = load_region_mask(self.som_dir / region["mask_path"])
            loaded.append({**region, "mask": mask})
        return loaded
