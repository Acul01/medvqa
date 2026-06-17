"""Set-of-Mark (SoM) image annotation and resolution utilities.

Workflow:
1. Run scripts/precompute_som.py to annotate all images of a dataset once.
2. Pass --use_som to any training/inference script to swap original images
   for their precomputed SoM versions at runtime via SomResolver.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PIL import Image


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
    """Overlay numbered bounding-box marks on *image* using SAM auto-segmentation.

    Returns:
        annotated_image: PIL.Image with coloured bbox + number overlays.
        regions: list of dicts with keys "label" (str), "bbox" ([x,y,w,h]).
    """
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError as e:
        raise ImportError(
            "segment-anything is required for SoM annotation. "
            "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from e

    import numpy as np
    from PIL import ImageDraw

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

    annotated = img_rgb.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    regions: list[dict] = []
    for idx, mask in enumerate(masks):
        label = str(idx + 1)
        x, y, w, h = (int(v) for v in mask["bbox"])
        cx, cy = x + w // 2, y + h // 2
        r, g, b = _label_color(idx)

        # Bounding box outline
        draw.rectangle([x, y, x + w, y + h], outline=(r, g, b, 210), width=2)

        # Numbered circle at region centre
        r_circle = 11
        draw.ellipse(
            [cx - r_circle, cy - r_circle, cx + r_circle, cy + r_circle],
            fill=(r, g, b, 200),
        )
        # Text: shift left ~3px per character so it stays centred
        tx = cx - 3 * len(label)
        draw.text((tx, cy - 7), label, fill=(255, 255, 255, 255))

        regions.append({"label": label, "bbox": [x, y, w, h]})

    annotated = Image.alpha_composite(annotated, overlay).convert("RGB")
    return annotated, regions


# ── SomResolver ──────────────────────────────────────────────────────────────

MANIFEST_FILENAME = "manifest.json"


class SomResolver:
    """Resolves original image paths to pre-annotated SoM image paths.

    The manifest (written by precompute_som.py) maps the original image
    filename stem+suffix → relative path of the SoM image inside som_dir.

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

    def resolve(self, image_path: str) -> Optional[Path]:
        """Return the SoM image Path for *image_path*, or None if unavailable."""
        key = self._manifest_key(image_path)
        for manifest in self._manifests.values():
            if key in manifest:
                return self.som_dir / manifest[key]["som_path"]
        return None

    def resolve_or_original(self, image_path: str) -> str:
        """Return SoM path as string if found, otherwise return *image_path* unchanged."""
        resolved = self.resolve(image_path)
        return str(resolved) if resolved is not None else image_path

    def get_regions(self, image_path: str) -> list[dict]:
        """Return region metadata list for *image_path*, or [] if not found."""
        key = self._manifest_key(image_path)
        for manifest in self._manifests.values():
            if key in manifest:
                return manifest[key].get("regions", [])
        return []
