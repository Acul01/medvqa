"""Sanity check: verify individual SoM region masks can be loaded and re-rendered.

Picks one image from a dataset's manifest, loads each region's mask
individually from disk, then renders (a) all masks together — should match
the precomputed preview image — and (b) an optional subset, to prove masks
can be selectively toggled on/off without re-running SAM.

Usage:
    python scripts/test_som_masks.py --dataset slake --som_dir 02_data/SOM_data/slake_som
    python scripts/test_som_masks.py --dataset slake --som_dir 02_data/SOM_data/slake_som \
        --image_index 3 --labels 1 3 5

    # Or look up a specific image directly (e.g. from a results JSON row):
    python scripts/test_som_masks.py --dataset vqarad --som_dir 02_data/SOM_data/ \
        --image_path 02_data/VQA-Rad/vqarad_images/train_0.jpg --labels 1 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from utils.som import MANIFEST_FILENAME, draw_region_marks, load_region_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test loading + rendering individual SoM region masks")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--som_dir", type=str, required=True)
    parser.add_argument("--image_index", type=int, default=0, help="Which manifest entry to test (0-based)")
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Look up a specific image by path instead of --image_index (e.g. image_path from a results JSON row).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Subset of region labels to render together (default: render all)",
    )
    parser.add_argument("--out_dir", type=str, default="som_mask_test_out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slug = args.dataset.lower()
    som_dir = Path(args.som_dir)
    manifest_path = som_dir / slug / MANIFEST_FILENAME

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    keys = list(manifest.keys())
    if not keys:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    if args.image_path:
        key = "/".join(Path(args.image_path).parts[-2:])
        if key not in manifest:
            raise ValueError(
                f"--image_path {args.image_path!r} (key {key!r}) not found in manifest. "
                f"Available keys e.g.: {keys[:5]}"
            )
    else:
        if args.image_index >= len(keys):
            raise ValueError(f"--image_index {args.image_index} out of range (0-{len(keys) - 1})")
        key = keys[args.image_index]

    entry = manifest[key]
    print(f"Testing image: {key}")
    print(f"  original_path: {entry['original_path']}")
    print(f"  regions in manifest: {len(entry['regions'])}")

    original_image = Image.open(entry["original_path"]).convert("RGB")

    # Load each region's mask individually from disk (the actual thing we're testing)
    regions = []
    for region in entry["regions"]:
        mask = load_region_mask(som_dir / region["mask_path"])
        regions.append({**region, "mask": mask})
        print(f"  loaded region {region['label']}: shape={mask.shape}, pixels_on={int(mask.sum())}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(key).stem

    # Render all loaded masks together — should visually match the precomputed preview image
    all_render = draw_region_marks(original_image, regions)
    all_path = out_dir / f"{stem}_all_reconstructed.png"
    all_render.save(all_path)
    print(f"\nSaved full reconstruction ({len(regions)} marks): {all_path}")
    print(f"Compare against precomputed preview:           {som_dir / entry['som_path']}")

    # Render only a subset, to prove individual masks can be toggled on/off
    if args.labels:
        subset = [r for r in regions if r["label"] in args.labels]
        if not subset:
            print(f"WARNING: none of labels {args.labels} found among {[r['label'] for r in regions]}")
        else:
            subset_render = draw_region_marks(original_image, subset)
            subset_path = out_dir / f"{stem}_subset_{'-'.join(args.labels)}.png"
            subset_render.save(subset_path)
            print(f"Saved subset render ({len(subset)} marks): {subset_path}")


if __name__ == "__main__":
    main()
