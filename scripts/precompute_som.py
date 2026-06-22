"""Pre-compute Set-of-Mark (SoM) annotations for all images of a dataset.

Annotated images and a manifest are saved to <som_dir>/<dataset_slug>/.
Running the script again on an already-processed dataset is safe: existing
images are skipped automatically.

Example usage:
    python scripts/precompute_som.py --dataset slake
    python scripts/precompute_som.py --dataset all --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth

Output layout:
    <som_dir>/
        <dataset>/
            images/
                <original_stem>_som.png
            manifest.json   # {filename -> {som_path, regions}}

Override som_dir via MEDVQA_SOM_DIR environment variable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# Ensure repo root is on sys.path when called as scripts/precompute_som.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.load_data import MedicalVQALoader
from utils.paths import get_data_root, get_mc_data_root, get_som_dir
from utils.som import MANIFEST_FILENAME, annotate_image_with_som, save_region_mask

DATASET_CHOICES = ["pathvqa", "slake", "vqarad", "medvqa2019", "all"]
ALL_DATASETS = ["pathvqa", "slake", "vqarad", "medvqa2019"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compute SoM annotations for all images in a dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DATASET_CHOICES,
        help="Dataset to annotate, or 'all' for every dataset.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Splits to collect images from (default: all available splits).",
    )
    parser.add_argument(
        "--som_dir",
        type=str,
        default=get_som_dir(),
        help="Root directory for SoM outputs (default: MEDVQA_SOM_DIR or 02_data/som).",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="01_external/checkpoints/sam_vit_b_01ec64.pth",
        help="Path to SAM checkpoint (.pth file).",
    )
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_b",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model variant (default: vit_b, fastest).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for SAM inference (cuda or cpu).",
    )
    parser.add_argument(
        "--max_marks",
        type=int,
        default=12,
        help="Maximum number of region marks per image.",
    )
    parser.add_argument(
        "--points_per_side",
        type=int,
        default=16,
        help="SAM grid density. Lower = faster but fewer regions.",
    )
    parser.add_argument(
        "--min_mask_area",
        type=int,
        default=400,
        help="Minimum pixel area to keep a region (filters tiny artefacts).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=get_data_root(),
    )
    parser.add_argument(
        "--mc_data_root",
        type=str,
        default=get_mc_data_root(),
    )
    return parser.parse_args()


def _collect_unique_image_paths(
    loader: MedicalVQALoader,
    dataset_slug: str,
    splits: list[str] | None,
) -> list[str]:
    """Return deduplicated list of image paths for the given dataset + splits."""
    canonical = loader.dataset_name_mapping.get(dataset_slug.lower(), dataset_slug)
    available_splits = loader.get_available_splits(canonical)
    use_splits = splits if splits else available_splits
    unknown = set(use_splits) - set(available_splits)
    if unknown:
        print(f"  Warning: splits {unknown} not available for {dataset_slug}, skipping.")
        use_splits = [s for s in use_splits if s in available_splits]

    seen: set[str] = set()
    paths: list[str] = []
    for split in use_splits:
        samples = loader.load_dataset(dataset_slug, split=split)
        for sample in samples:
            p = str(sample.image_path)
            if p not in seen:
                seen.add(p)
                paths.append(p)
    return paths


def _process_dataset(
    dataset_slug: str,
    loader: MedicalVQALoader,
    splits: list[str] | None,
    som_dir: Path,
    args: argparse.Namespace,
) -> None:
    print(f"\n=== {dataset_slug} ===")

    image_paths = _collect_unique_image_paths(loader, dataset_slug, splits)
    print(f"  Unique images found: {len(image_paths)}")

    dataset_out_dir = som_dir / dataset_slug
    images_out_dir = dataset_out_dir / "images"
    masks_out_dir = dataset_out_dir / "masks"
    images_out_dir.mkdir(parents=True, exist_ok=True)
    masks_out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_out_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest: dict = json.load(f)
    else:
        manifest = {}

    skipped = 0
    processed = 0
    failed = 0

    for img_path_str in tqdm(image_paths, desc=f"  Annotating {dataset_slug}"):
        img_path = Path(img_path_str)
        # Use last 2 path components as key to handle datasets where images share
        # the same filename across subdirectories (e.g. SLAKE: xmlab0/source.jpg)
        filename = "/".join(img_path.parts[-2:])

        if filename in manifest:
            skipped += 1
            continue

        if not img_path.exists():
            tqdm.write(f"  [WARN] Image not found, skipping: {img_path}")
            failed += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            annotated, regions = annotate_image_with_som(
                image,
                sam_checkpoint=args.sam_checkpoint,
                sam_model_type=args.sam_model_type,
                device=args.device,
                max_marks=args.max_marks,
                points_per_side=args.points_per_side,
                min_mask_region_area=args.min_mask_area,
            )

            # Preserve subdirectory structure to avoid name collisions
            subdir = img_path.parent.name
            base_name = f"{subdir}_{img_path.stem}"
            out_name = f"{base_name}_som.png"
            out_path = images_out_dir / out_name
            annotated.save(out_path, format="PNG")

            # Persist each region's mask individually so a later step can
            # render an arbitrary subset of marks without re-running SAM.
            region_entries = []
            for region in regions:
                mask_name = f"{base_name}_region{region['label']}.png"
                mask_out_path = masks_out_dir / mask_name
                save_region_mask(region["mask"], mask_out_path)
                region_entries.append(
                    {
                        "label": region["label"],
                        "bbox": region["bbox"],
                        "mask_path": str(dataset_slug / Path("masks") / mask_name),
                    }
                )

            # Store path relative to som_dir so it's portable
            rel_som_path = str((dataset_slug / Path("images") / out_name))
            manifest[filename] = {
                "som_path": rel_som_path,
                "original_path": img_path_str,
                "regions": region_entries,
            }
            processed += 1

        except Exception as exc:
            tqdm.write(f"  [ERROR] {img_path.name}: {exc}")
            failed += 1

    # Write manifest after each dataset (safe even if interrupted mid-run)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(
        f"  Done — processed: {processed}, skipped (already done): {skipped}, failed: {failed}"
    )
    print(f"  Manifest: {manifest_path}")


def main() -> None:
    args = parse_args()

    loader = MedicalVQALoader(
        data_root=args.data_root,
        mc_data_root=args.mc_data_root,
    )
    som_dir = Path(args.som_dir)
    som_dir.mkdir(parents=True, exist_ok=True)

    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    for dataset_slug in datasets:
        _process_dataset(dataset_slug, loader, args.splits, som_dir, args)

    print("\nAll done.")


if __name__ == "__main__":
    main()
