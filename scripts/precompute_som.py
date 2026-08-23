"""Pre-compute Set-of-Mark (SoM) annotations for all images of a dataset.

Annotated images and a manifest are saved to <som_dir>/<dataset_slug>/.
Running the script again on an already-processed dataset is safe: existing
images are skipped automatically.

Example usage:
    python scripts/precompute_som.py --dataset slake
    python scripts/precompute_som.py --dataset slake --backend medsam --max_marks 24
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
from utils.medsam_som import DEFAULT_MEDSAM_MODEL_ID
from utils.paths import get_data_root, get_mc_data_root, get_som_dir
from utils.som import (
    MANIFEST_FILENAME,
    annotate_image_with_som,
    build_som_generator,
    save_region_mask,
)

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
        "--backend",
        type=str,
        default="sam",
        choices=["sam", "medsam"],
        help="Segmentation backend: vanilla SAM Automatic, or MedSAM with a box grid.",
    )
    parser.add_argument(
        "--medsam_model_id",
        type=str,
        default=DEFAULT_MEDSAM_MODEL_ID,
        help=f"HuggingFace MedSAM id (default: {DEFAULT_MEDSAM_MODEL_ID}).",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="01_external/checkpoints/sam_vit_b_01ec64.pth",
        help="Path to SAM checkpoint (.pth file); only used with --backend sam.",
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
        help="Device for segmentation (cuda or cpu).",
    )
    parser.add_argument(
        "--max_marks",
        type=int,
        default=None,
        help="Maximum number of region marks per image "
        "(default: 12 for sam, 24 for medsam).",
    )
    parser.add_argument(
        "--points_per_side",
        type=int,
        default=None,
        help="SAM: grid density (default 16). MedSAM: coarse box-grid size "
        "(default 4; a denser grid of size+2 is always added).",
    )
    parser.add_argument(
        "--min_mask_area",
        type=int,
        default=400,
        help="Minimum pixel area to keep a region (filters tiny artefacts).",
    )
    parser.add_argument(
        "--bg_mean",
        type=float,
        default=18.0,
        help="MedSAM: gray intensity (0–255) treated as background. Boxes/masks "
        "at or below this mean are skipped (default: 18).",
    )
    parser.add_argument(
        "--min_fg_frac",
        type=float,
        default=0.08,
        help="MedSAM: minimum fraction of box pixels above --bg_mean to keep "
        "the prompt (default: 0.08).",
    )
    parser.add_argument(
        "--pred_iou_thresh",
        type=float,
        default=0.88,
        help="SAM predicted-IoU confidence threshold. Lower = more (less certain) regions kept.",
    )
    parser.add_argument(
        "--stability_score_thresh",
        type=float,
        default=0.95,
        help="SAM stability-score threshold. Lower = more (less stable) regions kept.",
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
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Limit to the first N unique images per dataset (for quick testing).",
    )
    args = parser.parse_args()
    if args.max_marks is None:
        args.max_marks = 24 if args.backend == "medsam" else 12
    if args.points_per_side is None:
        args.points_per_side = 4 if args.backend == "medsam" else 16
    return args


class _LazySomGenerator:
    """Builds the SoM mask generator on first use, then hands out the same one.

    Built lazily so a fully resumed run, where every image is already annotated,
    never pays for loading the checkpoint at all.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._generator = None

    def get(self):
        if self._generator is None:
            if self.args.backend == "medsam":
                print(
                    f"  Loading MedSAM from {self.args.medsam_model_id} "
                    f"(box grid points_per_side={self.args.points_per_side}) ..."
                )
            else:
                print(
                    f"  Loading SAM {self.args.sam_model_type} from {self.args.sam_checkpoint} ..."
                )
            self._generator = build_som_generator(
                backend=self.args.backend,
                sam_checkpoint=self.args.sam_checkpoint,
                sam_model_type=self.args.sam_model_type,
                device=self.args.device,
                points_per_side=self.args.points_per_side,
                pred_iou_thresh=self.args.pred_iou_thresh,
                stability_score_thresh=self.args.stability_score_thresh,
                min_mask_region_area=self.args.min_mask_area,
                medsam_model_id=self.args.medsam_model_id,
                max_marks=self.args.max_marks,
                bg_mean=self.args.bg_mean,
                min_fg_frac=self.args.min_fg_frac,
            )
        return self._generator


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
    generator_holder: _LazySomGenerator,
) -> None:
    print(f"\n=== {dataset_slug} ===")

    image_paths = _collect_unique_image_paths(loader, dataset_slug, splits)
    print(f"  Unique images found: {len(image_paths)}")
    if args.n_samples is not None:
        image_paths = image_paths[: args.n_samples]
        print(f"  Limited to n_samples={args.n_samples}")

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
                max_marks=args.max_marks,
                min_mask_region_area=args.min_mask_area,
                bg_mean=args.bg_mean,
                generator=generator_holder.get(),
            )

            # Preserve subdirectory structure to avoid name collisions
            subdir = img_path.parent.name
            base_name = f"{subdir}_{img_path.stem}"
            out_name = f"{base_name}_som.png"
            out_path = images_out_dir / out_name
            annotated.save(out_path, format="PNG")

            # Persist each region's mask individually so a later step can
            # render an arbitrary subset of marks without re-running segmentation.
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
            entry = {
                "som_path": rel_som_path,
                "original_path": img_path_str,
                "backend": args.backend,
                "regions": region_entries,
            }
            if args.backend == "medsam":
                entry["medsam_model_id"] = args.medsam_model_id
            else:
                entry["sam_model_type"] = args.sam_model_type
            manifest[filename] = entry
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

    print(
        f"SoM backend={args.backend}  max_marks={args.max_marks}  "
        f"points_per_side={args.points_per_side}"
    )

    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]
    generator_holder = _LazySomGenerator(args)

    for dataset_slug in datasets:
        _process_dataset(dataset_slug, loader, args.splits, som_dir, args, generator_holder)

    print("\nAll done.")


if __name__ == "__main__":
    main()
