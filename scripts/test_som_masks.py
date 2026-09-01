"""Export SoM visualization assets for one image.

Picks the first selection-manifest question for that image (deterministic:
sorted by selection key). Writes into --out_dir:

  - original.png
  - all_som.png
  - selected_som.png
  - meta.json   (includes the chosen question + labels)

Usage:
    python scripts/test_som_masks.py \\
      --dataset slake \\
      --image_path PATH/TO/image.jpg \\
      --out_dir som_viz_out/example1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from utils.paths import get_som_dir, get_som_selection_dir
from utils.som import SomResolver, draw_region_marks
from utils.som_selection import SELECTION_MANIFEST_FILENAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export original + all-SoM + selected-SoM + meta.json for one image."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the original image (as in the dataset / results JSON).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output folder (created if missing). Files are written directly here.",
    )
    parser.add_argument(
        "--som_dir",
        type=str,
        default=get_som_dir(),
        help=f"SoM root (default: {get_som_dir()}).",
    )
    parser.add_argument(
        "--selection_dir",
        type=str,
        default=get_som_selection_dir(),
        help=f"Selection root (default: {get_som_selection_dir()}).",
    )
    return parser.parse_args()


def _image_key(image_path: str) -> str:
    return "/".join(Path(image_path).parts[-2:])


def _first_selection_for_image(
    selection_manifest: dict, image_key: str
) -> tuple[str, str, dict]:
    """Return (selection_key, question, entry) for the first question of *image_key*.

    Order is sorted by full selection key so re-runs are stable.
    """
    prefix = f"{image_key}||"
    matches = sorted(k for k in selection_manifest if k.startswith(prefix))
    if not matches:
        raise SystemExit(
            f"No selection-manifest entries for image key {image_key!r}. "
            f"Run precompute_som_selection for this dataset first."
        )
    sel_key = matches[0]
    question = sel_key[len(prefix) :]
    return sel_key, question, selection_manifest[sel_key]


def main() -> None:
    args = parse_args()
    slug = args.dataset.lower()
    image_path = Path(args.image_path)
    key = _image_key(str(image_path))

    som = SomResolver(args.som_dir)
    som.load_dataset(slug)
    if not som.get_regions(str(image_path)):
        raise SystemExit(
            f"Image key {key!r} not found in SoM manifest under {args.som_dir}/{slug}/. "
            f"Check --dataset / --image_path."
        )

    if image_path.exists():
        original_path = image_path
    else:
        manifest_path = Path(args.som_dir) / slug / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_entry = json.load(f)[key]
        original_path = Path(manifest_entry["original_path"])
        if not original_path.exists():
            raise SystemExit(
                f"Neither --image_path nor manifest original_path exists:\n"
                f"  {image_path}\n  {original_path}"
            )

    selection_manifest_path = Path(args.selection_dir) / slug / SELECTION_MANIFEST_FILENAME
    if not selection_manifest_path.exists():
        raise SystemExit(f"Selection manifest not found: {selection_manifest_path}")

    with open(selection_manifest_path, "r", encoding="utf-8") as f:
        selection_manifest = json.load(f)

    sel_key, question, sel_entry = _first_selection_for_image(selection_manifest, key)
    selected_labels = [str(x) for x in sel_entry.get("selected_labels", [])]

    original = Image.open(original_path).convert("RGB")
    all_regions = som.load_region_masks(str(image_path))
    subset = [r for r in all_regions if r["label"] in set(selected_labels)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_out = out_dir / "original.png"
    all_out = out_dir / "all_som.png"
    selected_out = out_dir / "selected_som.png"
    meta_out = out_dir / "meta.json"

    original.save(original_out)
    draw_region_marks(original, all_regions).save(all_out)
    if selected_labels and subset:
        draw_region_marks(original, subset).save(selected_out)
    else:
        original.save(selected_out)

    # Other questions for this image (for reference in meta only).
    prefix = f"{key}||"
    other_questions = [
        k[len(prefix) :] for k in sorted(selection_manifest) if k.startswith(prefix)
    ]

    meta = {
        "dataset": slug,
        "image_key": key,
        "selection_key": sel_key,
        "image_path": str(image_path),
        "resolved_original_path": str(original_path),
        "question": question,
        "question_choice": "first_sorted_selection_key",
        "n_questions_for_image": len(other_questions),
        "all_questions_for_image": other_questions,
        "som_dir": str(args.som_dir),
        "selection_dir": str(args.selection_dir),
        "n_regions_all": len(all_regions),
        "all_region_labels": [r["label"] for r in all_regions],
        "regions": [
            {"label": r["label"], "bbox": r.get("bbox"), "mask_path": r.get("mask_path")}
            for r in all_regions
        ],
        "selected_labels": selected_labels,
        "selection_response": sel_entry.get("selection_response", ""),
        "n_regions_total": sel_entry.get("n_regions_total"),
        "split": sel_entry.get("split"),
        "selector_model": sel_entry.get("selector_model"),
        "outputs": {
            "original": original_out.name,
            "all_som": all_out.name,
            "selected_som": selected_out.name,
            "meta": meta_out.name,
        },
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out_dir}/")
    print(f"  {original_out.name}")
    print(f"  {all_out.name}  ({len(all_regions)} regions)")
    print(f"  {selected_out.name}  (labels={selected_labels or '[]'})")
    print(f"  {meta_out.name}")
    print(f"question (first): {question}")
    print(f"selection_key: {sel_key}")


if __name__ == "__main__":
    main()
