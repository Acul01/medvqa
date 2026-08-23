"""Pre-compute which SoM segments are relevant for each (image, question) pair.

Second stage of the SoM pipeline: scripts/precompute_som.py segments every image
once, this script asks an LVLM per question which of those segments matter and
records the answer. Running it offline (rather than during inference) is what
lets train, validation and test splits share one identical selector, so a model
trained on filtered images is evaluated on images produced the same way.

Selection is greedy so a re-run reproduces the manifest.

Example usage:
    python scripts/precompute_som_selection.py --dataset slake \
        --splits train validation test --resume

Output layout:
    <selection_dir>/
        <dataset>/
            selection_manifest.json   # {"<image_key>||<question>" -> entry}
            images/                   # only with --materialize, for eyeballing

Override selection_dir via MEDVQA_SOM_SELECTION_DIR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Ensure repo root is on sys.path when called as scripts/precompute_som_selection.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.build_pseudo_reasonings_openai import load_qwen_bundle
from utils.load_data import MedicalVQALoader, VQASample
from utils.paths import get_data_root, get_mc_data_root, get_som_dir, get_som_selection_dir
from utils.qwen_vl import build_segment_selection_conversation, prepare_qwen_inputs
from utils.som import SomResolver, draw_region_marks
from utils.som_select import DEFAULT_SELECTION_MAX_NEW_TOKENS, parse_selected_labels
from utils.som_selection import SELECTION_MANIFEST_FILENAME, selection_key

DATASET_CHOICES = ["pathvqa", "slake", "vqarad", "medvqa2019", "all"]
ALL_DATASETS = ["pathvqa", "slake", "vqarad", "medvqa2019"]
DEFAULT_SELECTOR_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask an LVLM which pre-computed SoM segments are relevant per question."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DATASET_CHOICES,
        help="Dataset to process, or 'all' for every dataset.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Splits to process (default: all available). Use 'train validation test'.",
    )
    parser.add_argument(
        "--som_dir",
        type=str,
        default=get_som_dir(),
        help="Root of precompute_som.py outputs (default: MEDVQA_SOM_DIR or 02_data/som).",
    )
    parser.add_argument(
        "--selection_dir",
        type=str,
        default=get_som_selection_dir(),
        help="Where to write selection manifests (default: MEDVQA_SOM_SELECTION_DIR or 02_data/som_selection).",
    )
    parser.add_argument(
        "--selector_model",
        type=str,
        default=DEFAULT_SELECTOR_MODEL,
        help=f"LVLM used to pick relevant segments (default: {DEFAULT_SELECTOR_MODEL}).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_SELECTION_MAX_NEW_TOKENS,
        help="Token budget for the selection answer (a short list of numbers).",
    )
    parser.add_argument("--data_root", type=str, default=get_data_root())
    parser.add_argument("--mc_data_root", type=str, default=get_mc_data_root())
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Limit to the first N QA pairs per split (for quick testing).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (image, question) pairs already present in the manifest.",
    )
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Also write the filtered images to disk. Debug aid only; the "
        "training and eval paths always re-render from the cached masks.",
    )
    return parser.parse_args()


class _MarkedImageCache:
    """Single-slot cache for the fully marked image of one source image.

    Rows are processed image-major, and several questions usually share an
    image, so re-loading the masks and re-drawing the overlay for every question
    would be pure waste.
    """

    def __init__(self, som: SomResolver):
        self.som = som
        self._path: str | None = None
        self._payload: tuple[Image.Image, Image.Image, list[dict]] | None = None

    def get(self, image_path: str) -> tuple[Image.Image, Image.Image, list[dict]]:
        """Return (original, fully marked, regions); regions is [] when unannotated."""
        if self._path == image_path and self._payload is not None:
            return self._payload

        original = Image.open(image_path).convert("RGB")
        regions = self.som.load_region_masks(image_path)
        marked = draw_region_marks(original, regions) if regions else original
        self._path = image_path
        self._payload = (original, marked, regions)
        return self._payload


def _select_segments(bundle, image: Image.Image, question: str, max_new_tokens: int) -> str:
    """Return the selector's raw answer for *question* on the fully marked *image*."""
    model, processor, device = bundle
    conversation = build_segment_selection_conversation(question, image)
    inputs = prepare_qwen_inputs(processor, conversation, add_generation_prompt=True)
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }
    prompt_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()


def _collect_samples(
    loader: MedicalVQALoader,
    dataset_slug: str,
    splits: list[str] | None,
    n_samples: int | None,
) -> list[tuple[str, VQASample]]:
    """Return (split, sample) pairs, ordered image-major so the cache hits."""
    canonical = loader.dataset_name_mapping.get(dataset_slug.lower(), dataset_slug)
    available_splits = loader.get_available_splits(canonical)
    use_splits = splits if splits else available_splits
    unknown = set(use_splits) - set(available_splits)
    if unknown:
        print(f"  Warning: splits {sorted(unknown)} not available for {dataset_slug}, skipping.")
        use_splits = [s for s in use_splits if s in available_splits]

    collected: list[tuple[str, VQASample]] = []
    for split in use_splits:
        samples = loader.load_dataset(dataset_slug, split=split)
        if n_samples is not None:
            samples = samples[:n_samples]
        print(f"  {split}: {len(samples)} QA pairs")
        collected.extend((split, sample) for sample in samples)

    collected.sort(key=lambda pair: pair[1].image_path)
    return collected


def _materialized_name(image_path: str, question: str) -> str:
    path = Path(image_path)
    digest = hashlib.sha1(question.strip().encode("utf-8")).hexdigest()[:8]
    return f"{path.parent.name}_{path.stem}_{digest}_som.png"


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def _process_dataset(
    dataset_slug: str,
    loader: MedicalVQALoader,
    som: SomResolver,
    bundle,
    args: argparse.Namespace,
) -> None:
    print(f"\n=== {dataset_slug} ===")

    dataset_out_dir = Path(args.selection_dir) / dataset_slug
    manifest_path = dataset_out_dir / SELECTION_MANIFEST_FILENAME
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest: dict = json.load(f)
    else:
        manifest = {}

    images_out_dir = dataset_out_dir / "images"
    if args.materialize:
        images_out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _collect_samples(loader, dataset_slug, args.splits, args.n_samples)
    cache = _MarkedImageCache(som)
    done_keys: set[str] = set()

    n_selected = n_none = n_no_regions = n_skipped = n_failed = 0
    selected_counts: list[int] = []

    for processed, (split, sample) in enumerate(
        tqdm(pairs, desc=f"  Selecting {dataset_slug}", unit="pair"), start=1
    ):
        key = selection_key(sample.image_path, sample.question)
        # The same QA pair can appear in several splits; select for it once.
        if key in done_keys or (args.resume and key in manifest):
            n_skipped += 1
            continue
        done_keys.add(key)

        try:
            original, marked, regions = cache.get(sample.image_path)
        except (OSError, ValueError) as exc:
            tqdm.write(f"  [WARN] {sample.image_path}: {exc}")
            n_failed += 1
            continue

        entry = {
            "selector_model": args.selector_model,
            "split": split,
            "n_regions_total": len(regions),
        }

        if not regions:
            entry.update({"selected_labels": [], "selection_response": ""})
            manifest[key] = entry
            n_no_regions += 1
            continue

        valid_labels = {region["label"] for region in regions}
        response = _select_segments(bundle, marked, sample.question, args.max_new_tokens)
        labels = parse_selected_labels(response, valid_labels)

        entry.update({"selected_labels": labels, "selection_response": response})
        manifest[key] = entry

        if labels:
            n_selected += 1
            selected_counts.append(len(labels))
            if args.materialize:
                subset = [r for r in regions if r["label"] in set(labels)]
                out_path = images_out_dir / _materialized_name(sample.image_path, sample.question)
                draw_region_marks(original, subset).save(out_path, format="PNG")
        else:
            n_none += 1

        if processed % args.save_every == 0:
            _write_manifest(manifest_path, manifest)

    _write_manifest(manifest_path, manifest)

    mean_selected = sum(selected_counts) / len(selected_counts) if selected_counts else 0.0
    print(
        f"  Done — selected: {n_selected}, none-relevant: {n_none}, "
        f"unannotated images: {n_no_regions}, skipped: {n_skipped}, failed: {n_failed}"
    )
    print(f"  Mean regions per selection: {mean_selected:.2f}")
    print(f"  Manifest: {manifest_path} ({len(manifest)} entries)")


def main() -> None:
    args = parse_args()

    loader = MedicalVQALoader(
        data_root=args.data_root,
        mc_data_root=args.mc_data_root,
    )
    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    # Check every manifest up front: a missing one should not cost the minutes it
    # takes to pull the selector weights onto the GPU.
    resolvers: dict[str, SomResolver] = {}
    for dataset_slug in datasets:
        resolver = SomResolver(args.som_dir)
        resolver.load_dataset(dataset_slug)
        resolvers[dataset_slug] = resolver

    print(f"Loading selector {args.selector_model} (dtype={args.dtype})...")
    bundle = load_qwen_bundle(args.selector_model, args.dtype)

    for dataset_slug in datasets:
        _process_dataset(dataset_slug, loader, resolvers[dataset_slug], bundle, args)

    print("\nAll done.")


if __name__ == "__main__":
    main()
