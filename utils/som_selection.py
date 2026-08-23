"""Question-aware resolution of the SoM segments a question actually needs.

Workflow:
1. scripts/precompute_som.py annotates every image once with *all* its segments
   and persists one mask PNG per segment.
2. scripts/precompute_som_selection.py asks an LVLM, per (image, question) pair,
   which of those segments are relevant, and records only the chosen labels.
3. This module turns that record back into an image: the original with just the
   selected segments outlined. Training, pseudo-reasoning generation and
   evaluation all go through it, so they render identically.

Only labels are persisted, never rendered images: re-drawing from the cached
per-region masks is cheap, whereas one PNG per QA pair would cost several GB on
the larger datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image

from utils.som import SomResolver, draw_region_marks

SELECTION_MANIFEST_FILENAME = "selection_manifest.json"


def selection_key(image_path: str | Path, question: str) -> str:
    """Manifest key for an (image, question) pair.

    Uses the same last-two-path-components convention as SomResolver, so keys
    stay valid across machines with different data roots.
    """
    parts = Path(image_path).parts[-2:]
    return f"{'/'.join(parts)}||{question.strip()}"


class SomSelectionResolver:
    """Renders images carrying only the segments selected for a given question.

    Usage:
        resolver = SomSelectionResolver("02_data/som", "02_data/som_selection")
        resolver.load_dataset("slake")
        image, labels = resolver.render_with_labels(sample.image_path, sample.question)
    """

    def __init__(self, som_dir: str | Path, selection_dir: str | Path):
        self.som = SomResolver(som_dir)
        self.selection_dir = Path(selection_dir)
        self._entries: dict[str, dict] = {}

    def load_dataset(self, dataset: str) -> None:
        """Load region masks metadata and the selection manifest for *dataset*."""
        slug = dataset.lower()
        self.som.load_dataset(slug)
        manifest_path = self.selection_dir / slug / SELECTION_MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"SoM selection manifest not found: {manifest_path}\n"
                f"Run scripts/precompute_som_selection.py --dataset {slug} first."
            )
        with open(manifest_path, "r", encoding="utf-8") as f:
            self._entries.update(json.load(f))

    def get_entry(self, image_path: str, question: str) -> Optional[dict]:
        return self._entries.get(selection_key(image_path, question))

    def get_selected_labels(self, image_path: str, question: str) -> Optional[list[str]]:
        """Selected labels; None if the pair was never processed, [] if nothing was relevant."""
        entry = self.get_entry(image_path, question)
        if entry is None:
            return None
        return [str(label) for label in entry.get("selected_labels", [])]

    def get_selection_response(self, image_path: str, question: str) -> str:
        entry = self.get_entry(image_path, question)
        return str(entry.get("selection_response", "")) if entry else ""

    def render_with_labels(self, image_path: str, question: str) -> tuple[Image.Image, list[str]]:
        """Return (image, labels actually drawn).

        Falls back to the untouched original whenever there is no manifest entry,
        the selector answered "none", or the masks are missing on disk.
        """
        image = Image.open(image_path).convert("RGB")
        labels = self.get_selected_labels(image_path, question)
        if not labels:
            return image, []
        regions = self.som.load_region_masks(image_path, labels=set(labels))
        if not regions:
            return image, []
        return draw_region_marks(image, regions), [region["label"] for region in regions]

    def render(self, image_path: str, question: str) -> Image.Image:
        return self.render_with_labels(image_path, question)[0]

    def coverage(self, pairs: Iterable[tuple[str, str]]) -> tuple[int, int]:
        """Return (pairs with a manifest entry, pairs total).

        Callers should report this: a manifest built for the wrong split would
        otherwise silently degrade into training on unannotated images.
        """
        total = 0
        found = 0
        for image_path, question in pairs:
            total += 1
            if self.get_entry(image_path, question) is not None:
                found += 1
        return found, total
