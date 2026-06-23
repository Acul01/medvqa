"""Question-aware filtering of precomputed SoM regions using BiomedCLIP.

SAM segments every region it can find in an image, independent of the
question being asked. This module scores each precomputed region against
the sample's question text via BiomedCLIP image-text similarity and keeps
only the top-k most relevant ones, so the final SoM overlay shown to the
VLM only marks regions plausibly relevant to answering the question.

The expensive step (SAM segmentation) stays precomputed via
scripts/precompute_som.py; this module only does cheap scoring + redrawing
at inference time.

Requires: pip install open_clip_torch

Usage:
    som_filter = SomQuestionFilter()
    image = som_filter.prepare_image(original_image, regions, question, top_k=5)
"""

from __future__ import annotations

import torch
from PIL import Image

from utils.som import SomResolver, draw_region_marks

DEFAULT_BIOMEDCLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


class SomQuestionFilter:
    """Scores precomputed SoM regions against a question using BiomedCLIP."""

    def __init__(self, model_name: str = DEFAULT_BIOMEDCLIP_MODEL, device: str = "cuda"):
        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "open_clip_torch is required for BiomedCLIP-based SoM filtering. "
                "Install with: pip install open_clip_torch"
            ) from e

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer([text]).to(self.device)
        features = self.model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _encode_images(self, crops: list[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(crop) for crop in crops]).to(self.device)
        features = self.model.encode_image(batch)
        return features / features.norm(dim=-1, keepdim=True)

    def score_regions(
        self,
        original_image: Image.Image,
        regions: list[dict],
        question: str,
    ) -> list[float]:
        """Return a relevance score per region (cosine similarity to *question*).

        *regions* only need a "bbox" key — masks are not required for scoring.
        """
        if not regions:
            return []

        crops = []
        for region in regions:
            x, y, w, h = region["bbox"]
            crops.append(original_image.crop((x, y, x + w, y + h)))

        image_features = self._encode_images(crops)
        text_features = self._encode_text(question)
        similarities = (image_features @ text_features.T).squeeze(-1)
        return similarities.float().cpu().tolist()

    def select_top_k(
        self,
        original_image: Image.Image,
        regions: list[dict],
        question: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Return the *top_k* regions most relevant to *question* (unchanged order)."""
        if len(regions) <= top_k:
            return regions
        scores = self.score_regions(original_image, regions, question)
        ranked = sorted(zip(regions, scores), key=lambda pair: pair[1], reverse=True)
        return [region for region, _ in ranked[:top_k]]

    def prepare_image(
        self,
        original_image: Image.Image,
        regions: list[dict],
        question: str,
        *,
        top_k: int = 5,
    ) -> Image.Image:
        """Score *regions* against *question*, keep top_k, render onto *original_image*.

        *regions* must include a loaded "mask" per region (e.g. via
        SomResolver.load_region_masks) for the final rendering step.
        """
        if not regions:
            return original_image
        selected = self.select_top_k(original_image, regions, question, top_k=top_k)
        return draw_region_marks(original_image, selected)

    def prepare_image_from_resolver(
        self,
        resolver: SomResolver,
        image_path: str,
        question: str,
        *,
        top_k: int = 5,
    ) -> Image.Image:
        """Per-sample entry point: score precomputed regions for *image_path*
        against *question*, then render only the top_k selected regions.

        Only loads masks for the selected regions from disk — cheap even
        when called once per (image, question) sample during inference.
        Falls back to the plain original image if no SoM regions exist.
        """
        original_image = Image.open(image_path).convert("RGB")
        regions = resolver.get_regions(image_path)
        if not regions:
            return original_image

        selected = self.select_top_k(original_image, regions, question, top_k=top_k)
        selected_labels = {region["label"] for region in selected}
        loaded_regions = resolver.load_region_masks(image_path, labels=selected_labels)
        return draw_region_marks(original_image, loaded_regions)
