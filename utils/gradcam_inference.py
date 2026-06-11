"""Shared GradCAM helpers for inference flows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from gradcam.gradcam_llava import LLaVAGradCAM
from gradcam.overlay import create_gradcam_overlay
from llava.constants import IMAGE_TOKEN_INDEX

from utils.mc_json_zeroshot import normalize_generation_sequences

__all__ = [
    "compute_reasoning_gradcam_overlay",
    "normalize_generation_sequences",
    "save_gradcam_artifacts",
]


def compute_reasoning_gradcam_overlay(
    model: torch.nn.Module,
    image: Image.Image,
    image_tensor: torch.Tensor,
    full_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    output_indices: Sequence[int],
) -> Tuple[torch.Tensor | None, Image.Image, str, str]:
    """Compute GradCAM from reasoning tokens and return (heatmap, overlay, status, error)."""
    try:
        gradcam = LLaVAGradCAM(model, target_layers=[-1, -2, -3, -4])
        heatmap = gradcam.compute_gradcam(
            image_tensor=image_tensor,
            input_ids=full_ids,
            attention_mask=attention_mask,
            target_token_indices=output_indices,
            image_token_index=IMAGE_TOKEN_INDEX,
            output_spatial_size=(image.height, image.width),
        )
        _, overlay_pil = create_gradcam_overlay(image, heatmap, alpha=0.4)
        return heatmap, overlay_pil, "ok", ""
    except Exception as e:  # noqa: BLE001
        _, overlay_pil = create_gradcam_overlay(image, None, alpha=0.4)
        return None, overlay_pil, "failed", str(e)


def save_gradcam_artifacts(base_dir: str | Path, sample_name: str, heatmap: torch.Tensor | None, overlay: Image.Image, logger: Any) -> Dict[str, str]:
    """Save overlay and optional heatmap npy without aborting on I/O failures."""
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    overlay_path = out_dir / f"{sample_name}_overlay.png"
    overlay.save(str(overlay_path))
    paths["gradcam_overlay_path"] = str(overlay_path)

    if heatmap is not None:
        heatmap_path = out_dir / f"{sample_name}_heatmap.npy"
        try:
            with open(heatmap_path, "wb") as f:
                np.save(f, heatmap.detach().cpu().numpy(), allow_pickle=False)
            paths["gradcam_heatmap_path"] = str(heatmap_path)
        except OSError as e:
            logger.warning("Could not save heatmap npy (%s); continuing without npy output.", e)
    return paths
