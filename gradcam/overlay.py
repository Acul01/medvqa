"""Colormap overlay for GradCAM heatmaps."""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from PIL import Image


def create_gradcam_overlay(
    original_image: Image.Image,
    heatmap: Union[np.ndarray, torch.Tensor, None],
    alpha: float = 0.4,
    colormap: str = "jet",
) -> Tuple[np.ndarray, Image.Image]:
    """
    Blend a normalized heatmap (values in ~[0, 1], spatially H×W) over ``original_image``.

    If ``heatmap`` is None or invalid, returns the original image unchanged (float RGB and PIL).

    Returns
    -------
    overlay_rgb : np.ndarray
        Float array, shape (H, W, 3), values in [0, 1].
    overlay_pil : PIL.Image.Image
        uint8 RGB image.
    """
    orig = np.asarray(original_image.convert("RGB"), dtype=np.float32) / 255.0
    h, w = orig.shape[:2]

    if heatmap is None:
        blank = np.clip(orig, 0.0, 1.0)
        return blank, Image.fromarray((blank * 255).round().astype(np.uint8))

    if hasattr(heatmap, "detach"):
        hm_t = heatmap.detach().float()
        if hm_t.ndim == 2:
            hm_t = hm_t.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"heatmap tensor must be 2D, got {heatmap.shape}")
    else:
        hm = np.asarray(heatmap, dtype=np.float32)
        if hm.ndim != 2:
            raise ValueError(f"heatmap must be 2D, got shape {hm.shape}")
        hm_t = torch.from_numpy(hm).view(1, 1, *hm.shape)

    if hm_t.shape[-2] != h or hm_t.shape[-1] != w:
        hm_t = F.interpolate(hm_t, size=(h, w), mode="bilinear", align_corners=False)

    hm_np = hm_t.squeeze().numpy()
    hm_np = hm_np - hm_np.min()
    hm_np = hm_np / (float(hm_np.max()) + 1e-8)

    try:
        cmap_fn = cm.get_cmap(colormap)
    except ValueError:
        cmap_fn = cm.get_cmap("jet")

    color = cmap_fn(hm_np)[..., :3].astype(np.float32)

    overlay = alpha * color + (1.0 - alpha) * orig
    overlay = np.clip(overlay, 0.0, 1.0)
    pil = Image.fromarray((overlay * 255).round().astype(np.uint8))
    return overlay, pil
