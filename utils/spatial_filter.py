"""
Spatial filter layer for Qwen2.5-VL visual tokens.

Shared between training (train_qwen_modified.py) and inference (inference_qwen.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Merger discovery helpers
# ---------------------------------------------------------------------------

def find_visual_merger(model: nn.Module) -> nn.Module:
    """Return the visual merger module, searching common attribute paths.

    Raises RuntimeError if not found.
    """
    candidate_paths = [
        "visual.merger",        # Qwen2.5-VL (some transformers versions)
        "model.visual.merger",  # Qwen2.5-VL (other transformers versions)
        "vision_model.merger",
    ]
    for path in candidate_paths:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError(
        f"Cannot find visual merger. Tried: {candidate_paths}. "
        "Inspect model architecture and extend candidate_paths."
    )


def get_visual_merger_output_dim(model: nn.Module) -> int:
    """Infer merger output dim (= LLM hidden size) from the model architecture."""
    # 1. Preferred: last Linear in visual merger outputs exactly the LLM hidden size
    try:
        merger = find_visual_merger(model)
        for m in reversed(list(merger.modules())):
            if isinstance(m, nn.Linear):
                return m.out_features
    except RuntimeError:
        pass
    # 2. LM head: Linear(hidden_size → vocab), so in_features = hidden_size
    try:
        return model.lm_head.in_features
    except AttributeError:
        pass
    # 3. Config fallback — attribute name differs across Qwen variants
    for attr in ("hidden_size", "text_config.hidden_size"):
        try:
            obj = model.config
            for part in attr.split("."):
                obj = getattr(obj, part)
            return int(obj)
        except AttributeError:
            pass
    raise RuntimeError(
        "Cannot infer visual merger output dimension. "
        "Check model.visual.merger or model.lm_head."
    )


def _get_spatial_merge_size(merger: nn.Module, base_model: nn.Module) -> int:
    """Resolve the spatial_merge_size used by the PatchMerger.

    Qwen2_5_VLPatchMerger receives spatial_merge_size in __init__ but stores
    it ONLY via self.hidden_size = context_dim * ms².  Try three paths before
    falling back to the Qwen2/2.5-VL default of 2.
    """
    ms = getattr(merger, "spatial_merge_size", None)
    if ms is None:
        try:
            ms = base_model.config.vision_config.spatial_merge_size
        except AttributeError:
            pass
    if ms is None:
        ms = 2  # Qwen2-VL and Qwen2.5-VL always use spatial_merge_size = 2
    return int(ms)


# ---------------------------------------------------------------------------
# Position encoding
# ---------------------------------------------------------------------------

def spatial_position_encoding(
    grid_h: int, grid_w: int, grid_t: int, device: torch.device
) -> torch.Tensor:
    """Normalised (y, x) ∈ [0, 1]² for every merged patch token.

    Returns shape (grid_t * grid_h * grid_w, 2).
    For temporal t > 1 spatial positions repeat across frames.
    """
    ys = torch.linspace(0, 1, grid_h, device=device)
    xs = torch.linspace(0, 1, grid_w, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    spatial = torch.stack([grid_y, grid_x], dim=-1).view(grid_h * grid_w, 2)
    return spatial.unsqueeze(0).expand(grid_t, -1, -1).reshape(grid_t * grid_h * grid_w, 2)


# ---------------------------------------------------------------------------
# Spatial filter layer
# ---------------------------------------------------------------------------

class SpatialFilterLayer(nn.Module):
    """Learnable per-patch spatial filter with position awareness.

    For each merged patch token the scorer receives both the visual feature
    vector AND the token's normalised (y, x) position in the image, so it can
    learn spatial priors in addition to purely visual patterns.

    Pipeline per token:
      1. concat(token_D, pos_2) → (D+2,)
      2. scorer MLP → Sigmoid  →  score ∈ (0, 1)
      3. filtered = token × score
      4. concat([original, filtered]) → Linear(2D, D) → output

    Initialisation: fusion layer = identity on original half, zero on filtered
    half → strict no-op at step 0; filter activates as gradients flow.
    """

    def __init__(self, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.d_model = d_model
        self.scorer = nn.Sequential(
            nn.Linear(d_model + 2, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Linear(d_model * 2, d_model)
        with torch.no_grad():
            self.fusion.weight[:, :d_model].copy_(torch.eye(d_model))
            self.fusion.weight[:, d_model:].zero_()
            self.fusion.bias.zero_()

    def forward(
        self,
        tokens: torch.Tensor,  # (N, D) — flat merger output for one image
        grid_h: int,
        grid_w: int,
        grid_t: int = 1,
    ) -> torch.Tensor:
        pos = spatial_position_encoding(grid_h, grid_w, grid_t, tokens.device)
        pos = pos.to(tokens.dtype)
        score = self.scorer(torch.cat([tokens, pos], dim=-1))
        filtered = tokens * score
        return self.fusion(torch.cat([tokens, filtered], dim=-1))


# ---------------------------------------------------------------------------
# Wrapper model
# ---------------------------------------------------------------------------

class QwenWithSpatialFilter(nn.Module):
    """Wrapper around Qwen2_5_VLForConditionalGeneration that installs a
    forward hook on visual.merger to apply SpatialFilterLayer transparently.
    """

    def __init__(self, base_model: nn.Module, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.model = base_model
        self.spatial_filter = SpatialFilterLayer(d_model, bottleneck)
        # Use object.__setattr__ so _merger is not registered as a submodule;
        # otherwise state_dict duplicates merger weights (shared memory) and
        # safetensors save fails.
        object.__setattr__(self, "_merger", find_visual_merger(base_model))
        self._merge_size = _get_spatial_merge_size(self._merger, base_model)

    def _make_hook(self, image_grid_thw: torch.Tensor):
        """Return a merger hook that applies SpatialFilterLayer per image.

        image_grid_thw: (B, 3) — t, h, w as returned by the processor.
          Older transformers: post-merger dims  → merge_size = 1 (no adjustment)
          Newer transformers: pre-merger dims   → divide h, w by merge_size
        """
        spatial_filter = self.spatial_filter
        ms = self._merge_size

        def hook(module, inp, output):  # noqa: ARG001
            result = output.clone()
            ptr = 0
            for thw in image_grid_thw:
                t = int(thw[0])
                h = int(thw[1]) // ms
                w = int(thw[2]) // ms
                n = t * h * w
                result[ptr : ptr + n] = spatial_filter(output[ptr : ptr + n], h, w, t)
                ptr += n
            return result

        return hook

    def forward(self, input_ids, attention_mask, pixel_values=None,
                image_grid_thw=None, labels=None, **kwargs):
        hook_fn = self._make_hook(image_grid_thw) if image_grid_thw is not None else None
        handle = self._merger.register_forward_hook(hook_fn) if hook_fn else None
        try:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,
                **kwargs,
            )
        finally:
            if handle is not None:
                handle.remove()

    def generate(self, **kwargs):
        image_grid_thw = kwargs.get("image_grid_thw")
        hook_fn = self._make_hook(image_grid_thw) if image_grid_thw is not None else None
        handle = self._merger.register_forward_hook(hook_fn) if hook_fn else None
        try:
            return self.model.generate(**kwargs)
        finally:
            if handle is not None:
                handle.remove()

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


# ---------------------------------------------------------------------------
# Convenience loader (used by inference)
# ---------------------------------------------------------------------------

def load_spatial_filter_checkpoint(
    base_model: nn.Module,
    filter_path: str,
    bottleneck: int = 64,
    *,
    strict: bool = True,
) -> QwenWithSpatialFilter:
    """Wrap base_model with SpatialFilterLayer and load weights from filter_path.

    Args:
        base_model:  Loaded Qwen2_5_VLForConditionalGeneration (eval mode).
        filter_path: Path to spatial_filter.pt produced by train_qwen_modified.py.
        bottleneck:  Must match the value used during training (default 64).
        strict:      Passed to load_state_dict.

    Returns:
        QwenWithSpatialFilter in eval mode with filter weights restored.
    """
    d_model = get_visual_merger_output_dim(base_model)
    wrapped = QwenWithSpatialFilter(base_model, d_model=d_model, bottleneck=bottleneck)
    state = torch.load(filter_path, map_location="cpu", weights_only=True)
    wrapped.spatial_filter.load_state_dict(state, strict=strict)
    # Move filter to the same device + dtype as the visual merger so that the
    # forward hook does not mix CPU filter weights with GPU merger outputs.
    merger_param = next(wrapped._merger.parameters())
    wrapped.spatial_filter.to(device=merger_param.device, dtype=merger_param.dtype)
    wrapped.eval()
    print(f"Spatial filter loaded from {filter_path}  (d_model={d_model}, bottleneck={bottleneck})")
    return wrapped
