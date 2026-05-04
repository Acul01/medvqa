"""
GradCAM on the CLIP ViT encoder inside LLaVA-Med.

Hooks target layers under
``vision_tower.vision_tower.vision_model.encoder.layers``.
``CLIPVisionTower.forward`` is often ``@torch.no_grad``; we temporarily replace
it with a grad-tracking forward during ``compute_gradcam``.
"""

from __future__ import annotations

import logging
import types
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _resolve_encoder_layers(model: torch.nn.Module):
    """Return (CLIPVisionTower wrapper, nn.ModuleList of ViT encoder layers)."""
    inner = model
    if hasattr(inner, "get_model"):
        inner = inner.get_model()
    vt = getattr(inner, "vision_tower", None)
    if vt is None and hasattr(inner, "get_vision_tower"):
        vt = inner.get_vision_tower()
    if vt is None:
        raise ValueError("Could not find vision_tower on model.")

    clip = getattr(vt, "vision_tower", None)
    if clip is None:
        raise ValueError("Vision tower has no inner vision_tower (CLIP).")

    vision_model = getattr(clip, "vision_model", None)
    if vision_model is None:
        raise ValueError("CLIP module has no vision_model.")

    encoder = getattr(vision_model, "encoder", None)
    if encoder is None:
        raise ValueError("vision_model has no encoder.")

    layers = getattr(encoder, "layers", None)
    if layers is None:
        raise ValueError("encoder has no layers ModuleList.")
    return vt, layers


def _num_patch_tokens(vision_tower_module) -> int:
    if hasattr(vision_tower_module, "num_patches"):
        return int(vision_tower_module.num_patches)
    inner = getattr(vision_tower_module, "vision_tower", None)
    cfg = getattr(inner, "config", None) if inner is not None else None
    if cfg is not None and hasattr(cfg, "image_size") and hasattr(cfg, "patch_size"):
        isz = int(cfg.image_size)
        psz = int(cfg.patch_size)
        return (isz // psz) ** 2
    return 576


def _patch_clip_vision_tower_forward(vision_tower_module) -> Tuple[Callable, Callable]:
    """Replace CLIPVisionTower.forward with a grad-tracking version."""
    old_forward = vision_tower_module.forward

    def forward_with_grad(self, images):
        if not self.is_loaded:
            raise RuntimeError("Vision tower is not loaded.")
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
            return image_features
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
        )
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    vision_tower_module.forward = types.MethodType(forward_with_grad, vision_tower_module)

    def restore():
        vision_tower_module.forward = old_forward

    return restore, vision_tower_module.forward


def _compact_positions_to_logits_pairs(
    input_ids: torch.Tensor,
    target_compact_indices: Sequence[int],
    logits_len: int,
    image_token_index: int,
    num_patch_tokens: int,
) -> List[Tuple[int, int]]:
    """
    Map compact ``input_ids`` positions (as from ``generate``) to (logit_index, token_id).

    Causal LM: logits[logit_index] scores the token at merged position logit_index + 1.
    """
    row = input_ids[0]
    img_idx = (row == image_token_index).nonzero(as_tuple=True)[0]

    def compact_to_expanded(compact_pos: int) -> int:
        if img_idx.numel() == 0:
            return compact_pos
        ip = int(img_idx[0])
        if compact_pos < ip:
            return compact_pos
        if compact_pos == ip:
            return -1
        return compact_pos + (num_patch_tokens - 1)

    pairs: List[Tuple[int, int]] = []
    for p in target_compact_indices:
        if p < 0 or p >= row.numel():
            continue
        tid = int(row[p].item())
        if tid == image_token_index:
            continue
        e = compact_to_expanded(p)
        if e <= 0:
            continue
        logit_idx = e - 1
        if logit_idx >= logits_len:
            continue
        pairs.append((logit_idx, tid))

    if not pairs:
        for p in target_compact_indices:
            if p <= 0 or p >= row.numel():
                continue
            tid = int(row[p].item())
            if tid == image_token_index:
                continue
            logit_idx = p - 1
            if logit_idx < logits_len:
                pairs.append((logit_idx, tid))

    return pairs


def _layer_cam_from_tensors(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    grid_hw: Tuple[int, int],
) -> torch.Tensor:
    """GradCAM from ViT layer output [B, S, C]; drop CLS (index 0)."""
    if activation.ndim != 3 or gradient.ndim != 3:
        raise ValueError(f"Expected 3D tensors, got act {activation.shape}, grad {gradient.shape}")

    act = activation[0, 1:, :]
    grad = gradient[0, 1:, :]

    h, w = grid_hw
    n = h * w
    if act.shape[0] < n or grad.shape[0] < n:
        raise ValueError(f"Not enough patch tokens: got {act.shape[0]}, need {n}")

    act = act[:n, :]
    grad = grad[:n, :]

    weights = grad.mean(dim=0, keepdim=True)
    cam = (weights * act).sum(dim=-1)
    cam = F.relu(cam)
    cam = cam.view(h, w)
    return cam


class _LayerHookState:
    __slots__ = ("activation", "gradient")

    def __init__(self):
        self.activation: Optional[torch.Tensor] = None
        self.gradient: Optional[torch.Tensor] = None


class LLaVAGradCAM:
    """
    GradCAM on CLIP ViT encoder layers inside LLaVA-Med.

    Parameters
    ----------
    model :
        ``LlavaMistralForCausalLM`` (or compatible).
    target_layers :
        Indices into ``encoder.layers`` (negative indices allowed).
        Default: last four layers.
    """

    def __init__(self, model: torch.nn.Module, target_layers: Optional[Sequence[int]] = None):
        self.model = model
        self.vision_tower_module, self.encoder_layers = _resolve_encoder_layers(model)
        n_layers = len(self.encoder_layers)
        if target_layers is None:
            self.target_layer_indices = [n_layers - 4, n_layers - 3, n_layers - 2, n_layers - 1]
        else:
            self.target_layer_indices = [i if i >= 0 else n_layers + i for i in target_layers]
        for idx in self.target_layer_indices:
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"Layer index {idx} out of range [0, {n_layers})")

        self._handles: List = []
        self._states: List[_LayerHookState] = []

    def _register_hooks(self) -> None:
        self._remove_hooks()
        self._states = []

        for idx in self.target_layer_indices:
            layer = self.encoder_layers[idx]
            state = _LayerHookState()
            self._states.append(state)

            def make_forward_hook(st: _LayerHookState):
                def hook(_module, _inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    st.activation = t

                return hook

            def make_backward_hook(st: _LayerHookState):
                def hook(_module, _grad_in, grad_out):
                    if grad_out[0] is None:
                        return
                    st.gradient = grad_out[0]

                return hook

            self._handles.append(layer.register_forward_hook(make_forward_hook(state)))
            self._handles.append(layer.register_full_backward_hook(make_backward_hook(state)))

    def _remove_hooks(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:  # noqa: S110
                pass
        self._handles.clear()

    def _vit_grid_hw(self) -> Tuple[int, int]:
        n = _num_patch_tokens(self.vision_tower_module)
        side = int(round(n**0.5))
        if side * side != n:
            raise ValueError(f"Patch count {n} is not a square grid.")
        return side, side

    def compute_gradcam(
        self,
        image_tensor: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_token_indices: Sequence[int],
        image_token_index: int = -200,
        output_spatial_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        GradCAM heatmap in [0, 1], float32 CPU, shape ``output_spatial_size`` (default 336×336).

        ``target_token_indices`` are **compact** indices into ``input_ids`` (same layout as
        ``model.generate`` output), e.g. positions of rationale tokens after the prompt.
        """
        self._register_hooks()
        restore_vt = None
        try:
            restore_vt, _ = _patch_clip_vision_tower_forward(self.vision_tower_module)
        except Exception as e:  # noqa: BLE001
            self._remove_hooks()
            raise RuntimeError(f"Failed to patch vision tower forward: {e}") from e

        device = input_ids.device
        dtype = getattr(self.model, "dtype", torch.bfloat16)
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device=device, dtype=dtype)
        input_ids = input_ids.to(device=device)
        attention_mask = attention_mask.to(device=device)

        num_patch = _num_patch_tokens(self.vision_tower_module)
        grid_hw = self._vit_grid_hw()

        if output_spatial_size is None:
            oh = int(image_tensor.shape[-2])
            ow = int(image_tensor.shape[-1])
            output_spatial_size = (oh, ow)

        self.model.zero_grad(set_to_none=True)
        try:
            with torch.enable_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor,
                    use_cache=False,
                    return_dict=True,
                )
                logits = outputs.logits[0]
                logits_len = logits.shape[0]

                pairs = _compact_positions_to_logits_pairs(
                    input_ids,
                    target_token_indices,
                    logits_len,
                    image_token_index,
                    num_patch,
                )
                if not pairs:
                    raise RuntimeError("No valid (logit_index, token_id) pairs for GradCAM loss.")

                loss = logits.new_zeros(())
                for li, tid in pairs:
                    loss = loss + logits[li, tid]

                loss.backward(retain_graph=True)

            cams = []
            for state in self._states:
                if state.activation is None or state.gradient is None:
                    continue
                act = state.activation
                grad = state.gradient
                if act.shape != grad.shape:
                    continue
                cam_l = _layer_cam_from_tensors(act.float(), grad.float(), grid_hw)
                cam_l = cam_l - cam_l.min()
                cam_l = cam_l / cam_l.max().clamp_min(1e-8)
                cams.append(cam_l)

            if not cams:
                raise RuntimeError("No layer CAMs collected (missing gradients?).")

            heatmap = torch.stack(cams, dim=0).mean(dim=0)
            heatmap = heatmap - heatmap.min()
            heatmap = heatmap / heatmap.max().clamp_min(1e-8)

            oh, ow = output_spatial_size
            heatmap_up = F.interpolate(
                heatmap.unsqueeze(0).unsqueeze(0),
                size=(oh, ow),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            heatmap_up = heatmap_up - heatmap_up.min()
            heatmap_up = heatmap_up / heatmap_up.max().clamp_min(1e-8)
            return heatmap_up.detach().float().cpu()
        finally:
            if restore_vt is not None:
                restore_vt()
            self._remove_hooks()
            self.model.zero_grad(set_to_none=True)
