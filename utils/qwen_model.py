"""Model loading helpers for Qwen2.5-VL and Qwen3-VL."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration

QWEN25_DEFAULT = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN3_DEFAULT = "Qwen/Qwen3-VL-8B-Instruct"

_MODEL_TYPE_TO_FAMILY = {
    "qwen2_5_vl": "qwen2.5",
    "qwen3_vl": "qwen3",
}


def resolve_local_model_path(path: str) -> str:
    """Return an absolute path for local checkpoints; leave HF repo ids unchanged."""
    if not path:
        return path
    if path.startswith(("./", "../")) or path.startswith("/") or os.path.isdir(path):
        return os.path.abspath(path)
    return path


def _family_from_model_type(model_type: str) -> str:
    family = _MODEL_TYPE_TO_FAMILY.get(model_type)
    if family is None:
        raise ValueError(
            f"Unsupported Qwen VL model_type: {model_type!r}. "
            f"Expected one of: {sorted(_MODEL_TYPE_TO_FAMILY)}"
        )
    return family


def family_from_path_hint(path: str) -> Optional[str]:
    """Best-effort family guess from a path string without loading config."""
    lowered = path.lower()
    if "qwen3" in lowered:
        return "qwen3"
    if "qwen2" in lowered or "qwen2.5" in lowered:
        return "qwen2.5"
    return None


def _read_adapter_base_model(model_path: str) -> Optional[str]:
    adapter_config = Path(resolve_local_model_path(model_path)) / "adapter_config.json"
    if not adapter_config.is_file():
        return None
    import json

    with open(adapter_config, encoding="utf-8") as handle:
        payload = json.load(handle)
    base = payload.get("base_model_name_or_path")
    return str(base).strip() if isinstance(base, str) and base.strip() else None


def detect_qwen_vl_family(model_path: str) -> str:
    """Return 'qwen2.5' or 'qwen3' based on the model config."""
    resolved = resolve_local_model_path(model_path)
    adapter_base = _read_adapter_base_model(resolved)
    if adapter_base is not None:
        hint = family_from_path_hint(adapter_base)
        if hint:
            return hint
        return detect_qwen_vl_family(adapter_base)

    hint = family_from_path_hint(resolved)
    config = AutoConfig.from_pretrained(resolved, trust_remote_code=False)
    model_type = getattr(config, "model_type", None)
    if model_type:
        return _family_from_model_type(str(model_type))
    if hint:
        return hint
    raise ValueError(f"Cannot determine Qwen VL family for: {model_path!r}")


def get_qwen_vl_model_class(model_path: str):
    """Return the transformers model class for a Qwen VL checkpoint."""
    family = detect_qwen_vl_family(model_path)
    if family == "qwen3":
        return Qwen3VLForConditionalGeneration
    return Qwen2_5_VLForConditionalGeneration


def default_base_model_for_path(model_path: str) -> str:
    """Return the default HF base id for processor/base-model fallbacks."""
    family = detect_qwen_vl_family(model_path)
    if family == "qwen3":
        return QWEN3_DEFAULT
    return QWEN25_DEFAULT


def is_local_model_dir(model_path: str) -> bool:
    return os.path.isdir(resolve_local_model_path(model_path))


def is_lora_checkpoint(model_path: str) -> bool:
    path = Path(resolve_local_model_path(model_path))
    return path.is_dir() and (path / "adapter_config.json").exists()


@contextmanager
def hub_cached_files_patch(model_path: str) -> Iterator[None]:
    """Patch transformers hub cached_files for local non-LoRA directories."""
    resolved = resolve_local_model_path(model_path)
    if not is_local_model_dir(resolved) or is_lora_checkpoint(resolved):
        yield
        return

    try:
        import transformers.utils.hub as hub_mod
    except ImportError:
        yield
        return

    original_cached_files = hub_mod.cached_files

    def safe_cached_files(path_or_repo_id=None, filenames=None, **kwargs):
        if (
            path_or_repo_id is not None
            and filenames is not None
            and isinstance(path_or_repo_id, (str, os.PathLike))
            and os.path.isdir(str(path_or_repo_id))
        ):
            base = str(path_or_repo_id)
            return [
                (os.path.join(base, fn) if os.path.exists(os.path.join(base, fn)) else None)
                for fn in filenames
            ]
        return original_cached_files(
            path_or_repo_id=path_or_repo_id, filenames=filenames, **kwargs
        )

    hub_mod.cached_files = safe_cached_files
    try:
        yield
    finally:
        hub_mod.cached_files = original_cached_files


def load_qwen_vl_model(
    model_path: str,
    *,
    torch_dtype: torch.dtype,
    device_map: str | dict = "auto",
    **kwargs: Any,
):
    """Load a Qwen2.5-VL or Qwen3-VL model from a HF id or local directory."""
    resolved = resolve_local_model_path(model_path)
    model_cls = get_qwen_vl_model_class(resolved)
    with hub_cached_files_patch(resolved):
        return model_cls.from_pretrained(
            resolved,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )


def load_qwen_vl_for_inference(
    model_path: str,
    *,
    torch_dtype: torch.dtype,
    device_map: str | dict = "auto",
    base_model: Optional[str] = None,
    **kwargs: Any,
) -> tuple[Any, str]:
    """Load a Qwen VL model for inference, merging LoRA adapters when present.

    Returns (model, resolved_model_path).
    """
    resolved = resolve_local_model_path(model_path)

    if is_lora_checkpoint(resolved):
        from peft import PeftModel

        if base_model is not None:
            base_path = resolve_local_model_path(base_model)
        else:
            base_path = default_base_model_for_path(resolved)
        base = load_qwen_vl_model(
            base_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )
        model = PeftModel.from_pretrained(base, resolved)
        model = model.merge_and_unload()
        return model, resolved

    model = load_qwen_vl_model(
        resolved,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **kwargs,
    )
    return model, resolved
