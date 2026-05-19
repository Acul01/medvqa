"""Shared helpers for MedVLM-R1 (Qwen2-VL-2B) off-the-shelf MC inference."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional

from huggingface_hub import hf_hub_download
from qwen_vl_utils import process_vision_info

from utils.mc_vqa import LETTER_POOL, parse_predicted_letter

DEFAULT_MODEL_PATH = "JZPeterPan/MedVLM-R1"

# MedVLM-R1 training (see train_script.sh: --max_pixels 401408)
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28  # 401408


def _medvlm_task_suffix() -> str:
    think = "redacted_thinking"
    return (
        "Your task:\n"
        f"1. Think through the question step by step, enclose your reasoning process in <{think}>...</{think}> tags.\n"
        "2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.\n"
        "3. No extra information or text outside of these tags."
    )


_ANSWER_TAG_PATTERN = re.compile(
    r"<answer>\s*([A-Z])\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)


def model_short_name(model_path: str) -> str:
    name_lower = Path(model_path).name.lower()
    if "medvlm" in name_lower:
        return "medvlm-r1"
    return Path(model_path).name


def load_qwen2vl_config(model_path: str, attn_implementation: str = "eager"):
    """Build Qwen2VLConfig with proper nested sub-config objects (not plain dicts)."""
    from transformers.models.qwen2_vl.configuration_qwen2_vl import (
        Qwen2VLConfig,
        Qwen2VLTextConfig,
        Qwen2VLVisionConfig,
    )

    config_path = ensure_medvlm_config_patched(model_path, attn_implementation=attn_implementation)
    with open(config_path, encoding="utf-8") as f:
        config_dict = json.load(f)

    text_cfg_dict = config_dict.pop("text_config", None) or {}
    vision_cfg_dict = config_dict.pop("vision_config", None)
    if text_cfg_dict.get("use_cache") is None:
        text_cfg_dict["use_cache"] = True

    config_dict["attn_implementation"] = attn_implementation

    config_kwargs = dict(config_dict)
    config_kwargs["text_config"] = Qwen2VLTextConfig(**text_cfg_dict)
    if vision_cfg_dict is not None:
        config_kwargs["vision_config"] = Qwen2VLVisionConfig(**vision_cfg_dict)
    return Qwen2VLConfig(**config_kwargs)


def ensure_medvlm_config_patched(model_path: str, attn_implementation: str = "eager") -> Path:
    """Patch cached config.json for reliable ``from_pretrained`` on MedVLM-R1.

    - ``text_config.use_cache``: null -> true (strict HF config validation)
    - ``attn_implementation``: avoid flash_attention_2 when flash_attn is not installed

    Patches the HF cache file instead of ``Qwen2VLConfig.from_dict`` (nested dict bug in 4.51).
    """
    config_path = Path(hf_hub_download(repo_id=model_path, filename="config.json"))
    with open(config_path, encoding="utf-8") as f:
        config_dict = json.load(f)
    text_cfg = config_dict.setdefault("text_config", {})
    changed = False
    if text_cfg.get("use_cache") is None:
        text_cfg["use_cache"] = True
        changed = True
    if config_dict.get("attn_implementation") == "flash_attention_2":
        config_dict["attn_implementation"] = attn_implementation
        changed = True
    if config_dict.pop("_attn_implementation", None) == "flash_attention_2":
        config_dict["attn_implementation"] = attn_implementation
        changed = True
    if changed:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
    return config_path


def configure_processor_pixels(processor) -> None:
    """Match MedVLM-R1 training resolution limits for stable vision token counts."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    image_processor.min_pixels = MIN_PIXELS
    image_processor.max_pixels = MAX_PIXELS


def format_mc_problem(question: str, options: Iterable[str]) -> str:
    """Huatuo-style MC problem string (question + inline A) B) C) options)."""
    option_list = [str(opt).strip() for opt in options]
    if len(option_list) < 2:
        raise ValueError("MC question requires at least 2 options")
    if len(option_list) > len(LETTER_POOL):
        raise ValueError("Too many options for letter mapping")

    option_lines = [f"{LETTER_POOL[i]}) {opt}" for i, opt in enumerate(option_list)]
    return f"{str(question).strip()}\n" + "\n".join(option_lines)


def build_medvlm_prompt(question: str, options: Iterable[str]) -> str:
    """Full user text: MC problem + MedVLM reasoning task template."""
    problem = format_mc_problem(question, options)
    return f"{problem}\n{_medvlm_task_suffix()}"


def extract_answer_letter(output: str, num_options: int) -> Optional[str]:
    """Parse single-letter MC answer from MedVLM output (<answer> tag preferred)."""
    match = _ANSWER_TAG_PATTERN.search(output or "")
    if match:
        letter = match.group(1).upper()
        if letter in LETTER_POOL[:num_options]:
            return letter
    return parse_predicted_letter(output, num_options=num_options)


def image_path_to_uri(image_path: str) -> str:
    """Absolute file:// URI for qwen_vl_utils / HF demo compatibility."""
    return Path(image_path).expanduser().resolve().as_uri()


def build_medvlm_conversation(image_path: str, prompt_text: str) -> List[dict]:
    """Qwen2-VL chat messages with image (file URI) + MedVLM prompt."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path_to_uri(image_path)},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def prepare_medvlm_inputs(
    processor,
    conversation: List[dict],
    *,
    add_generation_prompt: bool = True,
) -> dict:
    """Apply chat template, vision preprocessing, and processor tokenization."""
    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    image_inputs, video_inputs = process_vision_info(conversation)
    return processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
