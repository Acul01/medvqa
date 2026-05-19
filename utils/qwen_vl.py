"""Shared helpers for Qwen2.5-VL training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import torch
from qwen_vl_utils import process_vision_info

from utils.mc_vqa import format_mc_user_text

DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"


def model_short_name(model_path: str) -> str:
    name = Path(model_path).name.lower()
    if "qwen" in name:
        return "qwen2.5-vl"
    return Path(model_path).name


def resolve_processor_path(model_path: str, default_path: str = DEFAULT_MODEL_PATH) -> str:
    """Use base model processor when a local checkpoint omits preprocessor config."""
    path = Path(model_path)
    if path.exists() and not (path / "preprocessor_config.json").exists():
        return default_path
    return model_path


def build_mc_conversation(
    question: str,
    options: List[str],
    *,
    answer: Optional[str] = None,
    image: Any = None,
) -> List[dict]:
    """Build Qwen chat messages for MC VQA (user image + question, optional assistant answer)."""
    full_question = format_mc_user_text(question, options)
    image_content: dict = {"type": "image"}
    if image is not None:
        image_content["image"] = image

    conversation = [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": full_question},
            ],
        }
    ]
    if answer is not None:
        conversation.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            }
        )
    return conversation


def mask_labels_assistant_only(processor, input_ids: torch.Tensor, conversation: List[dict]) -> torch.Tensor:
    """Mask user/prompt tokens; loss is computed only on the assistant answer."""
    labels = input_ids.clone()
    user_only = [conversation[0]]
    prompt_text = processor.apply_chat_template(user_only, tokenize=False, add_generation_prompt=True)
    prompt_ids = processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    boundary = min(len(prompt_ids), labels.shape[0])
    labels[:boundary] = -100

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[input_ids == pad_id] = -100
    return labels


def prepare_qwen_inputs(
    processor,
    conversation: List[dict],
    *,
    add_generation_prompt: bool = False,
    max_length: Optional[int] = None,
    padding: bool | str = False,
    return_tensors: str = "pt",
) -> dict:
    """Apply chat template, vision preprocessing, and processor tokenization."""
    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    image_inputs, video_inputs = process_vision_info(conversation)

    processor_kwargs: dict = {
        "text": [text],
        "images": image_inputs,
        "videos": video_inputs,
        "padding": padding,
        "return_tensors": return_tensors,
    }
    if max_length is not None:
        processor_kwargs["max_length"] = max_length
        processor_kwargs["truncation"] = True

    return processor(**processor_kwargs)


def squeeze_batch_dim(batch: dict) -> dict:
    """Remove batch dimension from processor outputs for per-sample Dataset items."""
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == 1:
            out[key] = value.squeeze(0)
        else:
            out[key] = value
    return out
