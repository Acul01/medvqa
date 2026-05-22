"""Shared helpers for Qwen2.5-VL training and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from qwen_vl_utils import process_vision_info

from utils.mc_vqa import format_mc_user_text
from utils.reasoning_mc import END_REASONING_TOKEN, format_reasoning_block_for_training

DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_MC_EXPORT_SUFFIX_PSEUDO_VERIFY = "_pseudo_pr_qwen_verify"

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28  # 401408

REASONING_JSON_INSTRUCTION = (
    "\n\nRespond with a single JSON object with keys \"reasoning\" and \"answer\". "
    f"\"reasoning\" must follow REGION:/OBSERVATION:/REASONING: and end with {END_REASONING_TOKEN}. "
    "\"answer\" is one option letter. Output only JSON."
)

# Gray placeholder required by Qwen2.5-VL processor when verifying from text-only rationale (no clinical image).
PSEUDO_REASONING_VERIFY_PLACEHOLDER_SIZE = (224, 224)


def make_pseudo_reasoning_verify_placeholder(
    *,
    width: int = PSEUDO_REASONING_VERIFY_PLACEHOLDER_SIZE[0],
    height: int = PSEUDO_REASONING_VERIFY_PLACEHOLDER_SIZE[1],
) -> Any:
    """Uniform RGB gray image; pairing contract for ``build_pseudo_reasoning_verify_conversation``."""
    from PIL import Image

    return Image.new("RGB", (int(width), int(height)), color=(128, 128, 128))


def build_pseudo_reasoning_verify_conversation(
    question: str,
    options: List[str],
    reasoning_text: str,
    *,
    placeholder_image: Any,
) -> List[dict]:
    """User turn: meaningless placeholder image + rationale + MC question/options (matches ``format_mc_user_text`` tail)."""
    intro = (
        "You verify a medical multiple-choice item. "
        "The image shown is only a meaningless gray placeholder — do not use visual information.\n"
        "Base your answer solely on the textual reasoning provided below.\n\n"
        f"Reasoning:\n{(reasoning_text or '').strip()}\n\n"
    )
    rest = format_mc_user_text(question, options)
    full_text = intro + rest
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": placeholder_image},
                {"type": "text", "text": full_text},
            ],
        }
    ]


def model_short_name(model_path: str) -> str:
    name = Path(model_path).name.lower()
    if "qwen" in name:
        return "qwen2.5-vl"
    return Path(model_path).name


def is_pseudo_reasoning_verified(metadata: Optional[Dict[str, Any]]) -> bool:
    """True when ``pseudo_reasoning_test`` is verified (bool or common string forms)."""
    if not metadata:
        return False
    value = metadata.get("pseudo_reasoning_test")
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def pseudo_reasoning_raw_from_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """Read ``pseudo_reasoning_raw`` or reconstruct from parsed pseudo fields."""
    if not metadata:
        return None
    raw = metadata.get("pseudo_reasoning_raw")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    reg = str(metadata.get("pseudo_reasoning_region", "")).strip()
    obs = str(metadata.get("pseudo_reasoning_observation", "")).strip()
    exp = str(metadata.get("pseudo_reasoning_explanation", "")).strip()
    if reg or obs or exp:
        return format_reasoning_block_for_training(
            reg if reg else "unknown",
            obs if obs else "unknown",
            exp if exp else "unknown",
        )
    return None


def format_reasoning_answer_json(reasoning_raw: str, answer: str) -> str:
    """Compact assistant target: ``{\"reasoning\": \"...\", \"answer\": \"B\"}``."""
    payload = {
        "reasoning": (reasoning_raw or "").strip(),
        "answer": str(answer).strip(),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_qwen_processor_pixels(
    processor,
    *,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> None:
    """Cap vision resolution for stable token counts during Qwen2.5-VL training."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    image_processor.min_pixels = min_pixels
    image_processor.max_pixels = max_pixels


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


def build_mc_reasoning_conversation(
    question: str,
    options: List[str],
    *,
    reasoning_raw: str,
    answer: str,
    image: Any = None,
) -> List[dict]:
    """MC user turn + JSON assistant target with verified pseudo reasoning and answer letter."""
    full_question = format_mc_user_text(question, options) + REASONING_JSON_INSTRUCTION
    image_content: dict = {"type": "image"}
    if image is not None:
        image_content["image"] = image

    assistant_text = format_reasoning_answer_json(reasoning_raw, answer)
    return [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": full_question},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_text}],
        },
    ]


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
    truncation: bool = False,
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
        if truncation:
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
