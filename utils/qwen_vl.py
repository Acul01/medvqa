"""Shared helpers for Qwen2.5-VL training and inference."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from qwen_vl_utils import process_vision_info

from utils.mc_vqa import LETTER_POOL, format_mc_user_text, parse_predicted_letter

DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_MC_EXPORT_SUFFIX_PSEUDO_VERIFY = "_pseudo_pr_qwen_verify"

MC_JSON_RESPONSE_INSTRUCTION = (
    "\n\nRespond with ONLY valid JSON in this exact schema: "
    '{"reasoning":"<string>","answer":"<single MC option letter>"} '
    "No markdown fences, no other text before or after."
)


def configure_qwen_processor_pixels(_processor: Any) -> None:
    """Optional hook to align image resolution with training; no-op unless extended."""
    pass


def pseudo_reasoning_raw_from_metadata(metadata: Optional[Dict[str, Any]]) -> str:
    if not metadata:
        return ""
    raw = metadata.get("pseudo_reasoning_raw")
    return str(raw).strip() if isinstance(raw, str) else ""


def is_pseudo_reasoning_verified(metadata: Optional[Dict[str, Any]]) -> bool:
    if not metadata:
        return False
    v = metadata.get("pseudo_reasoning_test")
    if v is True:
        return True
    if isinstance(v, str) and v.strip().lower() in {"true", "1", "yes"}:
        return True
    return False


def _normalize_mc_letter(answer: Optional[str]) -> str:
    a = str(answer or "").strip().upper()
    if len(a) == 1 and a in LETTER_POOL:
        return a
    raise ValueError(f"Invalid MC answer letter: {answer!r}")


def build_mc_reasoning_inference_conversation(
    question: str,
    options: List[str],
    *,
    image: Any = None,
) -> List[dict]:
    """User turn: MC block + instructions to emit JSON reasoning+answer (inference — no gold reasoning)."""
    full_question = format_mc_user_text(question, options) + MC_JSON_RESPONSE_INSTRUCTION
    image_content: dict = {"type": "image"}
    if image is not None:
        image_content["image"] = image
    return [{"role": "user", "content": [image_content, {"type": "text", "text": full_question}]}]


def build_mc_reasoning_conversation(
    question: str,
    options: List[str],
    *,
    reasoning_raw: str,
    answer: str,
    image: Any = None,
) -> List[dict]:
    """Training: user sees MC + reference pseudo reasoning; assistant learns JSON {\"reasoning\", \"answer\"}."""
    mc = format_mc_user_text(question, options)
    user_text = mc + MC_JSON_RESPONSE_INSTRUCTION
    letter = _normalize_mc_letter(answer)
    assistant_text = json.dumps(
        {"reasoning": reasoning_raw.strip(), "answer": letter}, ensure_ascii=False
    )

    image_content: dict = {"type": "image"}
    if image is not None:
        image_content["image"] = image

    return [
        {
            "role": "user",
            "content": [image_content, {"type": "text", "text": user_text}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]


@dataclass
class McReasoningJsonParseResult:
    reasoning_text: Optional[str]
    predicted_letter: Optional[str]
    json_parse_ok: bool


def _strip_optional_json_fence(text: str) -> str:
    s = (text or "").strip()
    fence = re.match(r"^\s*```(?:json)?\s*\r?\n(.*?)\r?\n```\s*$", s, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return s


def parse_mc_reasoning_json_output(raw: str, num_options: int) -> McReasoningJsonParseResult:
    """Parse {\"reasoning\", \"answer\"} from model output; fallback to parse_predicted_letter."""
    stripped = _strip_optional_json_fence(raw)
    clean_json_fragment = stripped
    obj = None

    try:
        obj = json.loads(clean_json_fragment)
    except json.JSONDecodeError:
        start = stripped.find("{")
        if start == -1:
            letter = parse_predicted_letter(stripped, num_options=num_options)
            return McReasoningJsonParseResult(reasoning_text=None, predicted_letter=letter, json_parse_ok=False)
        try:
            obj, _end = json.JSONDecoder().raw_decode(stripped[start:])
        except json.JSONDecodeError:
            letter = parse_predicted_letter(stripped, num_options=num_options)
            return McReasoningJsonParseResult(reasoning_text=None, predicted_letter=letter, json_parse_ok=False)

    if not isinstance(obj, dict):
        letter = parse_predicted_letter(stripped, num_options=num_options)
        return McReasoningJsonParseResult(reasoning_text=None, predicted_letter=letter, json_parse_ok=False)

    reasoning_val = obj.get("reasoning")
    reasoning_txt = reasoning_val.strip() if isinstance(reasoning_val, str) else None

    ans_val = obj.get("answer")
    letter: Optional[str] = None
    if ans_val is not None:
        if isinstance(ans_val, str):
            letter = parse_predicted_letter(ans_val, num_options=num_options)
            if letter is None and len(ans_val.strip()) == 1:
                c = ans_val.strip().upper()
                if c in LETTER_POOL[:num_options]:
                    letter = c

    json_ok = "reasoning" in obj and "answer" in obj
    if letter is None:
        letter = parse_predicted_letter(stripped, num_options=num_options)

    return McReasoningJsonParseResult(reasoning_text=reasoning_txt, predicted_letter=letter, json_parse_ok=json_ok)


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
