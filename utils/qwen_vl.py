"""Shared helpers for Qwen-VL (2.5 / 3) training and inference."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from qwen_vl_utils import process_vision_info

from utils.mc_vqa import LETTER_POOL, format_mc_user_text, parse_predicted_letter
from utils.reasoning_mc import format_reasoning_block_for_training

DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN3_DEFAULT_MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MC_EXPORT_SUFFIX_PSEUDO = "_pseudo"
DEFAULT_MC_EXPORT_SUFFIX_PSEUDO_VERIFY = "_pseudo_pr_qwen_verify"
QWEN_VERIFY_FIELD = "qwen_verify"
# Alias used by simple_qwen_verify.py and some *_verify MC exports
QWEN_VERIFY_FIELD_ALIASES = ("is_correct", "pseudo_reasoning_test")
LABEL_IGNORE_INDEX = -100

# Cap vision resolution for stable token counts during batched training (512×28×28 patches).
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28

MC_JSON_RESPONSE_INSTRUCTION = (
    "\n\nAnalyze the image and respond with ONLY valid JSON using this exact schema:\n"
    '{"reasoning":"REGION: <anatomical region>\\nOBSERVATION: <what you observe in that region>\\nREASONING: <why this observation explains the correct answer>","answer":"<single uppercase MC letter>"}\n'
    "No text before or after the JSON."
)

MC_JSON_RESPONSE_INSTRUCTION_ANSWER_FIRST = (
    "\n\nAnalyze the image and respond with ONLY valid JSON using this exact schema:\n"
    '{"answer":"<single uppercase MC letter>","reasoning":"REGION: <anatomical region>\\nOBSERVATION: <what you observe in that region>\\nREASONING: <why this observation explains the correct answer>"}\n'
    "No text before or after the JSON."
)


def configure_qwen_processor_pixels(processor: Any) -> None:
    """Limit image resolution so vision token counts stay stable across batches."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    image_processor.min_pixels = MIN_PIXELS
    image_processor.max_pixels = MAX_PIXELS


def pseudo_reasoning_raw_from_metadata(metadata: Optional[Dict[str, Any]]) -> str:
    if not metadata:
        return ""
    raw = metadata.get("pseudo_reasoning_raw")
    return str(raw).strip() if isinstance(raw, str) else ""


def _parse_bool_field(value: Any) -> Optional[bool]:
    if value is True:
        return True
    if value is False:
        return False
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
    return None


def qwen_verify_from_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[bool]:
    """Return verify pass/fail (true/false) or None if not yet verified."""
    if not metadata:
        return None
    for field in (QWEN_VERIFY_FIELD, *QWEN_VERIFY_FIELD_ALIASES):
        v = _parse_bool_field(metadata.get(field))
        if v is not None:
            return v
    return None


def is_qwen_verify_done(metadata: Optional[Dict[str, Any]]) -> bool:
    return qwen_verify_from_metadata(metadata) is not None


def is_pseudo_reasoning_verified(metadata: Optional[Dict[str, Any]]) -> bool:
    """True when Qwen verify passed (qwen_verify or is_correct=True)."""
    return qwen_verify_from_metadata(metadata) is True


def reasoning_text_from_pseudo_row(row: Dict[str, Any]) -> Optional[str]:
    """Extract reasoning text from a pseudo-MC JSON row (raw or parsed fields)."""
    raw = row.get("pseudo_reasoning_raw")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    reg = str(row.get("pseudo_region", "")).strip()
    obs = str(row.get("pseudo_observation", "")).strip()
    exp = str(row.get("pseudo_reasoning", "")).strip()
    if reg or obs or exp:
        return format_reasoning_block_for_training(
            reg if reg else "unknown",
            obs if obs else "unknown",
            exp if exp else "unknown",
        )
    return None


def build_pseudo_reasoning_verify_conversation(
    question: str,
    options: List[str],
    reasoning_text: str,
) -> List[dict]:
    """User turn: reasoning + MC question/options; no image (reasoning refers to an unseen image)."""
    option_list = [str(opt).strip() for opt in options]
    if len(option_list) < 2:
        raise ValueError("MC question requires at least 2 options")
    option_lines = [f"{LETTER_POOL[i]}) {opt}" for i, opt in enumerate(option_list)]
    allowed = ", ".join(LETTER_POOL[: len(option_list)])
    verify_prompt = (
        "You answer a multiple-choice medical question. "
        "The reasoning below describes visual findings from a medical image. "
        "The image itself is not shown. Base your choice solely on the reasoning text.\n\n"
        f"Reasoning:\n{reasoning_text.strip()}\n\n"
        f"Question:\n{str(question).strip()}\n\n"
        f"Options:\n" + "\n".join(option_lines) + "\n\n"
        f"Respond with ONLY the single MC option letter ({allowed}) that is most supported by the reasoning."
        "Do not output any other text."
    )
    return [{"role": "user", "content": [{"type": "text", "text": verify_prompt}]}]


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
    answer_first: bool = False,
) -> List[dict]:
    """User turn: MC block + instructions to emit JSON reasoning+answer (inference — no gold reasoning)."""
    instruction = MC_JSON_RESPONSE_INSTRUCTION_ANSWER_FIRST if answer_first else MC_JSON_RESPONSE_INSTRUCTION
    full_question = format_mc_user_text(question, options) + instruction
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
    answer_first: bool = False,
) -> List[dict]:
    """Training: user sees MC + reference pseudo reasoning; assistant learns JSON {\"reasoning\", \"answer\"} or {\"answer\", \"reasoning\"}."""
    instruction = MC_JSON_RESPONSE_INSTRUCTION_ANSWER_FIRST if answer_first else MC_JSON_RESPONSE_INSTRUCTION
    mc = format_mc_user_text(question, options)
    user_text = mc + instruction
    letter = _normalize_mc_letter(answer)
    payload = (
        {"answer": letter, "reasoning": reasoning_raw.strip()}
        if answer_first
        else {"reasoning": reasoning_raw.strip(), "answer": letter}
    )
    assistant_text = json.dumps(payload, ensure_ascii=False)

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
    lowered = model_path.lower()
    if "qwen3" in lowered:
        return "qwen3-vl"
    if "qwen2" in lowered or "qwen" in lowered:
        return "qwen2.5-vl"
    return Path(model_path).name


def resolve_processor_path(model_path: str, default_path: str | None = None) -> str:
    """Use base model processor when a local checkpoint omits preprocessor config."""
    if default_path is None:
        from utils.qwen_model import default_base_model_for_path, family_from_path_hint

        if family_from_path_hint(model_path) is not None or Path(model_path).exists():
            try:
                default_path = default_base_model_for_path(model_path)
            except (OSError, ValueError):
                default_path = DEFAULT_MODEL_PATH
        else:
            default_path = DEFAULT_MODEL_PATH

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


def build_segment_selection_conversation(question: str, image: Any) -> List[dict]:
    """User turn: image with numbered SoM regions + ask which ones are relevant."""
    from utils.som_select import build_segment_selection_prompt

    prompt_text = build_segment_selection_prompt(question)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
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
    truncation: bool = True,
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
        processor_kwargs["truncation"] = truncation

    return processor(**processor_kwargs)


@dataclass
class QwenVLDataCollator:
    """Pad text fields; concatenate vision tensors (Qwen2.5-VL / Qwen3-VL batch layout)."""

    tokenizer: Any

    def __call__(self, instances: Sequence[dict]) -> dict:
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=LABEL_IGNORE_INDEX
        )

        batch: dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(pad_id),
        }

        if instances and "mm_token_type_ids" in instances[0]:
            mm_ids = []
            for inst in instances:
                mm = inst.get("mm_token_type_ids")
                if mm is None:
                    continue
                if mm.ndim == 0:
                    mm = mm.unsqueeze(0)
                elif mm.ndim > 1:
                    mm = mm.reshape(-1)
                mm_ids.append(mm)
            if mm_ids:
                batch["mm_token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
                    mm_ids, batch_first=True, padding_value=0
                )

        if instances and "pixel_values" in instances[0]:
            pixel_values = [inst["pixel_values"] for inst in instances if inst.get("pixel_values") is not None]
            if pixel_values:
                batch["pixel_values"] = torch.cat(pixel_values, dim=0)

        if instances and "image_grid_thw" in instances[0]:
            grids = [inst["image_grid_thw"] for inst in instances if inst.get("image_grid_thw") is not None]
            if grids:
                rows = []
                for grid in grids:
                    if grid.ndim == 1:
                        rows.append(grid.unsqueeze(0))
                    else:
                        rows.append(grid)
                batch["image_grid_thw"] = torch.cat(rows, dim=0).reshape(-1, 3)

        return batch


def squeeze_batch_dim(batch: dict) -> dict:
    """Remove batch dimension from processor outputs for per-sample Dataset items."""
    out = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            out[key] = value
            continue
        # Keep mm_token_type_ids aligned with input_ids (same leading batch dim).
        if key in {"input_ids", "attention_mask", "mm_token_type_ids"}:
            if value.shape[0] == 1:
                out[key] = value.squeeze(0)
            else:
                out[key] = value
            continue
        if value.shape[0] == 1:
            out[key] = value.squeeze(0)
        else:
            out[key] = value
    return out
