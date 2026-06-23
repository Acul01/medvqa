"""Shared helpers for zero-shot MC inference with JSON answer+rationale output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Literal, Optional, Tuple, cast

import torch

from utils.mc_vqa import LETTER_POOL, format_mc_user_text, parse_predicted_letter

JsonOrder = Literal["answer_first", "rationale_first", "rationale_first_2stage", "rationale_first_2stage_norat"]
JsonSingleShotOrder = Literal["answer_first", "rationale_first"]

DEFAULT_JSON_MAX_NEW_TOKENS = 256
DEFAULT_RATIONALE_MAX_NEW_TOKENS = 128
DEFAULT_STAGE2_MAX_NEW_TOKENS = 8

_ANSWER_PLACEHOLDER = "<single uppercase MC letter>"
_RATIONALE_PLACEHOLDER = "<brief medical rationale in 1-3 sentences>"

_JSON_SCHEMAS: dict[JsonSingleShotOrder, str] = {
    "answer_first": (
        f'{{"answer":"{_ANSWER_PLACEHOLDER}","rationale":"{_RATIONALE_PLACEHOLDER}"}}'
    ),
    "rationale_first": (
        f'{{"rationale":"{_RATIONALE_PLACEHOLDER}","answer":"{_ANSWER_PLACEHOLDER}"}}'
    ),
}

_MC_JSON_INSTRUCTION_PREFIX = (
    "\n\nAnalyze the image and respond with ONLY valid JSON using this exact schema:\n"
)
_MC_JSON_INSTRUCTION_SUFFIX = "\nNo text before or after the JSON."

_STAGE1_RATIONALE_SUFFIX = (
    "\n\nAnalyze the image and provide a brief medical rationale (1-3 sentences) "
    "explaining what visual findings are relevant to this question. "
    "Do not output an answer letter."
)


def is_json_single_shot(json_order: str) -> bool:
    return json_order in ("answer_first", "rationale_first")


def format_mc_question_options(question: str, options: Iterable[str]) -> str:
    """Question + MC options block without the answer-letter instruction."""
    option_list = [str(opt).strip() for opt in options]
    if len(option_list) < 2:
        raise ValueError("MC question requires at least 2 options")
    if len(option_list) > len(LETTER_POOL):
        raise ValueError("Too many options for letter mapping")
    option_lines = [f"{LETTER_POOL[i]}) {opt}" for i, opt in enumerate(option_list)]
    return f"{str(question).strip()}\n\nOptions:\n" + "\n".join(option_lines)


def build_stage1_rationale_prompt(question: str, options: List[str]) -> str:
    return format_mc_question_options(question, options) + _STAGE1_RATIONALE_SUFFIX


def build_stage2_answer_prompt(question: str, options: List[str], rationale: str) -> str:
    return (
        "You have analyzed this medical image and provided the following rationale:\n"
        f"{rationale.strip()}\n\n"
        "Based on this rationale, answer the following multiple-choice question.\n\n"
        f"{format_mc_user_text(question, options)}"
    )


def build_stage2_json_prompt(question: str, options: List[str], reasoning: str) -> str:
    """Stage-2 prompt for 2-stage CoT: inject stage-1 reasoning, request JSON {rationale, answer}."""
    return (
        "You have analyzed this medical image and provided the following reasoning:\n"
        f"{reasoning.strip()}\n\n"
        "Based on this reasoning, answer the following multiple-choice question "
        "and explain your conclusion.\n\n"
        + format_mc_user_text(question, options)
        + build_mc_json_instruction(json_order="rationale_first")
    )


def build_mc_json_instruction(*, json_order: JsonSingleShotOrder) -> str:
    """Return the JSON schema instruction suffix for the given field order."""
    return (
        _MC_JSON_INSTRUCTION_PREFIX
        + _JSON_SCHEMAS[json_order]
        + _MC_JSON_INSTRUCTION_SUFFIX
    )


def build_mc_json_user_text(
    question: str,
    options: List[str],
    *,
    json_order: JsonSingleShotOrder,
) -> str:
    """MC question block plus JSON output instructions."""
    return format_mc_user_text(question, options) + build_mc_json_instruction(json_order=json_order)


def _qwen_image_conversation(user_text: str, image: Any = None) -> List[dict]:
    image_content: dict = {"type": "image"}
    if image is not None:
        image_content["image"] = image
    return [
        {
            "role": "user",
            "content": [image_content, {"type": "text", "text": user_text}],
        }
    ]


def build_qwen_json_conversation(
    question: str,
    options: List[str],
    *,
    json_order: JsonSingleShotOrder,
    image: Any = None,
) -> List[dict]:
    """Qwen chat messages: user image + MC/JSON prompt (no assistant turn)."""
    full_question = build_mc_json_user_text(question, options, json_order=json_order)
    return _qwen_image_conversation(full_question, image=image)


def build_qwen_stage1_conversation(
    question: str,
    options: List[str],
    *,
    image: Any = None,
) -> List[dict]:
    return _qwen_image_conversation(
        build_stage1_rationale_prompt(question, options), image=image
    )


def build_qwen_stage2_conversation(
    question: str,
    options: List[str],
    *,
    rationale: str,
    image: Any = None,
) -> List[dict]:
    return _qwen_image_conversation(
        build_stage2_answer_prompt(question, options, rationale), image=image
    )


def build_qwen_stage2_json_conversation(
    question: str,
    options: List[str],
    *,
    reasoning: str,
    image: Any = None,
) -> List[dict]:
    """Stage-2 conversation for 2-stage CoT: inject stage-1 reasoning, request JSON output."""
    return _qwen_image_conversation(
        build_stage2_json_prompt(question, options, reasoning), image=image
    )


@dataclass
class McAnswerRationaleParseResult:
    predicted_letter: Optional[str]
    rationale_text: Optional[str]
    json_parse_ok: bool


def _strip_optional_json_fence(text: str) -> str:
    s = (text or "").strip()
    fence = re.match(r"^\s*```(?:json)?\s*\r?\n(.*?)\r?\n```\s*$", s, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return s


def parse_mc_answer_rationale_json(raw: str, num_options: int) -> McAnswerRationaleParseResult:
    """Parse {\"answer\", \"rationale\"} from model output; fallback to letter extraction."""
    stripped = _strip_optional_json_fence(raw)
    obj = None

    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        if start == -1:
            letter = parse_predicted_letter(stripped, num_options=num_options)
            return McAnswerRationaleParseResult(
                predicted_letter=letter, rationale_text=None, json_parse_ok=False
            )
        try:
            obj, _end = json.JSONDecoder().raw_decode(stripped[start:])
        except json.JSONDecodeError:
            letter = parse_predicted_letter(stripped, num_options=num_options)
            return McAnswerRationaleParseResult(
                predicted_letter=letter, rationale_text=None, json_parse_ok=False
            )

    if not isinstance(obj, dict):
        letter = parse_predicted_letter(stripped, num_options=num_options)
        return McAnswerRationaleParseResult(
            predicted_letter=letter, rationale_text=None, json_parse_ok=False
        )

    rationale_val = obj.get("rationale")
    rationale_txt = rationale_val.strip() if isinstance(rationale_val, str) else None

    ans_val = obj.get("answer")
    letter: Optional[str] = None
    if ans_val is not None:
        if isinstance(ans_val, str):
            letter = parse_predicted_letter(ans_val, num_options=num_options)
            if letter is None and len(ans_val.strip()) == 1:
                c = ans_val.strip().upper()
                if c in LETTER_POOL[:num_options]:
                    letter = c

    json_ok = "rationale" in obj and "answer" in obj
    if letter is None:
        letter = parse_predicted_letter(stripped, num_options=num_options)

    return McAnswerRationaleParseResult(
        predicted_letter=letter,
        rationale_text=rationale_txt,
        json_parse_ok=json_ok,
    )


def get_answer_letter_token_ids(tokenizer, num_opts: int = 10) -> dict[int, str]:
    """Return {token_id: letter} for common single-token encodings of A–J."""
    mapping: dict[int, str] = {}
    for letter in LETTER_POOL[:num_opts]:
        for variant in (letter, f" {letter}", f"▁{letter}"):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                mapping[ids[0]] = letter
    return mapping


def find_answer_token_step(token_ids: list[int], answer_token_ids: dict[int, str]) -> int | None:
    """Return the index of the last answer-letter token (backwards search)."""
    for i in reversed(range(len(token_ids))):
        if token_ids[i] in answer_token_ids:
            return i
    return None


def cast_json_single_shot(json_order: str) -> JsonSingleShotOrder:
    if not is_json_single_shot(json_order):
        raise ValueError(f"Not a JSON single-shot order: {json_order!r}")
    return cast(JsonSingleShotOrder, json_order)


def normalize_generation_sequences(
    raw_sequences: torch.Tensor, input_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return (full_ids, generated_ids, prompt_len) across HF sequence output variants."""
    prompt_len = int(input_ids.shape[1])
    has_prompt_prefix = raw_sequences.shape[1] >= prompt_len and torch.equal(
        raw_sequences[:, :prompt_len], input_ids
    )
    if has_prompt_prefix:
        full_ids = raw_sequences
        generated_ids = raw_sequences[:, prompt_len:]
    else:
        full_ids = torch.cat([input_ids, raw_sequences], dim=1)
        generated_ids = raw_sequences
    return full_ids, generated_ids, prompt_len
