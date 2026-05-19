"""Shared helpers for MedVLM-R1 (Qwen2-VL-2B) off-the-shelf MC inference."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, List, Optional

from qwen_vl_utils import process_vision_info

from utils.mc_vqa import LETTER_POOL, parse_predicted_letter

DEFAULT_MODEL_PATH = "JZPeterPan/MedVLM-R1"

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


def build_medvlm_conversation(image: Any, prompt_text: str) -> List[dict]:
    """Qwen2-VL chat messages with image + MedVLM prompt."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
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
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
