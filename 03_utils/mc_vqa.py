"""Utilities for multiple-choice medical VQA tasks."""

import re
from typing import Iterable, Optional

LETTER_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def correct_index_to_letter(correct_index: int) -> str:
    """Convert a zero-based option index to a letter label."""
    if correct_index < 0 or correct_index >= len(LETTER_POOL):
        raise ValueError(f"correct_index out of range: {correct_index}")
    return LETTER_POOL[correct_index]


def letter_to_index(letter: str) -> int:
    """Convert option letter to zero-based index."""
    value = (letter or "").strip().upper()
    if len(value) != 1 or value not in LETTER_POOL:
        raise ValueError(f"Invalid option letter: {letter!r}")
    return LETTER_POOL.index(value)


def format_mc_user_text(question: str, options: Iterable[str]) -> str:
    """Create MC prompt text with A/B/C... options and strict answer instruction."""
    option_list = [str(opt).strip() for opt in options]
    if len(option_list) < 2:
        raise ValueError("MC question requires at least 2 options")
    if len(option_list) > len(LETTER_POOL):
        raise ValueError("Too many options for letter mapping")

    option_lines = [f"{LETTER_POOL[i]}) {opt}" for i, opt in enumerate(option_list)]
    allowed = LETTER_POOL[: len(option_list)]
    instruction = (
        f"Answer with exactly one letter from {', '.join(allowed)}. "
        "Do not output any other text."
    )
    return f"{str(question).strip()}\n\nOptions:\n" + "\n".join(option_lines) + f"\n\n{instruction}"


def parse_predicted_letter(raw_answer: str, num_options: int) -> Optional[str]:
    """Extract the first valid option letter from model output."""
    if num_options < 1 or num_options > len(LETTER_POOL):
        raise ValueError(f"num_options out of range: {num_options}")
    text = (raw_answer or "").strip().upper()
    if not text:
        return None

    allowed = set(LETTER_POOL[:num_options])
    direct = text[:1]
    if direct in allowed:
        return direct

    match = re.search(r"\b([A-Z])\b", text)
    if match and match.group(1) in allowed:
        return match.group(1)

    punct_match = re.search(r"([A-Z])[^A-Z0-9]", text + " ")
    if punct_match and punct_match.group(1) in allowed:
        return punct_match.group(1)

    return None
