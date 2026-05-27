"""Shared REGION/OBSERVATION/REASONING helpers for MC medical VQA (inference + data gen + training)."""

from __future__ import annotations

import re
from typing import Dict, Tuple

END_REASONING_TOKEN = "<END_REASONING>"


def build_reasoning_prompt(
    question: str,
    *,
    correct_answer: str,
) -> str:
    """Build the prompt sent to GPT-4o-mini for pseudo-reasoning generation."""
    return (
        "You are a medical imaging expert analyzing an image and providing a reasoning for the correct answer.\n"
        f"Question: {question}\n"
        f"The correct answer is: {correct_answer}\n"
        "\nRespond in exactly this format:\n"
        "REGION: [anatomical region most relevant to this question]\n"
        "OBSERVATION: [what you observe in that region]\n"
        "REASONING: [why this observation explains the correct answer]\n\n"
        f"End your response with exactly this token: {END_REASONING_TOKEN}"
    )


def parse_reasoning_fields(text: str) -> Tuple[Dict[str, str], str]:
    source = text or ""
    marker_idx = source.find(END_REASONING_TOKEN)
    if marker_idx != -1:
        parse_source = source[:marker_idx]
        parse_mode = "best_effort_marker"
    else:
        parse_source = source
        parse_mode = "best_effort_no_marker"
    fields: Dict[str, str] = {"region": "unknown", "observation": "unknown", "reasoning": "unknown"}
    upper = parse_source.upper()
    markers = [("region", "REGION:"), ("observation", "OBSERVATION:"), ("reasoning", "REASONING:")]
    positions = []
    for key, marker in markers:
        idx = upper.find(marker)
        if idx != -1:
            positions.append((idx, key, marker))
    positions.sort(key=lambda x: x[0])
    for pos_idx, (start, key, marker) in enumerate(positions):
        content_start = start + len(marker)
        content_end = positions[pos_idx + 1][0] if pos_idx + 1 < len(positions) else len(parse_source)
        value = parse_source[content_start:content_end].strip()
        if value:
            fields[key] = re.sub(r"\s+", " ", value)
    return fields, parse_mode


def format_reasoning_block_for_training(region: str, observation: str, reasoning: str) -> str:
    """Three-line block matching stage-2 context (no END_REASONING token)."""
    return (
        f"REGION: {region}\n"
        f"OBSERVATION: {observation}\n"
        f"REASONING: {reasoning}"
    )
