"""VLM-driven selection of relevant SoM segments.

Instead of scoring regions with an external model, this asks the VLM under
test directly: given the image with all numbered SoM segments drawn, which
ones are relevant to answering the question? The model's own answer (a
list of region numbers) is parsed and used to filter the SoM overlay
before the main answering step, which is then nudged to refer back to the
selected region numbers in its explanation.
"""

from __future__ import annotations

import re
from typing import Sequence

DEFAULT_SELECTION_MAX_NEW_TOKENS = 32

SEGMENT_SELECTION_PROMPT_TEMPLATE = (
    "The image has numbered regions outlined in different colors.\n"
    "Question: {question}\n\n"
    "Which numbered regions are relevant for answering this question? "
    'Respond with ONLY the region numbers, comma-separated (e.g. "2, 5"). '
    'If none of the marked regions are relevant, respond with "none".'
)

SOM_ANSWER_HINT = (
    "\n\nThe numbered colored outlines in the image mark the regions you identified "
    "as relevant to this question. Base your answer on these marked regions and "
    'explicitly cite their numbers in your explanation (e.g. "region 3 shows...").'
)


SOM_SELECTED_HINT_TEMPLATE = (
    "\n\nThe numbered colored outlines in the image mark the regions ({labels}) that "
    "were pre-selected as relevant to this question. Base your answer on these "
    "marked regions and explicitly cite their numbers in your explanation "
    '(e.g. "region 3 shows...").'
)


def build_segment_selection_prompt(question: str) -> str:
    return SEGMENT_SELECTION_PROMPT_TEMPLATE.format(question=question.strip())


def question_with_som_hint(question: str) -> str:
    return f"{question}{SOM_ANSWER_HINT}"


def question_with_selected_som_hint(question: str, labels: Sequence[str]) -> str:
    """Append the hint used with pre-computed segment selection.

    Training and evaluation have to call this identically. A model fine-tuned on
    hinted prompts but evaluated on unhinted ones (or the reverse) reports the
    prompt shift as much as the effect of the marks themselves.
    """
    if not labels:
        return question
    listed = ", ".join(str(label) for label in labels)
    return f"{question}{SOM_SELECTED_HINT_TEMPLATE.format(labels=listed)}"


def parse_selected_labels(response_text: str, valid_labels: set[str]) -> list[str]:
    """Extract valid region-number tokens from the model's free-text response.

    Returns labels in first-seen order, de-duplicated, restricted to *valid_labels*.
    """
    found = re.findall(r"\d+", response_text or "")
    seen: set[str] = set()
    ordered: list[str] = []
    for label in found:
        if label in valid_labels and label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered
