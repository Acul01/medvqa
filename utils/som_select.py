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

DEFAULT_SELECTION_MAX_NEW_TOKENS = 32

SEGMENT_SELECTION_PROMPT_TEMPLATE = (
    "The image has numbered regions outlined in different colors.\n"
    "Question: {question}\n\n"
    "Which numbered regions are relevant for answering this question? "
    'Respond with ONLY the region numbers, comma-separated (e.g. "2, 5"). '
    'If none of the marked regions are relevant, respond with "none".'
)

SOM_ANSWER_HINT = (
    "\n\nNote: numbered colored outlines mark the regions identified as relevant "
    "to this question. Refer to these region numbers in your explanation where helpful."
)


def build_segment_selection_prompt(question: str) -> str:
    return SEGMENT_SELECTION_PROMPT_TEMPLATE.format(question=question.strip())


def question_with_som_hint(question: str) -> str:
    return f"{question}{SOM_ANSWER_HINT}"


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
