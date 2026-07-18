"""Modular answer and calibration losses for iterative calibration fine-tuning."""

from __future__ import annotations

import math
from typing import Any, Callable, Protocol, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.qwen_vl import LABEL_IGNORE_INDEX


class AnswerLoss(Protocol):
    """Compute supervised answer loss from model outputs and labels."""

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        ...


class CalibrationLoss(Protocol):
    """Compute calibration loss from model outputs and per-sample targets."""

    def __call__(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        calibration_targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        ...


def cross_entropy_answer_loss(logits: torch.Tensor, labels: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Standard masked cross-entropy on assistant tokens (labels != -100)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=LABEL_IGNORE_INDEX,
    )


class CrossEntropyAnswerLoss:
    """Callable wrapper for cross-entropy answer loss."""

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return cross_entropy_answer_loss(logits, labels, **kwargs)


def extract_answer_confidence(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    answer_letter_token_ids: dict[int, str] | None = None,
    find_answer_token_step: Callable[[list[int], dict[int, str]], int | None] | None = None,
    use_reasoning_json: bool = False,
) -> torch.Tensor:
    """Return max softmax probability at the answer-token step for each batch item.

    Falls back to the last supervised label position when letter-token lookup is unavailable.
    """
    batch_size = logits.size(0)
    confidences: list[torch.Tensor] = []

    for batch_idx in range(batch_size):
        sample_labels = labels[batch_idx].tolist()
        supervised = [i for i, value in enumerate(sample_labels) if value != LABEL_IGNORE_INDEX]
        if not supervised:
            confidences.append(torch.tensor(0.0, device=logits.device, dtype=logits.dtype))
            continue

        step: int | None = None
        if use_reasoning_json and answer_letter_token_ids and find_answer_token_step is not None:
            step = find_answer_token_step(sample_labels, answer_letter_token_ids)
            if step is not None and step > 0:
                step -= 1

        if step is None:
            step = supervised[-1] - 1

        step = max(0, min(step, logits.size(1) - 1))
        step_logits = logits[batch_idx, step].float()
        probs = torch.softmax(step_logits, dim=-1)
        confidences.append(probs.max())

    return torch.stack(confidences)


class ConfidenceBCELoss:
    """Binary cross-entropy between predicted confidence and correctness target.

    Reads the answer-token softmax probability from the vocabulary logits — the same
    distribution the answer cross-entropy loss optimizes. For samples where
    calibration_target=0 (miscalibration strategies select wrong-but-selected samples too),
    this creates a direct gradient conflict with the answer loss at the answer-letter token:
    CE pushes P(ground_truth_letter) up while this BCE pushes max-prob at the same position
    down. Kept for comparison; prefer HeadConfidenceLoss for calibration fine-tuning where
    the answer format must stay intact.
    """

    def __init__(
        self,
        *,
        answer_letter_token_ids: dict[int, str] | None = None,
        find_answer_token_step: Callable[[list[int], dict[int, str]], int | None] | None = None,
        use_reasoning_json: bool = False,
    ):
        self.answer_letter_token_ids = answer_letter_token_ids
        self.find_answer_token_step = find_answer_token_step
        self.use_reasoning_json = use_reasoning_json

    def __call__(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        calibration_targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        confidences = extract_answer_confidence(
            logits,
            labels,
            answer_letter_token_ids=self.answer_letter_token_ids,
            find_answer_token_step=self.find_answer_token_step,
            use_reasoning_json=self.use_reasoning_json,
        )
        targets = calibration_targets.to(device=logits.device, dtype=confidences.dtype)
        return F.binary_cross_entropy(confidences.clamp(1e-6, 1.0 - 1e-6), targets)


def extract_answer_hidden_states(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    *,
    answer_letter_token_ids: dict[int, str] | None = None,
    find_answer_token_step: Callable[[list[int], dict[int, str]], int | None] | None = None,
) -> torch.Tensor:
    """Return the last-layer hidden state at the answer-token position for each batch item.

    Same position convention as extract_answer_confidence (one step before the answer-letter
    label, matching how logits[i] predicts labels[i+1]), so a head trained here reads the exact
    slot queried at inference via the mirrored generation-time step index.
    """
    batch_size = hidden_states.size(0)
    vectors: list[torch.Tensor] = []

    for batch_idx in range(batch_size):
        sample_labels = labels[batch_idx].tolist()
        supervised = [i for i, value in enumerate(sample_labels) if value != LABEL_IGNORE_INDEX]
        if not supervised:
            vectors.append(torch.zeros(hidden_states.size(-1), device=hidden_states.device, dtype=hidden_states.dtype))
            continue

        step: int | None = None
        if answer_letter_token_ids and find_answer_token_step is not None:
            step = find_answer_token_step(sample_labels, answer_letter_token_ids)
            if step is not None and step > 0:
                step -= 1

        if step is None:
            step = supervised[-1] - 1

        step = max(0, min(step, hidden_states.size(1) - 1))
        vectors.append(hidden_states[batch_idx, step])

    return torch.stack(vectors)


class AnswerConfidenceHead(nn.Module):
    """Predicts P(generated MC answer is correct) from the hidden state at the answer position.

    Attached as a submodule on the Qwen model (not held by the loss) so its parameters share
    the model's optimizer and gradients flow back into the backbone. Reading hidden states
    instead of vocabulary logits decouples the calibration signal from the answer-token
    softmax that the answer cross-entropy also optimizes, avoiding the gradient conflict
    ConfidenceBCELoss has at that exact token.
    """

    def __init__(self, hidden_size: int, bottleneck: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (batch, hidden_size) -> raw logits (batch,); apply sigmoid for probability."""
        return self.net(hidden.to(self.net[0].weight.dtype)).squeeze(-1)


class HeadConfidenceLoss:
    """BCE between a learned AnswerConfidenceHead's prediction and the correctness target.

    Operates on hidden states, not vocabulary logits, so it does not compete with the answer
    cross-entropy loss for the same output distribution at the answer-letter token.
    """

    def __init__(
        self,
        confidence_head: AnswerConfidenceHead,
        *,
        answer_letter_token_ids: dict[int, str] | None = None,
        find_answer_token_step: Callable[[list[int], dict[int, str]], int | None] | None = None,
    ):
        self.confidence_head = confidence_head
        self.answer_letter_token_ids = answer_letter_token_ids
        self.find_answer_token_step = find_answer_token_step

    def __call__(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        calibration_targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        answer_hidden = extract_answer_hidden_states(
            hidden_states,
            labels,
            answer_letter_token_ids=self.answer_letter_token_ids,
            find_answer_token_step=self.find_answer_token_step,
        )
        head_logits = self.confidence_head(answer_hidden)
        targets = calibration_targets.to(device=head_logits.device, dtype=head_logits.dtype)
        return F.binary_cross_entropy_with_logits(head_logits, targets)


class CombinedCalibrationLoss:
    """Combine answer loss and weighted calibration loss."""

    def __init__(
        self,
        answer_loss: AnswerLoss,
        calibration_loss: CalibrationLoss,
        lambda_cal: float,
    ):
        self.answer_loss = answer_loss
        self.calibration_loss = calibration_loss
        self.lambda_cal = lambda_cal

    def __call__(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        calibration_targets: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        answer = self.answer_loss(logits, labels, **kwargs)
        calibration = self.calibration_loss(logits, hidden_states, labels, calibration_targets, **kwargs)
        total = answer + self.lambda_cal * calibration
        return total, answer, calibration


def compute_ece(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    *,
    n_bins: int = 15,
) -> float | None:
    """Expected Calibration Error via uniform binning over [0, 1]."""
    if len(confidences) != len(correctness):
        return None

    pairs = [
        (float(conf), float(corr))
        for conf, corr in zip(confidences, correctness)
        if isinstance(conf, (int, float)) and not math.isnan(conf)
    ]
    if len(pairs) < 2:
        return None

    effective_bins = min(n_bins, len(pairs))
    total = len(pairs)
    ece = 0.0
    for bin_idx in range(effective_bins):
        low = bin_idx / effective_bins
        high = (bin_idx + 1) / effective_bins
        bucket = [
            (conf, corr)
            for conf, corr in pairs
            if low <= conf < high or (bin_idx == effective_bins - 1 and conf == 1.0)
        ]
        if not bucket:
            continue
        avg_conf = sum(conf for conf, _ in bucket) / len(bucket)
        avg_acc = sum(corr for _, corr in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(avg_acc - avg_conf)
    return ece


def build_losses(
    answer_loss_name: str,
    calibration_loss_name: str,
    lambda_cal: float,
    *,
    answer_letter_token_ids: dict[int, str] | None = None,
    find_answer_token_step: Callable[[list[int], dict[int, str]], int | None] | None = None,
    use_reasoning_json: bool = False,
    confidence_head: AnswerConfidenceHead | None = None,
) -> CombinedCalibrationLoss:
    """Factory for combined answer + calibration loss."""
    answer_registry: dict[str, AnswerLoss] = {
        "cross_entropy": CrossEntropyAnswerLoss(),
    }
    calibration_registry: dict[str, CalibrationLoss] = {
        "confidence_bce": ConfidenceBCELoss(
            answer_letter_token_ids=answer_letter_token_ids,
            find_answer_token_step=find_answer_token_step,
            use_reasoning_json=use_reasoning_json,
        ),
    }
    if confidence_head is not None:
        calibration_registry["confidence_head"] = HeadConfidenceLoss(
            confidence_head,
            answer_letter_token_ids=answer_letter_token_ids,
            find_answer_token_step=find_answer_token_step,
        )

    if answer_loss_name not in answer_registry:
        raise ValueError(
            f"Unknown answer loss {answer_loss_name!r}. Choices: {sorted(answer_registry)}"
        )
    if calibration_loss_name not in calibration_registry:
        raise ValueError(
            f"Unknown calibration loss {calibration_loss_name!r}. Choices: {sorted(calibration_registry)}"
        )

    return CombinedCalibrationLoss(
        answer_loss=answer_registry[answer_loss_name],
        calibration_loss=calibration_registry[calibration_loss_name],
        lambda_cal=lambda_cal,
    )
