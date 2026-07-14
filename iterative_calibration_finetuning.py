"""Iterative calibration fine-tuning pipeline for Qwen2.5-VL on Medical VQA."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

from calibration_losses import AnswerConfidenceHead, build_losses, compute_ece
from utils.load_data import MedicalVQALoader, VQASample
from utils.mc_eval import run_mc_evaluation
from utils.mc_vqa import correct_index_to_letter, parse_predicted_letter
from utils.paths import get_data_root, get_mc_data_root, get_outputs_root
from utils.qwen_vl import (
    DEFAULT_MODEL_PATH,
    QwenVLDataCollator,
    build_mc_conversation,
    build_mc_reasoning_conversation,
    configure_qwen_processor_pixels,
    mask_labels_assistant_only,
    prepare_qwen_inputs,
    resolve_processor_path,
    squeeze_batch_dim,
)
from utils.mc_json_zeroshot import (
    DEFAULT_JSON_MAX_NEW_TOKENS,
    JsonSingleShotOrder,
    build_mc_zeroshot_training_conversation,
    build_qwen_json_conversation,
    parse_mc_answer_rationale_json,
)
from utils.save_results import clean_answer, save_results

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REASONING_JSON_MAX_NEW_TOKENS = 320

CONFIDENCE_HEAD_FILE = "confidence_head.pt"
CONFIDENCE_HEAD_BOTTLENECK = 256

HOLDOUT_RECORDS_FILE = "holdout_records.json"
UNSEEN_POOL_FILE = "unseen_pool.json"
ACTIVE_POOL_FILE = "active_pool.json"
RUN_STATE_FILE = "run_state.json"
PARTIAL_INFERENCE_FILE = "unseen_inference_partial.json"
POOL_PARTIAL_INFERENCE_FILE = "pool_inference_partial.json"
RUN_STATE_VERSION = 1

REQUIRED_ROW_FIELDS = (
    "image_path",
    "question",
    "options",
    "correct_index",
)


def _load_script_module(name: str, relative_path: str) -> Any:
    """Load a repo script that lives outside importable package paths."""
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inference_qwen_module() -> Any:
    return _load_script_module("inference_qwen", "04_evaluation/inference_qwen.py")


def _risk_coverage_module() -> Any:
    return _load_script_module("risk_coverage_entropy", "04_evaluation/risk_coverage_entropy.py")


@dataclass
class CalibrationRecord:
    """Unified record for calibration fine-tuning iterations."""

    image_path: str
    question: str
    ground_truth_answer: str
    generated_answer: str
    generated_rationale: str | None
    confidence: float
    entropy: float
    is_correct: bool
    options: list[str]
    correct_index: int
    question_type: str
    ground_truth_letter: str | None = None
    original_row: dict[str, Any] = field(default_factory=dict)
    # rationale-first mode ({"reasoning","answer"}): correct + highest confidence
    rf_confidence: float = float("nan")
    rf_entropy: float = float("nan")
    rf_is_correct: bool = False
    rf_rationale: str | None = None
    rf_generated_answer: str = ""
    rf_predicted_letter: str | None = None
    # answer-first mode ({"answer","reasoning"}): correct + lowest entropy
    af_confidence: float = float("nan")
    af_entropy: float = float("nan")
    af_is_correct: bool = False
    af_rationale: str | None = None
    af_generated_answer: str = ""
    af_predicted_letter: str | None = None
    # which prompting mode this sample was selected from (active_pseudo training)
    selection_mode: str | None = None

    def to_inference_row(self) -> dict[str, Any]:
        """Serialize to inference-compatible JSON row."""
        predicted_letter = self._resolved_predicted_letter()
        row = dict(self.original_row)
        row.update(
            {
                "image_path": self.image_path,
                "question": self.question,
                "ground_truth_answer": self.ground_truth_answer,
                "predicted_answer": self.generated_answer,
                "predicted_letter": predicted_letter,
                "predicted_reasoning_text": self.generated_rationale,
                "confidence": self.confidence,
                "entropy": self.entropy,
                "is_correct": self.is_correct,
                "options": self.options,
                "correct_index": self.correct_index,
                "question_type": self.question_type,
                "ground_truth_letter": self.ground_truth_letter,
                "selection_mode": self.selection_mode,
                "rationale_first": {
                    "confidence": self.rf_confidence,
                    "entropy": self.rf_entropy,
                    "is_correct": self.rf_is_correct,
                    "predicted_answer": self.rf_generated_answer,
                    "predicted_reasoning_text": self.rf_rationale,
                    "predicted_letter": self.rf_predicted_letter,
                },
                "answer_first": {
                    "confidence": self.af_confidence,
                    "entropy": self.af_entropy,
                    "is_correct": self.af_is_correct,
                    "predicted_answer": self.af_generated_answer,
                    "predicted_reasoning_text": self.af_rationale,
                    "predicted_letter": self.af_predicted_letter,
                },
            }
        )
        return row

    def _resolved_predicted_letter(self) -> str | None:
        """Best-effort predicted letter for MC eval exports."""
        for letter in (self.af_predicted_letter, self.rf_predicted_letter):
            if isinstance(letter, str) and letter.strip():
                return letter.strip().upper()
        if self.generated_answer:
            return parse_predicted_letter(self.generated_answer, num_options=len(self.options))
        return None


def record_key(record: CalibrationRecord) -> tuple[str, str]:
    """Stable identifier for deduplication across iterations."""
    return (record.image_path, record.question.strip())


def record_from_inference_row(row: dict[str, Any]) -> CalibrationRecord:
    """Convert an inference JSON row into a CalibrationRecord."""
    for key in REQUIRED_ROW_FIELDS:
        if key not in row:
            raise ValueError(f"Missing required field {key!r} in input row.")

    options = list(row["options"])
    correct_index = int(row["correct_index"])
    gt_letter = row.get("ground_truth_letter")
    if not isinstance(gt_letter, str) or not gt_letter.strip():
        gt_letter = correct_index_to_letter(correct_index)

    generated_answer = row.get("generated_answer") or row.get("predicted_answer") or ""
    generated_rationale = row.get("generated_rationale") or row.get("predicted_reasoning_text")
    confidence = float(row.get("confidence", float("nan")))
    entropy = float(row.get("entropy", float("nan")))

    is_correct = row.get("is_correct")
    if is_correct is None:
        pred_letter = row.get("predicted_letter")
        is_correct = pred_letter == gt_letter if isinstance(pred_letter, str) else False
    is_correct = bool(is_correct)

    return CalibrationRecord(
        image_path=str(row["image_path"]),
        question=str(row["question"]),
        ground_truth_answer=str(row.get("ground_truth_answer", "")),
        generated_answer=str(generated_answer),
        generated_rationale=str(generated_rationale) if generated_rationale else None,
        confidence=confidence,
        entropy=entropy,
        is_correct=is_correct,
        options=options,
        correct_index=correct_index,
        question_type=str(row.get("question_type", "unknown")),
        ground_truth_letter=str(gt_letter).strip().upper(),
        original_row=dict(row),
        rf_confidence=float(row.get("rationale_first", {}).get("confidence", float("nan"))),
        rf_entropy=float(row.get("rationale_first", {}).get("entropy", float("nan"))),
        rf_is_correct=bool(row.get("rationale_first", {}).get("is_correct", False)),
        rf_rationale=row.get("rationale_first", {}).get("predicted_reasoning_text"),
        rf_generated_answer=str(row.get("rationale_first", {}).get("predicted_answer", "")),
        rf_predicted_letter=row.get("rationale_first", {}).get("predicted_letter"),
        af_confidence=float(row.get("answer_first", {}).get("confidence", float("nan"))),
        af_entropy=float(row.get("answer_first", {}).get("entropy", float("nan"))),
        af_is_correct=bool(row.get("answer_first", {}).get("is_correct", False)),
        af_rationale=row.get("answer_first", {}).get("predicted_reasoning_text"),
        af_generated_answer=str(row.get("answer_first", {}).get("predicted_answer", "")),
        af_predicted_letter=row.get("answer_first", {}).get("predicted_letter"),
        selection_mode=row.get("selection_mode"),
    )


def save_records_json(path: Path, records: list[CalibrationRecord]) -> None:
    """Persist calibration records as inference-compatible JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([record.to_inference_row() for record in records], handle, indent=2, ensure_ascii=False)


def load_records_json(path: Path) -> list[CalibrationRecord]:
    """Load calibration records from a JSON file."""
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [record_from_inference_row(row) for row in rows]


def _args_state_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Store resume-critical CLI settings in run_state.json."""
    return {
        "input_json": args.input_json,
        "model_path": args.model_path,
        "selection_strategy": args.selection_strategy,
        "dataset": args.dataset,
        "data_root": args.data_root,
        "seed": args.seed,
        "iterations": args.iterations,
        "holdout_ratio": args.holdout_ratio,
        "seed_size": args.seed_size,
        "seed_rationale_first": args.seed_rationale_first,
        "seed_answer_first": args.seed_answer_first,
        "select_rationale_first": args.select_rationale_first,
        "select_answer_first": args.select_answer_first,
        "n_wrong_low_entropy": args.n_wrong_low_entropy,
        "n_correct_high_entropy": args.n_correct_high_entropy,
    }


def save_run_state(
    output_dir: Path,
    *,
    next_iteration: int,
    current_model_path: str,
    args: argparse.Namespace,
) -> None:
    """Persist run progress so a later job can resume with --resume_from."""
    state = {
        "version": RUN_STATE_VERSION,
        "next_iteration": next_iteration,
        "current_model_path": current_model_path,
        "selection_strategy": args.selection_strategy,
        "args": _args_state_summary(args),
    }
    path = output_dir / RUN_STATE_FILE
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)


def load_run_state(output_dir: Path) -> dict[str, Any]:
    """Load run_state.json from a previous run."""
    path = output_dir / RUN_STATE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No {RUN_STATE_FILE} in {output_dir}. Pass --output_dir for a fresh run, "
            f"or --resume_from with an existing output directory."
        )
    with open(path, encoding="utf-8") as handle:
        state = json.load(handle)
    return state


def bootstrap_run_state(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Create run_state.json from checkpoint artifacts when missing (legacy runs)."""
    state_path = output_dir / RUN_STATE_FILE
    if state_path.exists():
        return load_run_state(output_dir)

    holdout_path = output_dir / HOLDOUT_RECORDS_FILE
    if not holdout_path.exists():
        raise FileNotFoundError(
            f"Cannot resume {output_dir}: missing {RUN_STATE_FILE} and {HOLDOUT_RECORDS_FILE}. "
            "Runs started before resume support cannot be continued; start a new run with the same "
            f"--output_dir after pulling the resume update."
        )

    next_iteration = 0
    current_model_path = args.model_path
    for iteration in range(args.iterations):
        iter_dir = output_dir / f"iter_{iteration}"
        if not _iteration_complete(iter_dir):
            break
        with open(iter_dir / "metrics.json", encoding="utf-8") as handle:
            completed_metrics = json.load(handle)
        current_model_path = completed_metrics.get("model_path", current_model_path)
        next_iteration = iteration + 1

    state = {
        "version": RUN_STATE_VERSION,
        "next_iteration": next_iteration,
        "current_model_path": current_model_path,
        "selection_strategy": args.selection_strategy,
        "args": _args_state_summary(args),
    }
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)
    print(f"Bootstrapped {RUN_STATE_FILE} (next_iteration={next_iteration})")
    return state


def _validate_resume_args(args: argparse.Namespace, state: dict[str, Any]) -> None:
    """Warn when resumed CLI settings diverge from the saved run."""
    saved = state.get("args", {})
    checks = (
        "input_json",
        "selection_strategy",
        "dataset",
        "seed",
        "iterations",
    )
    for key in checks:
        if key in saved and getattr(args, key, None) != saved[key]:
            print(
                f"Warning: --{key}={getattr(args, key)!r} differs from saved "
                f"{saved[key]!r} in {RUN_STATE_FILE}"
            )


def _load_metrics_history(output_dir: Path) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    path = output_dir / "metrics_history.jsonl"
    if not path.exists():
        return history
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                history.append(json.loads(line))
    return history


def _iteration_complete(iter_dir: Path) -> bool:
    return (iter_dir / "metrics.json").exists()


def _has_trained_model(iter_dir: Path) -> bool:
    if (iter_dir / "model.safetensors").exists() or (iter_dir / "pytorch_model.bin").exists():
        return True
    return any(iter_dir.glob("model-*.safetensors"))


def _record_has_dual_inference(record: CalibrationRecord) -> bool:
    return math.isfinite(record.rf_confidence) and math.isfinite(record.af_confidence)


def _record_has_single_inference(record: CalibrationRecord) -> bool:
    return math.isfinite(record.confidence) and math.isfinite(record.entropy)


def _load_train_metrics(iter_dir: Path) -> dict[str, float]:
    path = iter_dir / "train_metrics.json"
    if not path.exists():
        return {"answer_loss": 0.0, "calibration_loss": 0.0, "train_loss": 0.0}
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "answer_loss": float(payload.get("answer_loss", 0.0)),
        "calibration_loss": float(payload.get("calibration_loss", 0.0)),
        "train_loss": float(payload.get("train_loss", 0.0)),
    }


def training_rationale(record: CalibrationRecord) -> str | None:
    """Return the pseudo rationale to use for training based on selection mode."""
    if record.selection_mode == "rationale_first":
        return record.rf_rationale
    if record.selection_mode == "answer_first":
        return record.af_rationale
    return record.generated_rationale


def uses_zeroshot_json(args: argparse.Namespace) -> bool:
    """entropy_calibration uses free-form rationale JSON (mc_json_zeroshot), not GPT REGION schema."""
    return args.selection_strategy == "entropy_calibration"


def zeroshot_json_order(args: argparse.Namespace) -> JsonSingleShotOrder:
    return "answer_first" if args.answer_first else "rationale_first"


def _parse_inference_output(
    generated_text: str,
    num_opts: int,
    *,
    zeroshot_json: bool,
    inference_mod: Any,
) -> tuple[str | None, str | None]:
    """Decode MC letter and rationale text from model output."""
    if zeroshot_json:
        parsed = parse_mc_answer_rationale_json(generated_text, num_opts)
        return parsed.predicted_letter, parsed.rationale_text
    predicted_letter, pred_reason_text, _ = inference_mod.decode_mc_prediction(
        generated_text,
        num_options=num_opts,
        use_reasoning_json=True,
    )
    return predicted_letter, pred_reason_text


def _build_inference_conversation(
    sample: VQASample,
    image: Image.Image,
    *,
    use_reasoning: bool,
    answer_first: bool,
    zeroshot_json: bool,
) -> list[dict[str, Any]]:
    """Build inference conversation (zero-shot JSON or GPT-structured reasoning JSON)."""
    if not use_reasoning:
        return build_mc_conversation(sample.question, sample.mc_options or [], image=image)
    if zeroshot_json:
        json_order: JsonSingleShotOrder = "answer_first" if answer_first else "rationale_first"
        return build_qwen_json_conversation(
            sample.question,
            sample.mc_options or [],
            json_order=json_order,
            image=image,
        )
    from utils.qwen_vl import build_mc_reasoning_inference_conversation

    return build_mc_reasoning_inference_conversation(
        sample.question,
        sample.mc_options or [],
        image=image,
        answer_first=answer_first,
    )


def record_to_vqa_sample(record: CalibrationRecord, *, dataset: str = "calibration") -> VQASample:
    """Convert a calibration record into a VQASample for Qwen training."""
    metadata = dict(record.original_row)
    rationale = training_rationale(record)
    if rationale:
        metadata["pseudo_reasoning_raw"] = rationale
        metadata["qwen_verify"] = True

    return VQASample(
        dataset=dataset,
        split="calibration",
        image_path=record.image_path,
        question=record.question,
        answer=record.ground_truth_letter or correct_index_to_letter(record.correct_index),
        metadata=metadata,
        mc_options=record.options,
        mc_correct_index=record.correct_index,
        question_type=record.question_type,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for iterative calibration fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Iterative calibration fine-tuning for Qwen2.5-VL on Medical VQA"
    )
    parser.add_argument("--input_json", type=str, required=True, help="Pre-built inference JSON")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_dir", type=str, default=str(Path(get_outputs_root()) / "iter_calib"))
    parser.add_argument(
        "--dataset",
        type=str,
        default="unknown",
        help="Dataset slug (pathvqa, slake, vqarad, medvqa2019) for image path resolution and logging",
    )
    parser.add_argument(
        "--selection_strategy",
        type=str,
        default="entropy_calibration",
        choices=["entropy_calibration", "active_pseudo", "miscalibration"],
        help=(
            "Sample selection: entropy_calibration (50%% wrong+low entropy, 50%% correct+high entropy), "
            "active_pseudo (confident correct pseudo-labels), or miscalibration (ratio-based)"
        ),
    )
    parser.add_argument(
        "--selection_ratio",
        type=float,
        default=0.10,
        help="Fraction of pool for miscalibration strategy (ignored for active_pseudo)",
    )
    parser.add_argument(
        "--random_ratio",
        type=float,
        default=0.20,
        help="Random fraction within miscalibration selection (ignored for active_pseudo)",
    )
    parser.add_argument("--seed_size", type=int, default=100, help="Initial pseudo seed set size (active_pseudo)")
    parser.add_argument(
        "--seed_rationale_first",
        type=int,
        default=50,
        help="Seed samples from rationale-first (correct, highest confidence)",
    )
    parser.add_argument(
        "--seed_answer_first",
        type=int,
        default=50,
        help="Seed samples from answer-first (correct, lowest entropy)",
    )
    parser.add_argument(
        "--select_rationale_first",
        type=int,
        default=50,
        help="Per-iteration selection from rationale-first (correct, highest confidence)",
    )
    parser.add_argument(
        "--select_answer_first",
        type=int,
        default=50,
        help="Per-iteration selection from answer-first (correct, lowest entropy)",
    )
    parser.add_argument(
        "--n_wrong_low_entropy",
        type=int,
        default=50,
        help="entropy_calibration: overconfident wrong samples (lowest entropy)",
    )
    parser.add_argument(
        "--n_correct_high_entropy",
        type=int,
        default=50,
        help="entropy_calibration: underconfident correct samples (highest entropy)",
    )
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--lambda_cal", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument(
        "--holdout_ratio",
        type=float,
        default=0.20,
        help="Fraction held out for evaluation; 0 skips holdout eval and uses full dataset as pool",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agreement_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--use_reasoning", action="store_true", default=False)
    parser.add_argument("--answer_first", action="store_true", default=False)
    parser.add_argument("--answer_loss", type=str, default="cross_entropy")
    parser.add_argument(
        "--calibration_loss",
        type=str,
        default="confidence_bce",
        choices=["confidence_bce", "confidence_head"],
        help=(
            "confidence_bce: BCE on the answer-letter softmax probability (same distribution "
            "the answer cross-entropy optimizes -- conflicts with it whenever a sample has "
            "calibration_target=0, degrading answer-format quality). "
            "confidence_head: BCE via a small learned head on the answer-token hidden state "
            "(decoupled from the vocabulary logits, avoids that conflict). Persisted as "
            "confidence_head.pt alongside the model checkpoint and warm-started across "
            "iterations like the model weights."
        ),
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing during training (reduces GPU memory)",
    )
    parser.add_argument(
        "--use_8bit_optimizer",
        action="store_true",
        help="Use 8-bit AdamW via bitsandbytes (reduces optimizer memory)",
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--data_root", type=str, default=get_data_root())
    parser.add_argument("--mc_data_root", type=str, default=get_mc_data_root())
    parser.add_argument("--max_pool_samples", type=int, default=None, help="Debug cap on pool size")
    parser.add_argument("--max_holdout_samples", type=int, default=None, help="Debug cap on holdout size")
    parser.add_argument("--skip_train", action="store_true", help="Run selection/eval only (debug)")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume an interrupted run from this output_dir (must contain run_state.json)",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=25,
        help="Save partial inference checkpoint every N samples (enables resume within an iteration)",
    )
    return parser.parse_args()


def resolve_record_image_paths(
    records: list[CalibrationRecord],
    *,
    data_root: str,
    dataset: str,
) -> list[CalibrationRecord]:
    """Resolve MC-style relative image paths via MedicalVQALoader (same as train/inference scripts)."""
    loader = MedicalVQALoader(data_root=data_root, mc_data_root=None)
    canonical = loader._normalize_dataset_name(dataset)
    config = loader.dataset_configs[canonical]

    resolved: list[CalibrationRecord] = []
    for record in records:
        image_path = str(loader._resolve_image_path(canonical, config, record.image_path))
        resolved.append(replace(record, image_path=image_path))
    return resolved


def load_dataset(
    path: str | Path,
    *,
    data_root: str | None = None,
    dataset: str | None = None,
) -> list[CalibrationRecord]:
    """Load and validate a pre-built inference or MC JSON dataset."""
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {json_path}")

    with open(json_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("input_json must contain a JSON list of samples.")

    records = [record_from_inference_row(row) for row in payload]
    if not records:
        raise ValueError("input_json contains no samples.")

    if dataset and dataset.strip().lower() != "unknown":
        root = data_root if data_root is not None else get_data_root()
        records = resolve_record_image_paths(records, data_root=root, dataset=dataset)
    elif not os.path.isabs(records[0].image_path) and not Path(records[0].image_path).exists():
        print(
            "Warning: relative image_path without --dataset; paths are used as-is. "
            "Pass --dataset vqarad (or set --data_root) to resolve MC-style paths."
        )
    return records


def split_dataset(
    records: list[CalibrationRecord],
    *,
    holdout_ratio: float,
    seed: int,
) -> tuple[list[CalibrationRecord], list[CalibrationRecord]]:
    """Split records into active-learning pool and holdout evaluation set."""
    if not 0.0 <= holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be in [0, 1).")

    if holdout_ratio == 0.0:
        return list(records), []

    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    holdout_count = max(1, round(holdout_ratio * len(records)))
    holdout_indices = set(indices[:holdout_count])
    pool = [records[i] for i in range(len(records)) if i not in holdout_indices]
    holdout = [records[i] for i in sorted(holdout_indices)]
    return pool, holdout


def compute_confidence_entropy(
    step_logits: list[torch.Tensor],
    generated_ids: torch.Tensor,
    *,
    use_reasoning_json: bool,
    num_opts: int,
    answer_letter_token_ids: dict[int, str],
) -> tuple[float, float]:
    """Compute mean confidence and entropy at the answer token."""
    inference_mod = _inference_qwen_module()
    return inference_mod._compute_answer_token_stats(
        step_logits,
        generated_ids,
        use_reasoning_json=use_reasoning_json,
        num_opts=num_opts,
        answer_letter_token_ids=answer_letter_token_ids,
    )


def compute_head_confidence_entropy(
    generation_hidden_states: Any,
    generated_ids: torch.Tensor,
    *,
    num_opts: int,
    answer_letter_token_ids: dict[int, str],
    confidence_head: Any,
) -> tuple[float, float]:
    """Compute (entropy, confidence) via the learned confidence head at the answer-token step.

    entropy is the binary entropy of the head's P(correct) output — monotonic in uncertainty
    (0 = certain either way, maximal at p=0.5) so it plugs directly into the existing
    AURC/risk-coverage code that expects a score_field="entropy" ranking.
    """
    inference_mod = _inference_qwen_module()
    if not generation_hidden_states:
        return float("nan"), float("nan")

    valid_answer_ids = {
        tid: letter for tid, letter in answer_letter_token_ids.items() if letter in "ABCDEFGHIJ"[:num_opts]
    }
    answer_step = inference_mod._find_answer_token_step(generated_ids.tolist(), valid_answer_ids)
    if answer_step is None or answer_step >= len(generation_hidden_states):
        return float("nan"), float("nan")

    hidden = generation_hidden_states[answer_step][-1][:, -1, :].float()
    confidence = float(torch.sigmoid(confidence_head(hidden)).item())
    confidence = min(max(confidence, 1e-6), 1.0 - 1e-6)
    entropy = -(confidence * math.log(confidence) + (1 - confidence) * math.log(1 - confidence))
    return entropy, confidence


def _move_inputs_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def load_qwen_model_and_processor(
    model_path: str,
    *,
    dtype: str = "bfloat16",
    base_model: str | None = None,
) -> tuple[Any, Any, torch.device]:
    """Load Qwen2.5-VL model and processor with LoRA/spatial-filter support."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved_path = os.path.abspath(model_path)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    load_dtype = dtype_map[dtype]

    is_local = os.path.isdir(resolved_path)
    is_lora = is_local and (Path(resolved_path) / "adapter_config.json").exists()

    _patches: list[tuple[Any, str, Any]] = []
    if is_local and not is_lora:
        try:
            import transformers.utils.hub as hub_mod

            original_cached_files = hub_mod.cached_files

            def safe_cached_files(path_or_repo_id=None, filenames=None, **kwargs):
                if (
                    path_or_repo_id is not None
                    and filenames is not None
                    and isinstance(path_or_repo_id, (str, os.PathLike))
                    and os.path.isdir(str(path_or_repo_id))
                ):
                    base = str(path_or_repo_id)
                    return [
                        (os.path.join(base, fn) if os.path.exists(os.path.join(base, fn)) else None)
                        for fn in filenames
                    ]
                return original_cached_files(path_or_repo_id=path_or_repo_id, filenames=filenames, **kwargs)

            hub_mod.cached_files = safe_cached_files
            _patches.append((hub_mod, "cached_files", original_cached_files))
        except (ImportError, AttributeError):
            pass

    try:
        if is_lora:
            from peft import PeftModel

            base_path = base_model or DEFAULT_MODEL_PATH
            base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_path, torch_dtype=load_dtype, device_map="auto"
            )
            model = PeftModel.from_pretrained(base, resolved_path)
            model = model.merge_and_unload()
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                resolved_path, torch_dtype=load_dtype, device_map="auto"
            )
    finally:
        for module, attr, original in _patches:
            setattr(module, attr, original)

    spatial_filter_ckpt = Path(resolved_path) / "spatial_filter.pt"
    if spatial_filter_ckpt.exists():
        from utils.spatial_filter import load_spatial_filter_checkpoint

        model = load_spatial_filter_checkpoint(model, str(spatial_filter_ckpt), bottleneck=64)

    confidence_head_ckpt = Path(resolved_path) / CONFIDENCE_HEAD_FILE
    if confidence_head_ckpt.exists():
        model = attach_confidence_head(model, resolved_path)

    processor_path = resolve_processor_path(base_model or resolved_path if is_lora else resolved_path)
    processor = AutoProcessor.from_pretrained(processor_path)
    configure_qwen_processor_pixels(processor)
    return model, processor, device


def attach_confidence_head(
    model: Any,
    model_path: str,
    *,
    bottleneck: int = CONFIDENCE_HEAD_BOTTLENECK,
) -> Any:
    """Attach an AnswerConfidenceHead submodule to `model`, warm-starting from
    confidence_head.pt in model_path if present, else fresh-initializing.
    """
    from utils.spatial_filter import get_visual_merger_output_dim

    hidden_size = get_visual_merger_output_dim(model)
    head = AnswerConfidenceHead(hidden_size, bottleneck=bottleneck)
    ckpt_path = Path(model_path) / CONFIDENCE_HEAD_FILE
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        head.load_state_dict(state)
        print(f"Confidence head loaded from {ckpt_path}")
    else:
        print("Confidence head freshly initialized (no confidence_head.pt found)")

    param = next(model.parameters())
    head.to(device=param.device, dtype=param.dtype)
    model.confidence_head = head
    return model


def _infer_single_mode(
    record: CalibrationRecord,
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    inference_mod: Any,
    answer_letter_token_ids: dict[int, str],
    answer_first: bool,
    agreement_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    zeroshot_json: bool = False,
) -> dict[str, Any]:
    """Run one MC reasoning inference pass (rationale-first or answer-first)."""
    sample = record_to_vqa_sample(record)
    image = Image.open(sample.image_path).convert("RGB")
    conversation = _build_inference_conversation(
        sample,
        image,
        use_reasoning=True,
        answer_first=answer_first,
        zeroshot_json=zeroshot_json,
    )
    inputs = prepare_qwen_inputs(processor, conversation, add_generation_prompt=True)
    inputs = _move_inputs_to_device(inputs, device)
    input_length = inputs["input_ids"].shape[1]

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    agreement_gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
    }

    generation_output = model.generate(
        **inputs,
        **gen_kwargs,
        return_dict_in_generate=True,
        output_logits=True,
    )
    generated_ids = generation_output.sequences[0][input_length:]
    generated_text = clean_answer(processor.decode(generated_ids, skip_special_tokens=True).strip())
    num_opts = len(sample.mc_options or [])
    predicted_letter, pred_reason_text = _parse_inference_output(
        generated_text,
        num_opts,
        zeroshot_json=zeroshot_json,
        inference_mod=inference_mod,
    )

    agreement_confidences: list[float] = []
    agreement_entropies: list[float] = []
    for _ in range(agreement_samples):
        sample_output = model.generate(
            **inputs,
            **agreement_gen_kwargs,
            return_dict_in_generate=True,
            output_logits=True,
        )
        sample_generated_ids = sample_output.sequences[0][input_length:]
        if sample_output.logits:
            entropy, confidence = compute_confidence_entropy(
                sample_output.logits,
                sample_generated_ids,
                use_reasoning_json=True,
                num_opts=num_opts,
                answer_letter_token_ids=answer_letter_token_ids,
            )
            agreement_entropies.append(entropy)
            agreement_confidences.append(confidence)

    confidence = inference_mod._mean_finite(agreement_confidences)
    entropy = inference_mod._mean_finite(agreement_entropies)
    if (not math.isfinite(confidence) or not math.isfinite(entropy)) and generation_output.logits:
        greedy_entropy, greedy_confidence = compute_confidence_entropy(
            generation_output.logits,
            generated_ids,
            use_reasoning_json=True,
            num_opts=num_opts,
            answer_letter_token_ids=answer_letter_token_ids,
        )
        if not math.isfinite(confidence):
            confidence = greedy_confidence
        if not math.isfinite(entropy):
            entropy = greedy_entropy
    ground_truth_letter = sample.answer
    is_correct = predicted_letter == ground_truth_letter

    return {
        "confidence": confidence,
        "entropy": entropy,
        "is_correct": is_correct,
        "rationale": pred_reason_text,
        "generated_answer": generated_text,
        "predicted_letter": predicted_letter,
        "ground_truth_letter": ground_truth_letter,
    }


def run_dual_mode_inference(
    records: list[CalibrationRecord],
    *,
    model_path: str,
    dtype: str,
    agreement_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    output_path: Path | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 25,
    zeroshot_json: bool = False,
) -> list[CalibrationRecord]:
    """Run rationale-first and answer-first inference; populate rf_* and af_* fields."""
    inference_mod = _inference_qwen_module()
    model, processor, device = load_qwen_model_and_processor(model_path, dtype=dtype)
    model.eval()

    answer_letter_token_ids = inference_mod._get_answer_letter_token_ids(processor.tokenizer)
    agreement_samples = max(1, agreement_samples)

    done_by_key: dict[tuple[str, str], CalibrationRecord] = {}
    if checkpoint_path is not None and checkpoint_path.exists():
        partial_records = load_records_json(checkpoint_path)
        done_by_key = {
            record_key(record): record
            for record in partial_records
            if _record_has_dual_inference(record)
        }
        if done_by_key:
            print(
                f"Resuming dual-mode inference from {checkpoint_path.name}: "
                f"{len(done_by_key)}/{len(records)} samples already done."
            )

    updated: list[CalibrationRecord] = []
    pending_since_checkpoint = 0
    with torch.no_grad():
        for record in tqdm(records, desc="Dual-mode inference", unit="sample"):
            key = record_key(record)
            if key in done_by_key:
                updated.append(done_by_key[key])
                continue

            rf = _infer_single_mode(
                record,
                model=model,
                processor=processor,
                device=device,
                inference_mod=inference_mod,
                answer_letter_token_ids=answer_letter_token_ids,
                answer_first=False,
                agreement_samples=agreement_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                zeroshot_json=zeroshot_json,
            )
            af = _infer_single_mode(
                record,
                model=model,
                processor=processor,
                device=device,
                inference_mod=inference_mod,
                answer_letter_token_ids=answer_letter_token_ids,
                answer_first=True,
                agreement_samples=agreement_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                zeroshot_json=zeroshot_json,
            )

            new_record = CalibrationRecord(
                image_path=record.image_path,
                question=record.question,
                ground_truth_answer=record.ground_truth_answer,
                generated_answer=af["generated_answer"],
                generated_rationale=af["rationale"],
                confidence=af["confidence"],
                entropy=af["entropy"],
                is_correct=af["is_correct"],
                options=record.options,
                correct_index=record.correct_index,
                question_type=record.question_type,
                ground_truth_letter=rf["ground_truth_letter"],
                original_row=record.original_row,
                rf_confidence=rf["confidence"],
                rf_entropy=rf["entropy"],
                rf_is_correct=rf["is_correct"],
                rf_rationale=rf["rationale"],
                rf_generated_answer=rf["generated_answer"],
                rf_predicted_letter=rf["predicted_letter"],
                af_confidence=af["confidence"],
                af_entropy=af["entropy"],
                af_is_correct=af["is_correct"],
                af_rationale=af["rationale"],
                af_generated_answer=af["generated_answer"],
                af_predicted_letter=af["predicted_letter"],
                selection_mode=record.selection_mode,
            )
            updated.append(new_record)
            done_by_key[key] = new_record
            pending_since_checkpoint += 1

            if (
                checkpoint_path is not None
                and checkpoint_every > 0
                and pending_since_checkpoint >= checkpoint_every
            ):
                save_records_json(checkpoint_path, updated)
                pending_since_checkpoint = 0

    if output_path is not None:
        save_records_json(output_path, updated)
        if checkpoint_path is not None and checkpoint_path.exists():
            checkpoint_path.unlink()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return updated


def run_inference(
    records: list[CalibrationRecord],
    *,
    model_path: str,
    dtype: str,
    use_reasoning: bool,
    answer_first: bool,
    agreement_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    output_path: Path | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 25,
    zeroshot_json: bool = False,
) -> list[CalibrationRecord]:
    """Run MC inference on records and refresh generated outputs and uncertainty."""
    inference_mod = _inference_qwen_module()

    model, processor, device = load_qwen_model_and_processor(model_path, dtype=dtype)
    model.eval()
    has_head = hasattr(model, "confidence_head")

    answer_letter_token_ids = inference_mod._get_answer_letter_token_ids(processor.tokenizer)
    agreement_samples = max(1, agreement_samples)
    agreement_gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "output_hidden_states": has_head,
    }
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False, "output_hidden_states": has_head}

    done_by_key: dict[tuple[str, str], CalibrationRecord] = {}
    if checkpoint_path is not None and checkpoint_path.exists():
        partial_records = load_records_json(checkpoint_path)
        done_by_key = {
            record_key(record): record
            for record in partial_records
            if _record_has_single_inference(record)
        }
        if done_by_key:
            print(
                f"Resuming inference from {checkpoint_path.name}: "
                f"{len(done_by_key)}/{len(records)} samples already done."
            )

    updated: list[CalibrationRecord] = []
    pending_since_checkpoint = 0
    with torch.no_grad():
        for record in tqdm(records, desc="Inference", unit="sample"):
            key = record_key(record)
            if key in done_by_key:
                updated.append(done_by_key[key])
                continue

            sample = record_to_vqa_sample(record)
            image = Image.open(sample.image_path).convert("RGB")
            conversation = _build_inference_conversation(
                sample,
                image,
                use_reasoning=use_reasoning,
                answer_first=answer_first,
                zeroshot_json=zeroshot_json,
            )
            inputs = prepare_qwen_inputs(processor, conversation, add_generation_prompt=True)
            inputs = _move_inputs_to_device(inputs, device)
            input_length = inputs["input_ids"].shape[1]

            generation_output = model.generate(
                **inputs,
                **gen_kwargs,
                return_dict_in_generate=True,
                output_logits=True,
            )
            generated_ids = generation_output.sequences[0][input_length:]
            generated_text = clean_answer(
                processor.decode(generated_ids, skip_special_tokens=True).strip()
            )
            num_opts = len(sample.mc_options or [])
            if use_reasoning:
                predicted_letter, pred_reason_text = _parse_inference_output(
                    generated_text,
                    num_opts,
                    zeroshot_json=zeroshot_json,
                    inference_mod=inference_mod,
                )
            else:
                predicted_letter, pred_reason_text, _ = inference_mod.decode_mc_prediction(
                    generated_text,
                    num_options=num_opts,
                    use_reasoning_json=False,
                )

            agreement_confidences: list[float] = []
            agreement_entropies: list[float] = []
            for _ in range(agreement_samples):
                sample_output = model.generate(
                    **inputs,
                    **agreement_gen_kwargs,
                    return_dict_in_generate=True,
                    output_logits=True,
                )
                sample_generated_ids = sample_output.sequences[0][input_length:]
                if has_head and sample_output.hidden_states:
                    entropy, confidence = compute_head_confidence_entropy(
                        sample_output.hidden_states,
                        sample_generated_ids,
                        num_opts=num_opts,
                        answer_letter_token_ids=answer_letter_token_ids,
                        confidence_head=model.confidence_head,
                    )
                    agreement_entropies.append(entropy)
                    agreement_confidences.append(confidence)
                elif not has_head and sample_output.logits:
                    entropy, confidence = compute_confidence_entropy(
                        sample_output.logits,
                        sample_generated_ids,
                        use_reasoning_json=use_reasoning,
                        num_opts=num_opts,
                        answer_letter_token_ids=answer_letter_token_ids,
                    )
                    agreement_entropies.append(entropy)
                    agreement_confidences.append(confidence)

            confidence = inference_mod._mean_finite(agreement_confidences)
            entropy = inference_mod._mean_finite(agreement_entropies)
            if not math.isfinite(confidence) or not math.isfinite(entropy):
                if has_head and generation_output.hidden_states:
                    greedy_entropy, greedy_confidence = compute_head_confidence_entropy(
                        generation_output.hidden_states,
                        generated_ids,
                        num_opts=num_opts,
                        answer_letter_token_ids=answer_letter_token_ids,
                        confidence_head=model.confidence_head,
                    )
                elif not has_head and generation_output.logits:
                    greedy_entropy, greedy_confidence = compute_confidence_entropy(
                        generation_output.logits,
                        generated_ids,
                        use_reasoning_json=use_reasoning,
                        num_opts=num_opts,
                        answer_letter_token_ids=answer_letter_token_ids,
                    )
                else:
                    greedy_entropy, greedy_confidence = float("nan"), float("nan")
                if not math.isfinite(confidence):
                    confidence = greedy_confidence
                if not math.isfinite(entropy):
                    entropy = greedy_entropy
            ground_truth_letter = sample.answer
            is_correct = predicted_letter == ground_truth_letter

            refreshed = CalibrationRecord(
                image_path=record.image_path,
                question=record.question,
                ground_truth_answer=record.ground_truth_answer,
                generated_answer=generated_text,
                generated_rationale=pred_reason_text if use_reasoning else record.generated_rationale,
                confidence=confidence,
                entropy=entropy,
                is_correct=is_correct,
                options=record.options,
                correct_index=record.correct_index,
                question_type=record.question_type,
                ground_truth_letter=ground_truth_letter,
                original_row=record.original_row,
                rf_predicted_letter=predicted_letter if use_reasoning and not answer_first else None,
                af_predicted_letter=predicted_letter if use_reasoning and answer_first else predicted_letter,
                rf_confidence=confidence,
                rf_entropy=entropy,
                rf_is_correct=is_correct,
                af_confidence=confidence,
                af_entropy=entropy,
                af_is_correct=is_correct,
            )
            updated.append(refreshed)
            done_by_key[key] = refreshed
            pending_since_checkpoint += 1

            if (
                checkpoint_path is not None
                and checkpoint_every > 0
                and pending_since_checkpoint >= checkpoint_every
            ):
                save_records_json(checkpoint_path, updated)
                pending_since_checkpoint = 0

    if output_path is not None:
        save_records_json(output_path, updated)
        if checkpoint_path is not None and checkpoint_path.exists():
            checkpoint_path.unlink()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return updated


def identify_calibration_errors(records: list[CalibrationRecord]) -> dict[str, Any]:
    """Flag miscalibrated samples using median splits within correct/wrong strata."""
    correct = [record for record in records if record.is_correct and math.isfinite(record.entropy)]
    wrong = [record for record in records if not record.is_correct and math.isfinite(record.confidence)]

    correct_entropy_median = (
        sorted(record.entropy for record in correct)[len(correct) // 2] if correct else float("nan")
    )
    wrong_confidence_median = (
        sorted(record.confidence for record in wrong)[len(wrong) // 2] if wrong else float("nan")
    )

    miscalibrated: list[CalibrationRecord] = []
    for record in records:
        if record.is_correct and math.isfinite(record.entropy) and record.entropy <= correct_entropy_median:
            miscalibrated.append(record)
        elif (
            not record.is_correct
            and math.isfinite(record.confidence)
            and record.confidence >= wrong_confidence_median
        ):
            miscalibrated.append(record)

    return {
        "n_miscalibrated": len(miscalibrated),
        "correct_entropy_median": correct_entropy_median,
        "wrong_confidence_median": wrong_confidence_median,
        "miscalibrated_fraction": len(miscalibrated) / len(records) if records else 0.0,
    }


def sample_calibration_dataset(
    pool: list[CalibrationRecord],
    *,
    selection_ratio: float,
    random_ratio: float,
    seed: int,
    iteration: int,
) -> tuple[list[CalibrationRecord], dict[str, Any]]:
    """Sample a calibration training set using stratified uncertainty selection."""
    if not pool:
        raise ValueError("Cannot sample from an empty pool.")
    if not 0.0 < selection_ratio <= 1.0:
        raise ValueError("selection_ratio must be in (0, 1].")
    if not 0.0 <= random_ratio < 1.0:
        raise ValueError("random_ratio must be in [0, 1).")

    rng = random.Random(seed + iteration)
    n_select = max(1, round(selection_ratio * len(pool)))
    n_random = round(random_ratio * n_select)
    n_stratified = n_select - n_random
    n_correct_high_entropy = n_stratified // 2
    n_wrong_high_confidence = n_stratified - n_correct_high_entropy

    correct_sorted = sorted(
        [record for record in pool if record.is_correct and math.isfinite(record.entropy)],
        key=lambda record: record.entropy,
        reverse=True,
    )
    wrong_sorted = sorted(
        [record for record in pool if not record.is_correct and math.isfinite(record.confidence)],
        key=lambda record: record.confidence,
        reverse=True,
    )

    selected: list[CalibrationRecord] = []
    selected.extend(correct_sorted[:n_correct_high_entropy])
    selected.extend(wrong_sorted[:n_wrong_high_confidence])

    selected_ids = {id(record) for record in selected}
    remaining = [record for record in pool if id(record) not in selected_ids]
    if n_random > 0 and remaining:
        n_random = min(n_random, len(remaining))
        selected.extend(rng.sample(remaining, n_random))

    n_selected = len(selected)
    stats = {
        "n_selected": n_selected,
        "n_pool": len(pool),
        "n_correct_high_entropy": min(n_correct_high_entropy, len(correct_sorted)),
        "n_wrong_high_confidence": min(n_wrong_high_confidence, len(wrong_sorted)),
        "n_random": n_selected
        - min(n_correct_high_entropy, len(correct_sorted))
        - min(n_wrong_high_confidence, len(wrong_sorted)),
        "fraction_correct_high_entropy": 0.0,
        "fraction_wrong_high_confidence": 0.0,
        "fraction_random": 0.0,
    }
    if n_selected:
        stats["fraction_correct_high_entropy"] = stats["n_correct_high_entropy"] / n_selected
        stats["fraction_wrong_high_confidence"] = stats["n_wrong_high_confidence"] / n_selected
        stats["fraction_random"] = stats["n_random"] / n_selected
    return selected, stats


def select_entropy_calibration(
    unseen: list[CalibrationRecord],
    *,
    n_wrong_low_entropy: int,
    n_correct_high_entropy: int,
) -> tuple[list[CalibrationRecord], list[CalibrationRecord], dict[str, Any]]:
    """Select miscalibrated samples: wrong+low entropy (overconfident) and correct+high
    entropy (underconfident).

    Sorts directly by confidence rather than entropy: for the confidence-head metric,
    entropy = binary_entropy(P(correct)) is symmetric around p=0 and p=1 (both "confidently
    wrong" and "correctly humble about being wrong" have low entropy), so entropy alone can't
    distinguish the miscalibration case we want from an already-well-calibrated one. Sorting
    wrong-records by confidence descending / correct-records by confidence ascending picks
    exactly "most overconfident wrong" / "most underconfident correct" without that ambiguity
    — which also matches the .md spec's original phrasing ("falsche Antwort + hohe Confidence").
    """
    wrong_sorted = sorted(
        [record for record in unseen if not record.is_correct and math.isfinite(record.confidence)],
        key=lambda record: record.confidence,
        reverse=True,
    )
    correct_sorted = sorted(
        [record for record in unseen if record.is_correct and math.isfinite(record.confidence)],
        key=lambda record: record.confidence,
    )

    wrong_selected = wrong_sorted[:n_wrong_low_entropy]
    correct_selected = correct_sorted[:n_correct_high_entropy]
    selected = wrong_selected + correct_selected

    selected_keys = {record_key(record) for record in selected}
    remaining = [record for record in unseen if record_key(record) not in selected_keys]

    n_selected = len(selected)
    stats = {
        "n_selected": n_selected,
        "n_wrong_low_entropy": len(wrong_selected),
        "n_correct_high_entropy": len(correct_selected),
        "n_unseen_remaining": len(remaining),
        "fraction_wrong_low_entropy": len(wrong_selected) / n_selected if n_selected else 0.0,
        "fraction_correct_high_entropy": len(correct_selected) / n_selected if n_selected else 0.0,
        "strategy": "entropy_calibration",
    }
    return selected, remaining, stats


def _tag_selected_record(record: CalibrationRecord, mode: str) -> CalibrationRecord:
    """Return a copy tagged with selection_mode for answer-first training."""
    rationale = record.rf_rationale if mode == "rationale_first" else record.af_rationale
    generated = record.rf_generated_answer if mode == "rationale_first" else record.af_generated_answer
    confidence = record.rf_confidence if mode == "rationale_first" else record.af_confidence
    entropy = record.rf_entropy if mode == "rationale_first" else record.af_entropy
    return CalibrationRecord(
        image_path=record.image_path,
        question=record.question,
        ground_truth_answer=record.ground_truth_answer,
        generated_answer=generated,
        generated_rationale=rationale,
        confidence=confidence,
        entropy=entropy,
        is_correct=True,
        options=record.options,
        correct_index=record.correct_index,
        question_type=record.question_type,
        ground_truth_letter=record.ground_truth_letter,
        original_row=record.original_row,
        rf_confidence=record.rf_confidence,
        rf_entropy=record.rf_entropy,
        rf_is_correct=record.rf_is_correct,
        rf_rationale=record.rf_rationale,
        rf_generated_answer=record.rf_generated_answer,
        rf_predicted_letter=record.rf_predicted_letter,
        af_confidence=record.af_confidence,
        af_entropy=record.af_entropy,
        af_is_correct=record.af_is_correct,
        af_rationale=record.af_rationale,
        af_generated_answer=record.af_generated_answer,
        af_predicted_letter=record.af_predicted_letter,
        selection_mode=mode,
    )


def _select_active_mode(
    records: list[CalibrationRecord],
    *,
    mode: str,
    count: int,
    exclude_keys: set[tuple[str, str]],
) -> list[CalibrationRecord]:
    """Select top-k correct pseudo samples for one prompting mode."""
    if mode == "rationale_first":
        candidates = [
            record
            for record in records
            if record.rf_is_correct and math.isfinite(record.rf_confidence)
        ]
        candidates.sort(key=lambda record: record.rf_confidence, reverse=True)
    elif mode == "answer_first":
        candidates = [
            record
            for record in records
            if record.af_is_correct and math.isfinite(record.af_entropy)
        ]
        candidates.sort(key=lambda record: record.af_entropy)
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    selected: list[CalibrationRecord] = []
    for record in candidates:
        key = record_key(record)
        if key in exclude_keys:
            continue
        selected.append(_tag_selected_record(record, mode))
        exclude_keys.add(key)
        if len(selected) >= count:
            break
    return selected


def build_seed_set(
    unseen: list[CalibrationRecord],
    *,
    seed_rationale_first: int,
    seed_answer_first: int,
) -> tuple[list[CalibrationRecord], list[CalibrationRecord], dict[str, Any]]:
    """Build initial pseudo seed set: correct + highest confidence (RF) and lowest entropy (AF)."""
    exclude_keys: set[tuple[str, str]] = set()
    rf_selected = _select_active_mode(
        unseen,
        mode="rationale_first",
        count=seed_rationale_first,
        exclude_keys=exclude_keys,
    )
    af_selected = _select_active_mode(
        unseen,
        mode="answer_first",
        count=seed_answer_first,
        exclude_keys=exclude_keys,
    )
    selected = rf_selected + af_selected
    selected_keys = {record_key(record) for record in selected}
    remaining = [record for record in unseen if record_key(record) not in selected_keys]

    stats = {
        "n_selected": len(selected),
        "n_rationale_first": len(rf_selected),
        "n_answer_first": len(af_selected),
        "n_unseen_remaining": len(remaining),
        "fraction_rationale_first": len(rf_selected) / len(selected) if selected else 0.0,
        "fraction_answer_first": len(af_selected) / len(selected) if selected else 0.0,
        "strategy": "active_pseudo_seed",
    }
    return selected, remaining, stats


def sample_active_calibration(
    unseen: list[CalibrationRecord],
    *,
    select_rationale_first: int,
    select_answer_first: int,
) -> tuple[list[CalibrationRecord], list[CalibrationRecord], dict[str, Any]]:
    """Select per-iteration pseudo samples from unseen pool (both modes, correct only)."""
    exclude_keys: set[tuple[str, str]] = set()
    rf_selected = _select_active_mode(
        unseen,
        mode="rationale_first",
        count=select_rationale_first,
        exclude_keys=exclude_keys,
    )
    af_selected = _select_active_mode(
        unseen,
        mode="answer_first",
        count=select_answer_first,
        exclude_keys=exclude_keys,
    )
    selected = rf_selected + af_selected
    selected_keys = {record_key(record) for record in selected}
    remaining = [record for record in unseen if record_key(record) not in selected_keys]

    stats = {
        "n_selected": len(selected),
        "n_rationale_first": len(rf_selected),
        "n_answer_first": len(af_selected),
        "n_unseen_remaining": len(remaining),
        "fraction_rationale_first": len(rf_selected) / len(selected) if selected else 0.0,
        "fraction_answer_first": len(af_selected) / len(selected) if selected else 0.0,
        "strategy": "active_pseudo",
    }
    return selected, remaining, stats


@dataclass
class CalibrationVLDataCollator(QwenVLDataCollator):
    """Extend Qwen collator to batch calibration targets."""

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, Any]:
        calibration_targets = torch.stack([inst.pop("calibration_target") for inst in instances])
        batch = super().__call__(instances)
        batch["calibration_target"] = calibration_targets
        return batch


class CalibrationTrainDataset(Dataset):
    """Qwen MC training dataset with per-sample calibration targets."""

    def __init__(
        self,
        records: list[CalibrationRecord],
        processor: Any,
        max_length: int = 512,
        *,
        use_reasoning: bool = False,
        answer_first: bool = False,
        zeroshot_json: bool = False,
        dataset_name: str = "calibration",
    ):
        self.records = records
        self.processor = processor
        self.max_length = max_length
        self.use_reasoning = use_reasoning
        self.answer_first = answer_first
        self.zeroshot_json = zeroshot_json
        self.dataset_name = dataset_name
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        sample = record_to_vqa_sample(record, dataset=self.dataset_name)
        image = Image.open(sample.image_path).convert("RGB")

        rationale = training_rationale(record)
        if self.use_reasoning and rationale:
            if self.zeroshot_json:
                json_order: JsonSingleShotOrder = (
                    "answer_first" if self.answer_first else "rationale_first"
                )
                conversation = build_mc_zeroshot_training_conversation(
                    sample.question,
                    sample.mc_options or [],
                    rationale_raw=rationale,
                    answer=sample.answer,
                    image=image,
                    json_order=json_order,
                )
            else:
                conversation = build_mc_reasoning_conversation(
                    sample.question,
                    sample.mc_options or [],
                    reasoning_raw=rationale,
                    answer=sample.answer,
                    image=image,
                    answer_first=self.answer_first,
                )
        else:
            conversation = build_mc_conversation(
                sample.question,
                sample.mc_options or [],
                answer=sample.answer,
                image=image,
            )

        batch = prepare_qwen_inputs(
            self.processor,
            conversation,
            add_generation_prompt=False,
            max_length=self.max_length,
            padding=False,
        )
        item = squeeze_batch_dim(batch)
        item["labels"] = mask_labels_assistant_only(self.processor, item["input_ids"], conversation)
        item["calibration_target"] = torch.tensor(float(record.is_correct), dtype=torch.float32)
        return item


def build_dataloader(
    records: list[CalibrationRecord],
    processor: Any,
    *,
    max_length: int,
    use_reasoning: bool,
    answer_first: bool,
    zeroshot_json: bool,
    dataset_name: str,
) -> tuple[CalibrationTrainDataset, CalibrationVLDataCollator]:
    """Build training dataset and collator for one calibration iteration."""
    train_dataset = CalibrationTrainDataset(
        records,
        processor,
        max_length=max_length,
        use_reasoning=use_reasoning,
        answer_first=answer_first,
        zeroshot_json=zeroshot_json,
        dataset_name=dataset_name,
    )
    collator = CalibrationVLDataCollator(tokenizer=processor.tokenizer)
    return train_dataset, collator


class CalibrationTrainer(Trainer):
    """Trainer with combined answer + calibration loss."""

    def __init__(self, combined_loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.combined_loss = combined_loss
        self._last_answer_loss = 0.0
        self._last_calibration_loss = 0.0
        self._answer_losses: list[float] = []
        self._calibration_losses: list[float] = []

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        calibration_targets = inputs.pop("calibration_target")
        labels = inputs.get("labels")
        outputs = model(**inputs, output_hidden_states=True)
        total, answer_loss, calibration_loss = self.combined_loss(
            outputs.logits,
            outputs.hidden_states[-1],
            labels,
            calibration_targets,
        )
        self._last_answer_loss = float(answer_loss.detach().item())
        self._last_calibration_loss = float(calibration_loss.detach().item())
        self._answer_losses.append(self._last_answer_loss)
        self._calibration_losses.append(self._last_calibration_loss)
        self.log(
            {
                "answer_loss": self._last_answer_loss,
                "calibration_loss": self._last_calibration_loss,
            }
        )
        return (total, outputs) if return_outputs else total


def build_losses_from_args(args: argparse.Namespace, processor: Any, model: Any) -> Any:
    """Build combined loss using inference helpers for answer-token lookup."""
    inference_mod = _inference_qwen_module()
    answer_letter_token_ids = inference_mod._get_answer_letter_token_ids(processor.tokenizer)
    return build_losses(
        args.answer_loss,
        args.calibration_loss,
        args.lambda_cal,
        answer_letter_token_ids=answer_letter_token_ids,
        find_answer_token_step=inference_mod._find_answer_token_step,
        use_reasoning_json=args.use_reasoning,
        confidence_head=getattr(model, "confidence_head", None),
    )


def train_iteration(
    selected_records: list[CalibrationRecord],
    *,
    model_path: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Fine-tune the model on the selected calibration subset (warm start)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    use_8bit_optimizer = args.use_8bit_optimizer
    if use_8bit_optimizer:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            print("Warning: bitsandbytes not installed, falling back to standard optimizer")
            use_8bit_optimizer = False

    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if args.calibration_loss == "confidence_head":
        model = attach_confidence_head(model, model_path)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    processor_path = resolve_processor_path(model_path)
    processor = AutoProcessor.from_pretrained(processor_path)
    configure_qwen_processor_pixels(processor)

    train_dataset, collator = build_dataloader(
        selected_records,
        processor,
        max_length=args.max_length,
        use_reasoning=args.use_reasoning,
        answer_first=args.answer_first,
        zeroshot_json=uses_zeroshot_json(args),
        dataset_name=args.dataset,
    )
    combined_loss = build_losses_from_args(args, processor, model)

    trainer = CalibrationTrainer(
        combined_loss,
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            save_strategy="epoch",
            report_to="none",
            seed=args.seed,
            remove_unused_columns=False,
            bf16=args.dtype == "bfloat16",
            fp16=args.dtype == "float16",
            optim="adamw_8bit" if use_8bit_optimizer else "adamw_torch",
        ),
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=processor.tokenizer,
    )

    train_result = trainer.train()

    confidence_head_state = None
    if args.calibration_loss == "confidence_head" and hasattr(model, "confidence_head"):
        # Detach before save_pretrained: an unregistered submodule would land in the
        # safetensors as unexpected keys that from_pretrained silently drops on reload
        # (same reasoning as spatial_filter.pt in 03_training/train_qwen_modified.py).
        confidence_head_state = model.confidence_head.state_dict()
        del model.confidence_head

    trainer.save_model()
    processor.save_pretrained(output_dir)

    if confidence_head_state is not None:
        torch.save(confidence_head_state, output_dir / CONFIDENCE_HEAD_FILE)
        print(f"Confidence head saved -> {output_dir / CONFIDENCE_HEAD_FILE}")

    answer_loss = (
        sum(trainer._answer_losses) / len(trainer._answer_losses)
        if trainer._answer_losses
        else float(train_result.training_loss)
    )
    calibration_loss = (
        sum(trainer._calibration_losses) / len(trainer._calibration_losses)
        if trainer._calibration_losses
        else 0.0
    )

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "train_loss": float(train_result.training_loss),
        "answer_loss": float(answer_loss),
        "calibration_loss": float(calibration_loss),
    }


def _apply_primary_eval_view(record: CalibrationRecord, *, primary: str = "answer_first") -> None:
    """Set top-level eval fields from the chosen prompting mode (in-place)."""
    if primary == "rationale_first":
        record.confidence = record.rf_confidence
        record.entropy = record.rf_entropy
        record.is_correct = record.rf_is_correct
        record.generated_answer = record.rf_generated_answer
        record.generated_rationale = record.rf_rationale
    else:
        record.confidence = record.af_confidence
        record.entropy = record.af_entropy
        record.is_correct = record.af_is_correct
        record.generated_answer = record.af_generated_answer
        record.generated_rationale = record.af_rationale


def _mode_metric_rows(records: list[CalibrationRecord], mode: str) -> list[dict[str, Any]]:
    """Build inference rows using one dual-mode pass for AURC / breakdown metrics."""
    rows: list[dict[str, Any]] = []
    for record in records:
        row = record.to_inference_row()
        if mode == "rationale_first":
            row["predicted_answer"] = record.rf_generated_answer
            row["predicted_letter"] = record.rf_predicted_letter
            row["predicted_reasoning_text"] = record.rf_rationale
            row["confidence"] = record.rf_confidence
            row["entropy"] = record.rf_entropy
            row["is_correct"] = record.rf_is_correct
        else:
            row["predicted_answer"] = record.af_generated_answer
            row["predicted_letter"] = record.af_predicted_letter
            row["predicted_reasoning_text"] = record.af_rationale
            row["confidence"] = record.af_confidence
            row["entropy"] = record.af_entropy
            row["is_correct"] = record.af_is_correct
        rows.append(row)
    return rows


def _accuracy_from_rows(rows: list[dict[str, Any]]) -> float:
    correct = 0
    total = 0
    for row in rows:
        gt = row.get("ground_truth_letter")
        if not isinstance(gt, str) or not gt.strip():
            idx = row.get("correct_index")
            if isinstance(idx, int) and idx >= 0:
                gt = correct_index_to_letter(idx)
            else:
                continue
        pred = row.get("predicted_letter")
        if not isinstance(pred, str) or not pred.strip():
            options = row.get("options")
            n = len(options) if isinstance(options, list) else 5
            pred = parse_predicted_letter(str(row.get("predicted_answer", "")), num_options=n)
        total += 1
        if pred == str(gt).strip().upper():
            correct += 1
    return correct / total if total else 0.0


def _mean_finite_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(row[field])
        for row in rows
        if isinstance(row.get(field), (int, float)) and math.isfinite(float(row[field]))
    ]
    return sum(values) / len(values) if values else None


def _count_finite_entropy(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if isinstance(row.get("entropy"), (int, float)) and math.isfinite(float(row["entropy"]))
    )


def _print_holdout_eval_summary(
    *,
    iteration: int,
    args: argparse.Namespace,
    model_path: str,
    holdout_path: Path,
    metrics: dict[str, Any],
    refreshed: list[CalibrationRecord],
    aurc_primary: float,
    aurc_by_mode: dict[str, float],
    ece: float | None,
) -> None:
    """Print a readable holdout evaluation summary."""
    eval_rows = [record.to_inference_row() for record in refreshed]
    n_valid_entropy = _count_finite_entropy(eval_rows)

    print("\n" + "=" * 72)
    print(f"Holdout Evaluation — iteration {iteration}")
    print("=" * 72)
    print(f"Dataset:           {args.dataset}")
    print(f"Model checkpoint:  {model_path}")
    print(f"Holdout samples:   {len(refreshed)}")
    print(f"Results JSON:      {holdout_path}")
    print(f"Valid entropy rows:{n_valid_entropy}/{len(refreshed)}")
    print("-" * 72)
    print(f"Primary ({'answer-first' if args.selection_strategy in ('active_pseudo', 'entropy_calibration') else 'configured'})")
    print(f"  Accuracy:        {metrics['overall']['accuracy']:.4f}")
    overall_entropy = metrics["overall"].get("entropy")
    overall_conf = metrics["overall"].get("confidence")
    if isinstance(overall_entropy, (int, float)) and math.isfinite(float(overall_entropy)):
        print(f"  Mean entropy:    {float(overall_entropy):.6f}")
    else:
        print("  Mean entropy:    N/A")
    if isinstance(overall_conf, (int, float)) and math.isfinite(float(overall_conf)):
        print(f"  Mean confidence: {float(overall_conf):.6f}")
    else:
        print("  Mean confidence: N/A")
    print(f"  AURC:            {aurc_primary:.6f}")
    print(f"  ECE:             {ece:.6f}" if ece is not None else "  ECE:             N/A")

    if args.selection_strategy == "active_pseudo":
        rf_rows = _mode_metric_rows(refreshed, "rationale_first")
        af_rows = _mode_metric_rows(refreshed, "answer_first")
        print("-" * 72)
        print("Dual-mode breakdown")
        print(f"  Rationale-first accuracy: {_accuracy_from_rows(rf_rows):.4f}")
        rf_entropy = _mean_finite_field(rf_rows, "entropy")
        rf_conf = _mean_finite_field(rf_rows, "confidence")
        print(
            f"  Rationale-first entropy/conf: "
            f"{rf_entropy:.6f} / {rf_conf:.6f}"
            if rf_entropy is not None and rf_conf is not None
            else "  Rationale-first entropy/conf: N/A"
        )
        print(f"  Rationale-first AURC:     {aurc_by_mode.get('rationale_first', float('nan')):.6f}")
        print(f"  Answer-first accuracy:    {_accuracy_from_rows(af_rows):.4f}")
        af_entropy = _mean_finite_field(af_rows, "entropy")
        af_conf = _mean_finite_field(af_rows, "confidence")
        print(
            f"  Answer-first entropy/conf: "
            f"{af_entropy:.6f} / {af_conf:.6f}"
            if af_entropy is not None and af_conf is not None
            else "  Answer-first entropy/conf: N/A"
        )
        print(f"  Answer-first AURC:        {aurc_by_mode.get('answer_first', float('nan')):.6f}")
    print("=" * 72)


def evaluate_iteration(
    holdout: list[CalibrationRecord],
    *,
    model_path: str,
    output_dir: Path,
    args: argparse.Namespace,
    iteration: int,
) -> dict[str, Any]:
    """Evaluate holdout accuracy, AURC, and ECE after an iteration."""
    max_tokens = args.max_new_tokens or (
        DEFAULT_JSON_MAX_NEW_TOKENS
        if uses_zeroshot_json(args)
        else DEFAULT_REASONING_JSON_MAX_NEW_TOKENS
        if args.selection_strategy in ("active_pseudo", "entropy_calibration") or args.use_reasoning
        else 8
    )

    zeroshot_json = uses_zeroshot_json(args)
    if args.selection_strategy == "active_pseudo":
        refreshed = run_dual_mode_inference(
            holdout,
            model_path=model_path,
            dtype=args.dtype,
            agreement_samples=args.agreement_samples,
            max_new_tokens=max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            output_path=None,
            zeroshot_json=zeroshot_json,
        )
        for record in refreshed:
            _apply_primary_eval_view(record, primary="answer_first")
    else:
        refreshed = run_inference(
            holdout,
            model_path=model_path,
            dtype=args.dtype,
            use_reasoning=args.use_reasoning,
            answer_first=args.answer_first,
            agreement_samples=args.agreement_samples,
            max_new_tokens=max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            output_path=None,
            zeroshot_json=zeroshot_json,
        )

    eval_rows = [record.to_inference_row() for record in refreshed]
    holdout_path = save_results(
        eval_rows,
        model_path=model_path,
        dataset=args.dataset,
        n_samples=len(refreshed),
        results_dir=output_dir,
        name_suffix=f"holdout_iter{iteration}",
    )

    metrics = run_mc_evaluation(holdout_path, metrics_output_path=output_dir / "holdout_eval.json")
    accuracy = float(metrics["overall"]["accuracy"])

    risk_mod = _risk_coverage_module()
    primary_rows = [record.to_inference_row() for record in refreshed]
    aurc_eval = risk_mod.evaluate_question_type(
        primary_rows,
        "overall",
        risk_metric="match",
        score_field="entropy",
        max_entropy=None,
        num_thresholds=200,
    )
    aurc = float(aurc_eval["aurc"])

    aurc_by_mode: dict[str, float] = {}
    if args.selection_strategy == "active_pseudo":
        for mode in ("rationale_first", "answer_first"):
            mode_rows = _mode_metric_rows(refreshed, mode)
            mode_eval = risk_mod.evaluate_question_type(
                mode_rows,
                "overall",
                risk_metric="match",
                score_field="entropy",
                max_entropy=None,
                num_thresholds=200,
            )
            aurc_by_mode[mode] = float(mode_eval["aurc"])

    confidences = [record.confidence for record in refreshed if math.isfinite(record.confidence)]
    correctness = [record.is_correct for record in refreshed if math.isfinite(record.confidence)]
    ece = compute_ece(confidences, correctness)

    _print_holdout_eval_summary(
        iteration=iteration,
        args=args,
        model_path=model_path,
        holdout_path=holdout_path,
        metrics=metrics,
        refreshed=refreshed,
        aurc_primary=aurc,
        aurc_by_mode=aurc_by_mode,
        ece=ece,
    )

    return {
        "accuracy": accuracy,
        "aurc": aurc,
        "aurc_rationale_first": aurc_by_mode.get("rationale_first"),
        "aurc_answer_first": aurc_by_mode.get("answer_first"),
        "ece": ece,
        "n_holdout": len(refreshed),
        "n_valid_entropy": _count_finite_entropy(primary_rows),
        "holdout_results_path": str(holdout_path),
        "mean_confidence": _mean_finite_field(primary_rows, "confidence"),
        "mean_entropy": _mean_finite_field(primary_rows, "entropy"),
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    """Run the iterative calibration fine-tuning loop."""
    args = parse_args()
    set_seed(args.seed)

    uses_unseen_pool = args.selection_strategy in ("active_pseudo", "entropy_calibration")

    if args.selection_strategy in ("active_pseudo", "entropy_calibration"):
        args.use_reasoning = True
        args.answer_first = True

    if args.selection_strategy == "active_pseudo":
        if args.seed_rationale_first + args.seed_answer_first != args.seed_size:
            print(
                f"Warning: seed_rationale_first ({args.seed_rationale_first}) + "
                f"seed_answer_first ({args.seed_answer_first}) != seed_size ({args.seed_size})"
            )
        if args.select_rationale_first + args.select_answer_first != args.seed_size:
            print(
                f"Warning: select_rationale_first ({args.select_rationale_first}) + "
                f"select_answer_first ({args.select_answer_first}) != seed_size ({args.seed_size})"
            )

    max_tokens = args.max_new_tokens or (
        DEFAULT_JSON_MAX_NEW_TOKENS
        if uses_zeroshot_json(args)
        else DEFAULT_REASONING_JSON_MAX_NEW_TOKENS
        if uses_unseen_pool or args.use_reasoning
        else 8
    )
    zeroshot_json = uses_zeroshot_json(args)

    if args.resume_from:
        output_dir = Path(args.resume_from)
        if not output_dir.is_dir():
            raise FileNotFoundError(f"--resume_from directory not found: {output_dir}")
        state = bootstrap_run_state(output_dir, args)
        _validate_resume_args(args, state)
        if state.get("selection_strategy") != args.selection_strategy:
            raise ValueError(
                f"selection_strategy mismatch: saved={state.get('selection_strategy')!r}, "
                f"cli={args.selection_strategy!r}"
            )

        holdout = load_records_json(output_dir / HOLDOUT_RECORDS_FILE)
        history = _load_metrics_history(output_dir)
        current_model_path = state["current_model_path"]
        start_iteration = int(state["next_iteration"])

        if uses_unseen_pool:
            unseen = load_records_json(output_dir / UNSEEN_POOL_FILE)
            pool: list[CalibrationRecord] = []
        else:
            pool_path = output_dir / ACTIVE_POOL_FILE
            pool = load_records_json(pool_path) if pool_path.exists() else []
            unseen = []
        print(
            f"Resuming from {output_dir} at iteration {start_iteration} "
            f"(model={current_model_path})"
        )
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        records = load_dataset(
            args.input_json,
            data_root=args.data_root,
            dataset=args.dataset,
        )
        if args.max_pool_samples is not None or args.max_holdout_samples is not None:
            records = records[: (args.max_pool_samples or 0) + (args.max_holdout_samples or 0)]

        pool, holdout = split_dataset(records, holdout_ratio=args.holdout_ratio, seed=args.seed)
        if args.max_pool_samples is not None:
            pool = pool[: args.max_pool_samples]
        if args.max_holdout_samples is not None:
            holdout = holdout[: args.max_holdout_samples]

        save_records_json(output_dir / HOLDOUT_RECORDS_FILE, holdout)
        if uses_unseen_pool:
            unseen = list(pool)
            save_records_json(output_dir / UNSEEN_POOL_FILE, unseen)
        else:
            unseen = []
            save_records_json(output_dir / ACTIVE_POOL_FILE, pool)

        current_model_path = args.model_path
        history = []
        start_iteration = 0
        save_run_state(
            output_dir,
            next_iteration=0,
            current_model_path=current_model_path,
            args=args,
        )

    for iteration in range(start_iteration, args.iterations):
        iter_dir = output_dir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Iteration {iteration} ({args.selection_strategy}) ===")

        if _iteration_complete(iter_dir):
            with open(iter_dir / "metrics.json", encoding="utf-8") as handle:
                completed_metrics = json.load(handle)
            current_model_path = completed_metrics.get("model_path", current_model_path)
            if uses_unseen_pool:
                unseen = load_records_json(output_dir / UNSEEN_POOL_FILE)
            else:
                pool_path = output_dir / ACTIVE_POOL_FILE
                if pool_path.exists():
                    pool = load_records_json(pool_path)
            print(f"Skipping completed iteration {iteration}")
            continue

        selected: list[CalibrationRecord] = []
        selection_stats: dict[str, Any] = {}
        calibration_diag: dict[str, Any] = {}

        if uses_unseen_pool:
            inferred_path = iter_dir / "unseen_inference.json"
            selected_path = iter_dir / "selected_samples.json"

            if inferred_path.exists():
                unseen = load_records_json(inferred_path)
                print(f"Loaded completed pool inference ({len(unseen)} samples).")
            elif args.selection_strategy == "active_pseudo":
                print(f"Running dual-mode inference on {len(unseen)} unseen samples...")
                unseen = run_dual_mode_inference(
                    unseen,
                    model_path=current_model_path,
                    dtype=args.dtype,
                    agreement_samples=args.agreement_samples,
                    max_new_tokens=max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    output_path=inferred_path,
                    checkpoint_path=iter_dir / PARTIAL_INFERENCE_FILE,
                    checkpoint_every=args.checkpoint_every,
                )
                save_run_state(
                    output_dir,
                    next_iteration=iteration,
                    current_model_path=current_model_path,
                    args=args,
                )
            else:
                print(f"Running answer-first inference on {len(unseen)} unseen samples...")
                unseen = run_inference(
                    unseen,
                    model_path=current_model_path,
                    dtype=args.dtype,
                    use_reasoning=True,
                    answer_first=True,
                    agreement_samples=args.agreement_samples,
                    max_new_tokens=max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    output_path=inferred_path,
                    checkpoint_path=iter_dir / PARTIAL_INFERENCE_FILE,
                    checkpoint_every=args.checkpoint_every,
                    zeroshot_json=zeroshot_json,
                )
                save_run_state(
                    output_dir,
                    next_iteration=iteration,
                    current_model_path=current_model_path,
                    args=args,
                )

            if selected_path.exists():
                selected = load_records_json(selected_path)
                unseen = load_records_json(output_dir / UNSEEN_POOL_FILE)
                stats_path = iter_dir / "selection_stats.json"
                if stats_path.exists():
                    with open(stats_path, encoding="utf-8") as handle:
                        selection_stats = json.load(handle)
                calibration_diag = identify_calibration_errors(unseen)
                print(f"Loaded selection ({len(selected)} samples, {len(unseen)} unseen remaining).")
            elif args.selection_strategy == "active_pseudo":
                if iteration == 0:
                    selected, unseen, selection_stats = build_seed_set(
                        unseen,
                        seed_rationale_first=args.seed_rationale_first,
                        seed_answer_first=args.seed_answer_first,
                    )
                else:
                    selected, unseen, selection_stats = sample_active_calibration(
                        unseen,
                        select_rationale_first=args.select_rationale_first,
                        select_answer_first=args.select_answer_first,
                    )
                calibration_diag = identify_calibration_errors(unseen)
                save_records_json(selected_path, selected)
                save_records_json(output_dir / UNSEEN_POOL_FILE, unseen)
                with open(iter_dir / "selection_stats.json", "w", encoding="utf-8") as handle:
                    json.dump(selection_stats, handle, indent=2, ensure_ascii=False)
                save_run_state(
                    output_dir,
                    next_iteration=iteration,
                    current_model_path=current_model_path,
                    args=args,
                )
            else:
                selected, unseen, selection_stats = select_entropy_calibration(
                    unseen,
                    n_wrong_low_entropy=args.n_wrong_low_entropy,
                    n_correct_high_entropy=args.n_correct_high_entropy,
                )
                calibration_diag = identify_calibration_errors(unseen)
                save_records_json(selected_path, selected)
                save_records_json(output_dir / UNSEEN_POOL_FILE, unseen)
                with open(iter_dir / "selection_stats.json", "w", encoding="utf-8") as handle:
                    json.dump(selection_stats, handle, indent=2, ensure_ascii=False)
                save_run_state(
                    output_dir,
                    next_iteration=iteration,
                    current_model_path=current_model_path,
                    args=args,
                )
        else:
            pool_inference_path = iter_dir / "pool_inference.json"
            selected_path = iter_dir / "selected_samples.json"

            if iteration > 0:
                if pool_inference_path.exists():
                    pool = load_records_json(pool_inference_path)
                    print(f"Loaded completed pool inference ({len(pool)} samples).")
                else:
                    print("Running inference on active-learning pool...")
                    pool = run_inference(
                        pool,
                        model_path=current_model_path,
                        dtype=args.dtype,
                        use_reasoning=args.use_reasoning,
                        answer_first=args.answer_first,
                        agreement_samples=args.agreement_samples,
                        max_new_tokens=max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        output_path=pool_inference_path,
                        checkpoint_path=iter_dir / POOL_PARTIAL_INFERENCE_FILE,
                        checkpoint_every=args.checkpoint_every,
                        zeroshot_json=zeroshot_json,
                    )
                    save_records_json(output_dir / ACTIVE_POOL_FILE, pool)
                    save_run_state(
                        output_dir,
                        next_iteration=iteration,
                        current_model_path=current_model_path,
                        args=args,
                    )

            calibration_diag = identify_calibration_errors(pool)

            if selected_path.exists():
                selected = load_records_json(selected_path)
                stats_path = iter_dir / "selection_stats.json"
                if stats_path.exists():
                    with open(stats_path, encoding="utf-8") as handle:
                        selection_stats = json.load(handle)
                print(f"Loaded selection ({len(selected)} samples).")
            else:
                selected, selection_stats = sample_calibration_dataset(
                    pool,
                    selection_ratio=args.selection_ratio,
                    random_ratio=args.random_ratio,
                    seed=args.seed,
                    iteration=iteration,
                )
                save_records_json(selected_path, selected)
                with open(iter_dir / "selection_stats.json", "w", encoding="utf-8") as handle:
                    json.dump(selection_stats, handle, indent=2, ensure_ascii=False)
                save_run_state(
                    output_dir,
                    next_iteration=iteration,
                    current_model_path=current_model_path,
                    args=args,
                )

        train_metrics = {"answer_loss": 0.0, "calibration_loss": 0.0, "train_loss": 0.0}
        if not args.skip_train:
            if not selected:
                print("Warning: no samples selected; skipping training.")
            elif _has_trained_model(iter_dir):
                print(f"Using existing trained model in {iter_dir}")
                current_model_path = str(iter_dir)
                train_metrics = _load_train_metrics(iter_dir)
            else:
                print(f"Training on {len(selected)} selected samples...")
                train_metrics = train_iteration(
                    selected,
                    model_path=current_model_path,
                    output_dir=iter_dir,
                    args=args,
                )
                current_model_path = str(iter_dir)
                with open(iter_dir / "train_metrics.json", "w", encoding="utf-8") as handle:
                    json.dump(train_metrics, handle, indent=2, ensure_ascii=False)
                save_run_state(
                    output_dir,
                    next_iteration=iteration,
                    current_model_path=current_model_path,
                    args=args,
                )
        else:
            print("Skipping training (--skip_train).")

        if holdout:
            print("Evaluating holdout...")
            eval_metrics = evaluate_iteration(
                holdout,
                model_path=current_model_path,
                output_dir=iter_dir,
                args=args,
                iteration=iteration,
            )
        else:
            print("Skipping holdout evaluation (--holdout_ratio 0).")
            eval_metrics = {
                "accuracy": None,
                "aurc": None,
                "aurc_rationale_first": None,
                "aurc_answer_first": None,
                "ece": None,
                "n_holdout": 0,
                "n_valid_entropy": 0,
                "holdout_results_path": None,
                "mean_confidence": None,
                "mean_entropy": None,
            }

        iteration_metrics: dict[str, Any] = {
            "iteration": iteration,
            "selection_strategy": args.selection_strategy,
            "model_path": current_model_path,
            "accuracy": eval_metrics["accuracy"],
            "answer_loss": train_metrics["answer_loss"],
            "calibration_loss": train_metrics["calibration_loss"],
            "train_loss": train_metrics["train_loss"],
            "aurc": eval_metrics["aurc"],
            "aurc_rationale_first": eval_metrics.get("aurc_rationale_first"),
            "aurc_answer_first": eval_metrics.get("aurc_answer_first"),
            "ece": eval_metrics["ece"],
            "n_holdout": eval_metrics.get("n_holdout"),
            "n_valid_entropy": eval_metrics.get("n_valid_entropy"),
            "mean_confidence": eval_metrics.get("mean_confidence"),
            "mean_entropy": eval_metrics.get("mean_entropy"),
            "holdout_results_path": eval_metrics.get("holdout_results_path"),
            "n_selected": selection_stats["n_selected"],
            "selection_stats": selection_stats,
            "calibration_diagnostics": calibration_diag,
        }

        if args.selection_strategy == "active_pseudo":
            iteration_metrics.update(
                {
                    "n_unseen_remaining": selection_stats.get("n_unseen_remaining", len(unseen)),
                    "fraction_rationale_first": selection_stats.get("fraction_rationale_first", 0.0),
                    "fraction_answer_first": selection_stats.get("fraction_answer_first", 0.0),
                }
            )
        elif args.selection_strategy == "entropy_calibration":
            iteration_metrics.update(
                {
                    "n_unseen_remaining": selection_stats.get("n_unseen_remaining", len(unseen)),
                    "n_wrong_low_entropy": selection_stats.get("n_wrong_low_entropy", 0),
                    "n_correct_high_entropy": selection_stats.get("n_correct_high_entropy", 0),
                    "fraction_wrong_low_entropy": selection_stats.get("fraction_wrong_low_entropy", 0.0),
                    "fraction_correct_high_entropy": selection_stats.get("fraction_correct_high_entropy", 0.0),
                }
            )
        else:
            iteration_metrics.update(
                {
                    "fraction_correct_high_entropy": selection_stats["fraction_correct_high_entropy"],
                    "fraction_wrong_high_confidence": selection_stats["fraction_wrong_high_confidence"],
                    "fraction_random": selection_stats["fraction_random"],
                }
            )

        with open(iter_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(iteration_metrics, handle, indent=2, ensure_ascii=False)
        _append_jsonl(output_dir / "metrics_history.jsonl", iteration_metrics)
        history.append(iteration_metrics)

        save_run_state(
            output_dir,
            next_iteration=iteration + 1,
            current_model_path=current_model_path,
            args=args,
        )

        print(json.dumps(iteration_metrics, indent=2))

        if uses_unseen_pool and not unseen:
            print("Unseen pool exhausted; stopping early.")
            break

    summary = {"iterations": args.iterations, "selection_strategy": args.selection_strategy, "history": history}
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\nCompleted. Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
