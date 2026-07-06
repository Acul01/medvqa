"""Iterative calibration fine-tuning pipeline for Qwen2.5-VL on Medical VQA."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import wandb
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

from calibration_losses import build_losses, compute_ece
from utils.load_data import VQASample
from utils.mc_eval import run_mc_evaluation
from utils.mc_vqa import correct_index_to_letter
from utils.paths import get_data_root, get_mc_data_root, get_outputs_root
from utils.qwen_vl import (
    DEFAULT_MODEL_PATH,
    QwenVLDataCollator,
    build_mc_conversation,
    build_mc_reasoning_conversation,
    configure_qwen_processor_pixels,
    mask_labels_assistant_only,
    model_short_name,
    prepare_qwen_inputs,
    resolve_processor_path,
    squeeze_batch_dim,
)
from utils.save_results import clean_answer

REPO_ROOT = Path(__file__).resolve().parent

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

    def to_inference_row(self) -> dict[str, Any]:
        """Serialize to inference-compatible JSON row."""
        row = dict(self.original_row)
        row.update(
            {
                "image_path": self.image_path,
                "question": self.question,
                "ground_truth_answer": self.ground_truth_answer,
                "predicted_answer": self.generated_answer,
                "predicted_reasoning_text": self.generated_rationale,
                "confidence": self.confidence,
                "entropy": self.entropy,
                "is_correct": self.is_correct,
                "options": self.options,
                "correct_index": self.correct_index,
                "question_type": self.question_type,
                "ground_truth_letter": self.ground_truth_letter,
            }
        )
        return row


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
    )


def record_to_vqa_sample(record: CalibrationRecord, *, dataset: str = "calibration") -> VQASample:
    """Convert a calibration record into a VQASample for Qwen training."""
    metadata = dict(record.original_row)
    if record.generated_rationale:
        metadata["pseudo_reasoning_raw"] = record.generated_rationale
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
    parser.add_argument("--dataset", type=str, default="unknown", help="Dataset name for logging only")
    parser.add_argument("--selection_ratio", type=float, required=True)
    parser.add_argument("--random_ratio", type=float, default=0.20)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--lambda_cal", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--holdout_ratio", type=float, default=0.20)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agreement_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--use_reasoning", action="store_true", default=False)
    parser.add_argument("--answer_first", action="store_true", default=False)
    parser.add_argument("--answer_loss", type=str, default="cross_entropy")
    parser.add_argument("--calibration_loss", type=str, default="confidence_bce")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--data_root", type=str, default=get_data_root())
    parser.add_argument("--mc_data_root", type=str, default=get_mc_data_root())
    parser.add_argument("--max_pool_samples", type=int, default=None, help="Debug cap on pool size")
    parser.add_argument("--max_holdout_samples", type=int, default=None, help="Debug cap on holdout size")
    parser.add_argument("--skip_train", action="store_true", help="Run selection/eval only (debug)")
    parser.add_argument("--wandb_project", type=str, default="medvqa")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def load_dataset(path: str | Path) -> list[CalibrationRecord]:
    """Load and validate a pre-built inference JSON dataset."""
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
    return records


def split_dataset(
    records: list[CalibrationRecord],
    *,
    holdout_ratio: float,
    seed: int,
) -> tuple[list[CalibrationRecord], list[CalibrationRecord]]:
    """Split records into active-learning pool and holdout evaluation set."""
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be between 0 and 1.")

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


def _move_inputs_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def load_qwen_model_and_processor(
    model_path: str,
    *,
    dtype: str = "bfloat16",
    base_model: str | None = None,
) -> tuple[Any, Any, torch.device]:
    """Load Qwen2.5-VL model and processor with LoRA/spatial-filter support."""
    inference_mod = _inference_qwen_module()
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

    processor_path = resolve_processor_path(base_model or resolved_path if is_lora else resolved_path)
    processor = AutoProcessor.from_pretrained(processor_path)
    configure_qwen_processor_pixels(processor)
    return model, processor, device


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
) -> list[CalibrationRecord]:
    """Run MC inference on records and refresh generated outputs and uncertainty."""
    inference_mod = _inference_qwen_module()
    from utils.qwen_vl import build_mc_reasoning_inference_conversation

    model, processor, device = load_qwen_model_and_processor(model_path, dtype=dtype)
    model.eval()

    answer_letter_token_ids = inference_mod._get_answer_letter_token_ids(processor.tokenizer)
    agreement_samples = max(1, agreement_samples)
    agreement_gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
    }
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}

    updated: list[CalibrationRecord] = []
    with torch.no_grad():
        for record in records:
            sample = record_to_vqa_sample(record)
            image = Image.open(sample.image_path).convert("RGB")
            conversation = (
                build_mc_reasoning_inference_conversation(
                    sample.question,
                    sample.mc_options or [],
                    image=image,
                    answer_first=answer_first,
                )
                if use_reasoning
                else build_mc_conversation(
                    sample.question,
                    sample.mc_options or [],
                    image=image,
                )
            )
            inputs = prepare_qwen_inputs(processor, conversation, add_generation_prompt=True)
            inputs = _move_inputs_to_device(inputs, device)
            input_length = inputs["input_ids"].shape[1]

            generation_output = model.generate(**inputs, **gen_kwargs, return_dict_in_generate=True)
            generated_ids = generation_output.sequences[0][input_length:]
            generated_text = clean_answer(
                processor.decode(generated_ids, skip_special_tokens=True).strip()
            )
            num_opts = len(sample.mc_options or [])
            predicted_letter, pred_reason_text, _ = inference_mod.decode_mc_prediction(
                generated_text,
                num_options=num_opts,
                use_reasoning_json=use_reasoning,
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
                        use_reasoning_json=use_reasoning,
                        num_opts=num_opts,
                        answer_letter_token_ids=answer_letter_token_ids,
                    )
                    agreement_entropies.append(entropy)
                    agreement_confidences.append(confidence)

            confidence = inference_mod._mean_finite(agreement_confidences)
            entropy = inference_mod._mean_finite(agreement_entropies)
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
            )
            updated.append(refreshed)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump([item.to_inference_row() for item in updated], handle, indent=2, ensure_ascii=False)

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
        dataset_name: str = "calibration",
    ):
        self.records = records
        self.processor = processor
        self.max_length = max_length
        self.use_reasoning = use_reasoning
        self.answer_first = answer_first
        self.dataset_name = dataset_name
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        sample = record_to_vqa_sample(record, dataset=self.dataset_name)
        image = Image.open(sample.image_path).convert("RGB")

        if self.use_reasoning and record.generated_rationale:
            conversation = build_mc_reasoning_conversation(
                sample.question,
                sample.mc_options or [],
                reasoning_raw=record.generated_rationale,
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
    dataset_name: str,
) -> tuple[CalibrationTrainDataset, CalibrationVLDataCollator]:
    """Build training dataset and collator for one calibration iteration."""
    train_dataset = CalibrationTrainDataset(
        records,
        processor,
        max_length=max_length,
        use_reasoning=use_reasoning,
        answer_first=answer_first,
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
        outputs = model(**inputs)
        total, answer_loss, calibration_loss = self.combined_loss(
            outputs.logits,
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


def build_losses_from_args(args: argparse.Namespace, processor: Any) -> Any:
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

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor_path = resolve_processor_path(model_path)
    processor = AutoProcessor.from_pretrained(processor_path)
    configure_qwen_processor_pixels(processor)

    train_dataset, collator = build_dataloader(
        selected_records,
        processor,
        max_length=args.max_length,
        use_reasoning=args.use_reasoning,
        answer_first=args.answer_first,
        dataset_name=args.dataset,
    )
    combined_loss = build_losses_from_args(args, processor)

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
            report_to="wandb",
            seed=args.seed,
            remove_unused_columns=False,
            bf16=args.dtype == "bfloat16",
            fp16=args.dtype == "float16",
        ),
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=processor.tokenizer,
    )

    train_result = trainer.train()
    trainer.save_model()
    processor.save_pretrained(output_dir)

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


def evaluate_iteration(
    holdout: list[CalibrationRecord],
    *,
    model_path: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate holdout accuracy, AURC, and ECE after an iteration."""
    holdout_path = output_dir / "holdout_inference.json"
    refreshed = run_inference(
        holdout,
        model_path=model_path,
        dtype=args.dtype,
        use_reasoning=args.use_reasoning,
        answer_first=args.answer_first,
        agreement_samples=args.agreement_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        output_path=holdout_path,
    )

    metrics = run_mc_evaluation(holdout_path, metrics_output_path=output_dir / "holdout_eval.json")
    accuracy = float(metrics["overall"]["accuracy"])

    risk_mod = _risk_coverage_module()
    inference_rows = [record.to_inference_row() for record in refreshed]
    aurc_eval = risk_mod.evaluate_question_type(
        inference_rows,
        "overall",
        risk_metric="match",
        score_field="entropy",
        max_entropy=None,
        num_thresholds=200,
    )
    aurc = float(aurc_eval["aurc"])

    confidences = [record.confidence for record in refreshed if math.isfinite(record.confidence)]
    correctness = [record.is_correct for record in refreshed if math.isfinite(record.confidence)]
    ece = compute_ece(confidences, correctness)

    return {
        "accuracy": accuracy,
        "aurc": aurc,
        "ece": ece,
        "n_holdout": len(refreshed),
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    """Run the iterative calibration fine-tuning loop."""
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_dataset(args.input_json)
    if args.max_pool_samples is not None or args.max_holdout_samples is not None:
        records = records[: (args.max_pool_samples or 0) + (args.max_holdout_samples or 0)]

    pool, holdout = split_dataset(records, holdout_ratio=args.holdout_ratio, seed=args.seed)
    if args.max_pool_samples is not None:
        pool = pool[: args.max_pool_samples]
    if args.max_holdout_samples is not None:
        holdout = holdout[: args.max_holdout_samples]

    run_name = args.wandb_run_name or f"{model_short_name(args.model_path)}_{args.dataset}_iter_calib"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        tags=["iterative_calibration", args.dataset],
    )

    current_model_path = args.model_path
    history: list[dict[str, Any]] = []

    for iteration in range(args.iterations):
        iter_dir = output_dir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Iteration {iteration} ===")

        if iteration > 0:
            print("Running inference on active-learning pool...")
            pool = run_inference(
                pool,
                model_path=current_model_path,
                dtype=args.dtype,
                use_reasoning=args.use_reasoning,
                answer_first=args.answer_first,
                agreement_samples=args.agreement_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                output_path=iter_dir / "pool_inference.json",
            )

        calibration_diag = identify_calibration_errors(pool)
        selected, selection_stats = sample_calibration_dataset(
            pool,
            selection_ratio=args.selection_ratio,
            random_ratio=args.random_ratio,
            seed=args.seed,
            iteration=iteration,
        )
        with open(iter_dir / "selected_samples.json", "w", encoding="utf-8") as handle:
            json.dump([record.to_inference_row() for record in selected], handle, indent=2, ensure_ascii=False)

        train_metrics = {"answer_loss": 0.0, "calibration_loss": 0.0, "train_loss": 0.0}
        if not args.skip_train:
            print(f"Training on {len(selected)} selected samples...")
            train_metrics = train_iteration(
                selected,
                model_path=current_model_path,
                output_dir=iter_dir,
                args=args,
            )
            current_model_path = str(iter_dir)
        else:
            print("Skipping training (--skip_train).")

        print("Evaluating holdout...")
        eval_metrics = evaluate_iteration(
            holdout,
            model_path=current_model_path,
            output_dir=iter_dir,
            args=args,
        )

        iteration_metrics = {
            "iteration": iteration,
            "model_path": current_model_path,
            "accuracy": eval_metrics["accuracy"],
            "answer_loss": train_metrics["answer_loss"],
            "calibration_loss": train_metrics["calibration_loss"],
            "train_loss": train_metrics["train_loss"],
            "aurc": eval_metrics["aurc"],
            "ece": eval_metrics["ece"],
            "n_selected": selection_stats["n_selected"],
            "fraction_correct_high_entropy": selection_stats["fraction_correct_high_entropy"],
            "fraction_wrong_high_confidence": selection_stats["fraction_wrong_high_confidence"],
            "fraction_random": selection_stats["fraction_random"],
            "selection_stats": selection_stats,
            "calibration_diagnostics": calibration_diag,
        }

        with open(iter_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(iteration_metrics, handle, indent=2, ensure_ascii=False)
        _append_jsonl(output_dir / "metrics_history.jsonl", iteration_metrics)
        history.append(iteration_metrics)

        wandb.log(
            {
                f"iter_{iteration}/accuracy": eval_metrics["accuracy"],
                f"iter_{iteration}/aurc": eval_metrics["aurc"],
                f"iter_{iteration}/answer_loss": train_metrics["answer_loss"],
                f"iter_{iteration}/calibration_loss": train_metrics["calibration_loss"],
                f"iter_{iteration}/n_selected": selection_stats["n_selected"],
            }
        )
        print(json.dumps(iteration_metrics, indent=2))

    summary = {"iterations": args.iterations, "history": history}
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    wandb.finish()
    print(f"\nCompleted {args.iterations} iteration(s). Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
