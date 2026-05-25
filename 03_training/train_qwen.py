"""
Training script for Qwen2.5-VL on Medical VQA datasets (MC).

Uses HuggingFace Qwen2_5_VLForConditionalGeneration + AutoProcessor + Trainer.
For standard LLaVA, use train.py. For LLaVA-Med, use train_med.py.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

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

from utils.load_data import MedicalVQALoader, VQASample
from utils.paths import get_data_root, get_mc_data_root, get_train_output_dir_qwen
from utils.qwen_vl import (
    DEFAULT_MC_EXPORT_SUFFIX_PSEUDO_VERIFY,
    DEFAULT_MODEL_PATH,
    build_mc_conversation,
    build_mc_reasoning_conversation,
    configure_qwen_processor_pixels,
    is_pseudo_reasoning_verified,
    mask_labels_assistant_only,
    model_short_name,
    prepare_qwen_inputs,
    pseudo_reasoning_raw_from_metadata,
    resolve_processor_path,
    squeeze_batch_dim,
)


FgrTrainLookup = Dict[Tuple[str, str], str]  # (image_path, question) → effective_image_path


def build_fgr_train_lookup(json_path: str) -> FgrTrainLookup:
    """Build (image_path, question) → effective_image_path from FGR region comparison JSON.

    For samples where ``overall_winner_supervised == 'baseline'`` the original
    image path is kept.  For l1/l2 winners the saved crop path is used instead.
    """
    path = Path(json_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FGR crops JSON not found: {path}")

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    rows: list[dict] = []
    if isinstance(payload, dict) and "samples" in payload:
        rows = payload["samples"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("FGR JSON must be a list or contain a 'samples' key")

    lookup: FgrTrainLookup = {}
    n_crops = 0
    n_baseline = 0
    n_missing = 0

    for row in rows:
        image_path = (row.get("image_path") or "").strip()
        question   = (row.get("question")    or "").strip()
        if not image_path or not question:
            continue

        winner     = row.get("overall_winner_supervised", "baseline")
        crop_path  = (row.get("overall_winner_supervised_crop") or "").strip()

        if winner == "baseline" or not crop_path:
            effective = image_path
            n_baseline += 1
        elif Path(crop_path).is_file():
            effective = crop_path
            n_crops += 1
        else:
            # Crop file not found on this machine — fall back to original
            effective = image_path
            n_missing += 1

        key = (str(Path(image_path).resolve()) if Path(image_path).exists() else image_path,
               question)
        lookup[key] = effective

    print(
        f"FGR train lookup: {len(lookup)} samples — "
        f"{n_crops} crops, {n_baseline} baseline (original), {n_missing} missing (fallback)"
    )
    return lookup


def filter_verified_reasoning_samples(samples: list[VQASample]) -> list[VQASample]:
    """Keep only rows with pseudo_reasoning_test=True and non-empty pseudo_reasoning_raw."""
    filtered: list[VQASample] = []
    for sample in samples:
        if not is_pseudo_reasoning_verified(sample.metadata):
            continue
        if not pseudo_reasoning_raw_from_metadata(sample.metadata):
            continue
        filtered.append(sample)
    return filtered


def resolve_train_mc_export_suffix(use_reasoning: bool, mc_export_suffix: str) -> str:
    if mc_export_suffix:
        return mc_export_suffix
    if use_reasoning:
        return DEFAULT_MC_EXPORT_SUFFIX_PSEUDO_VERIFY
    return ""


class QwenMCTrainDataset(Dataset):
    """MC training dataset for Qwen2.5-VL; loss only on assistant answer tokens.

    Parameters
    ----------
    fgr_lookup:
        Optional lookup built by ``build_fgr_train_lookup``.  When provided,
        each sample's image is replaced by its FGR supervised winning crop
        (or kept as the original image for baseline winners).
    """

    def __init__(
        self,
        samples,
        processor,
        max_length=512,
        *,
        use_reasoning: bool = False,
        fgr_lookup: Optional[FgrTrainLookup] = None,
    ):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.use_reasoning = use_reasoning
        self.fgr_lookup = fgr_lookup
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    def _resolve_image_path(self, sample: VQASample) -> str:
        """Return effective image path: FGR crop if available, else original."""
        if self.fgr_lookup is None:
            return sample.image_path
        p = Path(sample.image_path)
        key = (str(p.resolve()) if p.exists() else sample.image_path,
               sample.question.strip())
        return self.fgr_lookup.get(key, sample.image_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(self._resolve_image_path(sample)).convert("RGB")

        if not sample.mc_options:
            raise ValueError("MC options missing in sample; ensure mc_data_root points to MC exports.")

        if self.use_reasoning:
            reasoning_raw = pseudo_reasoning_raw_from_metadata(sample.metadata)
            if not reasoning_raw:
                raise ValueError("pseudo_reasoning_raw missing for reasoning training sample.")
            conversation = build_mc_reasoning_conversation(
                sample.question,
                sample.mc_options,
                reasoning_raw=reasoning_raw,
                answer=sample.answer,
                image=image,
            )
        else:
            conversation = build_mc_conversation(
                sample.question,
                sample.mc_options,
                answer=sample.answer,
                image=image,
            )
        batch = prepare_qwen_inputs(
            self.processor,
            conversation,
            add_generation_prompt=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=False,
        )
        item = squeeze_batch_dim(batch)
        input_ids = item["input_ids"]
        labels = mask_labels_assistant_only(self.processor, input_ids, conversation)
        item["labels"] = labels
        return item


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL on Medical VQA (MC)")

    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"HuggingFace model name or local path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--data_root", type=str, default=get_data_root())
    parser.add_argument("--mc_data_root", type=str, default=get_mc_data_root())
    parser.add_argument(
        "--mc_export_suffix",
        type=str,
        default="",
        help=(
            "MC JSON suffix after _mc_openai (e.g. _pseudo_pr_qwen_verify). "
            f"With --use_reasoning and unset, defaults to {DEFAULT_MC_EXPORT_SUFFIX_PSEUDO_VERIFY!r}."
        ),
    )
    parser.add_argument(
        "--use_reasoning",
        action="store_true",
        help=(
            "Train on verified pseudo reasoning (pseudo_reasoning_test=True) with JSON "
            "assistant targets {reasoning, answer}; val stays answer-only on standard MC."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["pathvqa", "slake", "vqarad", "medvqa2019", "all"],
    )
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation")
    parser.add_argument("--output_dir", type=str, default=get_train_output_dir_qwen())
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_8bit_optimizer", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    parser.add_argument(
        "--fgr_crops_json",
        type=str,
        default=None,
        help=(
            "Path to fgr_region_comparison JSON (output of fgr_region_comparison.py). "
            "When set, replaces each training sample's image with its supervised FGR "
            "winning crop; baseline-winner samples keep the original image."
        ),
    )

    parser.add_argument("--wandb_project", type=str, default="medvqa")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    ft_tag = "lora" if args.use_lora else "full-ft"
    if args.use_reasoning:
        ft_tag = f"reasoning-{ft_tag}"
    model_short = model_short_name(args.model_path)
    run_name = args.wandb_run_name or f"{model_short}_{args.dataset}_{ft_tag}_train"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=args.dataset,
        tags=["training", "qwen", model_short, args.dataset, ft_tag],
        config=vars(args),
    )

    if args.use_8bit_optimizer:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            print("Warning: bitsandbytes not installed, falling back to standard optimizer")
            args.use_8bit_optimizer = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    print(f"Loading model: {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    processor_path = resolve_processor_path(args.model_path)
    if processor_path != args.model_path:
        print(f"Processor not in checkpoint, loading from: {processor_path}")
    processor = AutoProcessor.from_pretrained(processor_path)
    configure_qwen_processor_pixels(processor)

    train_suffix = resolve_train_mc_export_suffix(args.use_reasoning, args.mc_export_suffix)
    train_loader = MedicalVQALoader(
        data_root=args.data_root,
        mc_data_root=args.mc_data_root,
        mc_export_suffix=train_suffix,
    )
    val_loader = MedicalVQALoader(
        data_root=args.data_root,
        mc_data_root=args.mc_data_root,
        mc_export_suffix="",
    )

    use_all = args.dataset == "all"
    if use_all:
        print(f"Loading combined datasets {args.train_split} split (mc suffix={train_suffix!r})...")
        train_samples = train_loader.load_combined(args.train_split)
    else:
        print(f"Loading {args.dataset} {args.train_split} split (mc suffix={train_suffix!r})...")
        train_samples = train_loader.load_dataset(args.dataset, split=args.train_split)

    # Optional FGR crop lookup (replaces training images with supervised winning crops)
    fgr_lookup: Optional[FgrTrainLookup] = None
    if args.fgr_crops_json:
        fgr_lookup = build_fgr_train_lookup(args.fgr_crops_json)

    train_total = len(train_samples)
    if args.use_reasoning:
        train_samples = filter_verified_reasoning_samples(train_samples)
        print(
            f"Train: {len(train_samples)} / {train_total} samples "
            f"(use_reasoning, pseudo_reasoning_test=True)"
        )
        if not train_samples:
            raise ValueError(
                "No verified reasoning samples after filtering. "
                "Ensure verify JSON exists and pseudo_reasoning_test=True rows are present."
            )
    else:
        print(f"Train: {len(train_samples)} samples")

    train_dataset = QwenMCTrainDataset(
        train_samples,
        processor,
        args.max_length,
        use_reasoning=args.use_reasoning,
        fgr_lookup=fgr_lookup,
    )
    if use_all and not args.use_reasoning:
        print(f"  per dataset: {dict(Counter(s.dataset for s in train_samples))}")

    val_dataset = None
    try:
        if use_all:
            print(
                f"Loading validation: PathVQA/SLAKE/Med-VQA split={args.val_split!r}, "
                "VQA-Rad split='test'."
            )
            val_samples = val_loader.load_combined(args.val_split, vqarad_split="test")
        else:
            val_samples = val_loader.load_dataset(args.dataset, split=args.val_split)
        val_dataset = QwenMCTrainDataset(val_samples, processor, args.max_length, use_reasoning=False)
        print(f"Validation: {len(val_dataset)} samples")
    except Exception as e:
        print(f"No validation split available: {e}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=bool(val_dataset),
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="wandb",
        seed=args.seed,
        optim="adamw_8bit" if args.use_8bit_optimizer else "adamw_torch",
        max_grad_norm=1.0,
        save_only_model=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor.tokenizer,
    )

    loss_desc = (
        "assistant JSON (reasoning + answer)"
        if args.use_reasoning
        else "assistant answer tokens only"
    )
    print(f"Starting Qwen2.5-VL training (loss on {loss_desc})...")
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print(f"Saving model to {args.output_dir}")
    trainer.save_model()
    trainer.save_state()
    processor.save_pretrained(args.output_dir)

    print("Training completed!")
    print(f"Training loss: {result.training_loss}")

    wandb.log(
        {
            "final_train_loss": result.training_loss,
            "train_samples": len(train_dataset),
            "train_samples_total_before_filter": train_total if args.use_reasoning else len(train_dataset),
            "train_samples_filtered": len(train_dataset) if args.use_reasoning else None,
            "val_samples": len(val_dataset) if val_dataset else 0,
        }
    )
    wandb.finish()


if __name__ == "__main__":
    main()
