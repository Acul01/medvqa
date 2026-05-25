"""
Training script for Qwen2.5-VL on Medical VQA datasets (MC).

Uses HuggingFace Qwen2_5_VLForConditionalGeneration + AutoProcessor + Trainer.
For standard LLaVA, use train.py. For LLaVA-Med, use train_med.py.
"""

import argparse
from collections import Counter
from pathlib import Path

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

from dataclasses import dataclass
from typing import Any

from utils.ablation import load_fgr_crop_lookup, normalize_ablation_image_key
from utils.load_data import MedicalVQALoader
from utils.paths import get_data_root, get_mc_data_root, get_train_output_dir_qwen
from utils.qwen_vl import (
    DEFAULT_MODEL_PATH,
    build_mc_conversation,
    mask_labels_assistant_only,
    model_short_name,
    prepare_qwen_inputs,
    resolve_processor_path,
    squeeze_batch_dim,
)


class QwenMCTrainDataset(Dataset):
    """MC training dataset for Qwen2.5-VL; loss only on assistant answer tokens."""

    def __init__(self, samples, processor, max_length=512, fgr_crop_lookup=None, data_root=None):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.fgr_crop_lookup = fgr_crop_lookup  # (image_key, question) -> crop_path; None = use original
        self.data_root = data_root
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.fgr_crop_lookup is not None:
            key = (
                normalize_ablation_image_key(sample.image_path, data_root=self.data_root),
                sample.question.strip(),
            )
            image_path = self.fgr_crop_lookup.get(key, sample.image_path)
        else:
            image_path = sample.image_path
        image = Image.open(image_path).convert("RGB")

        if not sample.mc_options:
            raise ValueError("MC options missing in sample; ensure mc_data_root points to MC exports.")

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
        )
        item = squeeze_batch_dim(batch)
        input_ids = item["input_ids"]
        labels = mask_labels_assistant_only(self.processor, input_ids, conversation)
        item["labels"] = labels
        return item


@dataclass
class QwenVLDataCollator:
    """Collator for Qwen2.5-VL with variable-size pixel_values.

    The Qwen2.5-VL processor encodes each image into a variable number of
    patches.  The default HuggingFace collator tries to *stack* tensors of
    equal shape, which fails when crops and full images are mixed in one batch.
    Here pixel_values are *concatenated* along the patch dimension (dim 0) and
    image_grid_thw is stacked so the model knows which patches belong to which
    image.  All token-level tensors (already padded to max_length by the
    dataset) are simply stacked.
    """

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = {}
        for key in ("input_ids", "attention_mask", "labels"):
            batch[key] = torch.stack([f[key] for f in features])
        # pixel_values: (n_patches_i, ...) — concatenate across samples
        batch["pixel_values"] = torch.cat([f["pixel_values"] for f in features], dim=0)
        # image_grid_thw: (1, 3) per sample — stack into (batch, 3)
        batch["image_grid_thw"] = torch.cat([f["image_grid_thw"] for f in features], dim=0)
        return batch


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
        help="Path to fgr_region_comparison_*_train.json. When given, replaces each "
             "training image with its supervised FGR winning crop (or the original "
             "image when the FGR winner is 'baseline').",
    )
    parser.add_argument("--wandb_project", type=str, default="medvqa")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    ft_tag = "lora" if args.use_lora else "full-ft"
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

    loader = MedicalVQALoader(data_root=args.data_root, mc_data_root=args.mc_data_root)
    use_all = args.dataset == "all"
    if use_all:
        print(f"Loading combined datasets {args.train_split} split...")
        train_samples = loader.load_combined(args.train_split)
    else:
        print(f"Loading {args.dataset} {args.train_split} split...")
        train_samples = loader.load_dataset(args.dataset, split=args.train_split)

    fgr_crop_lookup = None
    if args.fgr_crops_json:
        fgr_crop_lookup = load_fgr_crop_lookup(
            args.fgr_crops_json,
            crop_field="overall_winner_supervised_crop",
            data_root=args.data_root,
        )
        print(f"FGR crop training: {len(fgr_crop_lookup)} mappings loaded")

    train_dataset = QwenMCTrainDataset(
        train_samples, processor, args.max_length,
        fgr_crop_lookup=fgr_crop_lookup,
        data_root=args.data_root,
    )
    print(f"Train: {len(train_dataset)} samples")
    if use_all:
        print(f"  per dataset: {dict(Counter(s.dataset for s in train_samples))}")

    val_dataset = None
    try:
        if use_all:
            print(
                f"Loading validation: PathVQA/SLAKE/Med-VQA split={args.val_split!r}, "
                "VQA-Rad split='test'."
            )
            val_samples = loader.load_combined(args.val_split, vqarad_split="test")
        else:
            val_samples = loader.load_dataset(args.dataset, split=args.val_split)
        val_dataset = QwenMCTrainDataset(val_samples, processor, args.max_length)
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
        data_collator=QwenVLDataCollator(),
    )

    print("Starting Qwen2.5-VL training (loss on assistant answer tokens only)...")
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
            "val_samples": len(val_dataset) if val_dataset else 0,
        }
    )
    wandb.finish()


if __name__ == "__main__":
    main()
