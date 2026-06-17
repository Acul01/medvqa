"""
Training script for standard LLaVA on Medical VQA datasets.

Uses HuggingFace's LlavaForConditionalGeneration + AutoProcessor + Trainer.
The conversation format matches inference.py:
    USER: <image>\n{question}
    ASSISTANT: {answer}

The model handles image-token expansion internally during forward(),
so the dataset only needs a single <image> token in the input_ids.

For LLaVA-Med, use train_med.py instead.
"""

import argparse
import torch
import wandb
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import set_seed
from pathlib import Path

from collections import Counter

from utils.load_data import MedicalVQALoader
from utils.mc_vqa import format_mc_user_text
from utils.paths import get_data_root, get_mc_data_root, get_train_output_dir_llava, get_som_dir
from utils.som import SomResolver


DEFAULT_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"



# ── Dataset ─────────────────────────────────────────────────────────────────────

class LLaVATrainDataset(Dataset):
    """
    Training dataset for standard LLaVA.

    Uses the same conversation format as inference.py:
        USER: <image>\\n{question}
        ASSISTANT: {answer}

    The processor tokenises the text (with a single <image> token) and
    preprocesses the image into pixel_values.  During forward() the model
    replaces the <image> embedding with actual image features.

    Labels are set to -100 for all prompt tokens so the loss is only
    computed on the assistant's answer.
    """

    def __init__(
        self,
        samples,
        processor,
        max_length=512,
        dataset="vqarad",
        som_resolver: SomResolver | None = None,
    ):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.som_resolver = som_resolver
        # Ensure pad token is set
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        # Fix for community models missing patch_size
        if getattr(processor, "patch_size", None) is None:
            processor.patch_size = 14
        if getattr(processor, "vision_feature_select_strategy", None) is None:
            processor.vision_feature_select_strategy = "default"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = (
            self.som_resolver.resolve_or_original(sample.image_path)
            if self.som_resolver is not None
            else sample.image_path
        )
        image = Image.open(image_path).convert("RGB")

        # ── Build conversation (user + assistant) ────────────────────────
        if not sample.mc_options:
            raise ValueError("MC options missing in sample; ensure mc_data_root points to MC exports.")
        full_question = format_mc_user_text(sample.question, sample.mc_options)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": full_question},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample.answer},
                ],
            },
        ]

        # Full prompt (user + assistant, no generation prompt)
        full_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=False
        )

        # Process (tokenisation + image preprocessing)
        inputs = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)

        # ── Labels: mask prompt, keep only answer ────────────────────────
        labels = input_ids.clone()

        # Find "ASSISTANT:" marker in token IDs
        assistant_ids = self.processor.tokenizer.encode(
            "ASSISTANT:", add_special_tokens=False
        )
        ids_list = input_ids.tolist()

        answer_start = None
        for i in range(len(ids_list) - len(assistant_ids) + 1):
            if ids_list[i : i + len(assistant_ids)] == assistant_ids:
                answer_start = i + len(assistant_ids)
                break

        if answer_start is not None:
            labels[:answer_start] = -100
        else:
            # Fallback: tokenise prompt-only to find the boundary
            prompt_only_text = self.processor.apply_chat_template(
                [conversation[0]], add_generation_prompt=True
            )
            prompt_only_ids = self.processor.tokenizer(
                prompt_only_text, add_special_tokens=False
            )["input_ids"]
            labels[: len(prompt_only_ids)] = -100

        # Mask padding tokens
        labels[input_ids == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }


# ── Arguments ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune standard LLaVA on Medical VQA"
    )

    # Model
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"HuggingFace model name or local path (default: {DEFAULT_MODEL_PATH})")

    # Data
    parser.add_argument("--data_root", type=str, default=get_data_root(),
                        help="Root directory containing raw image folders ($MEDVQA_DATA_ROOT or 02_data)")
    parser.add_argument("--mc_data_root", type=str, default=get_mc_data_root(),
                        help="Root directory containing MC JSON/CSV exports ($MEDVQA_MC_ROOT or 02_data/data_mc_openai)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["pathvqa", "slake", "vqarad", "medvqa2019", "all"],
                        help="Dataset to use; 'all' = PathVQA+SLAKE+VQA-Rad+Med-VQA-2019 (prompt names from each sample)")
    parser.add_argument("--train_split", type=str, default="train",
                        help="Training split name")
    parser.add_argument("--val_split", type=str, default="validation",
                        help="Validation split name")

    # Training
    parser.add_argument("--output_dir", type=str, default=get_train_output_dir_llava(),
                        help="Output directory for checkpoints ($MEDVQA_OUTPUTS_DIR/llava)")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length (image tokens excluded, model handles expansion)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing (saves memory)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision (fp16)")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 precision")
    parser.add_argument("--use_8bit_optimizer", action="store_true",
                        help="Use 8-bit AdamW (requires bitsandbytes)")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume training from checkpoint path")

    # LoRA
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA instead of full fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha scaling (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout (default: 0.05)")
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=["q_proj", "v_proj"],
                        help="LoRA target modules (default: q_proj v_proj)")

    # W&B
    parser.add_argument("--wandb_project", type=str, default="medvqa",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (auto-generated if not set)")
    parser.add_argument(
        "--use_som",
        action="store_true",
        default=False,
        help="Replace images with pre-annotated SoM versions during training.",
    )
    parser.add_argument(
        "--som_dir",
        type=str,
        default=get_som_dir(),
        help="Root directory of SoM outputs (default: MEDVQA_SOM_DIR or 02_data/som).",
    )

    return parser.parse_args()


# ── Main ────────────────────────────────────────────────────────────────────────

def _model_short_name(model_path: str) -> str:
    name = Path(model_path).name.lower()
    if "llava-med" in name:
        return "llava-med"
    if "llava" in name:
        return "llava"
    return Path(model_path).name


def main():
    args = parse_args()
    set_seed(args.seed)

    # ── W&B ──────────────────────────────────────────────────────────────
    model_short = _model_short_name(args.model_path)
    ft_tag = "lora" if args.use_lora else "full-ft"
    run_name = args.wandb_run_name or f"{model_short}_{args.dataset}_{ft_tag}_train"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=args.dataset,
        tags=["training", model_short, args.dataset, ft_tag],
        config=vars(args),
    )

    # Check 8-bit optimizer
    if args.use_8bit_optimizer:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            print("Warning: bitsandbytes not installed, falling back to standard optimizer")
            args.use_8bit_optimizer = False

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load model ──────────────────────────────────────────────────────
    print(f"Loading model: {args.model_path}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # ── LoRA ─────────────────────────────────────────────────────────────
    if args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
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

    # ── Load processor ──────────────────────────────────────────────────
    # For fine-tuned checkpoints the processor config may not be saved –
    # fall back to the base model's processor.
    processor_path = args.model_path
    if Path(args.model_path).exists() and not Path(args.model_path, "preprocessor_config.json").exists():
        processor_path = DEFAULT_MODEL_PATH
        print(f"Processor not in checkpoint, loading from: {processor_path}")
    processor = AutoProcessor.from_pretrained(processor_path)

    # ── Load datasets ───────────────────────────────────────────────────
    loader = MedicalVQALoader(data_root=args.data_root, mc_data_root=args.mc_data_root)
    use_all = args.dataset == "all"

    som_resolver: SomResolver | None = None
    if args.use_som:
        som_resolver = SomResolver(args.som_dir)
        datasets_to_load = ["pathvqa", "slake", "vqarad", "medvqa2019"] if use_all else [args.dataset]
        for ds in datasets_to_load:
            som_resolver.load_dataset(ds)
        print(f"SoM enabled — images loaded from {args.som_dir}")

    if use_all:
        # Source tag per sample: VQASample.dataset. VQA-Rad val uses test split (no validation in loader).
        print(f"Loading combined PathVQA+SLAKE+VQA-Rad {args.train_split} split...")
        train_samples = loader.load_combined(args.train_split)
        train_dataset = LLaVATrainDataset(
            train_samples,
            processor,
            args.max_length,
            dataset=args.dataset,
            som_resolver=som_resolver,
        )
        print(f"Train: {len(train_dataset)} samples ({dict(Counter(s.dataset for s in train_samples))})")
    else:
        print(f"Loading {args.dataset} {args.train_split} split...")
        train_samples = loader.load_dataset(args.dataset, split=args.train_split)
        train_dataset = LLaVATrainDataset(
            train_samples, processor, args.max_length, dataset=args.dataset, som_resolver=som_resolver,
        )
        print(f"Train: {len(train_dataset)} samples")

    val_dataset = None
    try:
        if use_all:
            print(
                f"Loading validation: PathVQA/SLAKE split={args.val_split!r}, "
                "VQA-Rad split='test' (VQA-Rad has no validation split in this loader)."
            )
            val_samples = loader.load_combined(args.val_split, vqarad_split="test")
        else:
            val_samples = loader.load_dataset(args.dataset, split=args.val_split)
        val_dataset = LLaVATrainDataset(
            val_samples,
            processor,
            args.max_length,
            dataset=args.dataset,
            som_resolver=som_resolver,
        )
        if use_all:
            print(f"Validation: {len(val_dataset)} samples ({dict(Counter(s.dataset for s in val_samples))})")
        else:
            print(f"Validation: {len(val_dataset)} samples")
    except Exception as e:
        print(f"No validation split available: {e}")

    # ── Training arguments ──────────────────────────────────────────────
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

    # ── Trainer ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor.tokenizer,
    )

    # ── Train ───────────────────────────────────────────────────────────
    print("Starting training...")
    print(f"  Conversation format: USER: <image>\\n{{question}} ASSISTANT: {{answer}}")
    print(f"  Loss computed only on assistant response tokens")

    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model()
    trainer.save_state()

    print("Training completed!")
    print(f"Training loss: {result.training_loss}")

    wandb.log({
        "final_train_loss": result.training_loss,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset else 0,
    })
    wandb.finish()


if __name__ == "__main__":
    main()
