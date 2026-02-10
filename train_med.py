"""
Training script for LLaVA-Med on Medical VQA datasets.

Uses the original Microsoft LLaVA-Med codebase for model loading and
data processing, combined with HuggingFace's Trainer for the training loop.
The conversation format matches inference_med.py:
    [INST] <image>\n{question} [/INST]{answer}</s>

For standard LLaVA, use train.py instead.

Prerequisites:
    cd LLaVA-Med && pip install -e .
"""

import argparse
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import set_seed
from pathlib import Path

# Original LLaVA-Med codebase imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from utils.load_data import MedicalVQALoader


DEFAULT_MODEL_PATH = "microsoft/llava-med-v1.5-mistral-7b"


# ── Dataset ─────────────────────────────────────────────────────────────────────

class LLaVAMedTrainDataset(Dataset):
    """
    Training dataset for LLaVA-Med.

    Uses the same conversation format as inference_med.py:
        [INST] <image>\\n{question} [/INST]{answer}</s>

    tokenizer_image_token() replaces <image> with IMAGE_TOKEN_INDEX (-200).
    During forward() the model expands this token into actual image features
    via prepare_inputs_labels_for_multimodal().

    Labels are set to -100 for all prompt tokens (up to and including [/INST])
    so the loss is only computed on the assistant's answer.
    """

    def __init__(self, samples, tokenizer, image_processor, model_config,
                 conv_template="mistral_instruct", max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_template = conv_template
        self.max_length = max_length

        # Ensure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")

        # ── Process image ────────────────────────────────────────────────
        image_tensor = process_images([image], self.image_processor, self.model_config)
        image_tensor = image_tensor.squeeze(0)  # [C, H, W]

        # ── Build conversation (same as inference_med.py, but with answer) ─
        conv = conv_templates[self.conv_template].copy()
        question = f"{DEFAULT_IMAGE_TOKEN}\n{sample.question}"
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], sample.answer)
        prompt = conv.get_prompt()

        # ── Tokenise (replaces <image> text with IMAGE_TOKEN_INDEX) ──────
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )  # 1-D tensor of shape [seq_len]

        # ── Truncate / pad to max_length ─────────────────────────────────
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]

        attention_mask = torch.ones_like(input_ids)

        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_len,), self.pad_token_id, dtype=input_ids.dtype),
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(pad_len, dtype=attention_mask.dtype),
            ])

        # ── Labels: mask prompt, keep only answer ────────────────────────
        labels = input_ids.clone()

        # Find [/INST] marker (end of instruction → start of answer)
        inst_end_ids = self.tokenizer.encode("[/INST]", add_special_tokens=False)
        ids_list = input_ids.tolist()

        answer_start = None
        for i in range(len(ids_list) - len(inst_end_ids) + 1):
            if ids_list[i : i + len(inst_end_ids)] == inst_end_ids:
                answer_start = i + len(inst_end_ids)
                break

        if answer_start is not None:
            labels[:answer_start] = -100
        else:
            # Fallback: mask first 2/3
            labels[: len(labels) * 2 // 3] = -100

        # Mask padding tokens and IMAGE_TOKEN_INDEX (image feature positions)
        labels[input_ids == self.pad_token_id] = -100
        labels[input_ids == IMAGE_TOKEN_INDEX] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": image_tensor,
            "labels": labels,
        }


# ── Arguments ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLaVA-Med on Medical VQA"
    )

    # Model
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"LLaVA-Med model name or local path (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--model_base", type=str, default=None,
                        help="Base model path (only needed for LoRA checkpoints)")
    parser.add_argument("--conv_template", type=str, default="mistral_instruct",
                        help="Conversation template name")

    # Data
    parser.add_argument("--data_root", type=str, default="data",
                        help="Root directory containing datasets")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["pathvqa", "slake", "vqarad"],
                        help="Dataset to use")
    parser.add_argument("--train_split", type=str, default="train",
                        help="Training split name")
    parser.add_argument("--val_split", type=str, default="validation",
                        help="Validation split name")

    # Training
    parser.add_argument("--output_dir", type=str, default="./outputs/llava-med",
                        help="Output directory for checkpoints")
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

    return parser.parse_args()


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

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

    # ── Load model via LLaVA-Med codebase ───────────────────────────────
    print(f"Loading LLaVA-Med model: {args.model_path}")
    model_name = Path(args.model_path).name or "llava-med-v1.5-mistral-7b"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=model_name,
        device=str(device),
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    print(f"Model loaded (context_len={context_len})")

    # ── Load datasets ───────────────────────────────────────────────────
    loader = MedicalVQALoader(data_root=args.data_root)

    print(f"Loading {args.dataset} {args.train_split} split...")
    train_samples = loader.load_dataset(args.dataset, split=args.train_split)
    train_dataset = LLaVAMedTrainDataset(
        train_samples, tokenizer, image_processor, model.config,
        conv_template=args.conv_template, max_length=args.max_length,
    )
    print(f"Train: {len(train_dataset)} samples")

    val_dataset = None
    try:
        val_samples = loader.load_dataset(args.dataset, split=args.val_split)
        val_dataset = LLaVAMedTrainDataset(
            val_samples, tokenizer, image_processor, model.config,
            conv_template=args.conv_template, max_length=args.max_length,
        )
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
        remove_unused_columns=False,  # IMPORTANT: keep 'images' key in batch
        report_to="tensorboard",
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
        processing_class=tokenizer,
    )

    # ── Train ───────────────────────────────────────────────────────────
    print("Starting training...")
    print(f"  Format: [INST] <image>\\n{{question}} [/INST]{{answer}}")
    print(f"  Loss computed only on assistant response tokens")

    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model()
    trainer.save_state()

    print("Training completed!")
    print(f"Training loss: {result.training_loss}")


if __name__ == "__main__":
    main()
