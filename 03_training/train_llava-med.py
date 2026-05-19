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
from collections import Counter
import torch
import wandb
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
from utils.mc_vqa import format_mc_user_text
from utils.paths import get_data_root, get_mc_data_root, get_train_output_dir_llava_med


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

    def __init__(
        self,
        samples,
        tokenizer,
        image_processor,
        model_config,
        conv_template="mistral_instruct",
        max_length=512,
        dataset="vqarad",
    ):
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
        if not sample.mc_options:
            raise ValueError("MC options missing in sample; ensure mc_data_root points to MC exports.")
        mc_user_text = format_mc_user_text(sample.question, sample.mc_options)
        question = f"{DEFAULT_IMAGE_TOKEN}\n{mc_user_text}"

        # Build prompt WITH answer (full training sequence)
        conv_full = conv_templates[self.conv_template].copy()
        conv_full.append_message(conv_full.roles[0], question)
        conv_full.append_message(conv_full.roles[1], sample.answer)
        full_prompt = conv_full.get_prompt()

        # ── Tokenise (replaces <image> text with IMAGE_TOKEN_INDEX) ──────
        input_ids = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )  # 1-D tensor of shape [seq_len]

        content_len = len(input_ids)

        # Tokenize just the answer to find how many tokens it occupies.
        # +2 accounts for EOS and a potential separator/space token
        # that BPE may insert between [/INST] and the answer.
        answer_ids = self.tokenizer.encode(sample.answer, add_special_tokens=False)
        answer_token_len = len(answer_ids) + 2

        # ── Truncate / pad to max_length ─────────────────────────────────
        if content_len > self.max_length:
            input_ids = input_ids[: self.max_length]
            content_len = self.max_length

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

        # ── Labels: mask everything, then unmask only the answer at the end
        labels = torch.full_like(input_ids, -100)
        answer_start = max(0, content_len - answer_token_len)
        labels[answer_start:content_len] = input_ids[answer_start:content_len]

        # Safety: mask IMAGE_TOKEN_INDEX positions (shouldn't be in answer, but just in case)
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
    parser.add_argument("--output_dir", type=str, default=get_train_output_dir_llava_med(),
                        help="Output directory for checkpoints ($MEDVQA_OUTPUTS_DIR/llava-med)")
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

    # ── Load model via LLaVA-Med codebase ───────────────────────────────
    print(f"Loading LLaVA-Med model: {args.model_path}")
    model_name = Path(args.model_path).name or "llava-med-v1.5-mistral-7b"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=model_name,
        device=str(device),
    )

    # Model is loaded in fp16; cast to bf16 for consistent precision.
    # (fp32 causes OOM with Adam optimizer states; the original fp16 + bf16
    # AMP mix caused NaN. Pure bf16 avoids both issues.)
    model = model.to(torch.bfloat16)

    # ── Diagnostic: check truncation config ─────────────────────────
    tok_max = getattr(model.config, 'tokenizer_model_max_length', 'NOT SET')
    print(f"  model.config.tokenizer_model_max_length = {tok_max}")
    print(f"  tokenizer.model_max_length               = {tokenizer.model_max_length}")
    print(f"  context_len                               = {context_len}")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
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

    print(f"Model loaded (context_len={context_len})")

    # ── Load datasets ───────────────────────────────────────────────────
    loader = MedicalVQALoader(data_root=args.data_root, mc_data_root=args.mc_data_root)
    use_all = args.dataset == "all"
    if use_all:
        # Source tag per sample: VQASample.dataset. VQA-Rad val uses test split (no validation in loader).
        print(f"Loading combined PathVQA+SLAKE+VQA-Rad {args.train_split} split...")
        train_samples = loader.load_combined(args.train_split)
        train_dataset = LLaVAMedTrainDataset(
            train_samples,
            tokenizer,
            image_processor,
            model.config,
            conv_template=args.conv_template,
            max_length=args.max_length,
            dataset=args.dataset,
        )
        print(f"Train: {len(train_dataset)} samples ({dict(Counter(s.dataset for s in train_samples))})")
    else:
        print(f"Loading {args.dataset} {args.train_split} split...")
        train_samples = loader.load_dataset(args.dataset, split=args.train_split)
        train_dataset = LLaVAMedTrainDataset(
            train_samples,
            tokenizer,
            image_processor,
            model.config,
            conv_template=args.conv_template,
            max_length=args.max_length,
            dataset=args.dataset,
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
        val_dataset = LLaVAMedTrainDataset(
            val_samples,
            tokenizer,
            image_processor,
            model.config,
            conv_template=args.conv_template,
            max_length=args.max_length,
            dataset=args.dataset,
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
        evaluation_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=bool(val_dataset),
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,  # IMPORTANT: keep 'images' key in batch
        report_to="wandb",
        seed=args.seed,
        optim="adamw_8bit" if args.use_8bit_optimizer else "adamw_torch",
        max_grad_norm=1.0,
        save_only_model=True,
    )

    # ── Debug Trainer: log label stats inside forward pass ──────────
    class DebugTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels_in = inputs.get("labels")
            step = self.state.global_step
            if labels_in is not None and step < 5:
                n_active = (labels_in != -100).sum().item()
                print(f"  [Step {step}] Labels IN  -> active={n_active}, total={labels_in.numel()}")

            result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            loss_val = result[0].item() if return_outputs else result.item()

            if step < 5:
                print(f"  [Step {step}] Loss = {loss_val:.6f}")

            return result

    trainer = DebugTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # ── Sanity check: verify label masking on first sample ──────────
    sample0 = train_dataset[0]
    n_total = sample0["labels"].numel()
    n_masked = (sample0["labels"] == -100).sum().item()
    n_active = n_total - n_masked
    active_ids = sample0["labels"][sample0["labels"] != -100]
    decoded_active = tokenizer.decode(active_ids, skip_special_tokens=False)
    print(f"\n[Label sanity check - sample 0]")
    print(f"  Total tokens: {n_total}  |  Masked (-100): {n_masked}  |  Active (loss): {n_active}")
    print(f"  Active tokens decoded: '{decoded_active}'")
    if n_active == 0:
        raise RuntimeError("BUG: all labels are -100 -> loss will always be 0. Aborting.")

    # ── Train ───────────────────────────────────────────────────────────
    print("\nStarting training...")
    print(f"  Format: [INST] <image>\\n{{question}} [/INST]{{answer}}")
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
