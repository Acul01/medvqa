"""
Training script for standard LLaVA on Medical VQA datasets.

Uses HuggingFace's LlavaForConditionalGeneration + AutoProcessor + Trainer.
The conversation format matches inference_llava.py:
    USER: <image>\n{question}
    ASSISTANT: {answer}

With --use_reasoning the assistant target is JSON {reasoning, answer}
(or answer-first), matching train_qwen.py / utils/qwen_vl.py prompts.

The model handles image-token expansion internally during forward(),
so the dataset only needs a single <image> token in the input_ids.

For LLaVA-Med, use train_llava-med.py instead.
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import torch
import wandb
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

from utils.load_data import MedicalVQALoader, VQASample
from utils.mc_vqa import format_mc_user_text
from utils.paths import get_data_root, get_mc_data_root, get_som_dir, get_train_output_dir_llava
from utils.qwen_vl import (
    DEFAULT_MC_EXPORT_SUFFIX_PSEUDO_VERIFY,
    format_mc_reasoning_assistant_json,
    format_mc_reasoning_user_text,
    is_pseudo_reasoning_verified,
    pseudo_reasoning_raw_from_metadata,
)
from utils.som import SomResolver


DEFAULT_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"


# ── Dataset ─────────────────────────────────────────────────────────────────────

class LLaVATrainDataset(Dataset):
    """
    Training dataset for standard LLaVA.

    Uses the same conversation format as inference_llava.py:
        USER: <image>\\n{question}
        ASSISTANT: {answer | JSON}

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
        *,
        use_reasoning: bool = False,
        answer_first: bool = False,
        both_json_orders: bool = False,
    ):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.som_resolver = som_resolver
        self.use_reasoning = use_reasoning
        self.answer_first = answer_first
        self.both_json_orders = bool(both_json_orders and use_reasoning)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        if getattr(processor, "patch_size", None) is None:
            processor.patch_size = 14
        if getattr(processor, "vision_feature_select_strategy", None) is None:
            processor.vision_feature_select_strategy = "default"

        # (sample_index, answer_first) — dual mode doubles each sample as RF then AF.
        if self.both_json_orders:
            self._index: list[tuple[int, bool]] = [
                pair for i in range(len(samples)) for pair in ((i, False), (i, True))
            ]
        else:
            self._index = [(i, answer_first) for i in range(len(samples))]

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        sample_i, answer_first = self._index[idx]
        sample = self.samples[sample_i]
        image_path = (
            self.som_resolver.resolve_or_original(sample.image_path)
            if self.som_resolver is not None
            else sample.image_path
        )
        image = Image.open(image_path).convert("RGB")

        if not sample.mc_options:
            raise ValueError("MC options missing in sample; ensure mc_data_root points to MC exports.")

        if self.use_reasoning:
            reasoning_raw = pseudo_reasoning_raw_from_metadata(sample.metadata)
            if not reasoning_raw:
                raise ValueError("pseudo_reasoning_raw missing for reasoning training sample.")
            user_text = format_mc_reasoning_user_text(
                sample.question,
                sample.mc_options,
                answer_first=answer_first,
            )
            assistant_text = format_mc_reasoning_assistant_json(
                reasoning_raw,
                sample.answer,
                answer_first=answer_first,
            )
        else:
            user_text = format_mc_user_text(sample.question, sample.mc_options)
            assistant_text = sample.answer

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_text},
                ],
            },
        ]

        full_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=False
        )

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

        labels = input_ids.clone()

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
            prompt_only_text = self.processor.apply_chat_template(
                [conversation[0]], add_generation_prompt=True
            )
            prompt_only_ids = self.processor.tokenizer(
                prompt_only_text, add_special_tokens=False
            )["input_ids"]
            labels[: len(prompt_only_ids)] = -100

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

    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"HuggingFace model name or local path (default: {DEFAULT_MODEL_PATH})")

    parser.add_argument("--data_root", type=str, default=get_data_root(),
                        help="Root directory containing raw image folders ($MEDVQA_DATA_ROOT or 02_data)")
    parser.add_argument("--mc_data_root", type=str, default=get_mc_data_root(),
                        help="Root directory containing MC JSON/CSV exports ($MEDVQA_MC_ROOT or 02_data/data_mc_openai)")
    parser.add_argument("--mc_export_suffix", type=str, default="",
                        help="MC export filename suffix (e.g. _verify). Empty + --use_reasoning → _pseudo_pr_qwen_verify.")
    parser.add_argument(
        "--use_reasoning",
        action="store_true",
        help="Train on verified pseudo reasoning (qwen_verify=True or is_correct=True).",
    )
    parser.add_argument(
        "--answer_first",
        action="store_true",
        help="Use answer-first JSON schema {answer, reasoning}. Only effective with --use_reasoning.",
    )
    parser.add_argument(
        "--both_json_orders",
        action="store_true",
        help="With --use_reasoning: train on both RF and AF JSON for each sample (2x examples). "
        "Incompatible with --answer_first. Val stays rationale-first.",
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["pathvqa", "slake", "vqarad", "medvqa2019", "all"],
                        help="Dataset to use; 'all' = PathVQA+SLAKE+VQA-Rad+Med-VQA-2019 (prompt names from each sample)")
    parser.add_argument("--train_split", type=str, default="train",
                        help="Training split name")
    parser.add_argument("--val_split", type=str, default="validation",
                        help="Validation split name")

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
                        help="Maximum sequence length (use >=2048 with --use_reasoning)")
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
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Cap training set size: random subset (seed=42) after verify filter when --use_reasoning.",
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume training from checkpoint path")

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


def _subsample_train_samples(samples: list[VQASample], max_n: int, *, seed: int = 42) -> list[VQASample]:
    if max_n is None or max_n <= 0 or len(samples) <= max_n:
        return samples
    rng = random.Random(seed)
    indices = rng.sample(range(len(samples)), max_n)
    return [samples[i] for i in indices]


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.both_json_orders and not args.use_reasoning:
        raise ValueError("--both_json_orders requires --use_reasoning.")
    if args.both_json_orders and args.answer_first:
        raise ValueError("--both_json_orders is incompatible with --answer_first.")

    model_short = _model_short_name(args.model_path)
    if args.use_reasoning:
        if args.both_json_orders:
            reasoning_tag = "reasoning-both-"
        elif args.answer_first:
            reasoning_tag = "reasoning-answer-first-"
        else:
            reasoning_tag = "reasoning-"
    else:
        reasoning_tag = ""
    ft_tag = reasoning_tag + ("lora" if args.use_lora else "full-ft")
    run_name = args.wandb_run_name or f"{model_short}_{args.dataset}_{ft_tag}_train"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=args.dataset,
        tags=["training", model_short, args.dataset, ft_tag],
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

    processor_path = args.model_path
    if Path(args.model_path).exists() and not Path(args.model_path, "preprocessor_config.json").exists():
        processor_path = DEFAULT_MODEL_PATH
        print(f"Processor not in checkpoint, loading from: {processor_path}")
    processor = AutoProcessor.from_pretrained(processor_path)

    train_suffix = args.mc_export_suffix or (
        DEFAULT_MC_EXPORT_SUFFIX_PSEUDO_VERIFY if args.use_reasoning else ""
    )
    loader = MedicalVQALoader(
        data_root=args.data_root,
        mc_data_root=args.mc_data_root,
        mc_export_suffix=train_suffix,
    )
    use_all = args.dataset == "all"

    som_resolver: SomResolver | None = None
    if args.use_som:
        som_resolver = SomResolver(args.som_dir)
        datasets_to_load = ["pathvqa", "slake", "vqarad", "medvqa2019"] if use_all else [args.dataset]
        for ds in datasets_to_load:
            som_resolver.load_dataset(ds)
        print(f"SoM enabled — images loaded from {args.som_dir}")

    if use_all:
        print(f"Loading combined PathVQA+SLAKE+VQA-Rad {args.train_split} split (suffix={train_suffix!r})...")
        train_samples = loader.load_combined(args.train_split)
    else:
        print(f"Loading {args.dataset} {args.train_split} split (suffix={train_suffix!r})...")
        train_samples = loader.load_dataset(args.dataset, split=args.train_split)

    train_total = len(train_samples)
    train_samples_after_filter = train_total
    if args.use_reasoning:
        train_samples = [
            s
            for s in train_samples
            if is_pseudo_reasoning_verified(s.metadata)
            and pseudo_reasoning_raw_from_metadata(s.metadata)
        ]
        train_samples_after_filter = len(train_samples)
        print(
            f"Train: {len(train_samples)} / {train_total} "
            f"(use_reasoning, qwen_verify/is_correct=True)"
        )
        if not train_samples:
            raise ValueError(
                "No verified reasoning samples after filtering. "
                "Ensure MC JSON has qwen_verify=True or is_correct=True and non-empty pseudo_reasoning_raw."
            )
    else:
        print(f"Train: {len(train_samples)} samples")

    if args.max_train_samples is not None:
        if not args.use_reasoning:
            raise ValueError("--max_train_samples requires --use_reasoning (subset of verified samples).")
        before_cap = len(train_samples)
        train_samples = _subsample_train_samples(
            train_samples,
            args.max_train_samples,
            seed=42,
        )
        print(
            f"Train subset: {len(train_samples)} / {before_cap} "
            f"(max_train_samples={args.max_train_samples}, seed=42)"
        )

    if args.both_json_orders:
        print(
            f"Train examples: {len(train_samples) * 2} "
            f"({len(train_samples)} samples × 2 JSON orders RF+AF)"
        )

    train_dataset = LLaVATrainDataset(
        train_samples,
        processor,
        args.max_length,
        dataset=args.dataset,
        som_resolver=som_resolver,
        use_reasoning=args.use_reasoning,
        answer_first=args.answer_first,
        both_json_orders=args.both_json_orders,
    )
    if use_all and not args.use_reasoning:
        print(f"  per dataset: {dict(Counter(s.dataset for s in train_samples))}")

    val_dataset = None
    try:
        val_loader = MedicalVQALoader(
            data_root=args.data_root,
            mc_data_root=args.mc_data_root,
            mc_export_suffix="",
        )
        if use_all:
            print(
                f"Loading validation: PathVQA/SLAKE split={args.val_split!r}, "
                "VQA-Rad split='test' (VQA-Rad has no validation split in this loader)."
            )
            val_samples = val_loader.load_combined(args.val_split, vqarad_split="test")
        else:
            val_samples = val_loader.load_dataset(args.dataset, split=args.val_split)
        val_dataset = LLaVATrainDataset(
            val_samples,
            processor,
            args.max_length,
            dataset=args.dataset,
            som_resolver=som_resolver,
            use_reasoning=False,
            answer_first=False,
            both_json_orders=False,
        )
        if use_all:
            print(f"Validation: {len(val_dataset)} samples ({dict(Counter(s.dataset for s in val_samples))})")
        else:
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

    if args.use_reasoning:
        if args.both_json_orders:
            order = "both RF {reasoning,answer} and AF {answer,reasoning}"
        elif args.answer_first:
            order = "answer + reasoning"
        else:
            order = "reasoning + answer"
        loss_desc = f"assistant JSON ({order})"
    else:
        loss_desc = "assistant answer tokens only"
    print("Starting training...")
    print(f"  Conversation format: USER: <image>\\n{{question}} ASSISTANT: {{{loss_desc}}}")
    print(f"  Loss computed only on assistant response tokens")

    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print(f"Saving model to {args.output_dir}")
    trainer.save_model()
    trainer.save_state()

    print("Training completed!")
    print(f"Training loss: {result.training_loss}")

    wandb.log({
        "final_train_loss": result.training_loss,
        "train_samples": len(train_dataset),
        "train_unique_samples": len(train_samples),
        "train_samples_total_before_filter": train_total,
        "train_samples_after_verify_filter": train_samples_after_filter if args.use_reasoning else None,
        "max_train_samples": args.max_train_samples,
        "both_json_orders": bool(args.both_json_orders),
        "val_samples": len(val_dataset) if val_dataset else 0,
    })
    wandb.finish()


if __name__ == "__main__":
    main()
