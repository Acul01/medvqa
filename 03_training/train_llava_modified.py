"""Fine-tune standard LLaVA with spatial filter on Medical VQA (MC) via HuggingFace Trainer."""

import argparse
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

from utils.load_data import MedicalVQALoader
from utils.mc_vqa import format_mc_user_text
from utils.paths import get_data_root, get_mc_data_root, get_train_output_dir_llava
from utils.qwen_vl import (
    DEFAULT_MC_EXPORT_SUFFIX_PSEUDO_VERIFY,
    format_mc_reasoning_assistant_json,
    format_mc_reasoning_user_text,
    is_pseudo_reasoning_verified,
    pseudo_reasoning_raw_from_metadata,
)
from utils.spatial_filter import (
    LlavaWithSpatialFilter,
    get_llava_projector_output_dim,
)

DEFAULT_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"


class LLaVATrainDataset(Dataset):
    """MC LLaVA dataset: loss only on assistant tokens (JSON or answer letter)."""

    def __init__(
        self,
        samples,
        processor,
        max_length=512,
        *,
        use_reasoning: bool = False,
        answer_first: bool = False,
    ):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.use_reasoning = use_reasoning
        self.answer_first = answer_first
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        if getattr(processor, "patch_size", None) is None:
            processor.patch_size = 14
        if getattr(processor, "vision_feature_select_strategy", None) is None:
            processor.vision_feature_select_strategy = "default"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")

        if not sample.mc_options:
            raise ValueError("MC options missing; ensure mc_data_root points to MC exports.")

        if self.use_reasoning:
            reasoning_raw = pseudo_reasoning_raw_from_metadata(sample.metadata)
            if not reasoning_raw:
                raise ValueError("pseudo_reasoning_raw missing for reasoning training sample.")
            user_text = format_mc_reasoning_user_text(
                sample.question,
                sample.mc_options,
                answer_first=self.answer_first,
            )
            assistant_text = format_mc_reasoning_assistant_json(
                reasoning_raw,
                sample.answer,
                answer_first=self.answer_first,
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune standard LLaVA with spatial filter on Medical VQA (MC)"
    )
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--data_root", type=str, default=get_data_root())
    p.add_argument("--mc_data_root", type=str, default=get_mc_data_root())
    p.add_argument("--mc_export_suffix", type=str, default="")
    p.add_argument(
        "--use_reasoning",
        action="store_true",
        help="Train on verified pseudo reasoning (qwen_verify=True or is_correct=True).",
    )
    p.add_argument(
        "--answer_first",
        action="store_true",
        help="Use answer-first JSON schema {answer, reasoning}. Only effective with --use_reasoning.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["pathvqa", "slake", "vqarad", "medvqa2019", "all"],
    )
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--val_split", type=str, default="validation")
    p.add_argument("--output_dir", type=str, default=get_train_output_dir_llava())
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--use_8bit_optimizer", action="store_true")
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj"],
    )
    p.add_argument(
        "--filter_bottleneck",
        type=int,
        default=64,
        help="Hidden dim of the scorer MLP inside SpatialFilterLayer (default: 64).",
    )
    return p.parse_args()


def _load_samples(
    loader: MedicalVQALoader,
    dataset: str,
    split: str,
    *,
    vqarad_split: str | None = None,
):
    if dataset == "all":
        if vqarad_split is not None:
            return loader.load_combined(split, vqarad_split=vqarad_split)
        return loader.load_combined(split)
    return loader.load_dataset(dataset, split=split)


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.use_8bit_optimizer:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            print("Warning: bitsandbytes not installed, falling back to standard optimizer")
            args.use_8bit_optimizer = False

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    print(f"Loading model: {args.model_path}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    d_model = get_llava_projector_output_dim(model)
    n_params = (
        (d_model + 2) * args.filter_bottleneck
        + args.filter_bottleneck
        + d_model * 2 * d_model
        + d_model
    )
    print(
        f"SpatialFilterLayer: d_model={d_model}, "
        f"bottleneck={args.filter_bottleneck}, params≈{n_params:,}"
    )
    model = LlavaWithSpatialFilter(model, d_model=d_model, bottleneck=args.filter_bottleneck)

    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        inner = get_peft_model(model.model, lora_cfg)
        inner.print_trainable_parameters()
        model.model = inner

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
    print(f"Loading train ({args.train_split}, suffix={train_suffix!r})...")
    train_samples = _load_samples(loader, args.dataset, args.train_split)

    train_total = len(train_samples)
    if args.use_reasoning:
        train_samples = [
            s
            for s in train_samples
            if is_pseudo_reasoning_verified(s.metadata)
            and pseudo_reasoning_raw_from_metadata(s.metadata)
        ]
        print(
            f"Train: {len(train_samples)} / {train_total} "
            f"(use_reasoning, qwen_verify/is_correct=True)"
        )
        if not train_samples:
            raise ValueError("No verified reasoning samples after filtering.")
    else:
        print(f"Train: {len(train_samples)} samples")

    train_dataset = LLaVATrainDataset(
        train_samples,
        processor,
        args.max_length,
        use_reasoning=args.use_reasoning,
        answer_first=args.answer_first,
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
                f"Loading val: combined split={args.val_split!r}, vqarad_split='test'"
            )
        val_samples = _load_samples(
            val_loader,
            args.dataset,
            args.val_split,
            vqarad_split="test" if use_all else None,
        )
        val_dataset = LLaVATrainDataset(
            val_samples,
            processor,
            args.max_length,
        )
        print(f"Validation: {len(val_dataset)} samples")
    except Exception as e:
        print(f"No validation split available: {e}")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
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
            report_to="none",
            seed=args.seed,
            optim="adamw_8bit" if args.use_8bit_optimizer else "adamw_torch",
            max_grad_norm=1.0,
            save_only_model=True,
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor.tokenizer,
    )

    if args.use_reasoning:
        order = "answer + reasoning" if args.answer_first else "reasoning + answer"
        loss_desc = f"assistant JSON ({order})"
    else:
        loss_desc = "assistant answer tokens only"
    print(f"Starting LLaVA training with spatial filter (loss on {loss_desc})...")
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print(f"Saving model to {args.output_dir}")
    # Save base LLaVA directly (not through LlavaWithSpatialFilter).
    base_model = model.model
    if args.use_lora:
        base_model = base_model.merge_and_unload()
    base_model.save_pretrained(args.output_dir, max_shard_size="4GB", safe_serialization=True)
    trainer.save_state()
    processor.save_pretrained(args.output_dir)

    filter_path = Path(args.output_dir) / "spatial_filter.pt"
    torch.save(model.spatial_filter.state_dict(), filter_path)
    print(f"Spatial filter saved → {filter_path}")

    print(f"Training completed! loss={result.training_loss}")


if __name__ == "__main__":
    main()
