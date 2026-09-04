"""Fine-tune Qwen-VL (2.5 / 3) on Medical VQA (MC) via HuggingFace Trainer."""

import argparse
import random
from collections import Counter

import torch
import wandb
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

from utils.ablation import FgrTrainLookup, load_fgr_supervised_train_lookup, normalize_ablation_image_key
from utils.load_data import MedicalVQALoader, VQASample
from utils.paths import (
    get_data_root,
    get_mc_data_root,
    get_som_dir,
    get_som_selection_dir,
    get_train_output_dir_qwen,
)
from utils.som import SomResolver
from utils.som_select import question_with_selected_som_hint
from utils.som_selection import SomSelectionResolver
from utils.qwen_model import default_base_model_for_path, load_qwen_vl_model, resolve_local_model_path
from utils.qwen_vl import (
    DEFAULT_MC_EXPORT_SUFFIX_PSEUDO_VERIFY,
    DEFAULT_MODEL_PATH,
    QwenVLDataCollator,
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


class QwenMCTrainDataset(Dataset):
    """MC Qwen dataset; loss only on assistant tokens (JSON or answer letter)."""

    def __init__(
        self,
        samples,
        processor,
        max_length=512,
        *,
        use_reasoning: bool = False,
        answer_first: bool = False,
        both_json_orders: bool = False,
        fgr_lookup: FgrTrainLookup | None = None,
        data_root: str | None = None,
        som_resolver: SomResolver | None = None,
        som_selection: SomSelectionResolver | None = None,
    ):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.use_reasoning = use_reasoning
        self.answer_first = answer_first
        self.both_json_orders = bool(both_json_orders and use_reasoning)
        self.fgr_lookup = fgr_lookup
        self.data_root = data_root
        self.som_resolver = som_resolver
        self.som_selection = som_selection
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        # (sample_index, answer_first) — dual mode doubles each sample as RF then AF.
        if self.both_json_orders:
            self._index: list[tuple[int, bool]] = [
                pair for i in range(len(samples)) for pair in ((i, False), (i, True))
            ]
        else:
            self._index = [(i, answer_first) for i in range(len(samples))]

    def _resolve_image_path(self, sample: VQASample) -> str:
        if self.fgr_lookup is not None:
            key = (
                normalize_ablation_image_key(sample.image_path, data_root=self.data_root),
                sample.question.strip(),
            )
            path = self.fgr_lookup.get(key, sample.image_path)
        else:
            path = sample.image_path
        if self.som_resolver is not None:
            path = self.som_resolver.resolve_or_original(path)
        return path

    def _load_image(self, sample: VQASample) -> tuple[Image.Image, list[str]]:
        """Return (image, SoM labels drawn on it); labels are empty without selection."""
        if self.som_selection is not None:
            return self.som_selection.render_with_labels(sample.image_path, sample.question)
        return Image.open(self._resolve_image_path(sample)).convert("RGB"), []

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        sample_i, answer_first = self._index[idx]
        sample = self.samples[sample_i]
        image, som_labels = self._load_image(sample)
        # Must match the prompt inference builds for the same sample.
        question = question_with_selected_som_hint(sample.question, som_labels)

        if not sample.mc_options:
            raise ValueError("MC options missing; ensure mc_data_root points to MC exports.")

        if self.use_reasoning:
            reasoning_raw = pseudo_reasoning_raw_from_metadata(sample.metadata)
            if not reasoning_raw:
                raise ValueError("pseudo_reasoning_raw missing for reasoning training sample.")
            conversation = build_mc_reasoning_conversation(
                question,
                sample.mc_options,
                reasoning_raw=reasoning_raw,
                answer=sample.answer,
                image=image,
                answer_first=answer_first,
            )
        else:
            conversation = build_mc_conversation(
                question,
                sample.mc_options,
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
        item["labels"] = mask_labels_assistant_only(
            self.processor, item["input_ids"], conversation
        )
        return item


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen-VL on Medical VQA (MC)")
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
        help="Use answer-first JSON schema {answer, reasoning} instead of {reasoning, answer}. Only effective with --use_reasoning.",
    )
    p.add_argument(
        "--both_json_orders",
        action="store_true",
        help="With --use_reasoning: train on both RF and AF JSON for each sample (2x examples). "
        "Incompatible with --answer_first. Val stays rationale-first.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["pathvqa", "slake", "vqarad", "medvqa2019", "all"],
    )
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--val_split", type=str, default="validation")
    p.add_argument("--output_dir", type=str, default=get_train_output_dir_qwen())
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
    p.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Cap training set size: random subset (seed=42) after verify filter when --use_reasoning.",
    )
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    p.add_argument(
        "--fgr_crops_json",
        type=str,
        default=None,
        help="fgr_region_comparison JSON: supervised winning crop for training images.",
    )
    p.add_argument("--wandb_project", type=str, default="medvqa")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument(
        "--use_som",
        action="store_true",
        default=False,
        help="Replace images with pre-annotated SoM versions during training.",
    )
    p.add_argument(
        "--som_dir",
        type=str,
        default=get_som_dir(),
        help="Root directory of SoM outputs (default: MEDVQA_SOM_DIR or 02_data/som).",
    )
    p.add_argument(
        "--use_som_selection",
        action="store_true",
        default=False,
        help="Draw only the SoM segments pre-selected for each question (per image AND "
        "question, unlike --use_som). Run scripts/precompute_som_selection.py first.",
    )
    p.add_argument(
        "--selection_dir",
        type=str,
        default=get_som_selection_dir(),
        help="Root of selection manifests (default: MEDVQA_SOM_SELECTION_DIR or 02_data/som_selection).",
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
    if args.use_som and args.use_som_selection:
        raise ValueError(
            "--use_som (all marks per image) and --use_som_selection (marks selected "
            "per question) both replace the image; pick one."
        )
    if args.use_som_selection and args.fgr_crops_json:
        raise ValueError("--use_som_selection renders from the source image; --fgr_crops_json would be ignored.")

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
    model_short = model_short_name(args.model_path)
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"{model_short}_{args.dataset}_{ft_tag}_train",
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

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    resolved_model_path = resolve_local_model_path(args.model_path)
    print(f"Loading model: {resolved_model_path}")
    model = load_qwen_vl_model(
        resolved_model_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules,
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            ),
        )
        model.print_trainable_parameters()

    processor_path = resolve_processor_path(
        resolved_model_path,
        default_path=default_base_model_for_path(resolved_model_path),
    )
    if processor_path != resolved_model_path:
        print(f"Processor not in checkpoint, loading from: {processor_path}")
    processor = AutoProcessor.from_pretrained(processor_path)
    configure_qwen_processor_pixels(processor)

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

    fgr_lookup = None
    if args.fgr_crops_json:
        fgr_lookup = load_fgr_supervised_train_lookup(
            args.fgr_crops_json, data_root=args.data_root
        )

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

    som_datasets = ["pathvqa", "slake", "vqarad", "medvqa2019"] if use_all else [args.dataset]

    som_resolver: SomResolver | None = None
    if args.use_som:
        som_resolver = SomResolver(args.som_dir)
        for ds in som_datasets:
            som_resolver.load_dataset(ds)
        print(f"SoM enabled — images loaded from {args.som_dir}")

    som_selection: SomSelectionResolver | None = None
    if args.use_som_selection:
        som_selection = SomSelectionResolver(args.som_dir, args.selection_dir)
        for ds in som_datasets:
            som_selection.load_dataset(ds)
        found, total = som_selection.coverage(
            (s.image_path, s.question) for s in train_samples
        )
        print(
            f"SoM selection enabled — {found}/{total} train samples have a selection "
            f"({args.selection_dir})"
        )
        if found == 0:
            raise ValueError(
                "No train sample matches the selection manifest. Re-run "
                "scripts/precompute_som_selection.py for this dataset and split."
            )

    train_dataset = QwenMCTrainDataset(
        train_samples,
        processor,
        args.max_length,
        use_reasoning=args.use_reasoning,
        answer_first=args.answer_first,
        both_json_orders=args.both_json_orders,
        fgr_lookup=fgr_lookup,
        data_root=args.data_root,
        som_resolver=som_resolver,
        som_selection=som_selection,
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
        val_dataset = QwenMCTrainDataset(
            val_samples,
            processor,
            args.max_length,
            data_root=args.data_root,
            som_resolver=som_resolver,
            som_selection=som_selection,
            use_reasoning=False,
            answer_first=False,
            both_json_orders=False,
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
            report_to="wandb",
            seed=args.seed,
            optim="adamw_8bit" if args.use_8bit_optimizer else "adamw_torch",
            max_grad_norm=1.0,
            save_only_model=True,
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=QwenVLDataCollator(tokenizer=processor.tokenizer),
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
    print(f"Starting Qwen-VL training (loss on {loss_desc})...")
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print(f"Saving model to {args.output_dir}")
    trainer.save_model()
    trainer.save_state()
    processor.save_pretrained(args.output_dir)
    print(f"Training completed! loss={result.training_loss}")

    wandb.log(
        {
            "final_train_loss": result.training_loss,
            "train_samples": len(train_dataset),
            "train_unique_samples": len(train_samples),
            "train_samples_total_before_filter": train_total,
            "train_samples_after_verify_filter": train_samples_after_filter if args.use_reasoning else None,
            "max_train_samples": args.max_train_samples,
            "both_json_orders": bool(args.both_json_orders),
            "val_samples": len(val_dataset) if val_dataset else 0,
        }
    )
    wandb.finish()


if __name__ == "__main__":
    main()
