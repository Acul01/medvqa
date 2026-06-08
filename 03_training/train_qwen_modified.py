"""Fine-tune Qwen2.5-VL with spatial filter on Medical VQA (MC) via HuggingFace Trainer."""

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

from utils.load_data import MedicalVQALoader
from utils.paths import get_data_root, get_mc_data_root, get_train_output_dir_qwen
from utils.qwen_vl import (
    DEFAULT_MC_EXPORT_SUFFIX_PSEUDO_VERIFY,
    DEFAULT_MODEL_PATH,
    QwenVLDataCollator,
    build_mc_conversation,
    build_mc_reasoning_conversation,
    configure_qwen_processor_pixels,
    is_pseudo_reasoning_verified,
    mask_labels_assistant_only,
    prepare_qwen_inputs,
    pseudo_reasoning_raw_from_metadata,
    resolve_processor_path,
    squeeze_batch_dim,
)


# Spatial Filter Layer
def _get_visual_merger_output_dim(model) -> int:
    """Infer merger output dim (= LLM hidden size) from the model architecture."""
    # 1. Preferred: last Linear in visual.merger outputs exactly the LLM hidden size
    try:
        for m in reversed(list(model.visual.merger.modules())):
            if isinstance(m, nn.Linear):
                return m.out_features
    except AttributeError:
        pass
    # 2. LM head: Linear(hidden_size → vocab), so in_features = hidden_size
    try:
        return model.lm_head.in_features
    except AttributeError:
        pass
    # 3. Config fallback — attribute name differs across Qwen variants
    for attr in ("hidden_size", "text_config.hidden_size"):
        try:
            obj = model.config
            for part in attr.split("."):
                obj = getattr(obj, part)
            return int(obj)
        except AttributeError:
            pass
    raise RuntimeError(
        "Cannot infer visual merger output dimension. "
        "Check model.visual.merger or model.lm_head."
    )


def _spatial_position_encoding(
    grid_h: int, grid_w: int, grid_t: int, device: torch.device
) -> torch.Tensor:
    """
    Normalised (y, x) ∈ [0, 1]² for every merged patch token.

    Returns shape (grid_t * grid_h * grid_w, 2).
    For temporal t > 1 (Qwen2.5-VL duplicates static images internally),
    the spatial positions simply repeat across frames.
    """
    ys = torch.linspace(0, 1, grid_h, device=device)
    xs = torch.linspace(0, 1, grid_w, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    spatial = torch.stack([grid_y, grid_x], dim=-1).view(grid_h * grid_w, 2)  # (h*w, 2)
    return spatial.unsqueeze(0).expand(grid_t, -1, -1).reshape(grid_t * grid_h * grid_w, 2)


class SpatialFilterLayer(nn.Module):
    """
    Learnable per-patch spatial filter with position awareness.

    For each merged patch token the scorer receives both the visual feature
    vector AND the token's normalised (y, x) position in the image, so it
    can learn spatial priors ("centre patches tend to be diagnostically
    relevant") in addition to purely visual patterns.

    Pipeline per token:
      1. concat(token_D, pos_2) → (D+2,)
      2. scorer MLP → Sigmoid  →  score ∈ (0, 1)
      3. filtered = token × score
      4. concat([original, filtered]) → Linear(2D, D) → output

    Initialisation: fusion layer = identity on original half, zero on filtered
    half → strict no-op at step 0; filter activates as gradients flow.
    """

    def __init__(self, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.d_model = d_model
        # Input: visual feature (D) + normalised position (2)
        self.scorer = nn.Sequential(
            nn.Linear(d_model + 2, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
            nn.Sigmoid(),                        # score ∈ (0, 1) per patch
        )
        self.fusion = nn.Linear(d_model * 2, d_model)
        # Identity init: original passes through unchanged at step 0
        with torch.no_grad():
            self.fusion.weight[:, :d_model].copy_(torch.eye(d_model))
            self.fusion.weight[:, d_model:].zero_()
            self.fusion.bias.zero_()

    def forward(
        self,
        tokens: torch.Tensor,   # (N, D) — flat merger output for one image
        grid_h: int,
        grid_w: int,
        grid_t: int = 1,
    ) -> torch.Tensor:
        pos = _spatial_position_encoding(grid_h, grid_w, grid_t, tokens.device)
        pos = pos.to(tokens.dtype)                                # (N, 2)
        score = self.scorer(torch.cat([tokens, pos], dim=-1))     # (N, 1)
        filtered = tokens * score                                 # (N, D)
        return self.fusion(torch.cat([tokens, filtered], dim=-1)) # (N, D)


class QwenWithSpatialFilter(nn.Module):
    """
    Wrapper around Qwen2_5_VLForConditionalGeneration that installs a
    forward hook on visual.merger to apply SpatialFilterLayer to every batch
    of merged visual tokens. This is done transparently, without changing the LLM interface.
    """

    def __init__(self, base_model, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.model = base_model
        self.spatial_filter = SpatialFilterLayer(d_model, bottleneck)

    def _make_hook(self, image_grid_thw: torch.Tensor):
        """
        Return a merger hook that applies SpatialFilterLayer per image,
        passing each image's own (grid_h, grid_w, grid_t) for position encoding.

        image_grid_thw: (B, 3) — t, h, w after merger for each image in batch.
        merger output:  (Σ t_i*h_i*w_i, D) — all images concatenated.
        """
        spatial_filter = self.spatial_filter

        def hook(module, inp, output):  # noqa: ARG001
            result = output.clone()
            ptr = 0
            for thw in image_grid_thw:
                t, h, w = int(thw[0]), int(thw[1]), int(thw[2])
                n = t * h * w
                result[ptr : ptr + n] = spatial_filter(output[ptr : ptr + n], h, w, t)
                ptr += n
            return result

        return hook

    def forward(self, input_ids, attention_mask, pixel_values=None,
                image_grid_thw=None, labels=None, **kwargs):
        hook_fn = self._make_hook(image_grid_thw) if image_grid_thw is not None else None
        handle = self.model.visual.merger.register_forward_hook(hook_fn) if hook_fn else None
        try:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,
                **kwargs,
            )
        finally:
            if handle is not None:
                handle.remove()

    def generate(self, **kwargs):
        image_grid_thw = kwargs.get("image_grid_thw")
        hook_fn = self._make_hook(image_grid_thw) if image_grid_thw is not None else None
        handle = self.model.visual.merger.register_forward_hook(hook_fn) if hook_fn else None
        try:
            return self.model.generate(**kwargs)
        finally:
            if handle is not None:
                handle.remove()

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class QwenMCTrainDataset(Dataset):
    """MC Qwen dataset: loss only on assistant tokens (JSON or answer letter)."""

    def __init__(
        self,
        samples,
        processor,
        max_length=512,
        *,
        use_reasoning: bool = False,
    ):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.use_reasoning = use_reasoning
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

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
            padding=False,
        )
        item = squeeze_batch_dim(batch)
        item["labels"] = mask_labels_assistant_only(
            self.processor, item["input_ids"], conversation
        )
        return item


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-VL with spatial filter on Medical VQA (MC)"
    )
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--data_root", type=str, default=get_data_root())
    p.add_argument("--mc_data_root", type=str, default=get_mc_data_root())
    p.add_argument("--mc_export_suffix", type=str, default="")
    p.add_argument(
        "--use_reasoning",
        action="store_true",
        help="Train on qwen_verify=True pseudo reasoning (JSON assistant target).",
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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    d_model = _get_visual_merger_output_dim(model)
    n_params = (d_model + 2) * args.filter_bottleneck + args.filter_bottleneck + d_model * 2 * d_model + d_model
    print(f"SpatialFilterLayer: d_model={d_model}, bottleneck={args.filter_bottleneck}, params≈{n_params:,}")
    model = QwenWithSpatialFilter(model, d_model=d_model, bottleneck=args.filter_bottleneck)

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

    processor_path = resolve_processor_path(args.model_path)
    if processor_path != args.model_path:
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

    train_total = len(train_samples)
    if args.use_reasoning:
        train_samples = [
            s
            for s in train_samples
            if is_pseudo_reasoning_verified(s.metadata)
            and pseudo_reasoning_raw_from_metadata(s.metadata)
        ]
        print(f"Train: {len(train_samples)} / {train_total} (use_reasoning, qwen_verify=True)")
        if not train_samples:
            raise ValueError("No verified reasoning samples after filtering.")
    else:
        print(f"Train: {len(train_samples)} samples")

    train_dataset = QwenMCTrainDataset(
        train_samples,
        processor,
        args.max_length,
        use_reasoning=args.use_reasoning,
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
        data_collator=QwenVLDataCollator(tokenizer=processor.tokenizer),
        processing_class=processor.tokenizer,
    )

    loss_desc = (
        "assistant JSON (reasoning + answer)"
        if args.use_reasoning
        else "assistant answer tokens only"
    )
    print(f"Starting Qwen2.5-VL training with spatial filter (loss on {loss_desc})...")
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print(f"Saving model to {args.output_dir}")
    trainer.save_model()
    trainer.save_state()
    processor.save_pretrained(args.output_dir)

    filter_path = Path(args.output_dir) / "spatial_filter.pt"
    torch.save(model.spatial_filter.state_dict(), filter_path)
    print(f"Spatial filter saved → {filter_path}")

    print(f"Training completed! loss={result.training_loss}")


if __name__ == "__main__":
    main()
