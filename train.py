"""
Training script for LLaVA and LLaVA-Med on Medical VQA datasets
"""

import argparse
import torch
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from transformers.trainer_utils import set_seed

from utils.dataset import create_dataset

# Check for bitsandbytes (required for 8-bit optimizer)
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False


# Default model paths for each model type
DEFAULT_MODEL_PATHS = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "llava_med": "microsoft/llava-med-v1.5-mistral-7b"
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLaVA/LLaVA-Med on Medical VQA")
    
    # Model arguments
    parser.add_argument("--model", type=str, required=True, choices=["llava", "llava_med"],
                       help="Model type: 'llava' or 'llava_med'")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to pretrained model or model name from HuggingFace. "
                            "If not specified, uses default model for --model type.")
    
    # Data arguments
    parser.add_argument("--data_root", type=str, default="data",
                       help="Root directory containing datasets")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["pathvqa", "slake", "vqarad"],
                       help="Dataset to use")
    parser.add_argument("--train_split", type=str, default="train",
                       help="Training split name")
    parser.add_argument("--val_split", type=str, default="validation",
                       help="Validation split name")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                       help="Batch size per device for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                       help="Batch size per device for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluate every N steps")
    parser.add_argument("--max_length", type=int, default=768,
                       help="Maximum sequence length (must accommodate image tokens + text, reduce if OOM)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing to save memory")
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--bf16", action="store_true",
                       help="Use bfloat16 precision")
    parser.add_argument("--use_8bit_optimizer", action="store_true",
                       help="Use 8-bit optimizer (bitsandbytes) to save memory. Requires: pip install bitsandbytes")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    
    # Prompt tokens
    parser.add_argument("--question_token", type=str, default="<question>",
                       help="Token marking question start")
    parser.add_argument("--answer_token", type=str, default="<answer>",
                       help="Token marking answer start")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Resume training from checkpoint")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Determine model path
    if args.model_path is None:
        args.model_path = DEFAULT_MODEL_PATHS[args.model]
        print(f"No --model_path specified, using default: {args.model_path}")
    else:
        print(f"Using specified model path: {args.model_path}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and processor
    print(f"Loading model: {args.model_path}")
    print("(Model will be downloaded from HuggingFace Hub if not cached)")
    
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        device_map="auto"
    )
    
    # Enable gradient checkpointing if requested (saves memory)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled (saves memory)")
    
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # Determine number of image tokens from model config
    # LLaVA models typically have this in the config
    num_image_tokens = 576  # Default for LLaVA-1.5
    if hasattr(model.config, 'image_grid_pinpoints'):
        # LLaVA-1.5 uses image_grid_pinpoints to determine image size
        # Default is usually [[336, 672], [672, 336], [672, 672]] for 336x336 base
        # Which results in 576 tokens (24x24 patches)
        num_image_tokens = 576
    elif hasattr(model.config, 'num_image_tokens'):
        num_image_tokens = model.config.num_image_tokens
    elif hasattr(model.config, 'vision_config'):
        # Try to calculate from vision config
        if hasattr(model.config.vision_config, 'image_size'):
            image_size = model.config.vision_config.image_size
            patch_size = getattr(model.config.vision_config, 'patch_size', 14)
            num_patches_per_side = image_size // patch_size
            num_image_tokens = num_patches_per_side * num_patches_per_side
    
    print(f"Using {num_image_tokens} image tokens based on model configuration")
    
    # Warn if max_length might be too short for image tokens
    if args.max_length < num_image_tokens + 100:
        print(f"Warning: max_length ({args.max_length}) might be too short for {num_image_tokens} image tokens + text")
        print(f"Consider increasing --max_length to at least {num_image_tokens + 200}")
    
    # Create datasets
    print(f"Loading {args.dataset} dataset...")
    train_dataset = create_dataset(
        dataset_name=args.dataset,
        split=args.train_split,
        data_root=args.data_root,
        image_processor=processor.image_processor,
        tokenizer=processor.tokenizer,
        model_type=args.model,
        max_length=args.max_length,
        for_training=True,
        question_token=args.question_token,
        answer_token=args.answer_token,
        num_image_tokens=num_image_tokens
    )
    
    # Validation dataset (if available)
    val_dataset = None
    try:
        val_dataset = create_dataset(
            dataset_name=args.dataset,
            split=args.val_split,
            data_root=args.data_root,
            image_processor=processor.image_processor,
            tokenizer=processor.tokenizer,
            model_type=args.model,
            max_length=args.max_length,
            for_training=True,
            question_token=args.question_token,
            answer_token=args.answer_token,
            num_image_tokens=num_image_tokens
        )
        print(f"Validation dataset loaded: {len(val_dataset)} samples")
    except Exception as e:
        print(f"Warning: Could not load validation dataset: {e}")
    
    print(f"Training dataset loaded: {len(train_dataset)} samples")
    
    # Training arguments
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
        load_best_model_at_end=True if val_dataset else False,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="tensorboard",
        seed=args.seed,
        optim="adamw_8bit" if args.use_8bit_optimizer else "adamw_torch",  # 8-bit optimizer saves memory
        max_grad_norm=1.0,  # Gradient clipping
    )
    
    # #region agent log
    import json
    import os
    LOG_PATH = "/Users/luca/Desktop/Thesis+Work/medvqa/.cursor/debug.log"
    def _log(session_id, run_id, hypothesis_id, location, message, data):
        try:
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps({"sessionId": session_id, "runId": run_id, "hypothesisId": hypothesis_id, "location": location, "message": message, "data": data, "timestamp": os.times().elapsed}) + "\n")
        except: pass
    
    # Check first sample from dataset
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        _log("debug-session", "run1", "E", "train.py:183", "First dataset sample", {"keys": list(sample.keys()), "input_ids_shape": list(sample["input_ids"].shape) if "input_ids" in sample else None, "pixel_values_shape": list(sample["pixel_values"].shape) if "pixel_values" in sample else None})
    # #endregion
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer,
    )
    
    # Train
    print("Starting training...")
    print(f"Training format: [question_token; question; answer_token; answer]")
    print(f"Loss will be computed only for tokens after {args.answer_token}")
    
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    print(f"Saving final model to {args.output_dir}")
    trainer.save_model()
    trainer.save_state()
    
    print("Training completed!")
    print(f"Training loss: {train_result.training_loss}")


if __name__ == "__main__":
    main()
