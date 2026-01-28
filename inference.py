"""
Inference script for LLaVA and LLaVA-Med on Medical VQA datasets
"""

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import LlavaForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import json
from pathlib import Path

from utils.dataset import create_dataset
from utils.prompt_formatter import PromptFormatter


# Default model paths for each model type
DEFAULT_MODEL_PATHS = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "llava_med": "microsoft/llava-med-v1.5-mistral-7b"
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with LLaVA/LLaVA-Med on Medical VQA")
    
    # Model arguments
    parser.add_argument("--model", type=str, required=True, choices=["llava", "llava_med"],
                       help="Model type: 'llava' or 'llava_med'")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to model checkpoint or model name from HuggingFace. "
                            "If not specified, uses default model for --model type.")
    
    # Data arguments
    parser.add_argument("--data_root", type=str, default="data",
                       help="Root directory containing datasets")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["pathvqa", "slake", "vqarad"],
                       help="Dataset to use")
    parser.add_argument("--split", type=str, default="test",
                       help="Split to evaluate on")
    
    # Inference arguments
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum generation length")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p for generation")
    parser.add_argument("--do_sample", action="store_true",
                       help="Use sampling for generation")
    parser.add_argument("--num_beams", type=int, default=1,
                       help="Number of beams for beam search")
    
    # Output arguments
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output JSON file for results")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    
    # Prompt tokens
    parser.add_argument("--question_token", type=str, default="<question>",
                       help="Token marking question start")
    parser.add_argument("--answer_token", type=str, default="<answer>",
                       help="Token marking answer start")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
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
    if Path(args.model_path).exists():
        print("(Loading from local path)")
    else:
        print("(Model will be downloaded from HuggingFace Hub if not cached)")
    
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # Load processor - use original model if trained model doesn't have processor config
    processor_path = args.model_path
    if Path(args.model_path).exists() and not Path(args.model_path, "preprocessor_config.json").exists():
        # Trained model doesn't have processor config, use original model
        processor_path = DEFAULT_MODEL_PATHS[args.model]
        print(f"Processor config not found in trained model, loading from original model: {processor_path}")
    
    processor = AutoProcessor.from_pretrained(processor_path)
    
    # Determine number of image tokens from model config
    num_image_tokens = 576  # Default for LLaVA-1.5
    if hasattr(model.config, 'image_grid_pinpoints'):
        num_image_tokens = 576
    elif hasattr(model.config, 'num_image_tokens'):
        num_image_tokens = model.config.num_image_tokens
    elif hasattr(model.config, 'vision_config'):
        if hasattr(model.config.vision_config, 'image_size'):
            image_size = model.config.vision_config.image_size
            patch_size = getattr(model.config.vision_config, 'patch_size', 14)
            num_patches_per_side = image_size // patch_size
            num_image_tokens = num_patches_per_side * num_patches_per_side
    
    print(f"Using {num_image_tokens} image tokens based on model configuration")
    
    # Create dataset
    print(f"Loading {args.dataset} {args.split} split...")
    dataset = create_dataset(
        dataset_name=args.dataset,
        split=args.split,
        data_root=args.data_root,
        image_processor=processor.image_processor,
        tokenizer=processor.tokenizer,
        model_type=args.model,
        max_length=args.max_length,
        for_training=False,  # Inference mode
        question_token=args.question_token,
        answer_token=args.answer_token,
        num_image_tokens=num_image_tokens
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Inference format: [question_token; question; answer_token]")
    print(f"Model will generate answer after {args.answer_token}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers
    )
    
    # Initialize prompt formatter for answer extraction
    formatter = PromptFormatter(
        question_token=args.question_token,
        answer_token=args.answer_token,
        model_type=args.model
    )
    
    # Run inference
    results = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move to device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Generate
            outputs = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            
            # Decode generated text
            generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)
            
            # Process each sample in batch
            for i in range(len(batch["question"])):
                generated_text = generated_texts[i]
                
                # Extract answer from generated text
                answer = formatter.extract_answer_from_response(generated_text)
                
                result = {
                    "image_path": batch["image_path"][i],
                    "question": batch["question"][i],
                    "ground_truth_answer": batch["answer"][i] if "answer" in batch and batch["answer"][i] is not None else None,
                    "predicted_answer": answer,
                    "full_response": generated_text,
                }
                results.append(result)
    
    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")
    else:
        # Print first few results
        print("\nFirst 5 results:")
        for i, result in enumerate(results[:5]):
            print(f"\nSample {i+1}:")
            print(f"  Question: {result['question']}")
            print(f"  Ground Truth: {result['ground_truth_answer']}")
            print(f"  Predicted: {result['predicted_answer']}")
    
    print(f"\nInference completed! Processed {len(results)} samples.")


if __name__ == "__main__":
    main()
