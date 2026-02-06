"""
Inference script for LLaVA and LLaVA-Med on Medical VQA datasets.

Uses the HuggingFace processor directly for correct tokenization and
image-token handling (no manual padding, no manual image-token expansion).
"""

import argparse
import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import json
from pathlib import Path

from utils.load_data import MedicalVQALoader


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
    parser.add_argument("--max_new_tokens", type=int, default=50,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--do_sample", action="store_true", default=False,
                       help="Use sampling for generation (default: greedy)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation (only used with --do_sample)")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p for generation (only used with --do_sample)")
    parser.add_argument("--num_beams", type=int, default=1,
                       help="Number of beams for beam search")

    # Output arguments
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output JSON file for results")
    parser.add_argument("--n_samples", type=int, default=None,
                       help="Only process first N samples (for quick testing)")

    return parser.parse_args()


def build_conversation(question: str, model_id: str):
    """
    Build the chat conversation in the format expected by the model's chat template.

    - Standard LLaVA 1.5: multimodal content list with {"type": "image"} and {"type": "text"}
    - LLaVA-Med Mistral: plain string content with "<image>" placeholder

    A short-answer instruction is appended to the question so that the model
    produces concise answers suitable for VQA evaluation.
    """
    brief_instruction = "Answer with a single word or short phrase."
    full_question = f"{question}\n{brief_instruction}"

    if "llava-med-v1.5-mistral" in model_id or "llava_mistral" in model_id:
        return [{"role": "user", "content": f"<image>\n{full_question}"}]
    else:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": full_question},
                ],
            }
        ]


def main():
    args = parse_args()

    # ── Determine model path ──────────────────────────────────────────────
    if args.model_path is None:
        args.model_path = DEFAULT_MODEL_PATHS[args.model]
        print(f"No --model_path specified, using default: {args.model_path}")
    else:
        print(f"Using specified model path: {args.model_path}")

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading model: {args.model_path}")
    if Path(args.model_path).exists():
        print("(Loading from local path)")
    else:
        print("(Model will be downloaded from HuggingFace Hub if not cached)")

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model.eval()

    # ── Load processor ────────────────────────────────────────────────────
    # For fine-tuned models the processor config may not be saved alongside
    # the checkpoint – fall back to the original base model's processor.
    processor_path = args.model_path
    if Path(args.model_path).exists() and not Path(args.model_path, "preprocessor_config.json").exists():
        processor_path = DEFAULT_MODEL_PATHS[args.model]
        print(f"Processor config not found in trained model, loading from: {processor_path}")

    processor = AutoProcessor.from_pretrained(processor_path)

    # ── Load dataset samples ──────────────────────────────────────────────
    print(f"Loading {args.dataset} {args.split} split...")
    loader = MedicalVQALoader(data_root=args.data_root)
    samples = loader.load_dataset(args.dataset, split=args.split)
    if args.n_samples is not None:
        samples = samples[:args.n_samples]
    print(f"Loaded {len(samples)} samples")

    # ── Generation kwargs ─────────────────────────────────────────────────
    gen_kwargs = {"max_new_tokens": args.max_new_tokens}

    if args.do_sample:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
        })
    else:
        gen_kwargs["do_sample"] = False

    if args.num_beams > 1:
        gen_kwargs["num_beams"] = args.num_beams

    # ── Inference loop ────────────────────────────────────────────────────
    results = []
    print("Running inference...")

    with torch.no_grad():
        for sample in tqdm(samples):
            # 1. Load image
            image = Image.open(sample.image_path).convert("RGB")

            # 2. Build conversation and apply chat template → prompt string
            conversation = build_conversation(sample.question, args.model_path)
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

            # 3. Processor handles EVERYTHING: tokenisation, image-token
            #    expansion, pixel-value preprocessing – no manual padding.
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

            # 4. Generate
            output = model.generate(**inputs, **gen_kwargs)

            # 5. Decode only the newly generated tokens
            input_length = inputs["input_ids"].shape[1]
            generated_ids = output[0][input_length:]
            generated_text = processor.decode(generated_ids, skip_special_tokens=True).strip()

            # 6. Print live
            print(f"\n[Sample {len(results) + 1}]")
            print(f"  Question:  {sample.question}")
            print(f"  Answer:    {generated_text}")
            if sample.answer:
                print(f"  GT:        {sample.answer}")

            results.append({
                "image_path": sample.image_path,
                "question": sample.question,
                "ground_truth_answer": sample.answer,
                "predicted_answer": generated_text,
            })

    # ── Save / print results ──────────────────────────────────────────────
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")
    else:
        print("\nFirst 5 results:")
        for i, result in enumerate(results[:5]):
            print(f"\nSample {i+1}:")
            print(f"  Question:    {result['question']}")
            print(f"  Ground Truth: {result['ground_truth_answer']}")
            print(f"  Predicted:    {result['predicted_answer']}")

    print(f"\nInference completed! Processed {len(results)} samples.")


if __name__ == "__main__":
    main()
