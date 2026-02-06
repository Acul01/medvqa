"""
Inference script for standard LLaVA on Medical VQA datasets.

Uses the HuggingFace processor directly for correct tokenization and
image-token handling (no manual padding, no manual image-token expansion).

For LLaVA-Med, use inference_med.py instead.
"""

import argparse
import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import json
from pathlib import Path

from utils.load_data import MedicalVQALoader
from utils.save_results import save_results


DEFAULT_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with LLaVA on Medical VQA")

    # Model arguments
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                       help="Path to model checkpoint or HuggingFace model name "
                            f"(default: {DEFAULT_MODEL_PATH})")

    # Data arguments
    parser.add_argument("--data_root", type=str, default="data",
                       help="Root directory containing datasets")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["pathvqa", "slake", "vqarad"],
                       help="Dataset to use")
    parser.add_argument("--split", type=str, default="test",
                       help="Split to evaluate on")

    # Inference arguments
    parser.add_argument("--max_new_tokens", type=int, default=15,
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
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory to save result files (default: results)")
    parser.add_argument("--n_samples", type=int, default=None,
                       help="Only process first N samples (for quick testing)")
    parser.add_argument("--finetuned", action="store_true", default=False,
                       help="Flag: model is fine-tuned (recorded in history)")

    return parser.parse_args()


def build_conversation(question: str):
    """
    Build the chat conversation in LLaVA 1.5 format:
    multimodal content list with {"type": "image"} and {"type": "text"}.

    A short-answer instruction is appended so the model produces concise
    answers suitable for VQA evaluation.
    """
    brief_instruction = "Answer with a maximum of three words."
    full_question = f"{question}\n{brief_instruction}"

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

    # ── Model path ─────────────────────────────────────────────────────────
    print(f"Using model: {args.model_path}")

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
        processor_path = DEFAULT_MODEL_PATH
        print(f"Processor config not found in trained model, loading from: {processor_path}")

    processor = AutoProcessor.from_pretrained(processor_path)

    # Fix for community conversions that are missing patch_size in config
    if getattr(processor, "patch_size", None) is None:
        processor.patch_size = 14
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = "default"

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
            conversation = build_conversation(sample.question)
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

    # ── Print results summary ───────────────────────────────────────────────
    print("\nFirst 5 results:")
    for i, result in enumerate(results[:5]):
        print(f"\nSample {i+1}:")
        print(f"  Question:    {result['question']}")
        print(f"  Ground Truth: {result['ground_truth_answer']}")
        print(f"  Predicted:    {result['predicted_answer']}")

    # ── Accuracy ───────────────────────────────────────────────────────────
    exact = 0
    contains = 0
    total = 0
    for r in results:
        if r["ground_truth_answer"] is None:
            continue
        total += 1
        pred = r["predicted_answer"].lower().strip()
        gt = r["ground_truth_answer"].lower().strip()
        if pred == gt:
            exact += 1
        if gt in pred:
            contains += 1

    exact_acc = exact / total if total else 0.0
    contains_acc = contains / total if total else 0.0
    print(f"\nExact-match accuracy:    {exact}/{total} = {exact_acc:.4f} ({exact_acc * 100:.2f}%)" if total else "\nNo samples with ground truth.")
    print(f"Contains-match accuracy: {contains}/{total} = {contains_acc:.4f} ({contains_acc * 100:.2f}%)" if total else "")

    # ── Save results (JSON, CSV, XLSX) + history ─────────────────────────
    print(f"\nSaving results...")
    save_results(
        results=results,
        model_path=args.model_path,
        dataset=args.dataset,
        n_samples=args.n_samples,
        finetuned=args.finetuned,
        exact_acc=exact_acc,
        contains_acc=contains_acc,
        results_dir=args.results_dir,
    )

    print(f"Inference completed! Processed {len(results)} samples.")


if __name__ == "__main__":
    main()
