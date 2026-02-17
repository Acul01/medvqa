"""
Inference script for LLaVA-Med on Medical VQA datasets.

Uses the original Microsoft LLaVA-Med codebase (github.com/microsoft/LLaVA-Med)
for model loading and inference, since the model is not compatible with
HuggingFace's LlavaForConditionalGeneration.

Prerequisites:
    git clone https://github.com/microsoft/LLaVA-Med
    cd LLaVA-Med && pip install -e .
"""

import argparse
import torch
from PIL import Image
from tqdm import tqdm
import json
from pathlib import Path

# Original LLaVA-Med codebase imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from transformers import AutoTokenizer
from llava.model import LlavaMistralForCausalLM
from llava.mm_utils import get_model_name_from_path

from utils.load_data import MedicalVQALoader
from utils.save_results import save_results, clean_answer


DEFAULT_MODEL_PATH = "microsoft/llava-med-v1.5-mistral-7b"


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with LLaVA-Med on Medical VQA")

    # Model arguments
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                       help="Path to LLaVA-Med checkpoint or HuggingFace model name")
    parser.add_argument("--model_base", type=str, default=None,
                       help="Base model path (only needed for LoRA checkpoints)")
    parser.add_argument("--conv_template", type=str, default="mistral_instruct",
                       help="Conversation template name (default: mistral_instruct)")

    # Data arguments
    parser.add_argument("--data_root", type=str, default="data",
                       help="Root directory containing datasets")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["pathvqa", "slake", "vqarad"],
                       help="Dataset to use")
    parser.add_argument("--split", type=str, default="test",
                       help="Split to evaluate on")

    # Inference arguments
    parser.add_argument("--max_new_tokens", type=int, default=3,
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


def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading LLaVA-Med model: {args.model_path}")
    model_path = Path(args.model_path)

    if model_path.exists() and (model_path / "config.json").exists():
        # Fine-tuned local checkpoint: load directly
        print("(Loading fine-tuned model from local path)")
        model = LlavaMistralForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # Load tokenizer: try local first, fall back to base model
        if (model_path / "tokenizer_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        else:
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH)
        image_processor = model.get_vision_tower().image_processor
    else:
        # HuggingFace model: use original LLaVA-Med builder
        print("(Loading from HuggingFace Hub via LLaVA-Med builder)")
        model_name = model_path.name or "llava-med-v1.5-mistral-7b"
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path,
            model_base=args.model_base,
            model_name=model_name,
            device=str(device),
        )

    model.eval()
    print("Model loaded")

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
    brief_instruction = "Answer with a maximum of three words."
    print("Running inference...")

    with torch.no_grad():
        for sample in tqdm(samples):
            # 1. Load and process image
            image = Image.open(sample.image_path).convert("RGB")
            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = image_tensor.to(device, dtype=torch.float16)

            # 2. Build conversation using LLaVA-Med's conv_templates
            conv = conv_templates[args.conv_template].copy()
            question = f"{DEFAULT_IMAGE_TOKEN}\n{sample.question}\n{brief_instruction}"
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # 3. Tokenize with image token handling
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)

            # 4. Generate (input_ids as positional arg – required by LLaVA-Med's
            #    generate override which expects it as the 'inputs' parameter)
            output = model.generate(
                input_ids,
                images=image_tensor,
                attention_mask=torch.ones_like(input_ids),
                **gen_kwargs,
            )

            # 5. Decode generated tokens
            #    LLaVA-Med's generate() passes inputs_embeds (not input_ids)
            #    to super().generate(), so the output contains ONLY the newly
            #    generated tokens – no need to slice off the input portion.
            generated_ids = output[0]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generated_text = clean_answer(generated_text)

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
