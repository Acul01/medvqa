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
import wandb
from PIL import Image
from tqdm import tqdm
import json
from pathlib import Path

# Original LLaVA-Med codebase imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from transformers import AutoTokenizer, CLIPImageProcessor
from llava.model import LlavaMistralForCausalLM
from llava.mm_utils import get_model_name_from_path

from utils.load_data import MedicalVQALoader
from utils.save_results import save_results, clean_answer, compute_bleu_scores, compute_bert_scores


DEFAULT_MODEL_PATH = "microsoft/llava-med-v1.5-mistral-7b"

DATASET_INSTRUCTIONS = {
    "vqarad": (
        "You are a medical expert. Answer the question about the medical image concisely.\n"
        "Answer with 1–3 tokens. If it is a yes/no question, answer only 'yes' or 'no'."
    ),
    "_default": "Answer with a maximum of three words.",
}


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

    # W&B
    parser.add_argument("--wandb_project", type=str, default="medvqa",
                       help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="W&B run name (auto-generated if not set)")

    return parser.parse_args()


def _model_short_name(model_path: str) -> str:
    name = Path(model_path).name.lower()
    if "llava-med" in name:
        return "llava-med"
    if "llava" in name:
        return "llava"
    return Path(model_path).name


def main():
    args = parse_args()

    # ── W&B ──────────────────────────────────────────────────────────────
    model_short = _model_short_name(args.model_path)
    ft_tag = "finetuned" if args.finetuned else "pretrained"
    run_name = args.wandb_run_name or f"{model_short}_{args.dataset}_{ft_tag}_eval"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=args.dataset,
        tags=["inference", model_short, args.dataset, ft_tag],
        config=vars(args),
    )

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading LLaVA-Med model: {args.model_path}")
    model_path = Path(args.model_path)
    is_lora = model_path.exists() and (model_path / "adapter_config.json").exists()

    if is_lora:
        from peft import PeftModel
        base_path = args.model_base or DEFAULT_MODEL_PATH
        print(f"(Loading LoRA adapter, base model: {base_path})")
        base_model = LlavaMistralForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.float16, device_map="auto",
        )
        vision_tower = base_model.get_model().get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)

        model = PeftModel.from_pretrained(base_model, str(model_path))
        model = model.merge_and_unload()
        print("  LoRA adapter merged into base model")

        tokenizer = AutoTokenizer.from_pretrained(base_path)
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    elif model_path.exists() and (model_path / "config.json").exists():
        print("(Loading fine-tuned model from local path)")
        model = LlavaMistralForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="auto",
        )
        vision_tower = model.get_model().get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)

        if (model_path / "tokenizer_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        else:
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH)
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    else:
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
    instruction = DATASET_INSTRUCTIONS.get(args.dataset, DATASET_INSTRUCTIONS["_default"])
    results = []
    print("Running inference...")

    with torch.no_grad():
        for sample in tqdm(samples):
            # 1. Load and process image
            image = Image.open(sample.image_path).convert("RGB")
            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = image_tensor.to(device, dtype=torch.float16)

            # 2. Build conversation using LLaVA-Med's conv_templates
            conv = conv_templates[args.conv_template].copy()
            question = f"{DEFAULT_IMAGE_TOKEN}\n{sample.question}\n{instruction}"
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

    # ── Metrics ────────────────────────────────────────────────────────────
    exact = 0
    total = 0
    bleu_accum = {f"bleu_{n}": 0.0 for n in range(1, 5)}
    per_sample_bleu = []

    for r in results:
        if r["ground_truth_answer"] is None:
            per_sample_bleu.append({f"bleu_{n}": 0.0 for n in range(1, 5)})
            continue
        total += 1
        pred = r["predicted_answer"].lower().strip()
        gt = r["ground_truth_answer"].lower().strip()
        if pred == gt:
            exact += 1
        scores = compute_bleu_scores(r["predicted_answer"], r["ground_truth_answer"])
        per_sample_bleu.append(scores)
        for k in bleu_accum:
            bleu_accum[k] += scores[k]

    exact_acc = exact / total if total else 0.0
    avg_bleu = {k: v / total if total else 0.0 for k, v in bleu_accum.items()}

    # ── BERTScore (batch) ────────────────────────────────────────────────
    valid_preds = [r["predicted_answer"] for r in results if r["ground_truth_answer"] is not None]
    valid_refs = [r["ground_truth_answer"] for r in results if r["ground_truth_answer"] is not None]
    bert_result = compute_bert_scores(valid_preds, valid_refs)
    avg_bert = bert_result["avg"]
    bert_per_sample_f1 = bert_result["per_sample"]
    bert_idx = 0
    per_sample_bert_f1 = []
    for r in results:
        if r["ground_truth_answer"] is None:
            per_sample_bert_f1.append(0.0)
        else:
            per_sample_bert_f1.append(bert_per_sample_f1[bert_idx])
            bert_idx += 1

    print(f"\nExact-match accuracy: {exact}/{total} = {exact_acc:.4f} ({exact_acc * 100:.2f}%)" if total else "\nNo samples with ground truth.")
    for n in range(1, 5):
        print(f"BLEU-{n}:                {avg_bleu[f'bleu_{n}']:.4f}")
    print(f"BERTScore F1:           {avg_bert['bert_f1']:.4f}")

    # ── Save results (JSON, CSV, XLSX) + history ─────────────────────────
    print(f"\nSaving results...")
    save_results(
        results=results,
        model_path=args.model_path,
        dataset=args.dataset,
        n_samples=args.n_samples,
        finetuned=args.finetuned,
        exact_acc=exact_acc,
        bleu_scores=avg_bleu,
        bert_scores=avg_bert,
        results_dir=args.results_dir,
    )

    # ── W&B: summary metrics + predictions table ────────────────────────
    wandb.summary["exact_match_accuracy"] = exact_acc
    wandb.summary["bleu_1"] = avg_bleu["bleu_1"]
    wandb.summary["bleu_2"] = avg_bleu["bleu_2"]
    wandb.summary["bleu_3"] = avg_bleu["bleu_3"]
    wandb.summary["bleu_4"] = avg_bleu["bleu_4"]
    wandb.summary["bert_precision"] = avg_bert["bert_precision"]
    wandb.summary["bert_recall"] = avg_bert["bert_recall"]
    wandb.summary["bert_f1"] = avg_bert["bert_f1"]
    wandb.summary["total_samples"] = total
    table = wandb.Table(columns=[
        "image_path", "question", "ground_truth", "predicted",
        "exact_match", "bleu_1", "bleu_2", "bleu_3", "bleu_4", "bert_f1",
    ])
    for r, bleu, bf1 in zip(results, per_sample_bleu, per_sample_bert_f1):
        gt = (r["ground_truth_answer"] or "").lower().strip()
        pred = r["predicted_answer"].lower().strip()
        table.add_data(
            r["image_path"], r["question"],
            r["ground_truth_answer"], r["predicted_answer"],
            pred == gt,
            round(bleu["bleu_1"], 4), round(bleu["bleu_2"], 4),
            round(bleu["bleu_3"], 4), round(bleu["bleu_4"], 4),
            round(bf1, 4),
        )
    wandb.log({"predictions": table})
    wandb.finish()

    print(f"Inference completed! Processed {len(results)} samples.")


if __name__ == "__main__":
    main()
