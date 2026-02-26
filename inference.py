"""
Inference script for standard LLaVA on Medical VQA datasets.

Uses the HuggingFace processor directly for correct tokenization and
image-token handling (no manual padding, no manual image-token expansion).

For LLaVA-Med, use inference_med.py instead.
"""

import argparse
import torch
import wandb
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import json
from pathlib import Path

from utils.load_data import MedicalVQALoader
from utils.save_results import (
    save_results, clean_answer, normalize_for_comparison,
    compute_bleu_scores, compute_bert_scores,
    compute_rouge_scores, classify_answer_type, ROUGE_KEYS,
)


DEFAULT_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"

DATASET_INSTRUCTIONS = {
    "vqarad": "You are a medical expert. Answer the question about the medical image. Be concise: answer in as few words as possible.",
    "pathvqa": "You are a pathology expert. Answer concisely and factually.",
    "_default": "Answer with a maximum of three words.",
}

DATASET_MAX_NEW_TOKENS = {
    "vqarad": 10,
    "pathvqa": 16,
    "_default": 3,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with LLaVA on Medical VQA")

    # Model arguments
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                       help="Path to model checkpoint or HuggingFace model name "
                            f"(default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--base_model", type=str, default=None,
                       help="Base model for LoRA checkpoints (default: auto-detect or DEFAULT_MODEL_PATH)")

    # Data arguments
    parser.add_argument("--data_root", type=str, default="data",
                       help="Root directory containing datasets")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["pathvqa", "slake", "vqarad"],
                       help="Dataset to use")
    parser.add_argument("--split", type=str, default="test",
                       help="Split to evaluate on")

    # Inference arguments
    parser.add_argument("--max_new_tokens", type=int, default=None,
                       help="Maximum new tokens to generate (default: dataset-specific)")
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


def build_conversation(question: str, dataset: str):
    """
    Build the chat conversation in LLaVA 1.5 format:
    multimodal content list with {"type": "image"} and {"type": "text"}.
    """
    instruction = DATASET_INSTRUCTIONS.get(dataset, DATASET_INSTRUCTIONS["_default"])
    full_question = f"{question}\n{instruction}"

    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": full_question},
            ],
        }
    ]


def _model_short_name(model_path: str) -> str:
    name = Path(model_path).name.lower()
    if "llava-med" in name:
        return "llava-med"
    if "llava" in name:
        return "llava"
    return Path(model_path).name


def main():
    args = parse_args()

    if args.max_new_tokens is None:
        args.max_new_tokens = DATASET_MAX_NEW_TOKENS.get(args.dataset, DATASET_MAX_NEW_TOKENS["_default"])
    print(f"max_new_tokens: {args.max_new_tokens}")

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

    # ── Model path ─────────────────────────────────────────────────────────
    print(f"Using model: {args.model_path}")

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    is_lora = Path(args.model_path).exists() and (Path(args.model_path) / "adapter_config.json").exists()
    print(f"Loading model: {args.model_path}" + (" (LoRA adapter)" if is_lora else ""))

    if is_lora:
        from peft import PeftModel
        base_path = args.base_model or DEFAULT_MODEL_PATH
        print(f"  Base model: {base_path}")
        base_model = LlavaForConditionalGeneration.from_pretrained(
            base_path, torch_dtype=torch.float32, device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, args.model_path)
        model = model.merge_and_unload()
        print("  LoRA adapter merged into base model")
    else:
        if Path(args.model_path).exists():
            print("(Loading from local path)")
        else:
            print("(Model will be downloaded from HuggingFace Hub if not cached)")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float32, device_map="auto",
        )
    model.eval()

    # ── Load processor ────────────────────────────────────────────────────
    # For LoRA checkpoints or fine-tuned models the processor config may not
    # be saved -- fall back to the base model's processor.
    if is_lora:
        processor_path = args.base_model or DEFAULT_MODEL_PATH
    else:
        processor_path = args.model_path
    if Path(processor_path).exists() and not Path(processor_path, "preprocessor_config.json").exists():
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
            conversation = build_conversation(sample.question, args.dataset)
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

    # ── Per-sample metrics ──────────────────────────────────────────────
    per_sample_bleu = []
    per_sample_rouge = []
    per_sample_type = []

    # Accumulators: overall, yes_no, open
    groups = {"overall": {"exact": 0, "total": 0, "bleu": {f"bleu_{n}": 0.0 for n in range(1, 5)}, "rouge": {k: 0.0 for k in ROUGE_KEYS}},
              "yes_no": {"exact": 0, "total": 0},
              "open":   {"exact": 0, "total": 0, "bleu": {f"bleu_{n}": 0.0 for n in range(1, 5)}, "rouge": {k: 0.0 for k in ROUGE_KEYS}}}

    for r in results:
        if r["ground_truth_answer"] is None:
            per_sample_bleu.append({f"bleu_{n}": 0.0 for n in range(1, 5)})
            per_sample_rouge.append({k: 0.0 for k in ROUGE_KEYS})
            per_sample_type.append("unknown")
            continue

        atype = classify_answer_type(r["ground_truth_answer"])
        per_sample_type.append(atype)
        pred_norm = normalize_for_comparison(r["predicted_answer"])
        gt_norm = normalize_for_comparison(r["ground_truth_answer"])
        is_exact = pred_norm == gt_norm

        groups["overall"]["total"] += 1
        groups[atype]["total"] += 1
        if is_exact:
            groups["overall"]["exact"] += 1
            groups[atype]["exact"] += 1

        b_scores = compute_bleu_scores(r["predicted_answer"], r["ground_truth_answer"])
        per_sample_bleu.append(b_scores)
        r_scores = compute_rouge_scores(r["predicted_answer"], r["ground_truth_answer"])
        per_sample_rouge.append(r_scores)

        for k in groups["overall"]["bleu"]:
            groups["overall"]["bleu"][k] += b_scores[k]
        for k in groups["overall"]["rouge"]:
            groups["overall"]["rouge"][k] += r_scores[k]
        if atype == "open":
            for k in groups["open"]["bleu"]:
                groups["open"]["bleu"][k] += b_scores[k]
            for k in groups["open"]["rouge"]:
                groups["open"]["rouge"][k] += r_scores[k]

    # ── Averages ─────────────────────────────────────────────────────────
    def _safe_div(a, b): return a / b if b else 0.0

    total = groups["overall"]["total"]
    exact_acc = _safe_div(groups["overall"]["exact"], total)
    avg_bleu = {k: _safe_div(v, total) for k, v in groups["overall"]["bleu"].items()}
    avg_rouge = {k: _safe_div(v, total) for k, v in groups["overall"]["rouge"].items()}

    yesno_total = groups["yes_no"]["total"]
    yesno_acc = _safe_div(groups["yes_no"]["exact"], yesno_total)

    open_total = groups["open"]["total"]
    open_acc = _safe_div(groups["open"]["exact"], open_total)
    open_bleu = {k: _safe_div(v, open_total) for k, v in groups["open"]["bleu"].items()}
    open_rouge = {k: _safe_div(v, open_total) for k, v in groups["open"]["rouge"].items()}

    # ── BERTScore (batch) ────────────────────────────────────────────────
    valid_preds = [r["predicted_answer"] for r in results if r["ground_truth_answer"] is not None]
    valid_refs = [r["ground_truth_answer"] for r in results if r["ground_truth_answer"] is not None]
    valid_types = [t for t in per_sample_type if t != "unknown"]
    bert_result = compute_bert_scores(valid_preds, valid_refs)
    avg_bert = bert_result["avg"]
    bert_per_sample_f1 = bert_result["per_sample"]

    open_bert_f1s = [f1 for f1, t in zip(bert_per_sample_f1, valid_types) if t == "open"]
    open_bert_f1 = _safe_div(sum(open_bert_f1s), len(open_bert_f1s))

    bert_idx = 0
    per_sample_bert_f1 = []
    for r in results:
        if r["ground_truth_answer"] is None:
            per_sample_bert_f1.append(0.0)
        else:
            per_sample_bert_f1.append(bert_per_sample_f1[bert_idx])
            bert_idx += 1

    # ── Console output ───────────────────────────────────────────────────
    if yesno_total:
        print(f"\n=== yes/no (N={yesno_total}) ===")
        print(f"  Exact-match accuracy: {groups['yes_no']['exact']}/{yesno_total} = {yesno_acc:.4f} ({yesno_acc * 100:.2f}%)")

    if open_total:
        print(f"\n=== open (N={open_total}) ===")
        print(f"  Exact-match accuracy: {groups['open']['exact']}/{open_total} = {open_acc:.4f} ({open_acc * 100:.2f}%)")
        for n in range(1, 5):
            print(f"  BLEU-{n}:              {open_bleu[f'bleu_{n}']:.4f}")
        for k in ROUGE_KEYS:
            print(f"  {k:20s}  {open_rouge[k]:.4f}")
        print(f"  BERTScore F1:         {open_bert_f1:.4f}")

    if total:
        print(f"\n=== overall (N={total}) ===")
        print(f"  Exact-match accuracy: {groups['overall']['exact']}/{total} = {exact_acc:.4f} ({exact_acc * 100:.2f}%)")
        for n in range(1, 5):
            print(f"  BLEU-{n}:              {avg_bleu[f'bleu_{n}']:.4f}")
        for k in ROUGE_KEYS:
            print(f"  {k:20s}  {avg_rouge[k]:.4f}")
        print(f"  BERTScore F1:         {avg_bert['bert_f1']:.4f}")
    else:
        print("\nNo samples with ground truth.")

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
        rouge_scores=avg_rouge,
        bert_scores=avg_bert,
        yesno_acc=yesno_acc if yesno_total else None,
        open_acc=open_acc if open_total else None,
        results_dir=args.results_dir,
    )

    # ── W&B: summary metrics + predictions table ────────────────────────
    wandb.summary["overall/exact_match_accuracy"] = exact_acc
    for k, v in avg_bleu.items():
        wandb.summary[f"overall/{k}"] = v
    for k, v in avg_rouge.items():
        wandb.summary[f"overall/{k}"] = v
    wandb.summary["overall/bert_f1"] = avg_bert["bert_f1"]
    wandb.summary["overall/total_samples"] = total

    if yesno_total:
        wandb.summary["yesno/exact_match_accuracy"] = yesno_acc
        wandb.summary["yesno/total_samples"] = yesno_total

    if open_total:
        wandb.summary["open/exact_match_accuracy"] = open_acc
        for k, v in open_bleu.items():
            wandb.summary[f"open/{k}"] = v
        for k, v in open_rouge.items():
            wandb.summary[f"open/{k}"] = v
        wandb.summary["open/bert_f1"] = open_bert_f1
        wandb.summary["open/total_samples"] = open_total

    table = wandb.Table(columns=[
        "image_path", "question", "ground_truth", "predicted",
        "answer_type", "exact_match",
        "bleu_1", "bleu_2", "bleu_3", "bleu_4",
        "rouge1", "rouge2", "rougeL", "bert_f1",
    ])
    for r, bleu, rouge, bf1, atype in zip(results, per_sample_bleu, per_sample_rouge, per_sample_bert_f1, per_sample_type):
        gt_norm = normalize_for_comparison(r["ground_truth_answer"] or "")
        pred_norm = normalize_for_comparison(r["predicted_answer"])
        table.add_data(
            r["image_path"], r["question"],
            r["ground_truth_answer"], r["predicted_answer"],
            atype, pred_norm == gt_norm,
            round(bleu["bleu_1"], 4), round(bleu["bleu_2"], 4),
            round(bleu["bleu_3"], 4), round(bleu["bleu_4"], 4),
            round(rouge["rouge1"], 4), round(rouge["rouge2"], 4),
            round(rouge["rougeL"], 4),
            round(bf1, 4),
        )
    wandb.log({"predictions": table})
    wandb.finish()

    print(f"Inference completed! Processed {len(results)} samples.")


if __name__ == "__main__":
    main()
