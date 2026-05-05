"""Inference script for standard LLaVA on MC Medical VQA."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
import wandb
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from utils.load_data import MedicalVQALoader
from utils.mc_vqa import format_mc_user_text, parse_predicted_letter
from utils.save_results import clean_answer, save_results

DEFAULT_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"

DATASET_MAX_NEW_TOKENS = {
    "pathvqa": 8,
    "slake": 8,
    "vqarad": 8,
    "medvqa2019": 8,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with LLaVA on MC Medical VQA")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="data", help="Root directory containing raw image folders")
    parser.add_argument("--mc_data_root", type=str, default="data_mc_openai", help="Root directory containing MC JSON/CSV exports")
    parser.add_argument("--dataset", type=str, required=True, choices=["pathvqa", "slake", "vqarad", "medvqa2019"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--agreement_samples", type=int, default=5, help="Number of stochastic generations used to compute agreement")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--finetuned", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="medvqa")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def build_conversation(question: str, options: list[str]):
    user_text = format_mc_user_text(question, options)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
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


def _compute_token_entropies(step_scores, generated_ids, tokenizer) -> tuple[dict, float]:
    entropies = []
    for step_logits in step_scores[: generated_ids.shape[0]]:
        probs = torch.softmax(step_logits[0], dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
        entropies.append(entropy)

    special_tokens = set(getattr(tokenizer, "all_special_tokens", []))
    entropy_tokens = {}
    for token_id, entropy in zip(generated_ids.tolist(), entropies):
        token_id = int(token_id)
        raw_token = tokenizer.convert_ids_to_tokens(token_id)
        if raw_token in special_tokens:
            continue

        token_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip()
        if not token_text:
            continue

        if token_text in entropy_tokens:
            dup_idx = 2
            candidate = f"{token_text}#{dup_idx}"
            while candidate in entropy_tokens:
                dup_idx += 1
                candidate = f"{token_text}#{dup_idx}"
            token_text = candidate
        entropy_tokens[token_text] = entropy

    entropy_mean = sum(entropy_tokens.values()) / len(entropy_tokens) if entropy_tokens else float("nan")
    return entropy_tokens, entropy_mean


def _compute_generation_confidence(step_scores) -> float:
    token_confidences = []
    for step_logits in step_scores:
        probs = torch.softmax(step_logits[0], dim=-1)
        token_confidences.append(float(torch.max(probs).item()))
    return sum(token_confidences) / len(token_confidences) if token_confidences else float("nan")


def _compute_agreement(letter_predictions: list[str | None]) -> float:
    valid = [letter for letter in letter_predictions if letter]
    if not valid:
        return 0.0
    counts = Counter(valid)
    return max(counts.values()) / len(valid)


def _safe_div(a: int, b: int) -> float:
    return (a / b) if b else 0.0


def main():
    args = parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = DATASET_MAX_NEW_TOKENS[args.dataset]

    model_short = _model_short_name(args.model_path)
    ft_tag = "finetuned" if args.finetuned else "pretrained"
    run_name = args.wandb_run_name or f"{model_short}_{args.dataset}_{ft_tag}_eval_mc"
    wandb.init(project=args.wandb_project, name=run_name, group=args.dataset, tags=["inference", "mc", model_short, args.dataset, ft_tag], config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_lora = Path(args.model_path).exists() and (Path(args.model_path) / "adapter_config.json").exists()
    if is_lora:
        from peft import PeftModel

        base_path = args.base_model or DEFAULT_MODEL_PATH
        base_model = LlavaForConditionalGeneration.from_pretrained(base_path, torch_dtype=torch.float32, device_map="auto")
        model = PeftModel.from_pretrained(base_model, args.model_path)
        model = model.merge_and_unload()
    else:
        model = LlavaForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.float32, device_map="auto")
    model.eval()

    if is_lora:
        processor_path = args.base_model or DEFAULT_MODEL_PATH
    else:
        processor_path = args.model_path
    if Path(processor_path).exists() and not Path(processor_path, "preprocessor_config.json").exists():
        processor_path = DEFAULT_MODEL_PATH
    processor = AutoProcessor.from_pretrained(processor_path)
    if getattr(processor, "patch_size", None) is None:
        processor.patch_size = 14
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = "default"

    loader = MedicalVQALoader(data_root=args.data_root, mc_data_root=args.mc_data_root)
    samples = loader.load_dataset(args.dataset, split=args.split)
    if args.n_samples is not None:
        samples = samples[: args.n_samples]

    gen_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.do_sample:
        gen_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
    else:
        gen_kwargs["do_sample"] = False
    if args.num_beams > 1:
        gen_kwargs["num_beams"] = args.num_beams

    agreement_samples = max(1, int(args.agreement_samples))
    agreement_gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    results = []
    with torch.no_grad():
        for sample in tqdm(samples):
            if not sample.mc_options or sample.mc_correct_index is None:
                continue
            image = Image.open(sample.image_path).convert("RGB")
            conversation = build_conversation(sample.question, sample.mc_options)
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

            generation_output = model.generate(**inputs, **gen_kwargs, return_dict_in_generate=True, output_scores=True)
            input_length = inputs["input_ids"].shape[1]
            generated_ids = generation_output.sequences[0][input_length:]
            generated_text = clean_answer(processor.decode(generated_ids, skip_special_tokens=True).strip())
            predicted_letter = parse_predicted_letter(generated_text, num_options=len(sample.mc_options))
            ground_truth_letter = sample.answer

            entropy_tokens, entropy_mean = _compute_token_entropies(
                generation_output.scores,
                generated_ids=generated_ids,
                tokenizer=processor.tokenizer,
            )
            confidence = _compute_generation_confidence(generation_output.scores)

            sampled_predicted_letters = []
            if agreement_samples == 1:
                sampled_predicted_letters.append(predicted_letter)
            else:
                for _ in range(agreement_samples):
                    sample_output = model.generate(
                        **inputs,
                        **agreement_gen_kwargs,
                        return_dict_in_generate=True,
                        output_scores=False,
                    )
                    sample_generated_ids = sample_output.sequences[0][input_length:]
                    sample_text = clean_answer(processor.decode(sample_generated_ids, skip_special_tokens=True).strip())
                    sample_letter = parse_predicted_letter(sample_text, num_options=len(sample.mc_options))
                    sampled_predicted_letters.append(sample_letter)
            agreement = _compute_agreement(sampled_predicted_letters)
            is_correct = predicted_letter == ground_truth_letter
            qtype = sample.question_type or "unknown"

            results.append(
                {
                    "image_path": sample.image_path,
                    "question": sample.question,
                    "options": sample.mc_options,
                    "correct_index": sample.mc_correct_index,
                    "question_type": qtype,
                    "ground_truth_answer": (sample.metadata or {}).get("gt_answer_text", ""),
                    "ground_truth_letter": ground_truth_letter,
                    "predicted_answer": generated_text,
                    "predicted_letter": predicted_letter,
                    "is_correct": is_correct,
                    "entropy_tokens": entropy_tokens,
                    "entropy": entropy_mean,
                    "agreement": agreement,
                    "confidence": confidence,
                    "agreement_samples": sampled_predicted_letters,
                }
            )

    total = len(results)
    closed_rows = [r for r in results if r.get("question_type") == "closed"]
    open_rows = [r for r in results if r.get("question_type") == "open"]

    overall_correct = sum(1 for r in results if r.get("is_correct"))
    closed_correct = sum(1 for r in closed_rows if r.get("is_correct"))
    open_correct = sum(1 for r in open_rows if r.get("is_correct"))

    overall_acc = _safe_div(overall_correct, total)
    closed_acc = _safe_div(closed_correct, len(closed_rows)) if closed_rows else None
    open_acc = _safe_div(open_correct, len(open_rows)) if open_rows else None

    print(f"MC accuracy: {overall_correct}/{total} = {overall_acc:.4f}")
    if closed_acc is not None:
        print(f"Closed accuracy: {closed_correct}/{len(closed_rows)} = {closed_acc:.4f}")
    if open_acc is not None:
        print(f"Open accuracy: {open_correct}/{len(open_rows)} = {open_acc:.4f}")

    save_results(
        results=results,
        model_path=args.model_path,
        dataset=args.dataset,
        n_samples=args.n_samples,
        finetuned=args.finetuned,
        exact_acc=overall_acc,
        bleu_scores={},
        rouge_scores={},
        bert_scores={},
        yesno_acc=closed_acc,
        open_acc=open_acc,
        results_dir=args.results_dir,
        mc_overall_acc=overall_acc,
        mc_closed_acc=closed_acc,
        mc_open_acc=open_acc,
    )

    wandb.summary["mc_overall_accuracy"] = overall_acc
    if closed_acc is not None:
        wandb.summary["mc_closed_accuracy"] = closed_acc
    if open_acc is not None:
        wandb.summary["mc_open_accuracy"] = open_acc
    wandb.summary["total_samples"] = total

    table = wandb.Table(columns=["image_path", "question", "question_type", "ground_truth_letter", "predicted_letter", "is_correct", "options", "entropy", "agreement", "confidence"])
    for row in results:
        table.add_data(
            row["image_path"],
            row["question"],
            row.get("question_type"),
            row.get("ground_truth_letter"),
            row.get("predicted_letter"),
            bool(row.get("is_correct")),
            json.dumps(row.get("options", []), ensure_ascii=False),
            row.get("entropy", math.nan),
            row.get("agreement", math.nan),
            row.get("confidence", math.nan),
        )
    wandb.log({"predictions": table})
    wandb.finish()


if __name__ == "__main__":
    main()
