"""Inference script for Qwen2.5-VL on MC Medical VQA."""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
import wandb
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from utils.ablation import prepare_fgr_bbox_lookup, prepare_fgr_crop_lookup, resolve_ablation_image
from utils.fgr import draw_bbox_on_image
from utils.load_data import MedicalVQALoader
from utils.mc_eval import run_mc_evaluation
from utils.mc_vqa import parse_predicted_letter
from utils.paths import get_data_root, get_mc_data_root, get_results_dir
from utils.qwen_vl import (
    DEFAULT_MODEL_PATH,
    build_mc_conversation,
    model_short_name,
    prepare_qwen_inputs,
    resolve_processor_path,
)
from utils.save_results import clean_answer, save_results

DATASET_MAX_NEW_TOKENS = {
    "pathvqa": 8,
    "slake": 8,
    "vqarad": 8,
    "medvqa2019": 8,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with Qwen2.5-VL on MC Medical VQA")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=get_data_root())
    parser.add_argument("--mc_data_root", type=str, default=get_mc_data_root())
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["pathvqa", "slake", "vqarad", "medvqa2019"],
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument(
        "--agreement_samples",
        type=int,
        default=5,
        help="Number of stochastic generations used to compute agreement",
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="none",
        choices=["none", "black", "shuffle", "crop_unsupervised", "crop_supervised", "bbox_supervised"],
    )
    parser.add_argument("--ablation_seed", type=int, default=42)
    parser.add_argument(
        "--fgr_region_comparison_json",
        type=str,
        default=None,
        help="Required for crop_unsupervised/crop_supervised/bbox_supervised (output of fgr_region_comparison.py)",
    )
    parser.add_argument(
        "--bbox_color",
        type=str,
        default="red",
        help="Colour of the bounding box drawn on the image for bbox_supervised mode (default: red).",
    )
    parser.add_argument(
        "--bbox_line_width",
        type=int,
        default=4,
        help="Line thickness in pixels for the bounding box (default: 4).",
    )
    parser.add_argument("--results_dir", type=str, default=get_results_dir())
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--finetuned", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="medvqa")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def _compute_generation_confidence(step_scores, generated_ids, tokenizer, *, prompt_length: int | None = None) -> float:
    new_ids = generated_ids[prompt_length:] if prompt_length is not None else generated_ids
    n = min(len(step_scores), int(new_ids.shape[0]))
    special_tokens = set(getattr(tokenizer, "all_special_tokens", []))
    token_confidences = []
    for step_logits, token_id in zip(step_scores[:n], new_ids[:n].tolist()):
        if tokenizer.convert_ids_to_tokens(int(token_id)) in special_tokens:
            continue
        logits64 = step_logits[0].double()
        probs = torch.exp(torch.log_softmax(logits64, dim=-1))
        token_confidences.append(float(probs.max().item()))
    return sum(token_confidences) / len(token_confidences) if token_confidences else float("nan")


def _compute_agreement(letter_predictions: list[str | None]) -> float:
    valid = [letter for letter in letter_predictions if letter]
    if not valid:
        return 0.0
    counts = Counter(valid)
    return max(counts.values()) / len(valid)


def _safe_div(a: int, b: int) -> float:
    return (a / b) if b else 0.0


def _build_shuffle_assignment(samples, seed: int) -> dict[int, str]:
    image_paths = [sample.image_path for sample in samples]
    shuffled = image_paths.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return {idx: shuffled[idx] for idx in range(len(samples))}


def _move_inputs_to_device(inputs: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def main():
    args = parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = DATASET_MAX_NEW_TOKENS[args.dataset]

    model_short = model_short_name(args.model_path)
    ft_tag = "finetuned" if args.finetuned else "pretrained"
    run_name = args.wandb_run_name or f"{model_short}_{args.dataset}_{ft_tag}_eval_mc"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=args.dataset,
        tags=["inference", "mc", "qwen", model_short, args.dataset, ft_tag],
        config=vars(args),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_lora = Path(args.model_path).exists() and (Path(args.model_path) / "adapter_config.json").exists()
    if is_lora:
        from peft import PeftModel

        base_path = args.base_model or DEFAULT_MODEL_PATH
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_path, torch_dtype=torch.float32, device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, args.model_path)
        model = model.merge_and_unload()
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float32, device_map="auto"
        )
    model.eval()

    if is_lora:
        processor_path = resolve_processor_path(args.base_model or DEFAULT_MODEL_PATH)
    else:
        processor_path = resolve_processor_path(args.model_path)
    processor = AutoProcessor.from_pretrained(processor_path)

    loader = MedicalVQALoader(data_root=args.data_root, mc_data_root=args.mc_data_root)
    samples = loader.load_dataset(args.dataset, split=args.split)
    if args.n_samples is not None:
        samples = samples[: args.n_samples]
    shuffle_assignment = (
        _build_shuffle_assignment(samples, args.ablation_seed) if args.ablation_mode == "shuffle" else {}
    )
    fgr_crop_lookup = prepare_fgr_crop_lookup(
        args.ablation_mode,
        args.fgr_region_comparison_json,
        data_root=args.data_root,
    )
    fgr_bbox_lookup = prepare_fgr_bbox_lookup(
        args.ablation_mode,
        args.fgr_region_comparison_json,
        data_root=args.data_root,
    )

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
    skipped_fgr = 0
    with torch.no_grad():
        for sample_idx, sample in enumerate(tqdm(samples)):
            if not sample.mc_options or sample.mc_correct_index is None:
                continue

            source_image_path = sample.image_path
            ablation = resolve_ablation_image(
                ablation_mode=args.ablation_mode,
                source_image_path=source_image_path,
                question=sample.question,
                sample_idx=sample_idx,
                shuffle_assignment=shuffle_assignment,
                fgr_crop_lookup=fgr_crop_lookup,
                fgr_bbox_lookup=fgr_bbox_lookup,
                data_root=args.data_root,
            )
            if ablation.skipped:
                skipped_fgr += 1
                continue
            effective_image_path = ablation.effective_image_path
            image = Image.open(effective_image_path).convert("RGB")
            if args.ablation_mode == "black":
                image = Image.new("RGB", image.size, color=(0, 0, 0))
            elif args.ablation_mode == "bbox_supervised" and ablation.fgr_bbox is not None:
                image = draw_bbox_on_image(
                    image, ablation.fgr_bbox,
                    color=args.bbox_color,
                    line_width=args.bbox_line_width,
                )

            conversation = build_mc_conversation(
                sample.question,
                sample.mc_options,
                image=image,
            )
            inputs = prepare_qwen_inputs(
                processor,
                conversation,
                add_generation_prompt=True,
            )
            inputs = _move_inputs_to_device(inputs, device)
            input_length = inputs["input_ids"].shape[1]

            generation_output = model.generate(
                **inputs,
                **gen_kwargs,
                return_dict_in_generate=True,
                output_scores=True,
            )
            generated_ids = generation_output.sequences[0][input_length:]
            generated_text = clean_answer(
                processor.decode(generated_ids, skip_special_tokens=True).strip()
            )
            predicted_letter = parse_predicted_letter(
                generated_text, num_options=len(sample.mc_options)
            )
            ground_truth_letter = sample.answer

            fwd = model(**inputs, return_dict=True)
            first_logits = fwd.logits[0, -1, :].double()
            log_probs_fwd = torch.log_softmax(first_logits, dim=-1)
            probs_fwd = torch.exp(log_probs_fwd)
            entropy_val = -torch.where(
                probs_fwd > 0,
                probs_fwd * log_probs_fwd,
                torch.zeros_like(log_probs_fwd),
            ).sum().item()
            confidence = float(probs_fwd.max().item())
            first_tok_id = int(torch.argmax(first_logits).item())
            first_tok_text = processor.tokenizer.decode(
                [first_tok_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            ).strip() or f"tok_{first_tok_id}"
            entropy_tokens = {first_tok_text: entropy_val}
            entropy_mean = entropy_val

            sampled_predicted_letters = []
            agreement_confidences = []
            for _ in range(agreement_samples):
                sample_output = model.generate(
                    **inputs,
                    **agreement_gen_kwargs,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                sample_generated_ids = sample_output.sequences[0][input_length:]
                sample_text = clean_answer(
                    processor.decode(sample_generated_ids, skip_special_tokens=True).strip()
                )
                sample_letter = parse_predicted_letter(
                    sample_text, num_options=len(sample.mc_options)
                )
                sampled_predicted_letters.append(sample_letter)
                if sample_output.scores:
                    agreement_confidences.append(
                        _compute_generation_confidence(
                            sample_output.scores,
                            sample_generated_ids,
                            processor.tokenizer,
                        )
                    )

            agreement = _compute_agreement(sampled_predicted_letters)
            if agreement_confidences:
                confidence = sum(agreement_confidences) / len(agreement_confidences)

            is_correct = predicted_letter == ground_truth_letter
            qtype = sample.question_type or "unknown"

            row = {
                    "image_path": effective_image_path,
                    "source_image_path": source_image_path,
                    "ablation_mode": args.ablation_mode,
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
            if ablation.fgr_crop_path is not None:
                row["fgr_crop_path"] = ablation.fgr_crop_path
            results.append(row)

    if skipped_fgr:
        print(f"Skipped {skipped_fgr} samples (no FGR crop mapping)")

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

    results_path = save_results(
        results=results,
        model_path=args.model_path,
        dataset=args.dataset,
        n_samples=args.n_samples,
        results_dir=args.results_dir,
    )
    run_mc_evaluation(results_path)

    wandb.summary["mc_overall_accuracy"] = overall_acc
    if closed_acc is not None:
        wandb.summary["mc_closed_accuracy"] = closed_acc
    if open_acc is not None:
        wandb.summary["mc_open_accuracy"] = open_acc
    wandb.summary["total_samples"] = total

    table = wandb.Table(
        columns=[
            "image_path",
            "question",
            "question_type",
            "ground_truth_letter",
            "predicted_letter",
            "is_correct",
            "options",
            "entropy",
            "agreement",
            "confidence",
        ]
    )
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
