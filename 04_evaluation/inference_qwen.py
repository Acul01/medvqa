"""Inference script for Qwen2.5-VL on MC Medical VQA."""

import argparse
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import torch
import wandb
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from utils.load_data import MedicalVQALoader
from utils.mc_eval import run_mc_evaluation
from utils.mc_vqa import parse_predicted_letter
from utils.paths import get_data_root, get_mc_data_root, get_results_dir
from utils.qwen_vl import (
    DEFAULT_MODEL_PATH,
    build_mc_conversation,
    build_mc_reasoning_inference_conversation,
    configure_qwen_processor_pixels,
    model_short_name,
    parse_mc_reasoning_json_output,
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

DEFAULT_REASONING_JSON_MAX_NEW_TOKENS = 320


def decode_mc_prediction(generated_text: str, *, num_options: int, use_reasoning_json: bool):
    """Return (predicted_letter, reasoning_text_or_none, json_parse_ok_or_false)."""
    if use_reasoning_json:
        parsed = parse_mc_reasoning_json_output(generated_text, num_options=num_options)
        return parsed.predicted_letter, parsed.reasoning_text, parsed.json_parse_ok
    letter = parse_predicted_letter(generated_text, num_options=num_options)
    return letter, None, False


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
        choices=["none", "black", "shuffle"],
    )
    parser.add_argument("--ablation_seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default=get_results_dir())
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument(
        "--use_reasoning_json",
        action="store_true",
        help="Emit assistant JSON {\"reasoning\",\"answer\"} (match train_qwen --use_reasoning)",
    )
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


def _get_answer_letter_token_ids(tokenizer, num_opts: int = 10) -> dict[int, str]:
    """Return a mapping {token_id: letter} covering common encodings of A–J."""
    mapping: dict[int, str] = {}
    for letter in "ABCDEFGHIJ"[:num_opts]:
        for variant in [letter, f" {letter}", f"▁{letter}"]:  # bare / space / sentencepiece
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                mapping[ids[0]] = letter
    return mapping


def _find_answer_token_step(token_ids: list[int], answer_token_ids: dict[int, str]) -> int | None:
    """Return the index of the last answer-letter token in token_ids (backwards search)."""
    for i in reversed(range(len(token_ids))):
        if token_ids[i] in answer_token_ids:
            return i
    return None


def _compute_first_token_entropy(
    model, inputs: dict
) -> tuple[float, float, int]:
    """Forward pass over the prompt; returns (entropy, confidence, argmax_token_id)."""
    fwd = model(**inputs, return_dict=True)
    first_logits = fwd.logits[0, -1, :].double()
    log_probs = torch.log_softmax(first_logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.where(probs > 0, probs * log_probs, torch.zeros_like(log_probs)).sum().item()
    confidence = float(probs.max().item())
    top_id = int(torch.argmax(first_logits).item())
    return entropy, confidence, top_id


def main():
    args = parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = (
            DEFAULT_REASONING_JSON_MAX_NEW_TOKENS
            if args.use_reasoning_json
            else DATASET_MAX_NEW_TOKENS[args.dataset]
        )

    model_short = model_short_name(args.model_path)
    ft_tag = "finetuned" if args.finetuned else "pretrained"
    reason_tag = "reasoning_json" if args.use_reasoning_json else "answer_only"
    run_name = args.wandb_run_name or f"{model_short}_{args.dataset}_{ft_tag}_{reason_tag}_eval_mc"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=args.dataset,
        tags=["inference", "mc", "qwen", model_short, args.dataset, ft_tag, reason_tag],
        config=vars(args),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve to absolute path — newer huggingface_hub rejects relative paths
    # like "./05_outputs/..." in from_pretrained.
    model_path = os.path.abspath(args.model_path)

    is_lora = Path(model_path).exists() and (Path(model_path) / "adapter_config.json").exists()
    if is_lora:
        from peft import PeftModel

        base_path = args.base_model or DEFAULT_MODEL_PATH
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_path, torch_dtype=torch.float32, device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float32, device_map="auto"
        )
    model.eval()

    # Auto-detect spatial filter checkpoint produced by train_qwen_modified.py.
    # If spatial_filter.pt is present in the model directory, wrap the model so
    # that the filter is active during inference (same as during training).
    spatial_filter_ckpt = Path(model_path) / "spatial_filter.pt"
    if spatial_filter_ckpt.exists():
        from utils.spatial_filter import load_spatial_filter_checkpoint
        model = load_spatial_filter_checkpoint(
            model, str(spatial_filter_ckpt), bottleneck=64
        )
        print(f"Spatial filter applied ({spatial_filter_ckpt})")
    else:
        print("No spatial_filter.pt found — running without spatial filter.")

    if is_lora:
        processor_path = resolve_processor_path(args.base_model or DEFAULT_MODEL_PATH)
    else:
        processor_path = resolve_processor_path(model_path)
    processor = AutoProcessor.from_pretrained(processor_path)
    configure_qwen_processor_pixels(processor)

    loader = MedicalVQALoader(data_root=args.data_root, mc_data_root=args.mc_data_root)
    samples = loader.load_dataset(args.dataset, split=args.split)
    if args.n_samples is not None:
        samples = samples[: args.n_samples]
    shuffle_assignment = (
        _build_shuffle_assignment(samples, args.ablation_seed) if args.ablation_mode == "shuffle" else {}
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

    # Precompute answer-letter token-id mapping (used for reasoning-mode entropy)
    answer_letter_token_ids = _get_answer_letter_token_ids(processor.tokenizer)

    results = []
    with torch.no_grad():
        for sample_idx, sample in enumerate(tqdm(samples)):
            if not sample.mc_options or sample.mc_correct_index is None:
                continue

            source_image_path = sample.image_path
            effective_image_path = shuffle_assignment.get(sample_idx, source_image_path)
            image = Image.open(effective_image_path).convert("RGB")
            if args.ablation_mode == "black":
                image = Image.new("RGB", image.size, color=(0, 0, 0))

            conversation = (
                build_mc_reasoning_inference_conversation(
                    sample.question,
                    sample.mc_options,
                    image=image,
                )
                if args.use_reasoning_json
                else build_mc_conversation(
                    sample.question,
                    sample.mc_options,
                    image=image,
                )
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
            num_opts = len(sample.mc_options)
            predicted_letter, pred_reason_text, json_ok = decode_mc_prediction(
                generated_text, num_options=num_opts, use_reasoning_json=args.use_reasoning_json
            )
            ground_truth_letter = sample.answer

            if args.use_reasoning_json and generation_output.scores:
                # Reasoning mode: entropy at the final answer-letter token position
                valid_answer_ids = {
                    tid: l for tid, l in answer_letter_token_ids.items()
                    if l in "ABCDEFGHIJ"[:num_opts]
                }
                answer_step = _find_answer_token_step(generated_ids.tolist(), valid_answer_ids)
                if answer_step is not None:
                    ans_logits = generation_output.scores[answer_step][0].double()
                    ans_log_probs = torch.log_softmax(ans_logits, dim=-1)
                    ans_probs = torch.exp(ans_log_probs)
                    entropy_val = -(ans_probs * ans_log_probs.clamp(min=1e-12)).sum().item()
                    confidence = float(ans_probs.max().item())
                    entropy_tokens = {predicted_letter or "?": entropy_val}
                    entropy_mean = entropy_val
                else:
                    # Fallback: no answer token found – use first-token forward pass
                    entropy_val, confidence, first_tok_id = _compute_first_token_entropy(model, inputs)
                    first_tok_text = processor.tokenizer.decode(
                        [first_tok_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
                    ).strip() or f"tok_{first_tok_id}"
                    entropy_tokens = {first_tok_text: entropy_val}
                    entropy_mean = entropy_val
            else:
                # Standard mode: entropy of first generated token via forward pass
                entropy_val, confidence, first_tok_id = _compute_first_token_entropy(model, inputs)
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
                sample_letter, _, _ = decode_mc_prediction(
                    sample_text, num_options=num_opts, use_reasoning_json=args.use_reasoning_json
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

            row_out = {
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
                    "use_reasoning_json": args.use_reasoning_json,
                }
            if args.use_reasoning_json:
                row_out["predicted_reasoning_text"] = pred_reason_text
                row_out["predicted_json_parse_ok"] = json_ok
            results.append(row_out)

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
