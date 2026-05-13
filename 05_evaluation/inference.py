"""Inference script for standard LLaVA on MC Medical VQA."""

import argparse
import gc
import json
import math
import random
import re
import warnings
from collections import Counter
from pathlib import Path

import torch
import wandb
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from utils.load_data import MedicalVQALoader
from utils.mc_vqa import LETTER_POOL, format_mc_user_text, parse_predicted_letter
from utils.save_results import clean_answer, save_results

DEFAULT_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"
END_REASONING_TOKEN = "<END_REASONING>"

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
    parser.add_argument("--use_two_stage_reasoning", action="store_true", default=False, help="Run 2-stage reasoning before final MC answer")
    parser.add_argument("--use_gradcam", action="store_true", default=False, help="Attempt GradCAM in 2-stage reasoning flow (currently unsupported for this backend; safe fallback)")
    parser.add_argument("--gradcam_dir", type=str, default=None, help="Directory for GradCAM artifacts when enabled")
    parser.add_argument("--gradcam_chunk_size", type=int, default=100, help="Chunk size for automatic gradcam inference splitting to reduce OOM risk")
    parser.add_argument("--reasoning_max_new_tokens", type=int, default=192, help="Maximum new tokens for stage-1 reasoning generation")
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="none",
        choices=["none", "black", "shuffle"],
        help="Image ablation mode: none, black image, or sample-level image assignment shuffle",
    )
    parser.add_argument("--ablation_seed", type=int, default=42, help="Seed used for ablation shuffle mode")
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


def build_text_only_conversation(user_text: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]


def _format_mc_options_block(options: list[str]) -> str:
    lines = [f"{chr(65 + idx)}) {str(opt).strip()}" for idx, opt in enumerate(options)]
    return "\n".join(lines)


def build_reasoning_prompt(question: str) -> str:
    return (
        "You are a medical imaging expert analyzing an image.\n\n"
        f"Question: {question}\n\n"
        "Provide your reasoning in exactly this format:\n"
        "REGION: [identify the anatomical region relevant to this question]\n"
        "OBSERVATION: [describe what you observe in that region]\n"
        "REASONING: [explain how this observation relates to the question]\n\n"
        f"End your response with exactly this token: {END_REASONING_TOKEN}"
    )


def build_final_prompt(question: str, options: list[str], region: str, observation: str, reasoning: str) -> str:
    return (
        "You are a medical imaging expert analyzing an image.\n\n"
        "You have already analyzed this image and provided the following reasoning:\n"
        f"REGION: {region}\n"
        f"OBSERVATION: {observation}\n"
        f"REASONING: {reasoning}\n\n"
        "Based on this analysis, answer the following question \n"
        "by selecting exactly one option.\n\n"
        f"Question: {question}\n"
        f"Options: {_format_mc_options_block(options)}\n\n"
        "Answer with the option letter only (e.g. A, B, C, D or A, B for Yes/No questions)."
    )


def parse_reasoning_fields(text: str) -> tuple[dict, str]:
    source = text or ""
    marker_idx = source.find(END_REASONING_TOKEN)
    if marker_idx != -1:
        parse_source = source[:marker_idx]
        parse_mode = "best_effort_marker"
    else:
        parse_source = source
        parse_mode = "best_effort_no_marker"
    fields = {"region": "unknown", "observation": "unknown", "reasoning": "unknown"}
    upper = parse_source.upper()
    markers = [("region", "REGION:"), ("observation", "OBSERVATION:"), ("reasoning", "REASONING:")]
    positions = []
    for key, marker in markers:
        idx = upper.find(marker)
        if idx != -1:
            positions.append((idx, key, marker))
    positions.sort(key=lambda x: x[0])
    for pos_idx, (start, key, marker) in enumerate(positions):
        content_start = start + len(marker)
        content_end = positions[pos_idx + 1][0] if pos_idx + 1 < len(positions) else len(parse_source)
        value = parse_source[content_start:content_end].strip()
        if value:
            fields[key] = re.sub(r"\s+", " ", value)
    return fields, parse_mode


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


def _stage2_letter_probs_from_first_step(step_scores, tokenizer, num_options: int) -> dict[str, float]:
    probs_map: dict[str, float] = {}
    if not step_scores:
        for i in range(num_options):
            probs_map[LETTER_POOL[i]] = float("nan")
        return probs_map

    first_logits = step_scores[0][0]
    dist = torch.softmax(first_logits.float(), dim=-1)
    for i in range(num_options):
        letter = LETTER_POOL[i]
        candidate_ids = []
        for token_variant in (letter, f" {letter}"):
            ids = tokenizer.encode(token_variant, add_special_tokens=False)
            if ids:
                candidate_ids.append(ids[0])
        probs = [float(dist[tid].item()) for tid in candidate_ids if 0 <= tid < dist.shape[-1]]
        probs_map[letter] = max(probs) if probs else float("nan")
    return probs_map


def _pick_letter_from_probs(probs_map: dict[str, float]) -> str | None:
    valid = [(k, v) for k, v in probs_map.items() if isinstance(v, (int, float)) and not math.isnan(v)]
    if not valid:
        return None
    valid.sort(key=lambda x: x[1], reverse=True)
    return valid[0][0]


def _compute_agreement(letter_predictions: list[str | None]) -> float:
    valid = [letter for letter in letter_predictions if letter]
    if not valid:
        return 0.0
    counts = Counter(valid)
    return max(counts.values()) / len(valid)


def _nanmean(values: list[float]) -> float:
    valid = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return sum(valid) / len(valid) if valid else float("nan")


def _safe_div(a: int, b: int) -> float:
    return (a / b) if b else 0.0


def _build_shuffle_assignment(samples, seed: int) -> dict[int, str]:
    image_paths = [sample.image_path for sample in samples]
    shuffled = image_paths.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return {idx: shuffled[idx] for idx in range(len(samples))}


def main():
    args = parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = DATASET_MAX_NEW_TOKENS[args.dataset]
    if args.use_gradcam and not args.use_two_stage_reasoning:
        warnings.warn("--use_gradcam is ignored unless --use_two_stage_reasoning is set.", UserWarning, stacklevel=1)
    gradcam_effective = bool(args.use_gradcam and args.use_two_stage_reasoning)
    gradcam_dir = args.gradcam_dir or str(Path(args.results_dir) / "gradcam_overlays")
    gradcam_chunk_size = max(1, int(args.gradcam_chunk_size))
    _ = gradcam_dir

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
    shuffle_assignment = _build_shuffle_assignment(samples, args.ablation_seed) if args.ablation_mode == "shuffle" else {}

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

    chunk_ranges: list[tuple[int, int]]
    if gradcam_effective:
        chunk_ranges = [(start, min(start + gradcam_chunk_size, len(samples))) for start in range(0, len(samples), gradcam_chunk_size)]
        print(f"[gradcam-chunk] active: {len(chunk_ranges)} chunks (size={gradcam_chunk_size}, total={len(samples)})")
    else:
        chunk_ranges = [(0, len(samples))]

    results = []
    with torch.no_grad():
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_ranges, start=1):
            chunk_samples = samples[chunk_start:chunk_end]
            for local_idx, sample in enumerate(tqdm(chunk_samples, leave=not gradcam_effective)):
                sample_idx = chunk_start + local_idx
                if not sample.mc_options or sample.mc_correct_index is None:
                    continue
                source_image_path = sample.image_path
                effective_image_path = shuffle_assignment.get(sample_idx, source_image_path)
                image = Image.open(effective_image_path).convert("RGB")
                if args.ablation_mode == "black":
                    image = Image.new("RGB", image.size, color=(0, 0, 0))
                reasoning_raw = ""
                reasoning_region = "unknown"
                reasoning_observation = "unknown"
                reasoning_explanation = "unknown"
                reasoning_parse_mode = "disabled"
                gradcam_status = "disabled"
                gradcam_error = ""
                gradcam_overlay_path = ""
                gradcam_heatmap_path = ""

                if args.use_two_stage_reasoning:
                    if gradcam_effective:
                        gradcam_status = "failed"
                        gradcam_error = "unsupported_backend"
                    reasoning_prompt = build_reasoning_prompt(sample.question)
                    reasoning_conv = build_text_only_conversation(reasoning_prompt)
                    reasoning_text = processor.apply_chat_template(reasoning_conv, add_generation_prompt=True)
                    reasoning_inputs = processor(images=image, text=reasoning_text, return_tensors="pt").to(device)
                    reasoning_output = model.generate(
                        **reasoning_inputs,
                        max_new_tokens=args.reasoning_max_new_tokens,
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=False,
                    )
                    reasoning_input_len = reasoning_inputs["input_ids"].shape[1]
                    reasoning_ids = reasoning_output.sequences[0][reasoning_input_len:]
                    reasoning_raw = clean_answer(processor.decode(reasoning_ids, skip_special_tokens=True).strip())
                    parsed_reasoning, reasoning_parse_mode = parse_reasoning_fields(reasoning_raw)
                    reasoning_region = parsed_reasoning["region"]
                    reasoning_observation = parsed_reasoning["observation"]
                    reasoning_explanation = parsed_reasoning["reasoning"]
                    final_user_prompt = build_final_prompt(
                        sample.question,
                        sample.mc_options,
                        reasoning_region,
                        reasoning_observation,
                        reasoning_explanation,
                    )
                    conversation = build_text_only_conversation(final_user_prompt)
                else:
                    conversation = build_conversation(sample.question, sample.mc_options)

                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
                input_length = inputs["input_ids"].shape[1]

                generation_output = model.generate(**inputs, **gen_kwargs, return_dict_in_generate=True, output_scores=True)
                generated_ids = generation_output.sequences[0][input_length:]
                generated_text = clean_answer(processor.decode(generated_ids, skip_special_tokens=True).strip())
                stage2_option_probs = {}
                stage2_decoding_mode = "text_parse"
                if args.use_two_stage_reasoning:
                    stage2_option_probs = _stage2_letter_probs_from_first_step(
                        generation_output.scores,
                        tokenizer=processor.tokenizer,
                        num_options=len(sample.mc_options),
                    )
                    predicted_letter = _pick_letter_from_probs(stage2_option_probs)
                    generated_text = predicted_letter or ""
                    stage2_decoding_mode = "letter_logits_only"
                else:
                    predicted_letter = parse_predicted_letter(generated_text, num_options=len(sample.mc_options))
                ground_truth_letter = sample.answer

                entropy_tokens, entropy_mean = _compute_token_entropies(
                    generation_output.scores,
                    generated_ids=generated_ids,
                    tokenizer=processor.tokenizer,
                )

                sampled_predicted_letters = []
                sampled_entropies = []
                sampled_confidences = []
                for _ in range(agreement_samples):
                    sample_output = model.generate(
                        **inputs,
                        **agreement_gen_kwargs,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    sample_generated_ids = sample_output.sequences[0][input_length:]
                    if args.use_two_stage_reasoning:
                        sample_probs = _stage2_letter_probs_from_first_step(
                            sample_output.scores,
                            tokenizer=processor.tokenizer,
                            num_options=len(sample.mc_options),
                        )
                        sample_letter = _pick_letter_from_probs(sample_probs)
                    else:
                        sample_text = clean_answer(processor.decode(sample_generated_ids, skip_special_tokens=True).strip())
                        sample_letter = parse_predicted_letter(sample_text, num_options=len(sample.mc_options))
                    sampled_predicted_letters.append(sample_letter)
                    _, sample_entropy = _compute_token_entropies(
                        sample_output.scores,
                        generated_ids=sample_generated_ids,
                        tokenizer=processor.tokenizer,
                    )
                    sampled_entropies.append(sample_entropy)
                    sampled_confidences.append(_compute_generation_confidence(sample_output.scores))

                entropy_mean = _nanmean(sampled_entropies)
                confidence = _nanmean(sampled_confidences)
                agreement = _compute_agreement(sampled_predicted_letters)
                is_correct = predicted_letter == ground_truth_letter
                qtype = sample.question_type or "unknown"

                results.append(
                    {
                        "image_path": effective_image_path,
                        "source_image_path": source_image_path,
                        "ablation_mode": args.ablation_mode,
                        "use_gradcam": gradcam_effective,
                        "gradcam_status": gradcam_status,
                        "gradcam_error": gradcam_error,
                        "gradcam_overlay_path": gradcam_overlay_path,
                        "gradcam_heatmap_path": gradcam_heatmap_path,
                        "reasoning_raw": reasoning_raw,
                        "reasoning_region": reasoning_region,
                        "reasoning_observation": reasoning_observation,
                        "reasoning_explanation": reasoning_explanation,
                        "reasoning_parse_mode": reasoning_parse_mode,
                        "stage2_decoding_mode": stage2_decoding_mode,
                        "stage2_option_probs": stage2_option_probs,
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

            if gradcam_effective:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                print(f"[gradcam-chunk] completed {chunk_idx}/{len(chunk_ranges)} ({chunk_end - chunk_start} samples)")

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
