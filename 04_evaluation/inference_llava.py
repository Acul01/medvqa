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

from utils.ablation import prepare_fgr_crop_lookup, resolve_ablation_image
from utils.load_data import MedicalVQALoader
from utils.mc_vqa import LETTER_POOL, format_mc_user_text, parse_predicted_letter
from utils.paths import get_data_root, get_mc_data_root, get_results_dir, get_som_dir
from utils.qwen_vl import format_mc_reasoning_user_text, parse_mc_reasoning_json_output
from utils.som import SomResolver, draw_region_marks
from utils.som_select import (
    DEFAULT_SELECTION_MAX_NEW_TOKENS,
    build_segment_selection_prompt,
    parse_selected_labels,
    question_with_som_hint,
)
from utils.save_results import clean_answer, save_results
from utils.mc_eval import add_aurc_cli_arguments, apply_aurc_to_wandb_summary, run_mc_evaluation

DEFAULT_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"
END_REASONING_TOKEN = "<END_REASONING>"
DEFAULT_REASONING_JSON_MAX_NEW_TOKENS = 320

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
    parser.add_argument("--data_root", type=str, default=get_data_root(), help="Root directory containing raw image folders")
    parser.add_argument("--mc_data_root", type=str, default=get_mc_data_root(), help="Root directory containing MC JSON/CSV exports")
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
        "--use_reasoning_json",
        action="store_true",
        help='Emit assistant JSON {"reasoning","answer"} (match train_llava --use_reasoning)',
    )
    parser.add_argument(
        "--answer_first",
        action="store_true",
        help='Use answer-first JSON schema {"answer","reasoning"} (match train_llava --answer_first). Only effective with --use_reasoning_json.',
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="none",
        choices=["none", "black", "shuffle", "crop_unsupervised", "crop_supervised"],
        help="Image ablation mode: none, black, shuffle, or FGR winning crop (supervised/unsupervised)",
    )
    parser.add_argument("--ablation_seed", type=int, default=42, help="Seed used for ablation shuffle mode")
    parser.add_argument(
        "--fgr_region_comparison_json",
        type=str,
        default=None,
        help="Required for crop_unsupervised/crop_supervised (output of fgr_region_comparison.py)",
    )
    parser.add_argument("--results_dir", type=str, default=get_results_dir())
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--finetuned", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="medvqa")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    add_aurc_cli_arguments(parser)
    parser.add_argument(
        "--use_som",
        action="store_true",
        default=False,
        help="Replace images with pre-annotated SoM versions (run scripts/precompute_som.py first).",
    )
    parser.add_argument(
        "--som_dir",
        type=str,
        default=get_som_dir(),
        help="Root directory of SoM outputs (default: MEDVQA_SOM_DIR or 02_data/som).",
    )
    return parser.parse_args()


def build_conversation(question: str, options: list[str], *, use_reasoning_json: bool = False, answer_first: bool = False):
    if use_reasoning_json:
        user_text = format_mc_reasoning_user_text(question, options, answer_first=answer_first)
    else:
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


def decode_mc_prediction(generated_text: str, *, num_options: int, use_reasoning_json: bool):
    """Return (predicted_letter, reasoning_text_or_none, json_parse_ok_or_false)."""
    if use_reasoning_json:
        parsed = parse_mc_reasoning_json_output(generated_text, num_options=num_options)
        return parsed.predicted_letter, parsed.reasoning_text, parsed.json_parse_ok
    letter = parse_predicted_letter(generated_text, num_options=num_options)
    return letter, None, False


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


def _compute_token_entropies(
    step_scores,
    generated_ids,
    tokenizer,
    *,
    prompt_length: int | None = None,
) -> tuple[dict, float]:
    """Pair each `step_scores` step with the token actually chosen at that step.

    Pass `prompt_length` when `generated_ids` is the full sequence (prompt + new);
    omit it when `generated_ids` is already only the generated suffix.
    """
    new_ids = generated_ids[prompt_length:] if prompt_length is not None else generated_ids
    n = min(len(step_scores), int(new_ids.shape[0]))
    entropies = []
    for step_logits in step_scores[:n]:
        logits64 = step_logits[0].double()
        log_probs = torch.log_softmax(logits64, dim=-1)
        probs = torch.exp(log_probs)
        # 0 * (-inf) = NaN in IEEE 754; Shannon convention: 0 * log(0) = 0
        entropy = -torch.where(probs > 0, probs * log_probs, torch.zeros_like(log_probs)).sum().item()
        entropies.append(entropy)

    special_tokens = set(getattr(tokenizer, "all_special_tokens", []))
    entropy_tokens = {}
    for token_id, entropy in zip(new_ids[:n].tolist(), entropies):
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
    if args.use_reasoning_json and args.use_two_stage_reasoning:
        raise ValueError("--use_reasoning_json and --use_two_stage_reasoning cannot be used together.")
    if args.max_new_tokens is None:
        args.max_new_tokens = (
            DEFAULT_REASONING_JSON_MAX_NEW_TOKENS
            if args.use_reasoning_json
            else DATASET_MAX_NEW_TOKENS[args.dataset]
        )
    if args.use_gradcam and not args.use_two_stage_reasoning:
        warnings.warn("--use_gradcam is ignored unless --use_two_stage_reasoning is set.", UserWarning, stacklevel=1)
    gradcam_effective = bool(args.use_gradcam and args.use_two_stage_reasoning)
    gradcam_dir = args.gradcam_dir or str(Path(args.results_dir) / "gradcam_overlays")
    gradcam_chunk_size = max(1, int(args.gradcam_chunk_size))
    _ = gradcam_dir

    model_short = _model_short_name(args.model_path)
    ft_tag = "finetuned" if args.finetuned else "pretrained"
    if args.use_reasoning_json:
        reason_tag = "reasoning_json_answer_first" if args.answer_first else "reasoning_json"
    else:
        reason_tag = "answer_only"
    run_name = args.wandb_run_name or f"{model_short}_{args.dataset}_{ft_tag}_{reason_tag}_eval_mc"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=args.dataset,
        tags=["inference", "mc", model_short, args.dataset, ft_tag, reason_tag],
        config=vars(args),
    )

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

    som_resolver: SomResolver | None = None
    if args.use_som:
        som_resolver = SomResolver(args.som_dir)
        som_resolver.load_dataset(args.dataset)
        print(f"SoM enabled — model selects relevant regions per question, from {args.som_dir}/{args.dataset}")

    fgr_crop_lookup = prepare_fgr_crop_lookup(
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

    chunk_ranges: list[tuple[int, int]]
    if gradcam_effective:
        chunk_ranges = [(start, min(start + gradcam_chunk_size, len(samples))) for start in range(0, len(samples), gradcam_chunk_size)]
        print(f"[gradcam-chunk] active: {len(chunk_ranges)} chunks (size={gradcam_chunk_size}, total={len(samples)})")
    else:
        chunk_ranges = [(0, len(samples))]

    results = []
    skipped_fgr = 0
    with torch.no_grad():
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_ranges, start=1):
            chunk_samples = samples[chunk_start:chunk_end]
            for local_idx, sample in enumerate(tqdm(chunk_samples, leave=not gradcam_effective)):
                sample_idx = chunk_start + local_idx
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
                    data_root=args.data_root,
                )
                if ablation.skipped:
                    skipped_fgr += 1
                    continue
                effective_image_path = ablation.effective_image_path
                question_for_prompt = sample.question
                som_selected_regions: list[str] = []
                som_selection_response = ""

                if som_resolver is not None and args.ablation_mode == "none":
                    region_meta = som_resolver.get_regions(effective_image_path)
                    original_image = Image.open(effective_image_path).convert("RGB")
                    if region_meta:
                        all_regions = som_resolver.load_region_masks(effective_image_path)
                        full_marked_image = draw_region_marks(original_image, all_regions)
                        valid_labels = {r["label"] for r in region_meta}

                        selection_prompt_text = build_segment_selection_prompt(sample.question)
                        selection_conv = build_text_only_conversation(selection_prompt_text)
                        selection_prompt = processor.apply_chat_template(
                            selection_conv, add_generation_prompt=True
                        )
                        selection_inputs = processor(
                            images=full_marked_image, text=selection_prompt, return_tensors="pt"
                        ).to(device)
                        selection_len = selection_inputs["input_ids"].shape[1]
                        selection_out = model.generate(
                            **selection_inputs,
                            max_new_tokens=DEFAULT_SELECTION_MAX_NEW_TOKENS,
                            do_sample=False,
                        )
                        selection_generated_ids = selection_out[0][selection_len:]
                        selection_text = clean_answer(
                            processor.decode(selection_generated_ids, skip_special_tokens=True).strip()
                        )
                        som_selection_response = selection_text
                        som_selected_regions = parse_selected_labels(selection_text, valid_labels)

                        if som_selected_regions:
                            selected_regions = [r for r in all_regions if r["label"] in som_selected_regions]
                            image = draw_region_marks(original_image, selected_regions)
                            question_for_prompt = question_with_som_hint(sample.question)
                        else:
                            image = original_image
                    else:
                        image = original_image
                else:
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
                    reasoning_prompt = build_reasoning_prompt(question_for_prompt)
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
                        question_for_prompt,
                        sample.mc_options,
                        reasoning_region,
                        reasoning_observation,
                        reasoning_explanation,
                    )
                    conversation = build_text_only_conversation(final_user_prompt)
                else:
                    conversation = build_conversation(
                        question_for_prompt,
                        sample.mc_options,
                        use_reasoning_json=args.use_reasoning_json,
                        answer_first=args.answer_first,
                    )

                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
                input_length = inputs["input_ids"].shape[1]

                generation_output = model.generate(**inputs, **gen_kwargs, return_dict_in_generate=True, output_scores=True)
                generated_ids = generation_output.sequences[0][input_length:]
                generated_text = clean_answer(processor.decode(generated_ids, skip_special_tokens=True).strip())
                stage2_option_probs = {}
                stage2_decoding_mode = "text_parse"
                predicted_reasoning_text = None
                predicted_json_parse_ok = False
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
                    predicted_letter, predicted_reasoning_text, predicted_json_parse_ok = decode_mc_prediction(
                        generated_text,
                        num_options=len(sample.mc_options),
                        use_reasoning_json=args.use_reasoning_json,
                    )
                ground_truth_letter = sample.answer

                # Forward pass AFTER generate — raw logits free of all logit processors.
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
                for _ in range(agreement_samples):
                    sample_output = model.generate(
                        **inputs,
                        **agreement_gen_kwargs,
                        return_dict_in_generate=True,
                        output_scores=args.use_two_stage_reasoning,
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
                        sample_letter, _, _ = decode_mc_prediction(
                            sample_text,
                            num_options=len(sample.mc_options),
                            use_reasoning_json=args.use_reasoning_json,
                        )
                    sampled_predicted_letters.append(sample_letter)

                agreement = _compute_agreement(sampled_predicted_letters)
                is_correct = predicted_letter == ground_truth_letter
                qtype = sample.question_type or "unknown"

                row = {
                        "image_path": effective_image_path,
                        "source_image_path": source_image_path,
                        "ablation_mode": args.ablation_mode,
                        "use_gradcam": gradcam_effective,
                        "gradcam_status": gradcam_status,
                        "gradcam_error": gradcam_error,
                        "gradcam_overlay_path": gradcam_overlay_path,
                        "gradcam_heatmap_path": gradcam_heatmap_path,
                        "use_reasoning_json": args.use_reasoning_json,
                        "answer_first": bool(args.answer_first and args.use_reasoning_json),
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
                        "som_selected_regions": som_selected_regions,
                        "som_selection_response": som_selection_response,
                    }
                if args.use_reasoning_json:
                    row["predicted_reasoning_text"] = predicted_reasoning_text
                    row["predicted_json_parse_ok"] = predicted_json_parse_ok
                if ablation.fgr_crop_path is not None:
                    row["fgr_crop_path"] = ablation.fgr_crop_path
                results.append(row)

            if gradcam_effective:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                print(f"[gradcam-chunk] completed {chunk_idx}/{len(chunk_ranges)} ({chunk_end - chunk_start} samples)")

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
    metrics = run_mc_evaluation(
        results_path,
        compute_aurc=not args.skip_aurc,
        combined_alpha=args.combined_alpha,
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
