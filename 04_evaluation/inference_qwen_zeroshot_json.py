"""Zero-shot inference for Qwen2.5-VL with JSON answer+rationale output."""

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

from utils.load_data import MedicalVQALoader
from utils.mc_eval import run_mc_evaluation
from utils.mc_json_zeroshot import (
    DEFAULT_JSON_MAX_NEW_TOKENS,
    DEFAULT_RATIONALE_MAX_NEW_TOKENS,
    build_qwen_json_conversation,
    build_qwen_stage1_conversation,
    build_qwen_stage2_conversation,
    build_qwen_stage2_json_conversation,
    cast_json_single_shot,
    find_answer_token_step,
    get_answer_letter_token_ids,
    is_json_single_shot,
    parse_mc_answer_rationale_json,
)
from utils.mc_vqa import parse_predicted_letter
from utils.paths import get_data_root, get_mc_data_root, get_results_dir, get_som_dir
from utils.som import SomResolver, draw_region_marks
from utils.som_select import (
    DEFAULT_SELECTION_MAX_NEW_TOKENS,
    parse_selected_labels,
    question_with_som_hint,
)
from utils.qwen_vl import (
    DEFAULT_MODEL_PATH,
    build_segment_selection_conversation,
    configure_qwen_processor_pixels,
    model_short_name,
    prepare_qwen_inputs,
    resolve_processor_path,
)
from utils.save_results import clean_answer, save_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot Qwen2.5-VL inference with JSON answer+rationale output"
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_root", type=str, default=get_data_root())
    parser.add_argument("--mc_data_root", type=str, default=get_mc_data_root())
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["pathvqa", "slake", "vqarad", "medvqa2019"],
    )
    parser.add_argument(
        "--json_order",
        type=str,
        required=True,
        choices=["answer_first", "rationale_first", "rationale_first_2stage", "rationale_first_2stage_norat"],
        help="Output mode: JSON single-shot field order or 2-stage CoT",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument(
        "--rationale_max_new_tokens",
        type=int,
        default=DEFAULT_RATIONALE_MAX_NEW_TOKENS,
        help="Stage-1 rationale max tokens (rationale_first_2stage only)",
    )
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--agreement_samples", type=int, default=5)
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="none",
        choices=["none", "black", "shuffle"],
    )
    parser.add_argument("--ablation_seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default=get_results_dir())
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default="medvqa")
    parser.add_argument("--wandb_run_name", type=str, default=None)
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


def _mean_finite(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def _compute_agreement(letter_predictions: list[str | None]) -> float:
    valid = [letter for letter in letter_predictions if letter]
    if not valid:
        return 0.0
    counts = Counter(valid)
    return max(counts.values()) / len(valid)


def _majority_letter(letters: list[str | None]) -> str | None:
    valid = [l for l in letters if l]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


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


def _compute_first_token_entropy(model, inputs: dict) -> tuple[float, float, int]:
    fwd = model(**inputs, return_dict=True)
    first_logits = fwd.logits[0, -1, :].double()
    log_probs = torch.log_softmax(first_logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.where(probs > 0, probs * log_probs, torch.zeros_like(log_probs)).sum().item()
    confidence = float(probs.max().item())
    top_id = int(torch.argmax(first_logits).item())
    return entropy, confidence, top_id


def _compute_answer_token_stats(
    generation_output,
    generated_ids,
    *,
    num_opts: int,
    answer_letter_token_ids: dict[int, str],
    model,
    inputs: dict,
) -> tuple[float, float]:
    """Returns (entropy, confidence) at the answer token from raw (pre-processor) logits.

    Both metrics are derived from the same logit vector so they are directly comparable.
    Falls back to a forward pass on `inputs` when the answer token cannot be located.
    """
    valid_answer_ids = {
        tid: letter
        for tid, letter in answer_letter_token_ids.items()
        if letter in "ABCDEFGHIJ"[:num_opts]
    }
    answer_step = find_answer_token_step(generated_ids.tolist(), valid_answer_ids)
    if answer_step is not None and generation_output.logits and answer_step < len(generation_output.logits):
        logits = generation_output.logits[answer_step][0].double()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs)).sum().item()
        confidence = float(probs.max().item())
        return entropy, confidence
    entropy, confidence, _ = _compute_first_token_entropy(model, inputs)
    return entropy, confidence


def _decode_new_tokens(processor, sequences, input_length: int) -> str:
    generated_ids = sequences[0][input_length:]
    return clean_answer(processor.decode(generated_ids, skip_special_tokens=True).strip())


def main():
    args = parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = DEFAULT_JSON_MAX_NEW_TOKENS

    model_short = model_short_name(args.model_path)
    run_name = (
        args.wandb_run_name
        or f"{model_short}_{args.dataset}_zeroshot_{args.json_order}_eval_mc"
    )
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=args.dataset,
        tags=[
            "inference",
            "mc",
            "qwen",
            "zeroshot",
            "json",
            model_short,
            args.dataset,
            args.json_order,
        ],
        config=vars(args),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float32, device_map="auto"
    )
    model.eval()

    processor_path = resolve_processor_path(args.model_path)
    processor = AutoProcessor.from_pretrained(processor_path)
    configure_qwen_processor_pixels(processor)

    loader = MedicalVQALoader(data_root=args.data_root, mc_data_root=args.mc_data_root)
    samples = loader.load_dataset(args.dataset, split=args.split)
    if args.n_samples is not None:
        samples = samples[: args.n_samples]
    shuffle_assignment = (
        _build_shuffle_assignment(samples, args.ablation_seed)
        if args.ablation_mode == "shuffle"
        else {}
    )

    som_resolver: SomResolver | None = None
    if args.use_som:
        som_resolver = SomResolver(args.som_dir)
        som_resolver.load_dataset(args.dataset)
        print(f"SoM enabled — model selects relevant regions per question, from {args.som_dir}/{args.dataset}")

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
    stage1_gen_kwargs = {"max_new_tokens": args.rationale_max_new_tokens, "do_sample": False}

    answer_letter_token_ids = get_answer_letter_token_ids(processor.tokenizer)

    results = []
    with torch.no_grad():
        for sample_idx, sample in enumerate(tqdm(samples)):
            if not sample.mc_options or sample.mc_correct_index is None:
                continue

            source_image_path = sample.image_path
            effective_image_path = shuffle_assignment.get(sample_idx, source_image_path)
            question_for_prompt = sample.question

            if som_resolver is not None and args.ablation_mode == "none":
                region_meta = som_resolver.get_regions(effective_image_path)
                original_image = Image.open(effective_image_path).convert("RGB")
                if region_meta:
                    all_regions = som_resolver.load_region_masks(effective_image_path)
                    full_marked_image = draw_region_marks(original_image, all_regions)
                    valid_labels = {r["label"] for r in region_meta}

                    selection_conv = build_segment_selection_conversation(
                        sample.question, full_marked_image
                    )
                    selection_inputs = _move_inputs_to_device(
                        prepare_qwen_inputs(processor, selection_conv, add_generation_prompt=True),
                        device,
                    )
                    selection_len = selection_inputs["input_ids"].shape[1]
                    selection_out = model.generate(
                        **selection_inputs,
                        max_new_tokens=DEFAULT_SELECTION_MAX_NEW_TOKENS,
                        do_sample=False,
                    )
                    selection_text = _decode_new_tokens(processor, selection_out, selection_len)
                    selected_labels = parse_selected_labels(selection_text, valid_labels)

                    if selected_labels:
                        selected_regions = [r for r in all_regions if r["label"] in selected_labels]
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

            num_opts = len(sample.mc_options)
            ground_truth_letter = sample.answer

            if is_json_single_shot(args.json_order):
                json_order = cast_json_single_shot(args.json_order)
                conversation = build_qwen_json_conversation(
                    question_for_prompt,
                    sample.mc_options,
                    json_order=json_order,
                    image=image,
                )
                inputs = prepare_qwen_inputs(processor, conversation, add_generation_prompt=True)
                inputs = _move_inputs_to_device(inputs, device)
                input_length = inputs["input_ids"].shape[1]

                sample_results = []
                for _ in range(agreement_samples):
                    sample_output = model.generate(
                        **inputs,
                        **agreement_gen_kwargs,
                        return_dict_in_generate=True,
                        output_logits=True,
                    )
                    sample_generated_ids = sample_output.sequences[0][input_length:]
                    sample_text = clean_answer(
                        processor.decode(sample_generated_ids, skip_special_tokens=True).strip()
                    )
                    sample_parsed = parse_mc_answer_rationale_json(sample_text, num_options=num_opts)
                    ent, conf = _compute_answer_token_stats(
                        sample_output,
                        sample_generated_ids,
                        num_opts=num_opts,
                        answer_letter_token_ids=answer_letter_token_ids,
                        model=model,
                        inputs=inputs,
                    )
                    sample_results.append((sample_text, sample_parsed, ent, conf))

                sampled_predicted_letters = [r[1].predicted_letter for r in sample_results]
                predicted_letter = _majority_letter(sampled_predicted_letters)
                matching = next(
                    (r for r in reversed(sample_results) if r[1].predicted_letter == predicted_letter),
                    sample_results[-1],
                )
                agreement = _compute_agreement(sampled_predicted_letters)
                entropy_mean = _mean_finite([r[2] for r in sample_results])
                confidence = _mean_finite([r[3] for r in sample_results])
                entropy_tokens = {predicted_letter or "?": entropy_mean}

                row_out = {
                    "image_path": effective_image_path,
                    "source_image_path": source_image_path,
                    "ablation_mode": args.ablation_mode,
                    "question": sample.question,
                    "options": sample.mc_options,
                    "correct_index": sample.mc_correct_index,
                    "question_type": sample.question_type or "unknown",
                    "ground_truth_answer": (sample.metadata or {}).get("gt_answer_text", ""),
                    "ground_truth_letter": ground_truth_letter,
                    "predicted_answer": matching[0],
                    "predicted_letter": predicted_letter,
                    "is_correct": predicted_letter == ground_truth_letter,
                    "entropy_tokens": entropy_tokens,
                    "entropy": entropy_mean,
                    "agreement": agreement,
                    "confidence": confidence,
                    "agreement_samples": sampled_predicted_letters,
                    "json_order": args.json_order,
                    "predicted_rationale_text": matching[1].rationale_text,
                    "predicted_json_parse_ok": matching[1].json_parse_ok,
                    "zeroshot_json": True,
                }
            else:
                # Stage 1: greedy reasoning (always do_sample=False)
                conv1 = build_qwen_stage1_conversation(
                    question_for_prompt, sample.mc_options, image=image
                )
                inputs1 = _move_inputs_to_device(
                    prepare_qwen_inputs(processor, conv1, add_generation_prompt=True), device
                )
                len1 = inputs1["input_ids"].shape[1]
                out1 = model.generate(
                    **inputs1,
                    **stage1_gen_kwargs,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
                stage1_raw = _decode_new_tokens(processor, out1.sequences, len1)

                if args.json_order == "rationale_first_2stage_norat":
                    # Stage 2: letter only (N samples)
                    sample_results = []
                    for _ in range(agreement_samples):
                        agr_conv2 = build_qwen_stage2_conversation(
                            question_for_prompt,
                            sample.mc_options,
                            rationale=stage1_raw,
                            image=image,
                        )
                        agr_inputs2 = _move_inputs_to_device(
                            prepare_qwen_inputs(processor, agr_conv2, add_generation_prompt=True),
                            device,
                        )
                        agr_len2 = agr_inputs2["input_ids"].shape[1]
                        agr_out2 = model.generate(
                            **agr_inputs2,
                            **agreement_gen_kwargs,
                            return_dict_in_generate=True,
                            output_logits=True,
                        )
                        agr_generated_ids = agr_out2.sequences[0][agr_len2:]
                        agr_stage2 = _decode_new_tokens(processor, agr_out2.sequences, agr_len2)
                        agr_letter = parse_predicted_letter(agr_stage2, num_options=num_opts)
                        ent, conf = _compute_answer_token_stats(
                            agr_out2,
                            agr_generated_ids,
                            num_opts=num_opts,
                            answer_letter_token_ids=answer_letter_token_ids,
                            model=model,
                            inputs=agr_inputs2,
                        )
                        sample_results.append((agr_stage2, agr_letter, ent, conf))

                    sampled_predicted_letters = [r[1] for r in sample_results]
                    predicted_letter = _majority_letter(sampled_predicted_letters)
                    matching = next(
                        (r for r in reversed(sample_results) if r[1] == predicted_letter),
                        sample_results[-1],
                    )
                    agreement = _compute_agreement(sampled_predicted_letters)
                    entropy_mean = _mean_finite([r[2] for r in sample_results])
                    confidence = _mean_finite([r[3] for r in sample_results])
                    entropy_tokens = {predicted_letter or "?": entropy_mean}

                    row_out = {
                        "image_path": effective_image_path,
                        "source_image_path": source_image_path,
                        "ablation_mode": args.ablation_mode,
                        "question": sample.question,
                        "options": sample.mc_options,
                        "correct_index": sample.mc_correct_index,
                        "question_type": sample.question_type or "unknown",
                        "ground_truth_answer": (sample.metadata or {}).get("gt_answer_text", ""),
                        "ground_truth_letter": ground_truth_letter,
                        "predicted_answer": matching[0],
                        "predicted_letter": predicted_letter,
                        "is_correct": predicted_letter == ground_truth_letter,
                        "entropy_tokens": entropy_tokens,
                        "entropy": entropy_mean,
                        "agreement": agreement,
                        "confidence": confidence,
                        "agreement_samples": sampled_predicted_letters,
                        "json_order": args.json_order,
                        "stage1_raw": stage1_raw,
                        "stage2_raw": matching[0],
                        "zeroshot_json": False,
                        "two_stage_cot": True,
                    }
                else:
                    # Stage 2: N sampled JSON runs based on greedy stage-1 reasoning
                    sample_results = []
                    for _ in range(agreement_samples):
                        agr_conv2 = build_qwen_stage2_json_conversation(
                            question_for_prompt,
                            sample.mc_options,
                            reasoning=stage1_raw,
                            image=image,
                        )
                        agr_inputs2 = _move_inputs_to_device(
                            prepare_qwen_inputs(processor, agr_conv2, add_generation_prompt=True),
                            device,
                        )
                        agr_len2 = agr_inputs2["input_ids"].shape[1]
                        agr_out2 = model.generate(
                            **agr_inputs2,
                            **agreement_gen_kwargs,
                            return_dict_in_generate=True,
                            output_logits=True,
                        )
                        agr_generated_ids = agr_out2.sequences[0][agr_len2:]
                        agr_stage2 = _decode_new_tokens(processor, agr_out2.sequences, agr_len2)
                        agr_parsed2 = parse_mc_answer_rationale_json(agr_stage2, num_options=num_opts)
                        ent, conf = _compute_answer_token_stats(
                            agr_out2,
                            agr_generated_ids,
                            num_opts=num_opts,
                            answer_letter_token_ids=answer_letter_token_ids,
                            model=model,
                            inputs=agr_inputs2,
                        )
                        sample_results.append((agr_stage2, agr_parsed2, ent, conf))

                    sampled_predicted_letters = [r[1].predicted_letter for r in sample_results]
                    predicted_letter = _majority_letter(sampled_predicted_letters)
                    matching = next(
                        (r for r in reversed(sample_results) if r[1].predicted_letter == predicted_letter),
                        sample_results[-1],
                    )
                    agreement = _compute_agreement(sampled_predicted_letters)
                    entropy_mean = _mean_finite([r[2] for r in sample_results])
                    confidence = _mean_finite([r[3] for r in sample_results])
                    entropy_tokens = {predicted_letter or "?": entropy_mean}

                    row_out = {
                        "image_path": effective_image_path,
                        "source_image_path": source_image_path,
                        "ablation_mode": args.ablation_mode,
                        "question": sample.question,
                        "options": sample.mc_options,
                        "correct_index": sample.mc_correct_index,
                        "question_type": sample.question_type or "unknown",
                        "ground_truth_answer": (sample.metadata or {}).get("gt_answer_text", ""),
                        "ground_truth_letter": ground_truth_letter,
                        "predicted_answer": matching[0],
                        "predicted_letter": predicted_letter,
                        "is_correct": predicted_letter == ground_truth_letter,
                        "entropy_tokens": entropy_tokens,
                        "entropy": entropy_mean,
                        "agreement": agreement,
                        "confidence": confidence,
                        "agreement_samples": sampled_predicted_letters,
                        "json_order": args.json_order,
                        "predicted_rationale_text": matching[1].rationale_text,
                        "stage1_raw": stage1_raw,
                        "stage2_raw": matching[0],
                        "predicted_json_parse_ok": matching[1].json_parse_ok,
                        "zeroshot_json": False,
                        "two_stage_cot": True,
                    }

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
        name_suffix=args.json_order,
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
