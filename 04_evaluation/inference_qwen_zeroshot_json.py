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
    DEFAULT_STAGE2_MAX_NEW_TOKENS,
    build_qwen_json_conversation,
    build_qwen_stage1_conversation,
    build_qwen_stage2_conversation,
    cast_json_single_shot,
    find_answer_token_step,
    get_answer_letter_token_ids,
    is_json_single_shot,
    parse_mc_answer_rationale_json,
)
from utils.mc_vqa import parse_predicted_letter
from utils.paths import get_data_root, get_mc_data_root, get_results_dir
from utils.qwen_vl import (
    DEFAULT_MODEL_PATH,
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
        choices=["answer_first", "rationale_first", "rationale_first_2stage"],
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


def _compute_first_token_entropy(model, inputs: dict) -> tuple[float, float, int]:
    fwd = model(**inputs, return_dict=True)
    first_logits = fwd.logits[0, -1, :].double()
    log_probs = torch.log_softmax(first_logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.where(probs > 0, probs * log_probs, torch.zeros_like(log_probs)).sum().item()
    confidence = float(probs.max().item())
    top_id = int(torch.argmax(first_logits).item())
    return entropy, confidence, top_id


def _compute_answer_token_entropy(
    generation_output,
    generated_ids,
    *,
    num_opts: int,
    answer_letter_token_ids: dict[int, str],
    predicted_letter: str | None,
    model,
    inputs: dict,
    tokenizer,
) -> tuple[float, float, dict, float]:
    valid_answer_ids = {
        tid: letter
        for tid, letter in answer_letter_token_ids.items()
        if letter in "ABCDEFGHIJ"[:num_opts]
    }
    answer_step = find_answer_token_step(generated_ids.tolist(), valid_answer_ids)
    if answer_step is not None and generation_output.scores:
        ans_logits = generation_output.scores[answer_step][0].double()
        ans_log_probs = torch.log_softmax(ans_logits, dim=-1)
        ans_probs = torch.exp(ans_log_probs)
        entropy_val = -torch.where(
            ans_probs > 0, ans_probs * ans_log_probs, torch.zeros_like(ans_log_probs)
        ).sum().item()
        confidence = float(ans_probs.max().item())
        entropy_tokens = {predicted_letter or "?": entropy_val}
        return entropy_val, confidence, entropy_tokens, entropy_val

    entropy_val, confidence, first_tok_id = _compute_first_token_entropy(model, inputs)
    first_tok_text = tokenizer.decode(
        [first_tok_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    ).strip() or f"tok_{first_tok_id}"
    entropy_tokens = {first_tok_text: entropy_val}
    return entropy_val, confidence, entropy_tokens, entropy_val


def _decode_new_tokens(processor, sequences, input_length: int) -> str:
    generated_ids = sequences[0][input_length:]
    return clean_answer(processor.decode(generated_ids, skip_special_tokens=True).strip())


def main():
    args = parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = (
            DEFAULT_STAGE2_MAX_NEW_TOKENS
            if args.json_order == "rationale_first_2stage"
            else DEFAULT_JSON_MAX_NEW_TOKENS
        )

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
    agreement_stage1_kwargs = {
        "max_new_tokens": args.rationale_max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    answer_letter_token_ids = get_answer_letter_token_ids(processor.tokenizer)

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

            num_opts = len(sample.mc_options)
            ground_truth_letter = sample.answer

            if is_json_single_shot(args.json_order):
                json_order = cast_json_single_shot(args.json_order)
                conversation = build_qwen_json_conversation(
                    sample.question,
                    sample.mc_options,
                    json_order=json_order,
                    image=image,
                )
                inputs = prepare_qwen_inputs(processor, conversation, add_generation_prompt=True)
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
                parsed = parse_mc_answer_rationale_json(generated_text, num_options=num_opts)
                predicted_letter = parsed.predicted_letter

                entropy_mean, confidence, entropy_tokens, _ = _compute_answer_token_entropy(
                    generation_output,
                    generated_ids,
                    num_opts=num_opts,
                    answer_letter_token_ids=answer_letter_token_ids,
                    predicted_letter=predicted_letter,
                    model=model,
                    inputs=inputs,
                    tokenizer=processor.tokenizer,
                )

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
                    sample_parsed = parse_mc_answer_rationale_json(sample_text, num_options=num_opts)
                    sampled_predicted_letters.append(sample_parsed.predicted_letter)
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
                    "predicted_answer": generated_text,
                    "predicted_letter": predicted_letter,
                    "is_correct": predicted_letter == ground_truth_letter,
                    "entropy_tokens": entropy_tokens,
                    "entropy": entropy_mean,
                    "agreement": agreement,
                    "confidence": confidence,
                    "agreement_samples": sampled_predicted_letters,
                    "json_order": args.json_order,
                    "predicted_rationale_text": parsed.rationale_text,
                    "predicted_json_parse_ok": parsed.json_parse_ok,
                    "zeroshot_json": True,
                }
            else:
                conv1 = build_qwen_stage1_conversation(
                    sample.question, sample.mc_options, image=image
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
                rationale_text = _decode_new_tokens(processor, out1.sequences, len1)

                conv2 = build_qwen_stage2_conversation(
                    sample.question,
                    sample.mc_options,
                    rationale=rationale_text,
                    image=image,
                )
                inputs2 = _move_inputs_to_device(
                    prepare_qwen_inputs(processor, conv2, add_generation_prompt=True), device
                )
                len2 = inputs2["input_ids"].shape[1]
                out2 = model.generate(
                    **inputs2,
                    **gen_kwargs,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                stage2_text = _decode_new_tokens(processor, out2.sequences, len2)
                predicted_letter = parse_predicted_letter(stage2_text, num_options=num_opts)

                entropy_val, confidence, first_tok_id = _compute_first_token_entropy(model, inputs2)
                first_tok_text = processor.tokenizer.decode(
                    [first_tok_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
                ).strip() or f"tok_{first_tok_id}"
                entropy_tokens = {first_tok_text: entropy_val}
                entropy_mean = entropy_val

                sampled_predicted_letters = []
                agreement_confidences = []
                for _ in range(agreement_samples):
                    agr_out1 = model.generate(
                        **inputs1,
                        **agreement_stage1_kwargs,
                        return_dict_in_generate=True,
                        output_scores=False,
                    )
                    agr_rationale = _decode_new_tokens(processor, agr_out1.sequences, len1)
                    agr_conv2 = build_qwen_stage2_conversation(
                        sample.question,
                        sample.mc_options,
                        rationale=agr_rationale,
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
                        output_scores=True,
                    )
                    agr_stage2 = _decode_new_tokens(processor, agr_out2.sequences, agr_len2)
                    sampled_predicted_letters.append(
                        parse_predicted_letter(agr_stage2, num_options=num_opts)
                    )
                    if agr_out2.scores:
                        agreement_confidences.append(
                            _compute_generation_confidence(
                                agr_out2.scores,
                                agr_out2.sequences[0][agr_len2:],
                                processor.tokenizer,
                            )
                        )

                agreement = _compute_agreement(sampled_predicted_letters)
                if agreement_confidences:
                    confidence = sum(agreement_confidences) / len(agreement_confidences)

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
                    "predicted_answer": stage2_text,
                    "predicted_letter": predicted_letter,
                    "is_correct": predicted_letter == ground_truth_letter,
                    "entropy_tokens": entropy_tokens,
                    "entropy": entropy_mean,
                    "agreement": agreement,
                    "confidence": confidence,
                    "agreement_samples": sampled_predicted_letters,
                    "json_order": args.json_order,
                    "predicted_rationale_text": rationale_text,
                    "stage1_raw": rationale_text,
                    "predicted_json_parse_ok": False,
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
