"""Zero-shot inference for LLaVA-Med with JSON answer+rationale output."""

from __future__ import annotations

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
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

from utils.load_data import MedicalVQALoader
from utils.mc_eval import add_aurc_cli_arguments, apply_aurc_to_wandb_summary, resolve_eval_runs_csv, run_mc_evaluation
from utils.mc_json_zeroshot import (
    DEFAULT_JSON_MAX_NEW_TOKENS,
    DEFAULT_RATIONALE_MAX_NEW_TOKENS,
    build_mc_json_user_text,
    build_stage1_rationale_prompt,
    build_stage2_answer_prompt,
    build_stage2_json_prompt,
    cast_json_single_shot,
    find_answer_token_step,
    get_answer_letter_token_ids,
    is_json_single_shot,
    normalize_generation_sequences,
    parse_mc_answer_rationale_json,
)
from utils.mc_vqa import parse_predicted_letter
from utils.paths import get_data_root, get_mc_data_root, get_results_dir
from utils.inference_resume import (
    add_inference_resume_cli_arguments,
    sample_inference_key,
    setup_inference_checkpoint,
)
from utils.save_results import clean_answer, save_results

DEFAULT_MODEL_PATH = "microsoft/llava-med-v1.5-mistral-7b"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot LLaVA-Med inference with JSON answer+rationale output"
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype. bfloat16 halves memory vs float32 (default: bfloat16).",
    )
    parser.add_argument("--conv_template", type=str, default="mistral_instruct")
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
    add_aurc_cli_arguments(parser)
    add_inference_resume_cli_arguments(parser)
    return parser.parse_args()


def _model_short_name(model_path: str) -> str:
    name = Path(model_path).name.lower()
    if "llava-med" in name:
        return "llava-med"
    if "llava" in name:
        return "llava"
    return Path(model_path).name


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


def _mean_finite(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


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


def _compute_first_token_entropy(
    model,
    input_ids: torch.Tensor,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, int]:
    fwd = model(
        input_ids,
        images=image_tensor,
        attention_mask=torch.ones_like(input_ids),
        return_dict=True,
    )
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
    input_ids: torch.Tensor,
    image_tensor: torch.Tensor,
    device: torch.device,
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

    entropy_val, confidence, first_tok_id = _compute_first_token_entropy(
        model, input_ids, image_tensor, device
    )
    first_tok_text = tokenizer.decode(
        [first_tok_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    ).strip() or f"tok_{first_tok_id}"
    entropy_tokens = {first_tok_text: entropy_val}
    return entropy_val, confidence, entropy_tokens, entropy_val


def _prepare_llava_med_inputs(
    user_text: str,
    image: Image.Image,
    *,
    conv_template: str,
    tokenizer,
    image_processor,
    model,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    conv = conv_templates[conv_template].copy()
    question = f"{DEFAULT_IMAGE_TOKEN}\n{user_text}"
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    image_tensor = process_images([image], image_processor, model.config).to(
        device, dtype=next(model.parameters()).dtype
    )
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    return input_ids, image_tensor, input_ids.shape[1]


def _decode_llava_med_new(tokenizer, sequences: torch.Tensor, input_ids: torch.Tensor) -> str:
    _, generated_ids, _ = normalize_generation_sequences(sequences, input_ids)
    return clean_answer(tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip())


def _generated_suffix(sequences: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Return generated token ids only (handles HF variants with/without prompt prefix)."""
    _, generated_ids, _ = normalize_generation_sequences(sequences, input_ids)
    return generated_ids[0]


def load_llava_med_pretrained(model_path: str, device: torch.device, torch_dtype: torch.dtype):
    """Load pretrained LLaVA-Med from HuggingFace (zero-shot, no LoRA)."""
    model_name = Path(model_path).name or "llava-med-v1.5-mistral-7b"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device=str(device),
    )
    model = model.to(device=device, dtype=torch_dtype)
    vision_tower = model.get_model().get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch_dtype)
    model.eval()
    return tokenizer, model, image_processor


def main():
    args = parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = DEFAULT_JSON_MAX_NEW_TOKENS

    model_short = _model_short_name(args.model_path)
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
            "llava-med",
            "zeroshot",
            "json",
            model_short,
            args.dataset,
            args.json_order,
        ],
        config=vars(args),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    load_dtype = dtype_map[args.dtype]
    tokenizer, model, image_processor = load_llava_med_pretrained(
        args.model_path, device, load_dtype
    )

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

    answer_letter_token_ids = get_answer_letter_token_ids(tokenizer)

    ckpt_manager, _ = setup_inference_checkpoint(
        args,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        partial_path=args.partial_path,
        name_suffix=args.json_order,
    )

    with torch.no_grad():
        for sample_idx, sample in enumerate(tqdm(samples)):
            if not sample.mc_options or sample.mc_correct_index is None:
                continue

            source_image_path = sample.image_path
            if ckpt_manager.try_skip(sample_inference_key(source_image_path, sample.question)):
                continue

            effective_image_path = shuffle_assignment.get(sample_idx, source_image_path)
            image = Image.open(effective_image_path).convert("RGB")
            if args.ablation_mode == "black":
                image = Image.new("RGB", image.size, color=(0, 0, 0))

            num_opts = len(sample.mc_options)
            ground_truth_letter = sample.answer

            if is_json_single_shot(args.json_order):
                json_order = cast_json_single_shot(args.json_order)
                user_text = build_mc_json_user_text(
                    sample.question,
                    sample.mc_options,
                    json_order=json_order,
                )
                input_ids, image_tensor, _ = _prepare_llava_med_inputs(
                    user_text,
                    image,
                    conv_template=args.conv_template,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    model=model,
                    device=device,
                )

                sample_results = []
                for _ in range(agreement_samples):
                    sample_output = model.generate(
                        input_ids,
                        images=image_tensor,
                        attention_mask=torch.ones_like(input_ids),
                        **agreement_gen_kwargs,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    sample_generated_ids = _generated_suffix(sample_output.sequences, input_ids)
                    sample_text = clean_answer(
                        tokenizer.decode(sample_generated_ids, skip_special_tokens=True).strip()
                    )
                    sample_parsed = parse_mc_answer_rationale_json(sample_text, num_options=num_opts)
                    ent, conf, _, _ = _compute_answer_token_entropy(
                        sample_output,
                        sample_generated_ids,
                        num_opts=num_opts,
                        answer_letter_token_ids=answer_letter_token_ids,
                        predicted_letter=sample_parsed.predicted_letter,
                        model=model,
                        input_ids=input_ids,
                        image_tensor=image_tensor,
                        device=device,
                        tokenizer=tokenizer,
                    )
                    sample_results.append((sample_text, sample_parsed, ent, conf))

                sampled_predicted_letters = [r[1].predicted_letter for r in sample_results]
                predicted_letter = _majority_letter(sampled_predicted_letters)
                matching = next(
                    (r for r in reversed(sample_results) if r[1].predicted_letter == predicted_letter),
                    sample_results[-1],
                )
                agreement = _compute_agreement(sampled_predicted_letters)
                majority_results = [r for r in sample_results if r[1].predicted_letter == predicted_letter]
                entropy_mean = _mean_finite([r[2] for r in majority_results])
                confidence = _mean_finite([r[3] for r in majority_results])
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
                stage1_text = build_stage1_rationale_prompt(sample.question, sample.mc_options)
                input_ids1, image_tensor1, _ = _prepare_llava_med_inputs(
                    stage1_text,
                    image,
                    conv_template=args.conv_template,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    model=model,
                    device=device,
                )
                out1 = model.generate(
                    input_ids1,
                    images=image_tensor1,
                    attention_mask=torch.ones_like(input_ids1),
                    **stage1_gen_kwargs,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
                stage1_raw = _decode_llava_med_new(tokenizer, out1.sequences, input_ids1)

                if args.json_order == "rationale_first_2stage_norat":
                    # Stage 2: letter only (N samples)
                    sample_results = []
                    for _ in range(agreement_samples):
                        agr_stage2_prompt = build_stage2_answer_prompt(
                            sample.question, sample.mc_options, stage1_raw
                        )
                        agr_ids2, agr_img2, _ = _prepare_llava_med_inputs(
                            agr_stage2_prompt,
                            image,
                            conv_template=args.conv_template,
                            tokenizer=tokenizer,
                            image_processor=image_processor,
                            model=model,
                            device=device,
                        )
                        agr_out2 = model.generate(
                            agr_ids2,
                            images=agr_img2,
                            attention_mask=torch.ones_like(agr_ids2),
                            **agreement_gen_kwargs,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                        agr_gen_ids = _generated_suffix(agr_out2.sequences, agr_ids2)
                        agr_stage2 = clean_answer(
                            tokenizer.decode(agr_gen_ids, skip_special_tokens=True).strip()
                        )
                        agr_letter = parse_predicted_letter(agr_stage2, num_options=num_opts)
                        ent, conf, _, _ = _compute_answer_token_entropy(
                            agr_out2,
                            agr_gen_ids,
                            num_opts=num_opts,
                            answer_letter_token_ids=answer_letter_token_ids,
                            predicted_letter=agr_letter,
                            model=model,
                            input_ids=agr_ids2,
                            image_tensor=agr_img2,
                            device=device,
                            tokenizer=tokenizer,
                        )
                        sample_results.append((agr_stage2, agr_letter, ent, conf))

                    sampled_predicted_letters = [r[1] for r in sample_results]
                    predicted_letter = _majority_letter(sampled_predicted_letters)
                    matching = next(
                        (r for r in reversed(sample_results) if r[1] == predicted_letter),
                        sample_results[-1],
                    )
                    agreement = _compute_agreement(sampled_predicted_letters)
                    majority_results = [r for r in sample_results if r[1] == predicted_letter]
                    entropy_mean = _mean_finite([r[2] for r in majority_results])
                    confidence = _mean_finite([r[3] for r in majority_results])
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
                        agr_stage2_prompt = build_stage2_json_prompt(
                            sample.question, sample.mc_options, stage1_raw
                        )
                        agr_ids2, agr_img2, _ = _prepare_llava_med_inputs(
                            agr_stage2_prompt,
                            image,
                            conv_template=args.conv_template,
                            tokenizer=tokenizer,
                            image_processor=image_processor,
                            model=model,
                            device=device,
                        )
                        agr_out2 = model.generate(
                            agr_ids2,
                            images=agr_img2,
                            attention_mask=torch.ones_like(agr_ids2),
                            **agreement_gen_kwargs,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                        agr_gen_ids = _generated_suffix(agr_out2.sequences, agr_ids2)
                        agr_stage2 = clean_answer(
                            tokenizer.decode(agr_gen_ids, skip_special_tokens=True).strip()
                        )
                        agr_parsed2 = parse_mc_answer_rationale_json(agr_stage2, num_options=num_opts)
                        ent, conf, _, _ = _compute_answer_token_entropy(
                            agr_out2,
                            agr_gen_ids,
                            num_opts=num_opts,
                            answer_letter_token_ids=answer_letter_token_ids,
                            predicted_letter=agr_parsed2.predicted_letter,
                            model=model,
                            input_ids=agr_ids2,
                            image_tensor=agr_img2,
                            device=device,
                            tokenizer=tokenizer,
                        )
                        sample_results.append((agr_stage2, agr_parsed2, ent, conf))

                    sampled_predicted_letters = [r[1].predicted_letter for r in sample_results]
                    predicted_letter = _majority_letter(sampled_predicted_letters)
                    matching = next(
                        (r for r in reversed(sample_results) if r[1].predicted_letter == predicted_letter),
                        sample_results[-1],
                    )
                    agreement = _compute_agreement(sampled_predicted_letters)
                    majority_results = [r for r in sample_results if r[1].predicted_letter == predicted_letter]
                    entropy_mean = _mean_finite([r[2] for r in majority_results])
                    confidence = _mean_finite([r[3] for r in majority_results])
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

            ckpt_manager.append_new(row_out)

    ckpt_manager.print_summary()
    results = ckpt_manager.results
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

    ckpt_manager.finalize(complete=True)
    results_path = save_results(
        results=results,
        model_path=args.model_path,
        dataset=args.dataset,
        n_samples=args.n_samples,
        results_dir=args.results_dir,
        name_suffix=args.json_order,
    )
    metrics = run_mc_evaluation(
        results_path,
        compute_aurc=not args.skip_aurc,
        combined_alpha=args.combined_alpha,
        eval_runs_csv=resolve_eval_runs_csv(args),
    )

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
