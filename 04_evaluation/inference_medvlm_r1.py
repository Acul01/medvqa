"""Off-the-shelf inference for MedVLM-R1 (Qwen2-VL-2B) on MC Medical VQA."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter

import torch
import wandb
from tqdm import tqdm
from transformers import AutoProcessor, GenerationConfig, Qwen2VLForConditionalGeneration

from utils.load_data import MedicalVQALoader
from utils.mc_eval import run_mc_evaluation
from utils.medvlm_r1 import (
    DEFAULT_MODEL_PATH,
    build_medvlm_conversation,
    build_medvlm_prompt,
    configure_processor_pixels,
    extract_answer_letter,
    load_qwen2vl_config,
    model_short_name,
    prepare_medvlm_inputs,
)
from utils.paths import get_data_root, get_mc_data_root, get_results_dir
from utils.save_results import save_results

DEFAULT_MAX_NEW_TOKENS = 1024
ATTN_FALLBACK_ORDER = ("eager", "sdpa", "flash_attention_2")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run off-the-shelf MedVLM-R1 inference on MC Medical VQA"
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_root", type=str, default=get_data_root())
    parser.add_argument("--mc_data_root", type=str, default=get_mc_data_root())
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["slake", "vqarad", "medvqa2019"],
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--agreement_samples",
        type=int,
        default=5,
        help="Number of stochastic generations used to compute agreement",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        help='Attention backend: "eager", "sdpa", or "flash_attention_2" (auto-fallback on failure)',
    )
    parser.add_argument(
        "--no_kv_cache",
        action="store_true",
        help="Disable KV cache during generation (slower; workaround for mask length bugs)",
    )
    parser.add_argument("--results_dir", type=str, default=get_results_dir())
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default="medvqa")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Skip Weights & Biases logging",
    )
    return parser.parse_args()


def _compute_agreement(letter_predictions: list[str | None]) -> float:
    valid = [letter for letter in letter_predictions if letter]
    if not valid:
        return 0.0
    counts = Counter(valid)
    return max(counts.values()) / len(valid)


def _safe_div(a: int, b: int) -> float:
    return (a / b) if b else 0.0


def _move_inputs_to_device(inputs: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def _decode_new_tokens(processor, inputs, generated_ids) -> str:
    input_length = inputs["input_ids"].shape[1]
    new_ids = generated_ids[0][input_length:]
    return processor.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def _resolve_pad_token_id(processor) -> int:
    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    return int(pad_id) if pad_id is not None else 151643


def _build_generation_configs(
    processor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[GenerationConfig, GenerationConfig]:
    pad_token_id = _resolve_pad_token_id(processor)
    greedy_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_token_id,
    )
    sample_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=pad_token_id,
    )
    return greedy_cfg, sample_cfg


def _load_model_and_processor(model_path: str, attn_implementation: str):
    candidates = [attn_implementation]
    for fallback in ATTN_FALLBACK_ORDER:
        if fallback not in candidates:
            candidates.append(fallback)

    last_exc: Exception | None = None
    model = None
    chosen_attn: str | None = None
    for attn in candidates:
        if attn == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("Skipping flash_attention_2: package flash_attn is not installed.")
                continue

        try:
            config = load_qwen2vl_config(model_path, attn_implementation=attn)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation=attn,
            )
            chosen_attn = attn
            if attn != attn_implementation:
                print(f"Loaded model with attn_implementation={attn!r} (requested {attn_implementation!r}).")
            break
        except Exception as exc:
            last_exc = exc
            print(f"Warning: load failed with attn_implementation={attn!r}: {exc}")

    if model is None or chosen_attn is None:
        raise RuntimeError(f"Could not load MedVLM-R1 with any attention backend: {last_exc}") from last_exc

    processor = AutoProcessor.from_pretrained(model_path)
    configure_processor_pixels(processor)
    return model, processor


def main():
    args = parse_args()
    model_short = model_short_name(args.model_path)
    run_name = args.wandb_run_name or f"{model_short}_{args.dataset}_pretrained_eval_mc"

    if not args.disable_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            group=args.dataset,
            tags=["inference", "mc", "medvlm-r1", model_short, args.dataset, "pretrained"],
            config=vars(args),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = _load_model_and_processor(args.model_path, args.attn_implementation)
    model.eval()

    greedy_cfg, sample_cfg = _build_generation_configs(
        processor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    use_cache = not args.no_kv_cache

    loader = MedicalVQALoader(data_root=args.data_root, mc_data_root=args.mc_data_root)
    samples = loader.load_dataset(args.dataset, split=args.split)
    if args.n_samples is not None:
        samples = samples[: args.n_samples]

    agreement_samples = max(1, int(args.agreement_samples))

    results = []
    with torch.no_grad():
        for sample in tqdm(samples):
            if not sample.mc_options or sample.mc_correct_index is None:
                continue

            num_options = len(sample.mc_options)
            prompt_text = build_medvlm_prompt(sample.question, sample.mc_options)
            conversation = build_medvlm_conversation(sample.image_path, prompt_text)
            inputs = prepare_medvlm_inputs(processor, conversation)
            inputs = _move_inputs_to_device(inputs, device)

            generated_ids = model.generate(
                **inputs,
                generation_config=greedy_cfg,
                use_cache=use_cache,
            )
            generated_text = _decode_new_tokens(processor, inputs, generated_ids)
            predicted_letter = extract_answer_letter(generated_text, num_options=num_options)
            ground_truth_letter = sample.answer

            sampled_predicted_letters: list[str | None] = []
            for _ in range(agreement_samples):
                sample_ids = model.generate(
                    **inputs,
                    generation_config=sample_cfg,
                    use_cache=use_cache,
                )
                sample_text = _decode_new_tokens(processor, inputs, sample_ids)
                sampled_predicted_letters.append(
                    extract_answer_letter(sample_text, num_options=num_options)
                )

            agreement = _compute_agreement(sampled_predicted_letters)
            is_correct = predicted_letter == ground_truth_letter
            qtype = sample.question_type or "unknown"

            results.append(
                {
                    "image_path": sample.image_path,
                    "source_image_path": sample.image_path,
                    "question": sample.question,
                    "options": sample.mc_options,
                    "correct_index": sample.mc_correct_index,
                    "question_type": qtype,
                    "ground_truth_answer": (sample.metadata or {}).get("gt_answer_text", ""),
                    "ground_truth_letter": ground_truth_letter,
                    "predicted_answer": generated_text,
                    "predicted_letter": predicted_letter,
                    "is_correct": is_correct,
                    "entropy": float("nan"),
                    "confidence": float("nan"),
                    "agreement": agreement,
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

    results_path = save_results(
        results=results,
        model_path=args.model_path,
        dataset=args.dataset,
        n_samples=args.n_samples,
        results_dir=args.results_dir,
    )
    run_mc_evaluation(results_path)

    if not args.disable_wandb:
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
                "agreement",
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
                row.get("agreement", math.nan),
            )
        wandb.log({"predictions": table})
        wandb.finish()


if __name__ == "__main__":
    main()
