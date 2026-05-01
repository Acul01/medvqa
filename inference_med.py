"""Inference script for LLaVA-Med on MC Medical VQA."""

import argparse
import json
import math
from pathlib import Path

import torch
import wandb
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model import LlavaMistralForCausalLM
from llava.model.builder import load_pretrained_model

from utils.load_data import MedicalVQALoader
from utils.mc_vqa import format_mc_user_text, parse_predicted_letter
from utils.save_results import clean_answer, save_results

DEFAULT_MODEL_PATH = "microsoft/llava-med-v1.5-mistral-7b"
DATASET_MAX_NEW_TOKENS = {"pathvqa": 8, "slake": 8, "vqarad": 8, "medvqa2019": 8}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with LLaVA-Med on MC Medical VQA")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--conv_template", type=str, default="mistral_instruct")
    parser.add_argument("--data_root", type=str, default="data", help="Root directory containing raw image folders")
    parser.add_argument("--mc_data_root", type=str, default="data_mc_openai", help="Root directory containing MC JSON/CSV exports")
    parser.add_argument("--dataset", type=str, required=True, choices=["pathvqa", "slake", "vqarad", "medvqa2019"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--finetuned", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="medvqa")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


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

        token_text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()
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

    model_path = Path(args.model_path)
    is_lora = model_path.exists() and (model_path / "adapter_config.json").exists()

    if is_lora:
        from peft import PeftModel

        base_path = args.model_base or DEFAULT_MODEL_PATH
        base_model = LlavaMistralForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16, device_map="auto")
        vision_tower = base_model.get_model().get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)

        model = PeftModel.from_pretrained(base_model, str(model_path))
        model = model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(base_path)
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    elif model_path.exists() and (model_path / "config.json").exists():
        model = LlavaMistralForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.float16, device_map="auto")
        vision_tower = model.get_model().get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)

        if (model_path / "tokenizer_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        else:
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH)
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    else:
        model_name = model_path.name or "llava-med-v1.5-mistral-7b"
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=args.model_path,
            model_base=args.model_base,
            model_name=model_name,
            device=str(device),
        )

    model.eval()

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

    results = []
    with torch.no_grad():
        for sample in tqdm(samples):
            if not sample.mc_options or sample.mc_correct_index is None:
                continue

            image = Image.open(sample.image_path).convert("RGB")
            image_tensor = process_images([image], image_processor, model.config).to(device, dtype=torch.float16)

            conv = conv_templates[args.conv_template].copy()
            user_text = format_mc_user_text(sample.question, sample.mc_options)
            question = f"{DEFAULT_IMAGE_TOKEN}\n{user_text}"
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            generation_output = model.generate(
                input_ids,
                images=image_tensor,
                attention_mask=torch.ones_like(input_ids),
                **gen_kwargs,
                return_dict_in_generate=True,
                output_scores=True,
            )

            generated_ids = generation_output.sequences[0]
            generated_text = clean_answer(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
            predicted_letter = parse_predicted_letter(generated_text, num_options=len(sample.mc_options))
            ground_truth_letter = sample.answer
            entropy_tokens, entropy_mean = _compute_token_entropies(generation_output.scores, generated_ids, tokenizer)

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

    table = wandb.Table(columns=["image_path", "question", "question_type", "ground_truth_letter", "predicted_letter", "is_correct", "options", "entropy"])
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
        )
    wandb.log({"predictions": table})
    wandb.finish()


if __name__ == "__main__":
    main()
