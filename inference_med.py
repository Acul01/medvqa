"""Inference script for LLaVA-Med on MC Medical VQA."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

from gradcam.gradcam_llava import LLaVAGradCAM
from gradcam.overlay import create_gradcam_overlay
from utils.load_data import MedicalVQALoader
from utils.mc_vqa import LETTER_POOL, format_mc_user_text, parse_predicted_letter
from utils.save_results import clean_answer, save_results

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "microsoft/llava-med-v1.5-mistral-7b"
DATASET_MAX_NEW_TOKENS = {"pathvqa": 8, "slake": 8, "vqarad": 8, "medvqa2019": 8}
END_REASONING_TOKEN = "<END_REASONING>"


def parse_mc_options_cli(options_line: str) -> List[str]:
    """Parse ``'A) Yes B) No'``-style option text into plain option strings."""
    s = options_line.strip()
    chunks = re.split(r"\s+(?=[A-Z]\))", s)
    out: List[str] = []
    for c in chunks:
        t = re.sub(r"^[A-Z]\)\s*", "", c).strip()
        if t:
            out.append(t)
    if len(out) < 2:
        raise ValueError(f"Could not parse at least two MC options from: {options_line!r}")
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with LLaVA-Med on MC Medical VQA")
    parser.add_argument("--mode", type=str, default="eval", choices=["eval", "path2"], help="eval: dataset MC; path2: rationale + GradCAM + MC")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--conv_template", type=str, default="mistral_instruct")
    parser.add_argument("--data_root", type=str, default="data", help="Root directory containing raw image folders")
    parser.add_argument("--mc_data_root", type=str, default="data_mc_openai", help="Root directory containing MC JSON/CSV exports")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["pathvqa", "slake", "vqarad", "medvqa2019"],
        help="Required when mode=eval",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--agreement_samples", type=int, default=5, help="Number of stochastic generations used to compute agreement")
    parser.add_argument("--use_two_stage_reasoning", action="store_true", default=False, help="Run 2-stage reasoning before final MC answer")
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
    # path2-only
    parser.add_argument("--image", type=str, default=None, help="path2: path to input image")
    parser.add_argument("--question", type=str, default=None, help="path2: clinical question")
    parser.add_argument("--options", type=str, default=None, help='path2: e.g. "A) Yes B) No"')
    parser.add_argument("--path2_output", type=str, default=None, help="path2: optional JSON path for results + sidecar heatmap/overlay")
    parser.add_argument("--rationale_max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    if args.mode == "eval" and args.dataset is None:
        parser.error("--dataset is required when mode=eval")
    if args.mode == "path2":
        if not args.image or not args.question or not args.options:
            parser.error("path2 requires --image, --question, and --options")
    return args


def _model_short_name(model_path: str) -> str:
    name = Path(model_path).name.lower()
    if "llava-med" in name:
        return "llava-med"
    if "llava" in name:
        return "llava"
    return Path(model_path).name


def _format_mc_options_block(options: List[str]) -> str:
    lines = [f"{LETTER_POOL[idx]}) {str(opt).strip()}" for idx, opt in enumerate(options)]
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


def build_final_prompt(question: str, options: List[str], region: str, observation: str, reasoning: str) -> str:
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


def parse_reasoning_fields(text: str) -> Tuple[Dict[str, str], str]:
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


def load_llava_med_bundle(args) -> Tuple[torch.device, Any, Any, Any]:
    """Load tokenizer, model, image_processor for inference (same logic as legacy main)."""
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
    return device, model, tokenizer, image_processor


def process_image(
    image: Image.Image,
    image_processor: CLIPImageProcessor,
    model_config: Any,
    device: torch.device,
    model: torch.nn.Module,
) -> torch.Tensor:
    """Preprocess PIL image to model input tensor (B, C, H, W) in ``model.dtype``."""
    t = process_images([image.convert("RGB")], image_processor, model_config)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    dtype = getattr(model, "dtype", torch.float16)
    return t.to(device=device, dtype=dtype)


def generate_with_token_ids(
    image: Image.Image,
    rationale_prompt: str,
    model: torch.nn.Module,
    tokenizer: Any,
    image_processor: CLIPImageProcessor,
    device: torch.device,
    model_config: Any,
    conv_template: str,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """Generate rationale; return full token ids and compact indices for GradCAM."""
    image_tensor = process_image(image, image_processor, model_config, device, model)

    conv = conv_templates[conv_template].copy()
    question = f"{DEFAULT_IMAGE_TOKEN}\n{rationale_prompt}"
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    with torch.inference_mode():
        gen = model.generate(
            input_ids,
            images=image_tensor,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )

    full_ids = gen.sequences
    prompt_len = int(input_ids.shape[1])
    rationale_text = clean_answer(tokenizer.decode(full_ids[0], skip_special_tokens=True).strip())
    output_indices = list(range(prompt_len, int(full_ids.shape[1])))
    attention_mask = torch.ones(full_ids.shape[0], full_ids.shape[1], device=device, dtype=torch.long)

    return {
        "rationale": rationale_text,
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "output_indices": output_indices,
        "image_tensor": image_tensor,
    }


def generate_mc_answer(
    overlay_image: Image.Image,
    final_prompt: str,
    model: torch.nn.Module,
    image_processor: CLIPImageProcessor,
    tokenizer: Any,
    device: torch.device,
    model_config: Any,
    conv_template: str,
    num_options: int,
    max_new_tokens: int = 8,
) -> Tuple[str, Dict[str, float]]:
    """Greedy MC generation on (possibly overlaid) image; probs from first generated-token softmax over A.."""
    image_tensor = process_image(overlay_image, image_processor, model_config, device, model)

    conv = conv_templates[conv_template].copy()
    question = f"{DEFAULT_IMAGE_TOKEN}\n{final_prompt}"
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    with torch.inference_mode():
        gen = model.generate(
            input_ids,
            images=image_tensor,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    seq = gen.sequences[0]
    text = clean_answer(tokenizer.decode(seq, skip_special_tokens=True).strip())
    letter = parse_predicted_letter(text, num_options=num_options)

    probs: Dict[str, float] = {}
    if gen.scores and len(gen.scores) > 0:
        first_logits = gen.scores[0][0]
        dist = torch.softmax(first_logits.float(), dim=-1)

        for i in range(num_options):
            ch = LETTER_POOL[i]
            ids = tokenizer.encode(ch, add_special_tokens=False)
            tid = ids[0] if ids else None
            if tid is not None and tid < dist.shape[-1]:
                probs[ch] = float(dist[tid].item())
            else:
                probs[ch] = float("nan")
    else:
        for i in range(num_options):
            probs[LETTER_POOL[i]] = float("nan")

    return text, probs


def run_path2(
    image: Image.Image,
    question: str,
    mc_options: List[str],
    model: torch.nn.Module,
    tokenizer: Any,
    image_processor: CLIPImageProcessor,
    device: torch.device,
    conv_template: str = "mistral_instruct",
    rationale_max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """Rationale → GradCAM (last 4 ViT layers, averaged) → overlay → final MC answer."""
    rationale_prompt = f"Explain your reasoning for answering: {question}"
    tok = generate_with_token_ids(
        image,
        rationale_prompt,
        model,
        tokenizer,
        image_processor,
        device,
        model.config,
        conv_template,
        max_new_tokens=rationale_max_new_tokens,
    )
    rationale = tok["rationale"]

    heatmap: Optional[torch.Tensor] = None
    overlay_pil: Image.Image = image.convert("RGB")
    try:
        gradcam = LLaVAGradCAM(model, target_layers=[-1, -2, -3, -4])
        heatmap = gradcam.compute_gradcam(
            image_tensor=tok["image_tensor"],
            input_ids=tok["input_ids"],
            attention_mask=tok["attention_mask"],
            target_token_indices=tok["output_indices"],
            image_token_index=IMAGE_TOKEN_INDEX,
            output_spatial_size=(image.height, image.width),
        )
        _, overlay_pil = create_gradcam_overlay(image, heatmap, alpha=0.4)
    except Exception as e:  # noqa: BLE001
        logger.warning("GradCAM failed (%s); using original image without overlay.", e)
        heatmap = None
        _, overlay_pil = create_gradcam_overlay(image, None, alpha=0.4)

    mc_block = format_mc_user_text(question, mc_options)
    final_prompt = (
        f"The relevant region is highlighted in the image. "
        f"Your reasoning was: {rationale}. "
        f"Based on this, answer the following.\n\n{mc_block}"
    )
    answer, probs = generate_mc_answer(
        overlay_pil,
        final_prompt,
        model,
        image_processor,
        tokenizer,
        device,
        model.config,
        conv_template,
        num_options=len(mc_options),
    )
    letter = parse_predicted_letter(answer, num_options=len(mc_options))

    return {
        "rationale": rationale,
        "heatmap": heatmap,
        "overlay_image": overlay_pil,
        "answer": answer,
        "predicted_letter": letter,
        "probabilities": probs,
    }


def _run_path2_cli(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device, model, tokenizer, image_processor = load_llava_med_bundle(args)
    image = Image.open(args.image).convert("RGB")
    mc_options = parse_mc_options_cli(args.options)
    out = run_path2(
        image,
        args.question,
        mc_options,
        model,
        tokenizer,
        image_processor,
        device,
        conv_template=args.conv_template,
        rationale_max_new_tokens=args.rationale_max_new_tokens,
    )

    serial: Dict[str, Any] = {
        "rationale": out["rationale"],
        "answer": out["answer"],
        "predicted_letter": out["predicted_letter"],
        "probabilities": out["probabilities"],
    }
    if args.path2_output:
        base = Path(args.path2_output)
        base.parent.mkdir(parents=True, exist_ok=True)
        if out["heatmap"] is not None:
            np.save(str(base.with_name(base.stem + "_heatmap.npy")), out["heatmap"].numpy())
            serial["heatmap_npy"] = str(base.with_name(base.stem + "_heatmap.npy"))
        out["overlay_image"].save(str(base.with_name(base.stem + "_overlay.png")))
        serial["overlay_png"] = str(base.with_name(base.stem + "_overlay.png"))
        with open(base, "w", encoding="utf-8") as f:
            json.dump(serial, f, indent=2, ensure_ascii=False)
        logger.info("Wrote %s", base)
    print(json.dumps(serial, indent=2, ensure_ascii=False))


def main():
    args = parse_args()

    if args.mode == "path2":
        _run_path2_cli(args)
        return

    if args.max_new_tokens is None:
        args.max_new_tokens = DATASET_MAX_NEW_TOKENS[args.dataset]

    model_short = _model_short_name(args.model_path)
    ft_tag = "finetuned" if args.finetuned else "pretrained"
    run_name = args.wandb_run_name or f"{model_short}_{args.dataset}_{ft_tag}_eval_mc"
    wandb.init(project=args.wandb_project, name=run_name, group=args.dataset, tags=["inference", "mc", model_short, args.dataset, ft_tag], config=vars(args))

    device, model, tokenizer, image_processor = load_llava_med_bundle(args)

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
            image_tensor = process_images([image], image_processor, model.config).to(device, dtype=torch.float16)

            conv = conv_templates[args.conv_template].copy()
            reasoning_raw = ""
            reasoning_region = "unknown"
            reasoning_observation = "unknown"
            reasoning_explanation = "unknown"
            reasoning_parse_mode = "disabled"

            if args.use_two_stage_reasoning:
                reasoning_prompt = build_reasoning_prompt(sample.question)
                reasoning_question = f"{DEFAULT_IMAGE_TOKEN}\n{reasoning_prompt}"
                reasoning_conv = conv_templates[args.conv_template].copy()
                reasoning_conv.append_message(reasoning_conv.roles[0], reasoning_question)
                reasoning_conv.append_message(reasoning_conv.roles[1], None)
                reasoning_full_prompt = reasoning_conv.get_prompt()
                reasoning_input_ids = tokenizer_image_token(
                    reasoning_full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).to(device)
                reasoning_output = model.generate(
                    reasoning_input_ids,
                    images=image_tensor,
                    attention_mask=torch.ones_like(reasoning_input_ids),
                    max_new_tokens=args.reasoning_max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
                reasoning_generated_ids = reasoning_output.sequences[0]
                reasoning_raw = clean_answer(tokenizer.decode(reasoning_generated_ids, skip_special_tokens=True).strip())
                parsed_reasoning, reasoning_parse_mode = parse_reasoning_fields(reasoning_raw)
                reasoning_region = parsed_reasoning["region"]
                reasoning_observation = parsed_reasoning["observation"]
                reasoning_explanation = parsed_reasoning["reasoning"]
                final_user_text = build_final_prompt(
                    sample.question,
                    sample.mc_options,
                    reasoning_region,
                    reasoning_observation,
                    reasoning_explanation,
                )
            else:
                final_user_text = format_mc_user_text(sample.question, sample.mc_options)

            question = f"{DEFAULT_IMAGE_TOKEN}\n{final_user_text}"
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

            sampled_predicted_letters = []
            sampled_entropies = []
            sampled_confidences = []
            for _ in range(agreement_samples):
                sample_output = model.generate(
                    input_ids,
                    images=image_tensor,
                    attention_mask=torch.ones_like(input_ids),
                    **agreement_gen_kwargs,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                sample_generated_ids = sample_output.sequences[0]
                sample_text = clean_answer(tokenizer.decode(sample_generated_ids, skip_special_tokens=True).strip())
                sample_letter = parse_predicted_letter(sample_text, num_options=len(sample.mc_options))
                sampled_predicted_letters.append(sample_letter)
                _, sample_entropy = _compute_token_entropies(sample_output.scores, sample_generated_ids, tokenizer)
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
                    "reasoning_raw": reasoning_raw,
                    "reasoning_region": reasoning_region,
                    "reasoning_observation": reasoning_observation,
                    "reasoning_explanation": reasoning_explanation,
                    "reasoning_parse_mode": reasoning_parse_mode,
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
