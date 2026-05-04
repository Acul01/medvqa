#!/usr/bin/env python3
"""
Standalone layer comparison for LLaVA-Med GradCAM (4-bit load for a quick test).

Run from repo root::

    python gradcam/validate_layers.py --image path/to.jpg --question "..." --output gradcam_layer_comparison.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Repo root on sys.path when executed as ``python gradcam/validate_layers.py``
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import torch
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "microsoft/llava-med-v1.5-mistral-7b"


def parse_args():
    p = argparse.ArgumentParser(description="GradCAM ViT layer comparison (LLaVA-Med)")
    p.add_argument("--image", type=str, required=True, help="Path to a sample image (e.g. VQA-Rad)")
    p.add_argument("--question", type=str, default="Is there cardiomegaly?")
    p.add_argument("--output", type=str, default="gradcam_layer_comparison.png")
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--model_base", type=str, default=None)
    p.add_argument("--conv_template", type=str, default="mistral_instruct")
    p.add_argument("--rationale_max_new_tokens", type=int, default=128)
    p.add_argument("--no_4bit", action="store_true", help="Load full fp16 instead of 4-bit (slower, more VRAM)")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX as IMG_IDX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    from gradcam.gradcam_llava import LLaVAGradCAM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA not available; 4-bit load and GradCAM may fail or be very slow.")

    model_name = Path(args.model_path).name or "llava-med-v1.5-mistral-7b"
    if "llava" not in model_name.lower():
        model_name = "llava-med-v1.5-mistral-7b"

    load_4bit = not args.no_4bit
    try:
        tokenizer, model, image_processor, _ctx = load_pretrained_model(
            model_path=args.model_path,
            model_base=args.model_base,
            model_name=model_name,
            device_map="auto",
            load_4bit=load_4bit,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("4-bit load failed (%s); retrying fp16.", e)
        tokenizer, model, image_processor, _ctx = load_pretrained_model(
            model_path=args.model_path,
            model_base=args.model_base,
            model_name=model_name,
            device_map="auto",
            load_4bit=False,
        )

    model.eval()

    image = Image.open(args.image).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device=device, dtype=model.dtype)

    rationale_prompt = f"Explain your reasoning for answering: {args.question}"
    conv = conv_templates[args.conv_template].copy()
    question = f"{DEFAULT_IMAGE_TOKEN}\n{rationale_prompt}"
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids_prompt = tokenizer_image_token(
        prompt, tokenizer, IMG_IDX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        gen = model.generate(
            input_ids_prompt,
            images=image_tensor,
            attention_mask=torch.ones_like(input_ids_prompt),
            max_new_tokens=args.rationale_max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )

    full_ids = gen.sequences
    prompt_len = input_ids_prompt.shape[1]
    output_indices = list(range(prompt_len, int(full_ids.shape[1])))
    attention_mask = torch.ones(full_ids.shape[0], full_ids.shape[1], device=device, dtype=torch.long)

    _probe = LLaVAGradCAM(model, target_layers=[-1])
    n_layers = len(_probe.encoder_layers)
    layer_negs = [-1, -2, -3, -4]
    heatmaps = []
    titles = []

    for neg in layer_negs:
        idx = n_layers + neg
        titles.append(f"layer {idx}")
        try:
            cam = LLaVAGradCAM(model, target_layers=[neg])
            hm = cam.compute_gradcam(
                image_tensor=image_tensor,
                input_ids=full_ids,
                attention_mask=attention_mask,
                target_token_indices=output_indices,
                image_token_index=IMG_IDX,
                output_spatial_size=(image.height, image.width),
            )
            heatmaps.append(hm.numpy())
        except Exception as e:  # noqa: BLE001
            logger.warning("GradCAM failed for layer %s: %s", neg, e)
            heatmaps.append(torch.zeros(image.height, image.width).numpy())

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(image)
    axes[0].set_title("original")
    axes[0].axis("off")
    for ax, hm, title in zip(axes[1:], heatmaps, titles):
        ax.imshow(hm, cmap="jet")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
