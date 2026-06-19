"""SHAP-based visual explanation for LLaVA and Qwen MC-VQA inference.

Supports json_order modes: answer_only, answer_first, rationale_first, rationale_first_2stage.
For finetuned Qwen checkpoints with a spatial filter layer, pass --spatial_filter to the
model directory that contains spatial_filter.pt; the wrapper is loaded automatically.

Usage (answer-only, compare two finetuned Qwen checkpoints):
    python 04_evaluation/shap_visualize.py \
        --model qwen \
        --model_path path/to/finetuned_qwen \
        --dataset slake \
        --json_order answer_only \
        --n_samples 5 \
        --max_evals 300 \
        --output_dir shap_outputs/

    python 04_evaluation/shap_visualize.py \
        --model qwen \
        --model_path path/to/finetuned_qwen_spatial \
        --dataset slake \
        --json_order answer_only \
        --n_samples 5 \
        --max_evals 300 \
        --output_dir shap_outputs/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import shap
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.load_data import MedicalVQALoader
from utils.mc_json_zeroshot import (
    build_mc_json_user_text,
    build_qwen_json_conversation,
    build_qwen_stage1_conversation,
    build_qwen_stage2_conversation,
    build_stage1_rationale_prompt,
    build_stage2_answer_prompt,
    cast_json_single_shot,
    find_answer_token_step,
    get_answer_letter_token_ids,
    is_json_single_shot,
    parse_mc_answer_rationale_json,
)
from utils.mc_vqa import parse_predicted_letter
from utils.paths import get_data_root, get_mc_data_root
from utils.save_results import clean_answer

LLAVA_DEFAULT = "llava-hf/llava-1.5-7b-hf"
QWEN_DEFAULT = "Qwen/Qwen2.5-VL-7B-Instruct"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_llava(model_path: str, device: torch.device):
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="auto"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    if getattr(processor, "patch_size", None) is None:
        processor.patch_size = 14
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = "default"
    return model, processor


def load_qwen(model_path: str, device: torch.device):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from utils.qwen_vl import configure_qwen_processor_pixels

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="auto"
    )

    # Auto-load spatial filter if checkpoint is present (mirrors inference_qwen.py logic)
    spatial_filter_ckpt = Path(model_path) / "spatial_filter.pt"
    if spatial_filter_ckpt.exists():
        from utils.spatial_filter import load_spatial_filter_checkpoint
        model = load_spatial_filter_checkpoint(model, str(spatial_filter_ckpt), bottleneck=64)
        print(f"Spatial filter applied ({spatial_filter_ckpt})")
    else:
        print("No spatial_filter.pt found — running without spatial filter.")

    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    configure_qwen_processor_pixels(processor)
    return model, processor


# ---------------------------------------------------------------------------
# Score functions: return P(answer_letter | masked_image)
# These are called by SHAP hundreds of times, each time with a PIL image.
# ---------------------------------------------------------------------------

def _answer_token_prob(model, inputs: dict, answer_token_ids: dict) -> float:
    """Single forward pass → max P over valid answer tokens."""
    with torch.no_grad():
        logits = model(**inputs, return_dict=True).logits[0, -1, :].double()
    probs = torch.softmax(logits, dim=-1)
    valid_ids = list(answer_token_ids.keys())
    return float(probs[valid_ids].max().item())


def make_llava_scorer(model, processor, question: str, options: list[str],
                      json_order: str, device: torch.device):
    """Returns f(np_images) → scores for SHAP (answer_first / rationale_first)."""
    from transformers import LlavaForConditionalGeneration

    answer_token_ids = get_answer_letter_token_ids(processor.tokenizer)
    num_opts = len(options)
    valid_answer_ids = {
        tid: letter
        for tid, letter in answer_token_ids.items()
        if letter in "ABCDEFGHIJ"[:num_opts]
    }

    if is_json_single_shot(json_order):
        user_text = build_mc_json_user_text(
            question, options, json_order=cast_json_single_shot(json_order)
        )

        def score_fn(images_np: np.ndarray) -> np.ndarray:
            scores = []
            for img_np in images_np:
                img = Image.fromarray(img_np.astype(np.uint8))
                from inference_llava_zeroshot_json import build_conversation
                conv = build_conversation(user_text)
                prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
                inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
                scores.append(_answer_token_prob(model, inputs, valid_answer_ids))
            return np.array(scores)

    else:
        # rationale_first_2stage: generate rationale once on the original image,
        # then score the answer stage with masked images.
        raise NotImplementedError(
            "For rationale_first_2stage pass --precomputed_rationale or use "
            "make_llava_twostage_scorer()."
        )

    return score_fn


def make_llava_twostage_scorer(model, processor, question: str, options: list[str],
                                rationale: str, device: torch.device):
    """Stage-2 scorer: rationale is fixed, image is masked."""
    answer_token_ids = get_answer_letter_token_ids(processor.tokenizer)
    num_opts = len(options)
    valid_answer_ids = {
        tid: letter for tid, letter in answer_token_ids.items()
        if letter in "ABCDEFGHIJ"[:num_opts]
    }
    stage2_text = build_stage2_answer_prompt(question, options, rationale)

    def score_fn(images_np: np.ndarray) -> np.ndarray:
        from inference_llava_zeroshot_json import build_conversation
        scores = []
        for img_np in images_np:
            img = Image.fromarray(img_np.astype(np.uint8))
            conv = build_conversation(stage2_text)
            prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
            inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
            scores.append(_answer_token_prob(model, inputs, valid_answer_ids))
        return np.array(scores)

    return score_fn


def make_qwen_answer_only_scorer(model, processor, question: str, options: list[str],
                                  device: torch.device):
    """Answer-only scorer using plain MC conversation (no JSON, no rationale)."""
    from utils.qwen_vl import build_mc_conversation, prepare_qwen_inputs

    answer_token_ids = get_answer_letter_token_ids(processor.tokenizer)
    num_opts = len(options)
    valid_answer_ids = {
        tid: letter for tid, letter in answer_token_ids.items()
        if letter in "ABCDEFGHIJ"[:num_opts]
    }

    def score_fn(images_np: np.ndarray) -> np.ndarray:
        scores = []
        for img_np in images_np:
            img = Image.fromarray(img_np.astype(np.uint8))
            conv = build_mc_conversation(question, options, image=img)
            inputs = prepare_qwen_inputs(processor, conv, add_generation_prompt=True)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            scores.append(_answer_token_prob(model, inputs, valid_answer_ids))
        return np.array(scores)

    return score_fn


def make_qwen_scorer(model, processor, question: str, options: list[str],
                     json_order: str, device: torch.device):
    from utils.qwen_vl import prepare_qwen_inputs

    answer_token_ids = get_answer_letter_token_ids(processor.tokenizer)
    num_opts = len(options)
    valid_answer_ids = {
        tid: letter for tid, letter in answer_token_ids.items()
        if letter in "ABCDEFGHIJ"[:num_opts]
    }

    if is_json_single_shot(json_order):
        jorder = cast_json_single_shot(json_order)

        def score_fn(images_np: np.ndarray) -> np.ndarray:
            scores = []
            for img_np in images_np:
                img = Image.fromarray(img_np.astype(np.uint8))
                conv = build_qwen_json_conversation(question, options, json_order=jorder, image=img)
                inputs = prepare_qwen_inputs(processor, conv, add_generation_prompt=True)
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                scores.append(_answer_token_prob(model, inputs, valid_answer_ids))
            return np.array(scores)

    else:
        raise NotImplementedError(
            "For rationale_first_2stage use make_qwen_twostage_scorer()."
        )

    return score_fn


def make_qwen_twostage_scorer(model, processor, question: str, options: list[str],
                               rationale: str, device: torch.device):
    from utils.qwen_vl import prepare_qwen_inputs

    answer_token_ids = get_answer_letter_token_ids(processor.tokenizer)
    num_opts = len(options)
    valid_answer_ids = {
        tid: letter for tid, letter in answer_token_ids.items()
        if letter in "ABCDEFGHIJ"[:num_opts]
    }

    def score_fn(images_np: np.ndarray) -> np.ndarray:
        scores = []
        for img_np in images_np:
            img = Image.fromarray(img_np.astype(np.uint8))
            conv = build_qwen_stage2_conversation(question, options, rationale=rationale, image=img)
            inputs = prepare_qwen_inputs(processor, conv, add_generation_prompt=True)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            scores.append(_answer_token_prob(model, inputs, valid_answer_ids))
        return np.array(scores)

    return score_fn


# ---------------------------------------------------------------------------
# Generate rationale (stage-1) for 2-stage mode
# ---------------------------------------------------------------------------

def generate_rationale_llava(model, processor, image: Image.Image,
                              question: str, options: list[str], device: torch.device,
                              max_new_tokens: int = 256) -> str:
    from inference_llava_zeroshot_json import build_conversation, _decode_llava_new

    stage1_text = build_stage1_rationale_prompt(question, options)
    conv = build_conversation(stage1_text)
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             return_dict_in_generate=True, output_scores=False)
    return _decode_llava_new(processor, out.sequences, input_length)


def generate_rationale_qwen(model, processor, image: Image.Image,
                             question: str, options: list[str], device: torch.device,
                             max_new_tokens: int = 256) -> str:
    from utils.qwen_vl import prepare_qwen_inputs

    conv = build_qwen_stage1_conversation(question, options, image=image)
    inputs = prepare_qwen_inputs(processor, conv, add_generation_prompt=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             return_dict_in_generate=True, output_scores=False)
    generated_ids = out.sequences[0][input_length:]
    return clean_answer(processor.decode(generated_ids, skip_special_tokens=True).strip())


# ---------------------------------------------------------------------------
# Generate final answer (for plot label)
# ---------------------------------------------------------------------------

def generate_answer_qwen(model, processor, image: Image.Image,
                         question: str, options: list[str], device: torch.device) -> str:
    from utils.qwen_vl import build_mc_conversation, prepare_qwen_inputs

    conv = build_mc_conversation(question, options, image=image)
    inputs = prepare_qwen_inputs(processor, conv, add_generation_prompt=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                             return_dict_in_generate=True)
    generated_ids = out.sequences[0][input_length:]
    return clean_answer(processor.decode(generated_ids, skip_special_tokens=True).strip())


def generate_answer_llava(model, processor, image: Image.Image,
                          question: str, options: list[str], device: torch.device) -> str:
    from inference_llava_zeroshot_json import build_conversation, _decode_llava_new
    from utils.mc_json_zeroshot import build_mc_json_user_text

    user_text = build_mc_json_user_text(question, options, json_order="answer_first")
    conv = build_conversation(user_text)
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                             return_dict_in_generate=True)
    return _decode_llava_new(processor, out.sequences, input_length)


# ---------------------------------------------------------------------------
# SHAP explanation + plot
# ---------------------------------------------------------------------------

def explain_sample(score_fn, image: Image.Image, max_evals: int, mask_type: str,
                   output_path: Path, title: str):
    import matplotlib.pyplot as plt

    img_np = np.array(image.convert("RGB"))
    masker = shap.maskers.Image(mask_type, img_np.shape)
    explainer = shap.Explainer(score_fn, masker)
    shap_values = explainer(img_np[np.newaxis], max_evals=max_evals, batch_size=4)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_np)
    axes[0].axis("off")
    axes[0].set_title("Original")

    sv = shap_values.values[0]  # H x W x C  (single output → no trailing dim)
    heatmap = sv.mean(axis=-1) # H x W
    vmax = np.abs(heatmap).max() or 1.0
    axes[1].imshow(img_np, alpha=0.5)
    im = axes[1].imshow(heatmap, cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.6)
    axes[1].axis("off")
    axes[1].set_title("SHAP (answer token prob)")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=10, wrap=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SHAP visual explanations for MedVQA")
    parser.add_argument("--model", required=True, choices=["llava", "qwen"])
    parser.add_argument("--model_path", type=str, default=None,
                        help="HF model path/ID; defaults to standard path for the chosen model")
    parser.add_argument("--dataset", required=True,
                        choices=["pathvqa", "slake", "vqarad", "medvqa2019"])
    parser.add_argument("--json_order", required=True,
                        choices=["answer_only", "answer_first", "rationale_first", "rationale_first_2stage"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--sample_offset", type=int, default=0,
                        help="Skip first N samples (to pick different examples)")
    parser.add_argument("--max_evals", type=int, default=300,
                        help="SHAP partition evaluations (300-500 recommended)")
    parser.add_argument("--mask_type", default="blur(32,32)",
                        help="SHAP masker type: blur(H,W) or inpaint_ns or inpaint_telea")
    parser.add_argument("--rationale_max_tokens", type=int, default=256)
    parser.add_argument("--data_root", type=str, default=get_data_root())
    parser.add_argument("--mc_data_root", type=str, default=get_mc_data_root())
    parser.add_argument("--output_dir", type=str, default="shap_outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.model_path or (LLAVA_DEFAULT if args.model == "llava" else QWEN_DEFAULT)
    print(f"Loading {args.model} from {model_path} …")
    if args.model == "llava":
        model, processor = load_llava(model_path, device)
    else:
        model, processor = load_qwen(model_path, device)

    loader = MedicalVQALoader(data_root=args.data_root, mc_data_root=args.mc_data_root)
    samples = loader.load_dataset(args.dataset, split=args.split)
    samples = [s for s in samples if s.mc_options and s.mc_correct_index is not None]
    samples = samples[args.sample_offset: args.sample_offset + args.n_samples]

    out_dir = Path(args.output_dir) / args.model / args.dataset / args.json_order
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(samples):
        print(f"\n[{idx+1}/{len(samples)}] {sample.image_path}")
        image = Image.open(sample.image_path).convert("RGB")
        q, opts = sample.question, sample.mc_options

        # Build scorer
        if args.json_order == "answer_only":
            if args.model == "llava":
                # LLaVA answer-only reuses the answer_first JSON scorer with minimal prompt
                score_fn = make_llava_scorer(model, processor, q, opts, "answer_first", device)
            else:
                score_fn = make_qwen_answer_only_scorer(model, processor, q, opts, device)
        elif args.json_order == "rationale_first_2stage":
            print("  Generating stage-1 rationale …")
            if args.model == "llava":
                rationale = generate_rationale_llava(
                    model, processor, image, q, opts, device, args.rationale_max_tokens
                )
                score_fn = make_llava_twostage_scorer(model, processor, q, opts, rationale, device)
            else:
                rationale = generate_rationale_qwen(
                    model, processor, image, q, opts, device, args.rationale_max_tokens
                )
                score_fn = make_qwen_twostage_scorer(model, processor, q, opts, rationale, device)
            print(f"  Rationale: {rationale[:120]} …")
        else:
            if args.model == "llava":
                score_fn = make_llava_scorer(model, processor, q, opts, args.json_order, device)
            else:
                score_fn = make_qwen_scorer(model, processor, q, opts, args.json_order, device)

        # Generate prediction on the original (unmasked) image
        if args.model == "qwen":
            pred_raw = generate_answer_qwen(model, processor, image, q, opts, device)
        else:
            pred_raw = generate_answer_llava(model, processor, image, q, opts, device)
        pred_letter = parse_predicted_letter(pred_raw, num_options=len(opts))
        pred_text = opts[ord(pred_letter) - 65] if pred_letter else "?"
        print(f"  Predicted: {pred_letter}) {pred_text}")

        # Build title string
        opts_str = "  ".join(f"{chr(65+i)}) {o}" for i, o in enumerate(opts))
        gt = sample.answer or "?"
        pred_display = f"{pred_letter}) {pred_text}" if pred_letter else pred_raw[:40]
        title = f"Q: {q}\n{opts_str}\nGT: {gt}   Pred: {pred_display}"

        out_path = out_dir / f"sample_{args.sample_offset + idx:04d}.png"
        print(f"  Running SHAP ({args.max_evals} evals, mask={args.mask_type}) …")
        explain_sample(score_fn, image, args.max_evals, args.mask_type, out_path, title)

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
