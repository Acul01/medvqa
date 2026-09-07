"""SHAP-based visual explanation for LLaVA and Qwen MC-VQA inference.

Supports experiment modes that match the inference scripts:
  answer_only  — plain MC letter (noX / answer-only SFT)
  zeroshot     — JSON {answer, rationale} (inference_*_zeroshot_json.py)
  pseudo       — JSON {reasoning, answer} with REGION/OBS/REASONING
                 (inference_*.py --use_reasoning_json)

JSON field order is selected with --json_order (answer_first / rationale_first).
Zeroshot also supports rationale_first_2stage.

For finetuned Qwen checkpoints with a spatial filter layer, spatial_filter.pt in
the model directory is loaded automatically.

Usage:
    # answer-only SFT (same as inference_qwen.py without --use_reasoning_json)
    python 04_evaluation/shap_visualize.py \\
        --model qwen --model_path path/to/finetuned_qwen \\
        --dataset vqarad --experiment answer_only --json_order answer_only \\
        --n_samples 5 --max_evals 300 --output_dir shap_outputs/

    # zeroshot JSON answer-first (base model)
    python 04_evaluation/shap_visualize.py \\
        --model qwen --model_path Qwen/Qwen2.5-VL-7B-Instruct \\
        --dataset vqarad --experiment zeroshot --json_order answer_first \\
        --n_samples 5 --max_evals 300 --output_dir shap_outputs/

    # pseudo-reasoning SFT answer-first
    python 04_evaluation/shap_visualize.py \\
        --model qwen --model_path path/to/pseudo_both_ckpt \\
        --dataset vqarad --experiment pseudo --json_order answer_first \\
        --n_samples 5 --max_evals 300 --output_dir shap_outputs/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import shap
import torch
from PIL import Image
from transformers import StoppingCriteria, StoppingCriteriaList

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.load_data import MedicalVQALoader
from utils.mc_json_zeroshot import (
    DEFAULT_JSON_MAX_NEW_TOKENS,
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
from utils.mc_vqa import format_mc_user_text, parse_predicted_letter
from utils.paths import get_data_root, get_mc_data_root
from utils.save_results import clean_answer

LLAVA_DEFAULT = "llava-hf/llava-1.5-7b-hf"
QWEN_DEFAULT = "Qwen/Qwen2.5-VL-7B-Instruct"
PSEUDO_JSON_MAX_NEW_TOKENS = 320
ANSWER_ONLY_MAX_NEW_TOKENS = 10


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
    from transformers import AutoProcessor

    from utils.qwen_model import (
        default_base_model_for_path,
        is_lora_checkpoint,
        load_qwen_vl_for_inference,
        resolve_local_model_path,
    )
    from utils.qwen_vl import configure_qwen_processor_pixels, resolve_processor_path

    model, resolved = load_qwen_vl_for_inference(
        model_path, torch_dtype=torch.float32, device_map="auto"
    )

    spatial_filter_ckpt = Path(resolved) / "spatial_filter.pt"
    if spatial_filter_ckpt.exists():
        from utils.spatial_filter import load_spatial_filter_checkpoint
        model = load_spatial_filter_checkpoint(model, str(spatial_filter_ckpt), bottleneck=64)
        print(f"Spatial filter applied ({spatial_filter_ckpt})")
    else:
        print("No spatial_filter.pt found — running without spatial filter.")

    model.eval()
    if is_lora_checkpoint(resolved):
        processor_path = resolve_processor_path(
            default_base_model_for_path(resolved),
            default_path=default_base_model_for_path(resolved),
        )
    else:
        processor_path = resolve_processor_path(
            resolved,
            default_path=default_base_model_for_path(resolve_local_model_path(resolved)),
        )
    processor = AutoProcessor.from_pretrained(processor_path)
    configure_qwen_processor_pixels(processor)
    return model, processor


# ---------------------------------------------------------------------------
# Prompts (match the corresponding inference script)
# ---------------------------------------------------------------------------

def _pseudo_answer_first(json_order: str) -> bool:
    return json_order == "answer_first"


def build_qwen_prompt_conversation(
    question: str,
    options: list[str],
    image: Image.Image,
    experiment: str,
    json_order: str,
    rationale: str | None = None,
) -> list[dict]:
    from utils.qwen_vl import build_mc_conversation, build_mc_reasoning_inference_conversation

    if json_order == "rationale_first_2stage":
        return build_qwen_stage2_conversation(
            question, options, rationale=rationale or "", image=image
        )
    if experiment == "answer_only" or json_order == "answer_only":
        return build_mc_conversation(question, options, image=image)
    if experiment == "pseudo":
        return build_mc_reasoning_inference_conversation(
            question, options, image=image, answer_first=_pseudo_answer_first(json_order)
        )
    return build_qwen_json_conversation(
        question, options, json_order=cast_json_single_shot(json_order), image=image
    )


def build_llava_user_text(
    question: str,
    options: list[str],
    experiment: str,
    json_order: str,
    rationale: str | None = None,
) -> str:
    from utils.qwen_vl import format_mc_reasoning_user_text

    if json_order == "rationale_first_2stage":
        return build_stage2_answer_prompt(question, options, rationale or "")
    if experiment == "answer_only" or json_order == "answer_only":
        return format_mc_user_text(question, options)
    if experiment == "pseudo":
        return format_mc_reasoning_user_text(
            question, options, answer_first=_pseudo_answer_first(json_order)
        )
    return build_mc_json_user_text(
        question, options, json_order=cast_json_single_shot(json_order)
    )


def _llava_conversation(user_text: str) -> list[dict]:
    from inference_llava_zeroshot_json import build_conversation
    return build_conversation(user_text)


def _qwen_inputs(processor, conv: list[dict], device: torch.device) -> dict:
    from utils.qwen_vl import prepare_qwen_inputs

    inputs = prepare_qwen_inputs(processor, conv, add_generation_prompt=True)
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def _llava_inputs(processor, user_text: str, image: Image.Image, device: torch.device) -> dict:
    conv = _llava_conversation(user_text)
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    return processor(images=image, text=prompt, return_tensors="pt").to(device)


# ---------------------------------------------------------------------------
# Score functions: P(answer_letter | masked_image)
# ---------------------------------------------------------------------------

class _StopOnAnswerLetter(StoppingCriteria):
    """Stop generation once an MC letter token has been produced (answer_first JSON)."""

    def __init__(self, answer_token_ids: set[int], prompt_len: int):
        self.answer_token_ids = answer_token_ids
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        generated = input_ids[0, self.prompt_len :]
        return any(int(t) in self.answer_token_ids for t in generated.tolist())


def _answer_token_prob(model, inputs: dict, answer_token_ids: dict) -> float:
    """Single forward pass → max P over valid answer tokens (first generated token)."""
    with torch.no_grad():
        logits = model(**inputs, return_dict=True).logits[0, -1, :].double()
    probs = torch.softmax(logits, dim=-1)
    valid_ids = list(answer_token_ids.keys())
    return float(probs[valid_ids].max().item())


def _json_answer_token_prob(
    model,
    inputs: dict,
    answer_token_ids: dict,
    max_new_tokens: int,
    stop_on_letter: bool,
) -> float:
    """Generate JSON, then take max P(letter) at the answer-letter token step."""
    input_length = inputs["input_ids"].shape[1]
    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if stop_on_letter:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [_StopOnAnswerLetter(set(answer_token_ids), input_length)]
        )
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    generated_ids = out.sequences[0][input_length:]
    step = find_answer_token_step(generated_ids.tolist(), answer_token_ids)
    if step is None or out.scores is None or step >= len(out.scores):
        return 0.0
    logits = out.scores[step][0].double()
    probs = torch.softmax(logits, dim=-1)
    valid_ids = list(answer_token_ids.keys())
    return float(probs[valid_ids].max().item())


def _valid_answer_ids(processor, num_opts: int) -> dict[int, str]:
    answer_token_ids = get_answer_letter_token_ids(processor.tokenizer)
    return {
        tid: letter
        for tid, letter in answer_token_ids.items()
        if letter in "ABCDEFGHIJ"[:num_opts]
    }


def _uses_json_generate(experiment: str, json_order: str) -> bool:
    return experiment in ("zeroshot", "pseudo") and is_json_single_shot(json_order)


def make_qwen_scorer(
    model,
    processor,
    question: str,
    options: list[str],
    experiment: str,
    json_order: str,
    device: torch.device,
    max_new_tokens: int,
    rationale: str | None = None,
):
    valid_answer_ids = _valid_answer_ids(processor, len(options))
    json_generate = _uses_json_generate(experiment, json_order)
    stop_on_letter = json_generate and json_order == "answer_first"

    def score_fn(images_np: np.ndarray) -> np.ndarray:
        scores = []
        for img_np in images_np:
            img = Image.fromarray(img_np.astype(np.uint8))
            conv = build_qwen_prompt_conversation(
                question, options, img, experiment, json_order, rationale
            )
            inputs = _qwen_inputs(processor, conv, device)
            if json_generate:
                scores.append(
                    _json_answer_token_prob(
                        model, inputs, valid_answer_ids, max_new_tokens, stop_on_letter
                    )
                )
            else:
                scores.append(_answer_token_prob(model, inputs, valid_answer_ids))
        return np.array(scores)

    return score_fn


def make_llava_scorer(
    model,
    processor,
    question: str,
    options: list[str],
    experiment: str,
    json_order: str,
    device: torch.device,
    max_new_tokens: int,
    rationale: str | None = None,
):
    valid_answer_ids = _valid_answer_ids(processor, len(options))
    json_generate = _uses_json_generate(experiment, json_order)
    stop_on_letter = json_generate and json_order == "answer_first"
    user_text = build_llava_user_text(question, options, experiment, json_order, rationale)

    def score_fn(images_np: np.ndarray) -> np.ndarray:
        scores = []
        for img_np in images_np:
            img = Image.fromarray(img_np.astype(np.uint8))
            inputs = _llava_inputs(processor, user_text, img, device)
            if json_generate:
                scores.append(
                    _json_answer_token_prob(
                        model, inputs, valid_answer_ids, max_new_tokens, stop_on_letter
                    )
                )
            else:
                scores.append(_answer_token_prob(model, inputs, valid_answer_ids))
        return np.array(scores)

    return score_fn


# ---------------------------------------------------------------------------
# Generation (stage-1 rationale + plot-label prediction)
# ---------------------------------------------------------------------------

def generate_rationale_llava(model, processor, image: Image.Image,
                              question: str, options: list[str], device: torch.device,
                              max_new_tokens: int = 256) -> str:
    from inference_llava_zeroshot_json import _decode_llava_new

    stage1_text = build_stage1_rationale_prompt(question, options)
    inputs = _llava_inputs(processor, stage1_text, image, device)
    input_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             return_dict_in_generate=True, output_scores=False)
    return _decode_llava_new(processor, out.sequences, input_length)


def generate_rationale_qwen(model, processor, image: Image.Image,
                             question: str, options: list[str], device: torch.device,
                             max_new_tokens: int = 256) -> str:
    conv = build_qwen_stage1_conversation(question, options, image=image)
    inputs = _qwen_inputs(processor, conv, device)
    input_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             return_dict_in_generate=True, output_scores=False)
    generated_ids = out.sequences[0][input_length:]
    return clean_answer(processor.decode(generated_ids, skip_special_tokens=True).strip())


def generate_prediction_qwen(
    model,
    processor,
    image: Image.Image,
    question: str,
    options: list[str],
    experiment: str,
    json_order: str,
    device: torch.device,
    max_new_tokens: int,
    rationale: str | None = None,
) -> str:
    conv = build_qwen_prompt_conversation(
        question, options, image, experiment, json_order, rationale
    )
    inputs = _qwen_inputs(processor, conv, device)
    input_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )
    generated_ids = out.sequences[0][input_length:]
    return clean_answer(processor.decode(generated_ids, skip_special_tokens=True).strip())


def generate_prediction_llava(
    model,
    processor,
    image: Image.Image,
    question: str,
    options: list[str],
    experiment: str,
    json_order: str,
    device: torch.device,
    max_new_tokens: int,
    rationale: str | None = None,
) -> str:
    from inference_llava_zeroshot_json import _decode_llava_new

    user_text = build_llava_user_text(question, options, experiment, json_order, rationale)
    inputs = _llava_inputs(processor, user_text, image, device)
    input_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )
    return _decode_llava_new(processor, out.sequences, input_length)


def parse_prediction(raw: str, num_options: int, experiment: str, json_order: str) -> str | None:
    if json_order == "rationale_first_2stage" or experiment == "answer_only":
        return parse_predicted_letter(raw, num_options=num_options)
    if experiment == "pseudo":
        from utils.qwen_vl import parse_mc_reasoning_json_output
        return parse_mc_reasoning_json_output(raw, num_options=num_options).predicted_letter
    parsed = parse_mc_answer_rationale_json(raw, num_options=num_options)
    return parsed.predicted_letter


def default_max_new_tokens(experiment: str, json_order: str) -> int:
    if json_order == "rationale_first_2stage" or experiment == "answer_only":
        return ANSWER_ONLY_MAX_NEW_TOKENS
    if experiment == "pseudo":
        return PSEUDO_JSON_MAX_NEW_TOKENS
    return DEFAULT_JSON_MAX_NEW_TOKENS


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
    parser.add_argument(
        "--experiment",
        choices=["answer_only", "zeroshot", "pseudo"],
        default=None,
        help="Inference setup to match. Default: answer_only if --json_order answer_only, else zeroshot.",
    )
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
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Generation length for JSON scoring/pred (default: 10 answer-only, 256 zeroshot, 320 pseudo)",
    )
    parser.add_argument("--data_root", type=str, default=get_data_root())
    parser.add_argument("--mc_data_root", type=str, default=get_mc_data_root())
    parser.add_argument("--output_dir", type=str, default="shap_outputs")
    return parser.parse_args()


def resolve_experiment(experiment: str | None, json_order: str) -> str:
    if experiment is None:
        experiment = "answer_only" if json_order == "answer_only" else "zeroshot"
    if experiment == "answer_only" and json_order != "answer_only":
        raise ValueError("--experiment answer_only requires --json_order answer_only")
    if experiment == "zeroshot" and json_order == "answer_only":
        raise ValueError(
            "--experiment zeroshot requires --json_order answer_first, "
            "rationale_first, or rationale_first_2stage"
        )
    if experiment == "pseudo" and json_order not in ("answer_first", "rationale_first"):
        raise ValueError("--experiment pseudo requires --json_order answer_first or rationale_first")
    return experiment


def main():
    args = parse_args()
    experiment = resolve_experiment(args.experiment, args.json_order)
    max_new_tokens = args.max_new_tokens or default_max_new_tokens(experiment, args.json_order)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.model_path or (LLAVA_DEFAULT if args.model == "llava" else QWEN_DEFAULT)
    print(f"Loading {args.model} from {model_path} …")
    print(f"Experiment={experiment}  json_order={args.json_order}  max_new_tokens={max_new_tokens}")
    if args.model == "llava":
        model, processor = load_llava(model_path, device)
    else:
        model, processor = load_qwen(model_path, device)

    loader = MedicalVQALoader(data_root=args.data_root, mc_data_root=args.mc_data_root)
    samples = loader.load_dataset(args.dataset, split=args.split)
    samples = [s for s in samples if s.mc_options and s.mc_correct_index is not None]
    samples = samples[args.sample_offset: args.sample_offset + args.n_samples]

    out_dir = Path(args.output_dir) / args.model / args.dataset / experiment / args.json_order
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(samples):
        print(f"\n[{idx+1}/{len(samples)}] {sample.image_path}")
        image = Image.open(sample.image_path).convert("RGB")
        q, opts = sample.question, sample.mc_options
        rationale = None

        if args.json_order == "rationale_first_2stage":
            print("  Generating stage-1 rationale …")
            if args.model == "llava":
                rationale = generate_rationale_llava(
                    model, processor, image, q, opts, device, args.rationale_max_tokens
                )
            else:
                rationale = generate_rationale_qwen(
                    model, processor, image, q, opts, device, args.rationale_max_tokens
                )
            print(f"  Rationale: {rationale[:120]} …")

        scorer_kwargs = dict(
            model=model,
            processor=processor,
            question=q,
            options=opts,
            experiment=experiment,
            json_order=args.json_order,
            device=device,
            max_new_tokens=max_new_tokens,
            rationale=rationale,
        )
        if args.model == "llava":
            score_fn = make_llava_scorer(**scorer_kwargs)
        else:
            score_fn = make_qwen_scorer(**scorer_kwargs)

        if args.model == "qwen":
            pred_raw = generate_prediction_qwen(
                model, processor, image, q, opts, experiment, args.json_order,
                device, max_new_tokens, rationale,
            )
        else:
            pred_raw = generate_prediction_llava(
                model, processor, image, q, opts, experiment, args.json_order,
                device, max_new_tokens, rationale,
            )
        pred_letter = parse_prediction(pred_raw, len(opts), experiment, args.json_order)
        pred_text = opts[ord(pred_letter) - 65] if pred_letter else "?"
        print(f"  Predicted: {pred_letter}) {pred_text}")
        if experiment != "answer_only" and pred_raw:
            preview = pred_raw.replace("\n", " ")[:160]
            print(f"  Raw: {preview}")

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
