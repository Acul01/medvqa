#!/usr/bin/env python3
"""Qualitative Qwen2.5-VL sample compare across experiment specifications.

For each dataset, pick one closed and one open MC sample (image + question),
then run greedy inference for every Qwen2.5 experiment spec and write all
outputs into one folder.

Specs:
  answer-only:  base, full, lora
  zeroshot:     answer-first, rationale-first, 2-stage, 2-stage_no_rationale
  pseudo-sft:   reasonings-first, answer-first, AF+RF, SOM

Example (cluster):
  python 04_evaluation/compare_qwen25_samples.py --dry_run
  python 04_evaluation/compare_qwen25_samples.py --dtype bfloat16 \\
      --output_dir 06_results/sample_compare_qwen2.5
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.load_data import MedicalVQALoader, VQASample
from utils.mc_json_zeroshot import (
    DEFAULT_JSON_MAX_NEW_TOKENS,
    DEFAULT_RATIONALE_MAX_NEW_TOKENS,
    DEFAULT_STAGE2_MAX_NEW_TOKENS,
    build_qwen_json_conversation,
    build_qwen_stage1_conversation,
    build_qwen_stage2_conversation,
    build_qwen_stage2_json_conversation,
    cast_json_single_shot,
    is_json_single_shot,
    parse_mc_answer_rationale_json,
)
from utils.mc_vqa import parse_predicted_letter
from utils.paths import (
    get_data_root,
    get_mc_data_root,
    get_som_dir,
    get_som_selection_dir,
)
from utils.qwen_model import (
    default_base_model_for_path,
    is_lora_checkpoint,
    load_qwen_vl_for_inference,
    resolve_local_model_path,
)
from utils.qwen_vl import (
    DEFAULT_MODEL_PATH,
    build_mc_conversation,
    build_mc_reasoning_inference_conversation,
    configure_qwen_processor_pixels,
    parse_mc_reasoning_json_output,
    prepare_qwen_inputs,
    resolve_processor_path,
)
from utils.save_results import clean_answer
from utils.som_select import question_with_selected_som_hint
from utils.som_selection import SomSelectionResolver

ANSWER_ONLY_MAX_NEW_TOKENS = 8
PSEUDO_MAX_NEW_TOKENS = 512
DEFAULT_CONFIG = Path("configs/qwen25_sample_compare.json")
DEFAULT_DATASETS = ["slake", "vqarad", "medvqa2019", "pathvqa"]
QUESTION_TYPES = ("closed", "open")


@dataclass(frozen=True)
class Spec:
    family: str
    name: str
    mode: str
    ckpt_key: str
    json_order: str | None = None
    answer_first: bool = False
    use_som: bool = False

    @property
    def spec_id(self) -> str:
        return f"{self.family}/{self.name}"

    @property
    def filename(self) -> str:
        return f"{self.family}__{self.name}.json"


SPECS: tuple[Spec, ...] = (
    Spec("answer_only", "base", "answer_only", "base"),
    Spec("answer_only", "full", "answer_only", "answer_only_full"),
    Spec("answer_only", "lora", "answer_only", "answer_only_lora"),
    Spec("zeroshot", "answer_first", "zeroshot", "base", json_order="answer_first"),
    Spec("zeroshot", "rationale_first", "zeroshot", "base", json_order="rationale_first"),
    Spec("zeroshot", "2stage", "zeroshot", "base", json_order="rationale_first_2stage"),
    Spec(
        "zeroshot",
        "2stage_no_rationale",
        "zeroshot",
        "base",
        json_order="rationale_first_2stage_norat",
    ),
    Spec("pseudo_sft", "reasonings_first", "pseudo", "pseudo_rf", answer_first=False),
    Spec("pseudo_sft", "answer_first", "pseudo", "pseudo_af", answer_first=True),
    Spec("pseudo_sft", "af_rf", "pseudo_both", "pseudo_af_rf"),
    Spec("pseudo_sft", "som", "pseudo_som", "pseudo_som", answer_first=False, use_som=True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Qwen2.5-VL experiment specs on one closed and one open sample per dataset"
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--data_root", type=str, default=get_data_root())
    parser.add_argument("--mc_data_root", type=str, default=get_mc_data_root())
    parser.add_argument("--som_dir", type=str, default=get_som_dir())
    parser.add_argument("--selection_dir", type=str, default=get_som_selection_dir())
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=DEFAULT_DATASETS,
        help="Datasets to include (default: all in config / slake vqarad medvqa2019 pathvqa)",
    )
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="06_results/sample_compare_qwen2.5",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--max_new_tokens_answer_only", type=int, default=ANSWER_ONLY_MAX_NEW_TOKENS)
    parser.add_argument("--max_new_tokens_json", type=int, default=DEFAULT_JSON_MAX_NEW_TOKENS)
    parser.add_argument("--max_new_tokens_pseudo", type=int, default=PSEUDO_MAX_NEW_TOKENS)
    parser.add_argument("--rationale_max_new_tokens", type=int, default=DEFAULT_RATIONALE_MAX_NEW_TOKENS)
    parser.add_argument("--stage2_max_new_tokens", type=int, default=DEFAULT_STAGE2_MAX_NEW_TOKENS)
    parser.add_argument(
        "--specs",
        nargs="+",
        default=None,
        help="Subset of spec ids, e.g. answer_only/base zeroshot/2stage pseudo_sft/som",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip specs whose checkpoint is missing instead of aborting",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-run specs that already have a JSON")
    parser.add_argument("--reselect", action="store_true", help="Ignore previously saved sample picks")
    parser.add_argument("--dry_run", action="store_true", help="Print samples and resolved paths, then exit")
    parser.add_argument(
        "--samples_json",
        type=str,
        default=None,
        help="Optional JSON with explicit sample picks (same schema as selected_samples.json)",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def sample_payload(sample: VQASample, *, image_copy: str | None = None) -> dict[str, Any]:
    return {
        "dataset": sample.dataset,
        "split": sample.split,
        "question_type": sample.question_type,
        "image_path": sample.image_path,
        "image_copy": image_copy,
        "question": sample.question,
        "options": list(sample.mc_options or []),
        "correct_index": sample.mc_correct_index,
        "ground_truth_letter": sample.answer,
        "ground_truth_answer": (sample.metadata or {}).get("gt_answer_text", ""),
    }


def copy_image(src: str, dest_dir: Path, stem: str = "image") -> str:
    src_path = Path(src)
    suffix = src_path.suffix.lower() if src_path.suffix else ".jpg"
    dest = dest_dir / f"{stem}{suffix}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest)
    return dest.name


def is_usable_sample(sample: VQASample) -> bool:
    if not sample.mc_options or sample.mc_correct_index is None:
        return False
    if not str(sample.question or "").strip():
        return False
    return Path(sample.image_path).is_file()


def try_load_som_selection(som_dir: str, selection_dir: str, dataset: str) -> SomSelectionResolver | None:
    try:
        resolver = SomSelectionResolver(som_dir, selection_dir)
        resolver.load_dataset(dataset)
        return resolver
    except FileNotFoundError as exc:
        print(f"  SoM selection not available for {dataset}: {exc}")
        return None


def pick_typed_sample(
    samples: list[VQASample],
    question_type: str,
    rng: random.Random,
    som_selection: SomSelectionResolver | None,
) -> VQASample:
    typed = [s for s in samples if (s.question_type or "").strip().lower() == question_type]
    if not typed:
        raise ValueError(f"No usable {question_type} MC samples with an existing image")
    if som_selection is not None:
        with_som = [
            s
            for s in typed
            if som_selection.get_selected_labels(s.image_path, s.question) is not None
        ]
        if with_som:
            typed = with_som
    return rng.choice(typed)


def vqa_from_payload(row: dict[str, Any], *, split: str) -> VQASample:
    return VQASample(
        dataset=str(row["dataset"]),
        split=str(row.get("split") or split),
        image_path=str(row["image_path"]),
        question=str(row["question"]),
        answer=str(row.get("ground_truth_letter") or ""),
        metadata={"gt_answer_text": row.get("ground_truth_answer", "")},
        mc_options=list(row.get("options") or []),
        mc_correct_index=row.get("correct_index"),
        question_type=str(row.get("question_type") or ""),
    )


def select_or_load_samples(
    *,
    loader: MedicalVQALoader,
    datasets: list[str],
    split: str,
    seed: int,
    out_dir: Path,
    som_dir: str,
    selection_dir: str,
    reselect: bool,
    samples_json: Path | None,
) -> dict[str, dict[str, VQASample]]:
    selected_path = Path(samples_json) if samples_json else out_dir / "selected_samples.json"
    if selected_path.is_file() and not reselect:
        with open(selected_path, encoding="utf-8") as handle:
            stored = json.load(handle)
        picked: dict[str, dict[str, VQASample]] = {}
        for dataset, by_type in stored["datasets"].items():
            if dataset not in datasets:
                continue
            picked[dataset] = {
                qtype: vqa_from_payload(row, split=split) for qtype, row in by_type.items()
            }
        for dataset, by_type in picked.items():
            for qtype, sample in by_type.items():
                sample_dir = out_dir / dataset / qtype
                image_copy = copy_image(sample.image_path, sample_dir)
                payload = sample_payload(sample, image_copy=image_copy)
                dump_json(sample_dir / "sample.json", payload)
        print(f"Loaded previously selected samples from {selected_path}")
        return picked

    rng = random.Random(seed)
    picked = {}
    stored_datasets: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        print(f"Selecting samples for {dataset} …")
        samples = [s for s in loader.load_dataset(dataset, split=split) if is_usable_sample(s)]
        som_selection = try_load_som_selection(som_dir, selection_dir, dataset)
        by_type: dict[str, VQASample] = {}
        stored_datasets[dataset] = {}
        for qtype in QUESTION_TYPES:
            sample = pick_typed_sample(samples, qtype, rng, som_selection)
            by_type[qtype] = sample
            sample_dir = out_dir / dataset / qtype
            image_copy = copy_image(sample.image_path, sample_dir)
            payload = sample_payload(sample, image_copy=image_copy)
            stored_datasets[dataset][qtype] = payload
            dump_json(sample_dir / "sample.json", payload)
            print(f"  {qtype}: {sample.question[:80]!r}  ({Path(sample.image_path).name})")
        picked[dataset] = by_type

    dump_json(
        out_dir / "selected_samples.json",
        {"seed": seed, "split": split, "datasets": stored_datasets},
    )
    return picked


def latest_checkpoint_dir(path: Path) -> Path:
    if (path / "config.json").is_file() or (path / "adapter_config.json").is_file():
        return path
    numbered = []
    for child in path.glob("checkpoint-*"):
        try:
            numbered.append((int(child.name.split("-")[-1]), child))
        except ValueError:
            continue
    if numbered:
        numbered.sort()
        return numbered[-1][1]
    return path


def looks_like_checkpoint(path: Path) -> bool:
    if not path.is_dir():
        return False
    resolved = latest_checkpoint_dir(path)
    return (resolved / "config.json").is_file() or (resolved / "adapter_config.json").is_file()


def format_ckpt_candidates(templates: list[str] | str, dataset: str) -> list[Path]:
    if isinstance(templates, str):
        templates = [templates]
    return [Path(str(item).format(dataset=dataset)) for item in templates]


def resolve_checkpoint(
    ckpt_key: str,
    dataset: str,
    config: dict[str, Any],
) -> Path | str:
    if ckpt_key == "base":
        return str(config.get("base_model") or DEFAULT_MODEL_PATH)
    mapping = config.get("checkpoints") or {}
    templates = mapping.get(ckpt_key)
    if not templates:
        raise KeyError(f"No checkpoint templates for key {ckpt_key!r}")
    candidates = format_ckpt_candidates(templates, dataset)
    for candidate in candidates:
        if looks_like_checkpoint(candidate):
            return latest_checkpoint_dir(candidate)
    raise FileNotFoundError(
        f"No checkpoint for {ckpt_key} / {dataset}. Tried:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )


def filter_specs(selected: list[str] | None) -> list[Spec]:
    if not selected:
        return list(SPECS)
    wanted = set(selected)
    known = {spec.spec_id for spec in SPECS}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"Unknown spec ids: {unknown}. Known: {sorted(known)}")
    return [spec for spec in SPECS if spec.spec_id in wanted]


def move_inputs_to_device(inputs: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def generate_text(
    model,
    processor,
    conversation: list[dict],
    device: torch.device,
    max_new_tokens: int,
) -> str:
    inputs = prepare_qwen_inputs(processor, conversation, add_generation_prompt=True)
    inputs = move_inputs_to_device(inputs, device)
    input_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        sequences = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated_ids = sequences[0][input_length:]
    return clean_answer(processor.decode(generated_ids, skip_special_tokens=True).strip())


def load_bundle(model_path: str | Path, dtype: torch.dtype):
    resolved = resolve_local_model_path(str(model_path))
    model, resolved = load_qwen_vl_for_inference(
        resolved,
        torch_dtype=dtype,
        device_map="auto",
    )
    spatial_filter_ckpt = Path(resolved) / "spatial_filter.pt"
    if spatial_filter_ckpt.exists():
        from utils.spatial_filter import load_spatial_filter_checkpoint

        model = load_spatial_filter_checkpoint(model, str(spatial_filter_ckpt), bottleneck=64)
        print(f"  Spatial filter applied ({spatial_filter_ckpt})")
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
    device = next(model.parameters()).device
    return model, processor, device, resolved


def unload_bundle(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_image(
    sample: VQASample,
    spec: Spec,
    som_selection: SomSelectionResolver | None,
    sample_dir: Path,
) -> tuple[Image.Image, str, list[str], str]:
    question = sample.question
    som_labels: list[str] = []
    som_response = ""
    if spec.use_som:
        if som_selection is None:
            raise FileNotFoundError("SoM selection resolver is not loaded")
        image, som_labels = som_selection.render_with_labels(sample.image_path, sample.question)
        som_response = som_selection.get_selection_response(sample.image_path, sample.question)
        question = question_with_selected_som_hint(sample.question, som_labels)
        sample_dir.mkdir(parents=True, exist_ok=True)
        image.save(sample_dir / "image_som.png")
        return image, question, som_labels, som_response
    image = Image.open(sample.image_path).convert("RGB")
    return image, question, som_labels, som_response


def run_answer_only(model, processor, image, question, options, device, max_new_tokens) -> dict[str, Any]:
    conversation = build_mc_conversation(question, options, image=image)
    text = generate_text(model, processor, conversation, device, max_new_tokens)
    letter = parse_predicted_letter(text, num_options=len(options))
    return {
        "predicted_answer": text,
        "predicted_letter": letter,
        "predicted_reasoning_text": None,
        "predicted_json_parse_ok": False,
    }


def run_pseudo(model, processor, image, question, options, device, max_new_tokens, answer_first: bool) -> dict[str, Any]:
    conversation = build_mc_reasoning_inference_conversation(
        question, options, image=image, answer_first=answer_first
    )
    text = generate_text(model, processor, conversation, device, max_new_tokens)
    parsed = parse_mc_reasoning_json_output(text, num_options=len(options))
    return {
        "predicted_answer": text,
        "predicted_letter": parsed.predicted_letter,
        "predicted_reasoning_text": parsed.reasoning_text,
        "predicted_json_parse_ok": parsed.json_parse_ok,
        "answer_first": answer_first,
    }


def run_zeroshot(
    model,
    processor,
    image,
    question,
    options,
    device,
    json_order: str,
    max_new_tokens_json: int,
    rationale_max_new_tokens: int,
    stage2_max_new_tokens: int,
) -> dict[str, Any]:
    if is_json_single_shot(json_order):
        conversation = build_qwen_json_conversation(
            question,
            options,
            json_order=cast_json_single_shot(json_order),
            image=image,
        )
        text = generate_text(model, processor, conversation, device, max_new_tokens_json)
        parsed = parse_mc_answer_rationale_json(text, num_options=len(options))
        return {
            "predicted_answer": text,
            "predicted_letter": parsed.predicted_letter,
            "predicted_reasoning_text": parsed.rationale_text,
            "predicted_json_parse_ok": parsed.json_parse_ok,
            "json_order": json_order,
        }

    conv1 = build_qwen_stage1_conversation(question, options, image=image)
    stage1_raw = generate_text(model, processor, conv1, device, rationale_max_new_tokens)
    if json_order == "rationale_first_2stage_norat":
        conv2 = build_qwen_stage2_conversation(
            question, options, rationale=stage1_raw, image=image
        )
        stage2_raw = generate_text(model, processor, conv2, device, stage2_max_new_tokens)
        letter = parse_predicted_letter(stage2_raw, num_options=len(options))
        return {
            "predicted_answer": stage2_raw,
            "predicted_letter": letter,
            "predicted_reasoning_text": stage1_raw,
            "predicted_json_parse_ok": False,
            "json_order": json_order,
            "stage1_raw": stage1_raw,
            "stage2_raw": stage2_raw,
            "two_stage_cot": True,
        }

    conv2 = build_qwen_stage2_json_conversation(
        question, options, reasoning=stage1_raw, image=image
    )
    stage2_raw = generate_text(model, processor, conv2, device, max_new_tokens_json)
    parsed = parse_mc_answer_rationale_json(stage2_raw, num_options=len(options))
    return {
        "predicted_answer": stage2_raw,
        "predicted_letter": parsed.predicted_letter,
        "predicted_reasoning_text": parsed.rationale_text,
        "predicted_json_parse_ok": parsed.json_parse_ok,
        "json_order": json_order,
        "stage1_raw": stage1_raw,
        "stage2_raw": stage2_raw,
        "two_stage_cot": True,
    }


def merge_af_rf(rf_row: dict[str, Any], af_row: dict[str, Any], gt_letter: str) -> dict[str, Any]:
    rf_letter = rf_row.get("predicted_letter")
    af_letter = af_row.get("predicted_letter")
    letter = rf_letter if rf_letter == af_letter else rf_letter or af_letter
    return {
        "predicted_letter": letter,
        "predicted_letter_rf": rf_letter,
        "predicted_letter_af": af_letter,
        "letters_agree": rf_letter == af_letter and rf_letter is not None,
        "is_correct": bool(letter) and letter == gt_letter,
        "predicted_json_parse_ok": bool(rf_row.get("predicted_json_parse_ok") or af_row.get("predicted_json_parse_ok")),
        "predicted_answer_rf": rf_row.get("predicted_answer"),
        "predicted_answer_af": af_row.get("predicted_answer"),
        "predicted_reasoning_text_rf": rf_row.get("predicted_reasoning_text"),
        "predicted_reasoning_text_af": af_row.get("predicted_reasoning_text"),
        "json_order_merge": "rf+af",
        "rf": rf_row,
        "af": af_row,
    }


def wrap_result(
    *,
    sample: VQASample,
    spec: Spec,
    model_path: str,
    prediction: dict[str, Any],
    question_for_prompt: str,
    som_labels: list[str],
    som_response: str,
) -> dict[str, Any]:
    gt = sample.answer
    letter = prediction.get("predicted_letter")
    row = {
        "spec_id": spec.spec_id,
        "family": spec.family,
        "spec": spec.name,
        "mode": spec.mode,
        "model_path": model_path,
        "dataset": sample.dataset,
        "split": sample.split,
        "question_type": sample.question_type,
        "image_path": sample.image_path,
        "question": sample.question,
        "question_for_prompt": question_for_prompt,
        "options": list(sample.mc_options or []),
        "correct_index": sample.mc_correct_index,
        "ground_truth_letter": gt,
        "ground_truth_answer": (sample.metadata or {}).get("gt_answer_text", ""),
        "is_correct": bool(letter) and letter == gt,
        "som_selected_regions": som_labels,
        "som_selection_response": som_response,
        "greedy": True,
    }
    row.update(prediction)
    if "is_correct" in prediction:
        row["is_correct"] = prediction["is_correct"]
    return row


def result_path(out_dir: Path, dataset: str, qtype: str, spec: Spec) -> Path:
    return out_dir / dataset / qtype / spec.filename


def write_summary(out_dir: Path, datasets: list[str], specs: list[Spec]) -> None:
    lines = [
        "# Qwen2.5 sample compare",
        "",
        "Greedy inference on one closed and one open sample per dataset.",
        "",
    ]
    for dataset in datasets:
        lines.append(f"## {dataset}")
        lines.append("")
        for qtype in QUESTION_TYPES:
            sample_file = out_dir / dataset / qtype / "sample.json"
            if not sample_file.is_file():
                continue
            with open(sample_file, encoding="utf-8") as handle:
                sample = json.load(handle)
            image_rel = f"{dataset}/{qtype}/{sample.get('image_copy') or 'image.jpg'}"
            lines.append(f"### {qtype}")
            lines.append("")
            lines.append(f"![{dataset} {qtype}]({image_rel})")
            lines.append("")
            lines.append(f"- **Question:** {sample.get('question')}")
            opts = sample.get("options") or []
            opt_txt = ", ".join(f"{chr(65 + i)}) {opt}" for i, opt in enumerate(opts))
            lines.append(f"- **Options:** {opt_txt}")
            lines.append(
                f"- **GT:** {sample.get('ground_truth_letter')} "
                f"({sample.get('ground_truth_answer')})"
            )
            lines.append("")
            lines.append("| Spec | Letter | Correct | Preview |")
            lines.append("|---|---|---|---|")
            for spec in specs:
                path = result_path(out_dir, dataset, qtype, spec)
                if not path.is_file():
                    lines.append(f"| `{spec.spec_id}` | — | — | missing |")
                    continue
                with open(path, encoding="utf-8") as handle:
                    row = json.load(handle)
                if row.get("error"):
                    lines.append(f"| `{spec.spec_id}` | — | — | error: {row['error'][:80]} |")
                    continue
                letter = row.get("predicted_letter") or "?"
                correct = "yes" if row.get("is_correct") else "no"
                preview = (
                    row.get("predicted_reasoning_text")
                    or row.get("predicted_answer")
                    or row.get("stage1_raw")
                    or ""
                )
                if spec.mode == "pseudo_both":
                    preview = (
                        f"RF={row.get('predicted_letter_rf')} AF={row.get('predicted_letter_af')}"
                    )
                preview = str(preview).replace("\n", " ")[:160]
                lines.append(f"| `{spec.spec_id}` | {letter} | {correct} | {preview} |")
            lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def collect_all_results(out_dir: Path, datasets: list[str], specs: list[Spec]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for dataset in datasets:
        nested[dataset] = {}
        for qtype in QUESTION_TYPES:
            sample_file = out_dir / dataset / qtype / "sample.json"
            sample = None
            if sample_file.is_file():
                with open(sample_file, encoding="utf-8") as handle:
                    sample = json.load(handle)
            results = {}
            for spec in specs:
                path = result_path(out_dir, dataset, qtype, spec)
                if path.is_file():
                    with open(path, encoding="utf-8") as handle:
                        results[spec.spec_id] = json.load(handle)
            nested[dataset][qtype] = {"sample": sample, "results": results}
    return nested


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    datasets = args.datasets or list(config.get("datasets") or DEFAULT_DATASETS)
    split = args.split or str(config.get("split") or "test")
    specs = filter_specs(args.specs)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = MedicalVQALoader(data_root=args.data_root, mc_data_root=args.mc_data_root)
    samples_json = Path(args.samples_json) if args.samples_json else None
    picked = select_or_load_samples(
        loader=loader,
        datasets=datasets,
        split=split,
        seed=args.seed,
        out_dir=out_dir,
        som_dir=args.som_dir,
        selection_dir=args.selection_dir,
        reselect=args.reselect,
        samples_json=samples_json,
    )
    missing_ds = [ds for ds in datasets if ds not in picked or any(q not in picked[ds] for q in QUESTION_TYPES)]
    if missing_ds:
        raise ValueError(f"No closed/open sample pair for: {missing_ds}")

    jobs: list[tuple[str, str, Spec, str | Path | None, str | None]] = []
    missing = 0
    for dataset in datasets:
        for spec in specs:
            try:
                model_path = resolve_checkpoint(spec.ckpt_key, dataset, config)
                error = None
            except FileNotFoundError as exc:
                model_path = None
                error = str(exc)
                missing += 1
                print(f"MISSING {spec.spec_id} {dataset}: {exc}")
                if not args.skip_missing and not args.dry_run:
                    raise
            for qtype in QUESTION_TYPES:
                jobs.append((dataset, qtype, spec, model_path, error))

    dump_json(
        out_dir / "run_plan.json",
        {
            "datasets": datasets,
            "split": split,
            "specs": [spec.spec_id for spec in specs],
            "jobs": [
                {
                    "dataset": dataset,
                    "question_type": qtype,
                    "spec_id": spec.spec_id,
                    "model_path": str(model_path) if model_path is not None else None,
                    "error": error,
                }
                for dataset, qtype, spec, model_path, error in jobs
            ],
        },
    )

    if args.dry_run:
        print(f"\nDry run complete. Plan written to {out_dir / 'run_plan.json'}")
        print(f"Missing checkpoints: {missing}")
        return

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    load_dtype = dtype_map[args.dtype]
    som_by_dataset: dict[str, SomSelectionResolver | None] = {}

    grouped: dict[str, list[tuple[str, str, Spec]]] = {}
    error_jobs: list[tuple[str, str, Spec, str]] = []
    for dataset, qtype, spec, model_path, error in jobs:
        if error or model_path is None:
            error_jobs.append((dataset, qtype, spec, error or "missing checkpoint"))
            continue
        grouped.setdefault(str(model_path), []).append((dataset, qtype, spec))

    for dataset, qtype, spec, error in error_jobs:
        path = result_path(out_dir, dataset, qtype, spec)
        if path.is_file() and not args.overwrite:
            continue
        dump_json(
            path,
            {
                "spec_id": spec.spec_id,
                "dataset": dataset,
                "question_type": qtype,
                "error": error,
            },
        )

    for model_path, model_jobs in grouped.items():
        pending = []
        for dataset, qtype, spec in model_jobs:
            path = result_path(out_dir, dataset, qtype, spec)
            if path.is_file() and not args.overwrite:
                print(f"skip existing {path}")
                continue
            pending.append((dataset, qtype, spec))
        if not pending:
            continue

        print(f"\nLoading {model_path}  ({len(pending)} jobs)")
        model, processor, device, resolved = load_bundle(model_path, load_dtype)
        try:
            for dataset, qtype, spec in pending:
                sample = picked[dataset][qtype]
                sample_dir = out_dir / dataset / qtype
                if spec.use_som and dataset not in som_by_dataset:
                    som_by_dataset[dataset] = try_load_som_selection(
                        args.som_dir, args.selection_dir, dataset
                    )
                som_selection = som_by_dataset.get(dataset)
                try:
                    image, question_for_prompt, som_labels, som_response = prepare_image(
                        sample, spec, som_selection, sample_dir
                    )
                    if spec.mode == "answer_only":
                        prediction = run_answer_only(
                            model,
                            processor,
                            image,
                            question_for_prompt,
                            sample.mc_options or [],
                            device,
                            args.max_new_tokens_answer_only,
                        )
                    elif spec.mode == "zeroshot":
                        prediction = run_zeroshot(
                            model,
                            processor,
                            image,
                            question_for_prompt,
                            sample.mc_options or [],
                            device,
                            spec.json_order or "answer_first",
                            args.max_new_tokens_json,
                            args.rationale_max_new_tokens,
                            args.stage2_max_new_tokens,
                        )
                    elif spec.mode == "pseudo_both":
                        rf = run_pseudo(
                            model,
                            processor,
                            image,
                            question_for_prompt,
                            sample.mc_options or [],
                            device,
                            args.max_new_tokens_pseudo,
                            answer_first=False,
                        )
                        af = run_pseudo(
                            model,
                            processor,
                            image,
                            question_for_prompt,
                            sample.mc_options or [],
                            device,
                            args.max_new_tokens_pseudo,
                            answer_first=True,
                        )
                        prediction = merge_af_rf(rf, af, sample.answer)
                    else:
                        prediction = run_pseudo(
                            model,
                            processor,
                            image,
                            question_for_prompt,
                            sample.mc_options or [],
                            device,
                            args.max_new_tokens_pseudo,
                            answer_first=spec.answer_first,
                        )
                    row = wrap_result(
                        sample=sample,
                        spec=spec,
                        model_path=str(resolved),
                        prediction=prediction,
                        question_for_prompt=question_for_prompt,
                        som_labels=som_labels,
                        som_response=som_response,
                    )
                except Exception as exc:  # noqa: BLE001 — keep remaining specs running on cluster
                    print(f"  ERROR {dataset}/{qtype}/{spec.spec_id}: {exc}")
                    row = {
                        "spec_id": spec.spec_id,
                        "dataset": dataset,
                        "question_type": qtype,
                        "error": str(exc),
                    }
                out_path = result_path(out_dir, dataset, qtype, spec)
                dump_json(out_path, row)
                letter = row.get("predicted_letter")
                status = "ERR" if row.get("error") else ("OK" if row.get("is_correct") else "wrong")
                print(f"  {dataset}/{qtype}/{spec.spec_id}: letter={letter} [{status}]")
        finally:
            unload_bundle(model)

    all_results = collect_all_results(out_dir, datasets, specs)
    dump_json(out_dir / "all_results.json", all_results)
    write_summary(out_dir, datasets, specs)
    print(f"\nDone. Outputs in {out_dir}")
    print(f"  selected_samples.json, all_results.json, summary.md")


if __name__ == "__main__":
    main()
