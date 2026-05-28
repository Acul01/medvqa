"""Verify pseudo reasoning rows with Qwen2.5-VL (reasoning + MC options, no image)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from utils.build_mc_datasets_openai import write_records
from utils.build_pseudo_reasonings_openai import parse_options_cell, record_key
from utils.mc_vqa import correct_index_to_letter, parse_predicted_letter
from utils.qwen_vl import (
    DEFAULT_MC_EXPORT_SUFFIX_PSEUDO,
    DEFAULT_MC_EXPORT_SUFFIX_PSEUDO_VERIFY,
    DEFAULT_MODEL_PATH,
    QWEN_VERIFY_FIELD,
    build_pseudo_reasoning_verify_conversation,
    configure_qwen_processor_pixels,
    qwen_verify_from_metadata,
    is_qwen_verify_done,
    prepare_qwen_inputs,
    reasoning_text_from_pseudo_row,
    resolve_processor_path,
)


def verify_row_key(row: Dict[str, Any]) -> str:
    idx = row.get("pseudo_reasoning_index", 0)
    return f"{record_key(row)}||pr_idx={idx}"


def default_verify_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith(DEFAULT_MC_EXPORT_SUFFIX_PSEUDO):
        new_stem = (
            stem[: -len(DEFAULT_MC_EXPORT_SUFFIX_PSEUDO)] + DEFAULT_MC_EXPORT_SUFFIX_PSEUDO_VERIFY
        )
    else:
        new_stem = stem + DEFAULT_MC_EXPORT_SUFFIX_PSEUDO_VERIFY
    return input_path.with_name(new_stem + input_path.suffix)


def is_verify_done(row: Dict[str, Any]) -> bool:
    return is_qwen_verify_done(row)


def load_json_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return data


def parse_correct_index(row: Dict[str, Any]) -> Optional[int]:
    try:
        ci = int(row.get("correct_index", -1))
    except (TypeError, ValueError):
        return None
    return ci if ci >= 0 else None


def load_model_and_processor(
    model_path: str,
    base_model: Optional[str],
) -> Tuple[Any, Any, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(model_path)
    is_lora = path.exists() and (path / "adapter_config.json").exists()

    if is_lora:
        from peft import PeftModel

        base_path = base_model or DEFAULT_MODEL_PATH
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_path, torch_dtype=torch.float32, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        processor_path = resolve_processor_path(base_path)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float32, device_map="auto"
        )
        processor_path = resolve_processor_path(model_path)

    model.eval()
    processor = AutoProcessor.from_pretrained(processor_path)
    configure_qwen_processor_pixels(processor)
    return model, processor, device


def run_qwen_verify_generate(
    model,
    processor,
    device: torch.device,
    conversation: List[dict],
    *,
    max_new_tokens: int,
) -> str:
    inputs = prepare_qwen_inputs(processor, conversation, add_generation_prompt=True)
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }
    prompt_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    new_ids = out[0, prompt_len:]
    return processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def verify_row(
    row: Dict[str, Any],
    *,
    model,
    processor,
    device: torch.device,
    max_new_tokens: int,
) -> Dict[str, Any]:
    out = dict(row)
    options = parse_options_cell(row)
    question = str(row.get("question", "")).strip()
    ci = parse_correct_index(row)
    reasoning_text = reasoning_text_from_pseudo_row(row)

    if options is None or ci is None or ci >= len(options) or not question or not reasoning_text:
        out[QWEN_VERIFY_FIELD] = None
        out["pseudo_reasoning_verify_pred_letter"] = None
        out["pseudo_reasoning_verify_raw"] = None
        out.pop("pseudo_reasoning_test", None)
        return out

    conversation = build_pseudo_reasoning_verify_conversation(
        question,
        options,
        reasoning_text,
    )
    raw = run_qwen_verify_generate(
        model,
        processor,
        device,
        conversation,
        max_new_tokens=max_new_tokens,
    )
    pred = parse_predicted_letter(raw, num_options=len(options))
    gold = correct_index_to_letter(ci)
    out["pseudo_reasoning_verify_pred_letter"] = pred
    out["pseudo_reasoning_verify_raw"] = raw
    out[QWEN_VERIFY_FIELD] = pred is not None and pred == gold
    out.pop("pseudo_reasoning_test", None)
    return out


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    verified = [r for r in rows if is_qwen_verify_done(r)]
    true_n = sum(1 for r in verified if qwen_verify_from_metadata(r) is True)
    false_n = sum(1 for r in verified if qwen_verify_from_metadata(r) is False)
    skipped = len(rows) - len(verified)
    acc = true_n / len(verified) if verified else 0.0
    return {
        "total_rows": len(rows),
        "verified_rows": len(verified),
        "true": true_n,
        "false": false_n,
        "skipped_or_null": skipped,
        "accuracy": acc,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify pseudo reasoning MC rows with Qwen2.5-VL (reasoning text only, no image)"
    )
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--base_model", type=str, default=None, help="Base model when --model_path is LoRA")
    parser.add_argument("--max_new_tokens", type=int, default=12)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows where qwen_verify is already true/false",
    )
    parser.add_argument(
        "--write_csv",
        action="store_true",
        help="Also write sibling .csv via write_records",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output_json) if args.output_json else default_verify_output_path(input_path)
    csv_path = output_path.with_suffix(".csv")

    rows = load_json_rows(input_path)
    existing_by_key: Dict[str, Dict[str, Any]] = {}
    if args.resume and output_path.exists():
        for prev in load_json_rows(output_path):
            existing_by_key[verify_row_key(prev)] = prev

    model, processor, device = load_model_and_processor(args.model_path, args.base_model)

    out_rows: List[Dict[str, Any]] = []
    n_skipped = 0
    n_run = 0

    for row in tqdm(rows, desc="verify_pseudo_reasoning"):
        key = verify_row_key(row)
        if args.resume and key in existing_by_key and is_verify_done(existing_by_key[key]):
            out_rows.append(existing_by_key[key])
            n_skipped += 1
            continue

        verified = verify_row(
            row,
            model=model,
            processor=processor,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        out_rows.append(verified)
        n_run += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.write_csv:
        write_records(output_path, csv_path, out_rows)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out_rows, f, ensure_ascii=False, indent=2)

    stats = summarize(out_rows)
    print(f"Wrote {len(out_rows)} rows → {output_path}")
    print(
        f"Resume skipped {n_skipped} rows; newly verified {n_run}. "
        f"Summary: verified={stats['verified_rows']} true={stats['true']} "
        f"false={stats['false']} null/skipped={stats['skipped_or_null']} "
        f"accuracy={stats['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
