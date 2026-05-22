"""Verify pseudo reasoning: Qwen2.5-VL chooses MC letter from rationale + question + options (gray placeholder image)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from utils.build_mc_datasets_openai import write_records
from utils.build_pseudo_reasonings_openai import parse_options
from utils.mc_vqa import correct_index_to_letter, parse_predicted_letter
from utils.reasoning_mc import format_reasoning_block_for_training
from utils.qwen_vl import (
    DEFAULT_MODEL_PATH,
    build_pseudo_reasoning_verify_conversation,
    make_pseudo_reasoning_verify_placeholder,
    prepare_qwen_inputs,
    resolve_processor_path,
)
from utils.save_results import clean_answer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check whether Qwen predicts the golden MC letter using only pseudo reasoning + question + options.",
    )
    p.add_argument("--input_json", type=str, required=True, help="Pseudo-MC JSON list, e.g. *_mc_openai_pseudo.json")
    p.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Defaults to <input_stem>_pr_qwen_verify.json beside the input file",
    )
    p.add_argument("--no_csv", action="store_true", help="Do not write the companion CSV file")
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--max_new_tokens", type=int, default=12)
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--save_every", type=int, default=50)
    return p.parse_args()


def reasoning_text_from_row(row: Dict[str, Any]) -> str | None:
    raw = row.get("pseudo_reasoning_raw")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    reg = str(row.get("pseudo_reasoning_region", "")).strip()
    obs = str(row.get("pseudo_reasoning_observation", "")).strip()
    exp = str(row.get("pseudo_reasoning_explanation", "")).strip()
    if reg or obs or exp:
        return format_reasoning_block_for_training(
            reg if reg else "unknown",
            obs if obs else "unknown",
            exp if exp else "unknown",
        )
    return None


def _move_inputs_to_device(inputs: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def load_qwen_mc_model(model_path: str, base_model: str | None):
    """Load optional LoRA checkpoint merged into base; else full finetuned or hub model."""
    is_lora = Path(model_path).exists() and (Path(model_path) / "adapter_config.json").exists()
    if is_lora:
        from peft import PeftModel

        base_path = base_model or DEFAULT_MODEL_PATH
        base_m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_path, torch_dtype=torch.float32, device_map="auto"
        )
        model = PeftModel.from_pretrained(base_m, model_path)
        return model.merge_and_unload(), resolve_processor_path(base_model or DEFAULT_MODEL_PATH)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="auto"
    )
    return model, resolve_processor_path(model_path)


def _write_json_only(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_json)
    if not in_path.is_file():
        raise FileNotFoundError(in_path)

    with open(in_path, encoding="utf-8") as f:
        records_raw = json.load(f)
    if not isinstance(records_raw, list):
        raise ValueError("--input_json must contain a JSON list")

    rows: List[Dict[str, Any]] = [dict(r) for r in records_raw]
    if args.n_samples is not None:
        rows = rows[: args.n_samples]

    out_json = Path(args.output_json) if args.output_json else in_path.with_name(in_path.stem + "_pr_qwen_verify.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor_path = load_qwen_mc_model(args.model_path, args.base_model)
    model.eval()
    processor = AutoProcessor.from_pretrained(processor_path)
    placeholder = make_pseudo_reasoning_verify_placeholder()

    gen_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False}

    n_invalid_mc = 0
    n_no_reasoning = 0
    n_ok = 0
    n_fail = 0
    out_rows: List[Dict[str, Any]] = []
    out_csv = out_json.with_suffix(".csv")

    with torch.no_grad():
        for idx, row in enumerate(tqdm(rows, desc="verify_pseudo_reasoning", unit="row"), start=1):
            out = dict(row)
            options, correct_index = parse_options(row)
            reasoning = reasoning_text_from_row(row)

            if options is None or correct_index is None:
                n_invalid_mc += 1
                out["pseudo_reasoning_test"] = None
                out["pseudo_reasoning_verify_pred_letter"] = None
                out["pseudo_reasoning_verify_raw"] = None
                out_rows.append(out)
            elif reasoning is None:
                n_no_reasoning += 1
                out["pseudo_reasoning_test"] = None
                out["pseudo_reasoning_verify_pred_letter"] = None
                out["pseudo_reasoning_verify_raw"] = None
                out_rows.append(out)
            else:
                question = str(row.get("question", "")).strip() or " "
                conv = build_pseudo_reasoning_verify_conversation(
                    question,
                    options,
                    reasoning,
                    placeholder_image=placeholder,
                )
                inputs = prepare_qwen_inputs(processor, conv, add_generation_prompt=True)
                inputs = _move_inputs_to_device(inputs, device)
                input_len = inputs["input_ids"].shape[1]
                gen_out = model.generate(**inputs, **gen_kwargs)
                seq = gen_out.sequences[0] if hasattr(gen_out, "sequences") else gen_out[0]
                gen_ids = seq[input_len:]
                raw = clean_answer(processor.decode(gen_ids, skip_special_tokens=True).strip())
                pred = parse_predicted_letter(raw, num_options=len(options))
                gold = correct_index_to_letter(correct_index)
                match = pred is not None and pred == gold
                if match:
                    n_ok += 1
                else:
                    n_fail += 1
                out["pseudo_reasoning_test"] = match
                out["pseudo_reasoning_verify_pred_letter"] = pred
                out["pseudo_reasoning_verify_raw"] = raw
                out_rows.append(out)

            if idx % args.save_every == 0:
                if args.no_csv:
                    _write_json_only(out_json, out_rows)
                else:
                    write_records(out_json, out_csv, out_rows)

    if args.no_csv:
        _write_json_only(out_json, out_rows)
    else:
        write_records(out_json, out_csv, out_rows)

    n_verified = n_ok + n_fail
    acc = n_ok / n_verified if n_verified else 0.0
    print(
        f"Rows={len(rows)} invalid_mc={n_invalid_mc} no_reasoning={n_no_reasoning} "
        f"verified={n_verified} pseudo_reasoning_test_true={n_ok} false={n_fail} accuracy={acc:.4f}"
    )
    print(f"Wrote {out_json}")
    if not args.no_csv:
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
