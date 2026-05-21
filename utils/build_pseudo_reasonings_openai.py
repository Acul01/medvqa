"""
Augment MC JSON (from build_mc_datasets_openai.py) with GPT-4o-mini multimodal pseudo rationales.

Reads ``{slug}_{split}_mc_openai.json``, writes ``{slug}_{split}_mc_openai_pseudo.json`` (+ CSV).
Load enriched data with ``MedicalVQALoader(mc_export_suffix="_pseudo")``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from utils.build_mc_datasets_openai import build_openai_client, write_records
from utils.load_data import MedicalVQALoader
from utils.mc_vqa import correct_index_to_letter
from utils.paths import get_data_root, get_mc_data_root
from utils.reasoning_mc import END_REASONING_TOKEN, parse_reasoning_fields

DATASETS = ["pathvqa", "slake", "vqarad", "medvqa2019"]

CANONICAL_NAME = {
    "pathvqa": "PathVQA",
    "slake": "SLAKE",
    "vqarad": "VQA-Rad",
    "medvqa2019": "Med-VQA-2019",
}


def infer_splits(loader: MedicalVQALoader, dataset_slug: str) -> List[str]:
    canonical = CANONICAL_NAME[dataset_slug]
    return list(loader.dataset_configs[canonical]["annotation_files"].keys())


def row_sample_key(row: Dict[str, Any]) -> str:
    return "||".join(
        [
            str(row.get("dataset", "")),
            str(row.get("split", "")),
            str(row.get("image_path", "")),
            str(row.get("question", "")).strip(),
            str(row.get("answer", "")).strip(),
        ]
    )


def parse_options(row: Dict[str, Any]) -> Tuple[List[str] | None, int | None]:
    options = row.get("options")
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except json.JSONDecodeError:
            return None, None
    if not isinstance(options, list) or len(options) < 2:
        return None, None
    try:
        correct_index = int(row.get("correct_index"))
    except (TypeError, ValueError):
        return None, None
    if correct_index < 0 or correct_index >= len(options):
        return None, None
    return [str(o) for o in options], correct_index


def image_to_data_url(path: Path, max_side: int | None) -> Tuple[str, str]:
    raw = path.read_bytes()
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"

    if max_side is not None and max_side > 0:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Resize requires Pillow. Install with: pip install pillow"
            ) from exc
        im = Image.open(io.BytesIO(raw))
        im = im.convert("RGB")
        w, h = im.size
        scale = min(max_side / float(w), max_side / float(h), 1.0)
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        raw = buf.getvalue()
        mime = "image/jpeg"

    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}", mime


def build_user_prompt(question: str, options: List[str], correct_index: int) -> str:
    letter_lines = [
        f"{correct_index_to_letter(i)}. {options[i]}" for i in range(len(options))
    ]
    letter = correct_index_to_letter(correct_index)
    gold_text = options[correct_index]
    return (
        "You are a medical imaging expert.\n\n"
        f"Multiple-choice question: {question}\n\n"
        "Options:\n"
        + "\n".join(letter_lines)
        + "\n\n"
        f"The correct answer is option {letter}: \"{gold_text}\".\n"
        "Write a plausible clinical rationale that supports this answer given what is visible in the image. "
        "Do not contradict the given correct option.\n\n"
        "Respond in exactly this format (three lines plus closing marker):\n"
        "REGION: [anatomical region relevant to the question]\n"
        "OBSERVATION: [what you observe in the image for that region]\n"
        "REASONING: [how the observation supports the correct option]\n\n"
        f"End your response with exactly this token: {END_REASONING_TOKEN}"
    )


def request_pseudo_reasoning(
    client: Any,
    model: str,
    data_url: str,
    user_text: str,
    max_retries: int,
    sleep_between_retries_s: float,
) -> str:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You write concise medical imaging rationales in a fixed parseable format."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": user_text},
                        ],
                    },
                ],
                temperature=0.3,
            )
            content = resp.choices[0].message.content if resp.choices else ""
            text = (content or "").strip()
            if not text:
                raise ValueError("OpenAI returned empty content")
            return text
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < max_retries:
                time.sleep(sleep_between_retries_s)
    raise RuntimeError(f"OpenAI pseudo reasoning failed after {max_retries} retries: {last_err}")


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in {path}")
    return payload


def enrich_row_with_pseudo_fields(
    row: Dict[str, Any],
    *,
    loader: MedicalVQALoader,
    client: Any,
    model: str,
    max_retries: int,
    sleep_between_retries_s: float,
    max_image_side: int | None,
) -> Dict[str, Any]:
    out = dict(row)
    dataset_name = str(row.get("dataset", "")).strip()
    question = str(row.get("question", "")).strip()

    options, correct_index = parse_options(row)
    if options is None or correct_index is None:
        out["pseudo_reasoning_error"] = "invalid_mc_row_skip"
        return out

    if not dataset_name or not question:
        out["pseudo_reasoning_error"] = "missing_dataset_or_question"
        return out

    try:
        img_path = loader.resolve_mc_image_path(dataset_name, str(row.get("image_path", "")))
        if not img_path.is_file():
            raise FileNotFoundError(str(img_path))
        data_url, _mime = image_to_data_url(img_path, max_image_side)
        user_prompt = build_user_prompt(question, options, correct_index)
        raw = request_pseudo_reasoning(
            client=client,
            model=model,
            data_url=data_url,
            user_text=user_prompt,
            max_retries=max_retries,
            sleep_between_retries_s=sleep_between_retries_s,
        )
        parsed, parse_mode = parse_reasoning_fields(raw)
        out["pseudo_reasoning_raw"] = raw
        out["pseudo_reasoning_region"] = parsed["region"]
        out["pseudo_reasoning_observation"] = parsed["observation"]
        out["pseudo_reasoning_explanation"] = parsed["reasoning"]
        out["pseudo_reasoning_parse_mode"] = parse_mode
        out.pop("pseudo_reasoning_error", None)
    except Exception as exc:  # noqa: BLE001
        out["pseudo_reasoning_error"] = str(exc)
        out.pop("pseudo_reasoning_raw", None)
        out.pop("pseudo_reasoning_region", None)
        out.pop("pseudo_reasoning_observation", None)
        out.pop("pseudo_reasoning_explanation", None)
        out.pop("pseudo_reasoning_parse_mode", None)

    return out


def process_dataset_split(
    *,
    loader: MedicalVQALoader,
    dataset_slug: str,
    split: str,
    mc_root: Path,
    client: Any,
    args: argparse.Namespace,
) -> None:
    canonical = CANONICAL_NAME[dataset_slug]
    in_json = mc_root / dataset_slug / f"{dataset_slug}_{split}_mc_openai.json"
    out_base = mc_root / dataset_slug / f"{dataset_slug}_{split}_mc_openai_pseudo"
    out_json = out_base.with_suffix(".json")
    out_csv = out_base.with_suffix(".csv")

    input_records = load_json_list(in_json)
    if not input_records:
        print(f"[{dataset_slug}/{split}] skip: missing or empty {in_json}")
        return

    existing_by_key: Dict[str, Dict[str, Any]] = {}
    if args.resume and out_json.exists():
        for prev in load_json_list(out_json):
            existing_by_key[row_sample_key(prev)] = prev

    records: List[Dict[str, Any]] = []
    n_skip = 0
    n_ok = 0
    n_err = 0

    limit = args.n_samples
    iterable = input_records[:limit] if limit is not None else input_records

    pbar = tqdm(iterable, total=len(iterable), desc=f"{dataset_slug}/{split}", unit="sample")
    for idx, row in enumerate(pbar, start=1):
        key = row_sample_key(row)
        merged = dict(row)

        if args.resume and key in existing_by_key:
            prev = existing_by_key[key]
            raw_prev = prev.get("pseudo_reasoning_raw")
            if isinstance(raw_prev, str) and raw_prev.strip():
                merged_resume = dict(row)
                for pk, pv in prev.items():
                    if pk.startswith("pseudo_"):
                        merged_resume[pk] = pv
                records.append(merged_resume)
                n_skip += 1
                continue

        merged = enrich_row_with_pseudo_fields(
            merged,
            loader=loader,
            client=client,
            model=args.openai_model,
            max_retries=args.max_retries,
            sleep_between_retries_s=args.sleep_between_retries_s,
            max_image_side=args.max_image_side,
        )
        records.append(merged)
        if merged.get("pseudo_reasoning_raw"):
            n_ok += 1
        elif merged.get("pseudo_reasoning_error"):
            n_err += 1

        if idx % args.save_every == 0:
            write_records(out_json, out_csv, records)

    write_records(out_json, out_csv, records)
    print(
        f"[{dataset_slug}/{split}] total_in={len(iterable)} saved={len(records)} "
        f"ok_raw={n_ok} errors={n_err} resume_skip={n_skip}"
    )
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add GPT-4o-mini multimodal pseudo rationales to MC JSON.")
    p.add_argument(
        "--data_root",
        type=str,
        default=get_data_root(),
        help="Raw images root ($MEDVQA_DATA_ROOT or 02_data)",
    )
    p.add_argument(
        "--mc_root",
        type=str,
        default=get_mc_data_root(),
        help="MC JSON root ($MEDVQA_MC_ROOT or 02_data/data_mc_openai); reads *_mc_openai.json",
    )
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    p.add_argument("--openai_api_key_env", type=str, default="OPENAI_API_KEY")
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--request_timeout_s", type=float, default=120.0)
    p.add_argument("--sleep_between_retries_s", type=float, default=2.0)
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument(
        "--max_image_side",
        type=int,
        default=None,
        help="If set, resize longest side to this many pixels (JPEG) before sending; requires Pillow.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.getenv(args.openai_api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Missing API key in env var '{args.openai_api_key_env}'. Export it before running."
        )
    client = build_openai_client(api_key=api_key, timeout_s=args.request_timeout_s)

    mc_root = Path(args.mc_root)
    loader = MedicalVQALoader(data_root=args.data_root, mc_data_root=args.mc_root)

    print("Pseudo rationales with:")
    print(f"  data_root={args.data_root}")
    print(f"  mc_root={args.mc_root}")
    print(f"  datasets={args.datasets}")
    print(f"  openai_model={args.openai_model}")
    print(f"  n_samples={args.n_samples}")
    print(f"  resume={args.resume}")
    print(f"  max_image_side={args.max_image_side}")

    for slug in args.datasets:
        splits = infer_splits(loader, slug)
        for split in splits:
            process_dataset_split(
                loader=loader,
                dataset_slug=slug,
                split=split,
                mc_root=mc_root,
                client=client,
                args=args,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
