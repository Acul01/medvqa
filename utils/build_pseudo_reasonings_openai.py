"""
Generate pseudo REGION/OBSERVATION/REASONING annotations for MC training samples via OpenAI vision.

Key design choices vs. the original script:
- Gold label (correct answer + all MC options) is passed to GPT so the reasoning explicitly
  justifies the correct answer and rules out distractors. No verification step needed.
- N reasoning variants are generated per sample (default 5) using OpenAI's ``n`` parameter
  with temperature > 0 for diversity.  Each variant becomes a separate training row.
- Output is written to a new file (default suffix ``_pseudo_pr_qwen_verify``) so the source
  MC JSON is never overwritten.

Output JSON rows are identical to the source MC rows but add:
    pseudo_reasoning_raw        str   raw GPT output (REGION/OBS/REASON + <END_REASONING>)
    pseudo_region               str   parsed REGION field
    pseudo_observation          str   parsed OBSERVATION field
    pseudo_reasoning            str   parsed REASONING field
    pseudo_reasoning_parse_mode str
    pseudo_reasoning_openai_model str
    pseudo_reasoning_generated_at str
    pseudo_reasoning_index      int   0 … n_reasonings-1
    pseudo_reasoning_total      int   n_reasonings
    pseudo_reasoning_test       bool  True (gold-label guided → no separate verify needed)
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm

from utils.build_mc_datasets_openai import (
    DATASETS,
    ensure_output_dir,
    infer_splits,
    load_existing_records,
)
from utils.load_data import MedicalVQALoader
from utils.reasoning_mc import build_reasoning_prompt, parse_reasoning_fields


# ── helpers ──────────────────────────────────────────────────────────────────


def record_key(row: Dict[str, Any]) -> str:
    return "||".join(
        [
            str(row.get("dataset", "")),
            str(row.get("split", "")),
            str(row.get("image_path", "")),
            str(row.get("question", "")).strip(),
            str(row.get("answer", "")).strip(),
        ]
    )


def image_path_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")
    b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def parse_options_cell(row: Dict[str, Any]) -> Optional[List[str]]:
    options = row.get("options")
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except json.JSONDecodeError:
            return None
    if not isinstance(options, list) or len(options) < 2:
        return None
    return [str(x).strip() for x in options]


def correct_answer_text(row: Dict[str, Any]) -> Optional[str]:
    """Return the text of the correct MC option."""
    options = parse_options_cell(row)
    try:
        ci = int(row.get("correct_index", -1))
    except (TypeError, ValueError):
        return None
    if options is None or ci < 0 or ci >= len(options):
        return None
    return options[ci]


def build_openai_client(api_key: str, timeout_s: float):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install openai: pip install openai") from exc
    return OpenAI(api_key=api_key, timeout=timeout_s)


# ── cache ─────────────────────────────────────────────────────────────────────


def load_cache(cache_path: Path) -> Dict[str, List[str]]:
    """Return {record_key: [raw_0, raw_1, ...]}."""
    if not cache_path.exists():
        return {}
    out: Dict[str, List[str]] = {}
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            key = str(entry.get("key", ""))
            raws = entry.get("reasonings", [])
            if key and isinstance(raws, list):
                out[key] = [str(r) for r in raws]
    return out


def save_cache(cache_path: Path, cache: Dict[str, List[str]]) -> None:
    ensure_output_dir(cache_path.parent)
    with open(cache_path, "w", encoding="utf-8") as f:
        for key in sorted(cache.keys()):
            entry = {"key": key, "reasonings": cache[key]}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── OpenAI call ───────────────────────────────────────────────────────────────


def request_n_reasonings(
    client: Any,
    model: str,
    image_path: Path,
    question: str,
    options: List[str],
    correct_answer: str,
    *,
    n: int,
    temperature: float,
    max_completion_tokens: int,
    image_detail: str,
    max_retries: int,
    sleep_between_retries_s: float,
) -> List[str]:
    """Return exactly ``n`` raw reasoning strings from a single API call."""
    data_url = image_path_to_data_url(image_path)
    user_text = build_reasoning_prompt(
        question,
        correct_answer=correct_answer,
    )
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url, "detail": image_detail},
                            },
                        ],
                    }
                ],
                n=n,
                temperature=temperature,
                max_tokens=max_completion_tokens,
            )
            raws = []
            for choice in resp.choices:
                content = (choice.message.content or "").strip()
                if content:
                    raws.append(content)
            if not raws:
                raise ValueError("All choices returned empty content")
            # Pad with last entry if fewer than n were returned
            while len(raws) < n:
                raws.append(raws[-1])
            return raws[:n]
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < max_retries:
                time.sleep(sleep_between_retries_s)
    raise RuntimeError(f"OpenAI request failed after {max_retries} retries: {last_err}")


# ── row expansion ─────────────────────────────────────────────────────────────


def expand_row(base_row: Dict[str, Any], raws: List[str], model: str) -> List[Dict[str, Any]]:
    """Return one dict per reasoning variant, each a copy of base_row + reasoning fields."""
    generated_at = datetime.now(timezone.utc).isoformat()
    expanded: List[Dict[str, Any]] = []
    for idx, raw in enumerate(raws):
        row = copy.deepcopy(base_row)
        fields, parse_mode = parse_reasoning_fields(raw)
        row["pseudo_reasoning_raw"] = raw
        row["pseudo_region"] = fields["region"]
        row["pseudo_observation"] = fields["observation"]
        row["pseudo_reasoning"] = fields["reasoning"]
        row["pseudo_reasoning_parse_mode"] = parse_mode
        row["pseudo_reasoning_openai_model"] = model
        row["pseudo_reasoning_generated_at"] = generated_at
        row["pseudo_reasoning_index"] = idx
        row["pseudo_reasoning_total"] = len(raws)
        # Gold-label guided → no separate verification step needed
        row["pseudo_reasoning_test"] = True
        row.pop("pseudo_reasoning_error", None)
        expanded.append(row)
    return expanded


# ── write helpers ─────────────────────────────────────────────────────────────


def write_expanded_json(path: Path, records: List[Dict[str, Any]]) -> None:
    ensure_output_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# ── core processing ───────────────────────────────────────────────────────────


def resolve_record_image_path(loader: MedicalVQALoader, row: Dict[str, Any]) -> Path:
    dataset_name = loader._normalize_dataset_name(str(row.get("dataset", "")))
    config = loader.dataset_configs[dataset_name]
    return loader._resolve_image_path(
        dataset_name=dataset_name,
        config=config,
        image_path=str(row.get("image_path", "")),
    )


def _splits_for_dataset(args: argparse.Namespace, loader: MedicalVQALoader, slug: str) -> List[str]:
    available = infer_splits(loader, slug)
    if not args.split:
        return available
    unknown = [s for s in args.split if s not in available]
    if unknown:
        raise ValueError(f"Unknown --split {unknown!r} for {slug!r}. Valid: {available}")
    return list(args.split)


def process_dataset(args: argparse.Namespace, loader: MedicalVQALoader, slug: str) -> None:
    splits = _splits_for_dataset(args, loader, slug)
    in_dir = Path(args.mc_data_root) / slug
    out_root = Path(args.output_dir) if args.output_dir else Path(args.mc_data_root)
    out_dir = out_root / slug
    ensure_output_dir(out_dir)

    cache_path = out_dir / f"pseudo_reasoning_cache{args.output_suffix}.jsonl"
    cache: Dict[str, List[str]] = load_cache(cache_path) if args.resume else {}

    client = build_openai_client(api_key=args._api_key, timeout_s=args.request_timeout_s)

    for split in splits:
        json_in = in_dir / f"{slug}_{split}_mc_openai.json"
        # Output: separate file so source MC JSON is never modified
        json_out = out_dir / f"{slug}_{split}_mc_openai{args.output_suffix}.json"

        if not json_in.exists():
            print(f"Skip {slug}/{split}: missing {json_in}")
            continue

        source_records = load_existing_records(json_in)
        n_total = len(source_records)

        # Load already-expanded output to resume
        existing_keys: set[str] = set()
        out_records: List[Dict[str, Any]] = []
        if args.resume and json_out.exists():
            out_records = load_existing_records(json_out)
            existing_keys = {record_key(r) for r in out_records}
            print(f"  Resuming: {len(out_records)} expanded rows already in {json_out.name}")

        n_api = n_cache = n_skip = n_err = n_limit = 0

        for idx, row in enumerate(tqdm(source_records, desc=f"{slug}/{split}", unit="sample")):
            if args.n_samples is not None and idx >= args.n_samples:
                n_limit += 1
                continue

            # Skip rows without valid MC data
            try:
                ci = int(row.get("correct_index", -1))
            except (TypeError, ValueError):
                ci = -1
            options = parse_options_cell(row)
            answer_text = correct_answer_text(row)
            if ci < 0 or options is None or answer_text is None:
                n_skip += 1
                continue

            key = record_key(row)

            # Already processed in a previous run
            if args.resume and key in existing_keys:
                n_skip += 1
                continue

            # Cache hit
            if args.resume and key in cache and not args.overwrite:
                raws = cache[key]
                out_records.extend(expand_row(row, raws, args.openai_model))
                existing_keys.add(key)
                n_cache += 1
                continue

            # API call
            try:
                img_path = resolve_record_image_path(loader, row)
                if not img_path.is_file():
                    raise FileNotFoundError(f"Image not found: {img_path}")
                raws = request_n_reasonings(
                    client=client,
                    model=args.openai_model,
                    image_path=img_path,
                    question=str(row.get("question", "")),
                    options=options,
                    correct_answer=answer_text,
                    n=args.n_reasonings,
                    temperature=args.temperature,
                    max_completion_tokens=args.max_completion_tokens,
                    image_detail=args.image_detail,
                    max_retries=args.max_retries,
                    sleep_between_retries_s=args.sleep_between_retries_s,
                )
                cache[key] = raws
                out_records.extend(expand_row(row, raws, args.openai_model))
                existing_keys.add(key)
                n_api += 1
            except Exception as exc:  # noqa: BLE001
                err_row = copy.deepcopy(row)
                err_row["pseudo_reasoning_error"] = str(exc)
                out_records.append(err_row)
                n_err += 1

            if (idx + 1) % args.save_every == 0:
                write_expanded_json(json_out, out_records)
                save_cache(cache_path, cache)

        write_expanded_json(json_out, out_records)
        save_cache(cache_path, cache)

        extra = f" (limit first {args.n_samples})" if args.n_samples else ""
        print(
            f"[{slug}/{split}]{extra} source={n_total} "
            f"api={n_api} cache={n_cache} skipped={n_skip} errors={n_err} "
            f"→ {len(out_records)} expanded rows → {json_out}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate N pseudo reasonings per MC sample via OpenAI vision (gold-label guided). "
            "Output is written to a separate file; source MC JSON is never modified."
        )
    )
    p.add_argument("--data_root", type=str, default="02_data")
    p.add_argument(
        "--mc_data_root",
        type=str,
        default="02_data/data_mc_openai",
        help="Directory containing *_mc_openai.json per dataset subfolder.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Write expanded JSON and cache here. Default: same as --mc_data_root.",
    )
    p.add_argument(
        "--output_suffix",
        type=str,
        default="_pseudo_pr_qwen_verify",
        help="Suffix appended before .json in output filename (default: _pseudo_pr_qwen_verify).",
    )
    p.add_argument(
        "--split",
        nargs="+",
        default=None,
        metavar="SPLIT",
        help="Process only these splits (e.g. train). Default: all splits.",
    )
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    p.add_argument("--openai_api_key_env", type=str, default="OPENAI_API_KEY")
    p.add_argument(
        "--n_reasonings",
        type=int,
        default=5,
        help="Number of reasoning variants to generate per sample (OpenAI n parameter).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for diversity (must be > 0 when n > 1).",
    )
    p.add_argument("--max_completion_tokens", type=int, default=512)
    p.add_argument("--image_detail", type=str, default="low", choices=["low", "high", "auto"])
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--request_timeout_s", type=float, default=120.0)
    p.add_argument("--sleep_between_retries_s", type=float, default=2.0)
    p.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Process only the first N source rows per split.",
    )
    p.add_argument("--resume", action="store_true", help="Skip already-processed rows using cache.")
    p.add_argument("--overwrite", action="store_true", help="Regenerate even if cached.")
    p.add_argument("--save_every", type=int, default=25)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.n_reasonings > 1 and args.temperature == 0.0:
        print("Warning: temperature=0 with n>1 produces identical reasonings. Consider temperature>=0.5.")

    api_key = os.getenv(args.openai_api_key_env)
    if not api_key:
        raise EnvironmentError(f"Missing API key in env var '{args.openai_api_key_env}'.")
    args._api_key = api_key

    print(f"Generating {args.n_reasonings} reasoning(s) per sample with {args.openai_model}")
    print(f"  temperature={args.temperature}  output_suffix={args.output_suffix!r}")
    if args.output_dir:
        print(f"  output_dir={args.output_dir}")
    if args.split:
        print(f"  splits={args.split}")

    loader = MedicalVQALoader(data_root=args.data_root, mc_data_root=args.mc_data_root)
    for slug in args.datasets:
        process_dataset(args, loader, slug)
    print("Done.")


if __name__ == "__main__":
    main()
