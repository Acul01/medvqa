"""
Generate pseudo REGION/OBSERVATION/REASONING annotations for MC training samples.

Backends (selected via ``--model``):
- ``gpt-4o-mini`` — OpenAI vision API (``n`` completions per call)
- ``qwen3-vl-8b`` — local Qwen3-VL-8B-Instruct (``n`` sampled generations)

- Gold label (correct answer text) is passed so the reasoning justifies that answer.
- N reasoning variants per sample; each variant is a separate row.
- Output suffix default ``_pseudo`` → e.g. ``vqarad_train_mc_openai_pseudo.json``.
- Run ``04_evaluation/verify_pseudo_reasoning_qwen.py`` afterward to set ``qwen_verify``
  and write ``*_mc_openai_pseudo_pr_qwen_verify.json`` for ``train_qwen --use_reasoning``.

Output fields per expanded row (among others):
    pseudo_reasoning_raw, pseudo_region, pseudo_observation, pseudo_reasoning
    pseudo_reasoning_index, pseudo_reasoning_total
    (qwen_verify is set by the Qwen verify step, not here)
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from utils.build_mc_datasets_openai import (
    DATASETS,
    ensure_output_dir,
    infer_splits,
    load_existing_records,
)
from utils.load_data import MedicalVQALoader
from utils.paths import get_som_dir, get_som_selection_dir
from utils.reasoning_mc import build_reasoning_prompt, parse_reasoning_fields
from utils.som_selection import SomSelectionResolver


MODEL_CHOICES = {
    "gpt-4o-mini": {"backend": "openai", "id": "gpt-4o-mini"},
    "qwen3-vl-8b": {"backend": "qwen", "id": "Qwen/Qwen3-VL-8B-Instruct"},
}

_DTYPE_MAP = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
}


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


def pil_image_to_data_url(image: Any) -> str:
    """Data URL for an in-memory image; SoM renders have no file on disk."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    b64 = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


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


def load_qwen_bundle(model_id: str, dtype_name: str) -> Tuple[Any, Any, Any]:
    """Load Qwen VL model + processor. Returns (model, processor, device)."""
    import torch
    from transformers import AutoProcessor

    from utils.qwen_model import load_qwen_vl_model
    from utils.qwen_vl import configure_qwen_processor_pixels, resolve_processor_path

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype_name]
    model = load_qwen_vl_model(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()
    processor_path = resolve_processor_path(model_id)
    processor = AutoProcessor.from_pretrained(processor_path)
    configure_qwen_processor_pixels(processor)
    device = next(model.parameters()).device
    return model, processor, device


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
    image: Any = None,
    som_labels: Optional[List[str]] = None,
) -> List[str]:
    """Return exactly ``n`` raw reasoning strings from a single API call.

    *image* overrides *image_path* when the caller has a rendered image in
    memory (SoM-filtered marks).
    """
    _ = options  # gold answer text is used in the prompt; options kept for call-site parity
    data_url = (
        pil_image_to_data_url(image) if image is not None else image_path_to_data_url(image_path)
    )
    user_text = build_reasoning_prompt(
        question,
        correct_answer=correct_answer,
        som_labels=som_labels,
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


# ── Qwen call ─────────────────────────────────────────────────────────────────


def request_n_reasonings_qwen(
    model: Any,
    processor: Any,
    device: Any,
    image_path: Path,
    question: str,
    correct_answer: str,
    *,
    n: int,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
    sleep_between_retries_s: float,
    image: Any = None,
    som_labels: Optional[List[str]] = None,
) -> List[str]:
    """Return exactly ``n`` raw reasoning strings via local Qwen VL sampling.

    *image* overrides *image_path* when the caller has a rendered image in
    memory (SoM-filtered marks).
    """
    import torch
    from PIL import Image

    from utils.qwen_vl import prepare_qwen_inputs

    user_text = build_reasoning_prompt(
        question, correct_answer=correct_answer, som_labels=som_labels
    )
    if image is None:
        image = Image.open(image_path).convert("RGB")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        }
    ]

    do_sample = temperature > 0.0
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    raws: List[str] = []
    last_err: Optional[Exception] = None
    for _ in range(n):
        content: Optional[str] = None
        for attempt in range(1, max_retries + 1):
            try:
                inputs = prepare_qwen_inputs(
                    processor,
                    conversation,
                    add_generation_prompt=True,
                )
                inputs = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }
                prompt_len = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kwargs)
                new_ids = out[0, prompt_len:]
                content = processor.tokenizer.decode(
                    new_ids, skip_special_tokens=True
                ).strip()
                if not content:
                    raise ValueError("Qwen returned empty content")
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                content = None
                if attempt < max_retries:
                    time.sleep(sleep_between_retries_s)
        if content is None:
            raise RuntimeError(
                f"Qwen generation failed after {max_retries} retries: {last_err}"
            )
        raws.append(content)

    while len(raws) < n:
        raws.append(raws[-1])
    return raws[:n]


# ── row expansion ─────────────────────────────────────────────────────────────


def expand_row(
    base_row: Dict[str, Any],
    raws: List[str],
    model: str,
    *,
    som_selected_labels: Optional[List[str]] = None,
    som_selector_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
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
        if som_selected_labels is not None:
            row["som_selected_labels"] = som_selected_labels
            row["som_selector_model"] = som_selector_model
        row.pop("pseudo_reasoning_test", None)
        row.pop("qwen_verify", None)
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


def som_entry_for_row(
    som_selection: Optional[SomSelectionResolver],
    loader: MedicalVQALoader,
    row: Dict[str, Any],
) -> Dict[str, Any]:
    """Selection manifest entry for a row; empty dict when unavailable."""
    if som_selection is None:
        return {}
    try:
        img_path = resolve_record_image_path(loader, row)
    except (KeyError, ValueError):
        return {}
    entry = som_selection.get_entry(str(img_path), str(row.get("question", "")))
    return entry or {}


def _splits_for_dataset(args: argparse.Namespace, loader: MedicalVQALoader, slug: str) -> List[str]:
    available = infer_splits(loader, slug)
    if not args.split:
        return available
    unknown = [s for s in args.split if s not in available]
    if unknown:
        raise ValueError(f"Unknown --split {unknown!r} for {slug!r}. Valid: {available}")
    return list(args.split)


def process_dataset(
    args: argparse.Namespace,
    loader: MedicalVQALoader,
    slug: str,
    *,
    openai_client: Any = None,
    qwen_bundle: Optional[Tuple[Any, Any, Any]] = None,
) -> None:
    splits = _splits_for_dataset(args, loader, slug)
    in_dir = Path(args.mc_data_root) / slug
    out_root = Path(args.output_dir) if args.output_dir else Path(args.mc_data_root)
    out_dir = out_root / slug
    ensure_output_dir(out_dir)

    cache_path = out_dir / f"pseudo_reasoning_cache{args.output_suffix}.jsonl"
    cache: Dict[str, List[str]] = load_cache(cache_path) if args.resume else {}

    model_choice = args.model
    backend = MODEL_CHOICES[model_choice]["backend"]
    backend_model_id = MODEL_CHOICES[model_choice]["id"]

    som_selection: Optional[SomSelectionResolver] = None
    if args.use_som_selection:
        som_selection = SomSelectionResolver(args.som_dir, args.selection_dir)
        som_selection.load_dataset(slug)
        print(f"  SoM selection enabled — filtered marks from {args.selection_dir}/{slug}")

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

            som_entry = som_entry_for_row(som_selection, loader, row)
            som_labels = (
                [str(label) for label in som_entry.get("selected_labels", [])]
                if som_selection is not None
                else None
            )
            som_selector_model = som_entry.get("selector_model")

            # Cache hit
            if args.resume and key in cache and not args.overwrite:
                raws = cache[key]
                out_records.extend(
                    expand_row(
                        row,
                        raws,
                        model_choice,
                        som_selected_labels=som_labels,
                        som_selector_model=som_selector_model,
                    )
                )
                existing_keys.add(key)
                n_cache += 1
                continue

            # Generation call
            try:
                img_path = resolve_record_image_path(loader, row)
                if not img_path.is_file():
                    raise FileNotFoundError(f"Image not found: {img_path}")

                som_image = None
                if som_selection is not None:
                    # The drawn labels, not the manifest's, are what the generator sees.
                    som_image, som_labels = som_selection.render_with_labels(
                        str(img_path), str(row.get("question", ""))
                    )

                if backend == "openai":
                    raws = request_n_reasonings(
                        client=openai_client,
                        model=backend_model_id,
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
                        image=som_image,
                        som_labels=som_labels,
                    )
                else:
                    assert qwen_bundle is not None
                    qwen_model, qwen_processor, qwen_device = qwen_bundle
                    raws = request_n_reasonings_qwen(
                        model=qwen_model,
                        processor=qwen_processor,
                        device=qwen_device,
                        image_path=img_path,
                        question=str(row.get("question", "")),
                        correct_answer=answer_text,
                        n=args.n_reasonings,
                        temperature=args.temperature,
                        max_new_tokens=args.max_completion_tokens,
                        max_retries=args.max_retries,
                        sleep_between_retries_s=args.sleep_between_retries_s,
                        image=som_image,
                        som_labels=som_labels,
                    )
                cache[key] = raws
                out_records.extend(
                    expand_row(
                        row,
                        raws,
                        model_choice,
                        som_selected_labels=som_labels,
                        som_selector_model=som_selector_model,
                    )
                )
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
            "Generate N pseudo reasonings per MC sample (gold-label guided). "
            "Backend via --model: gpt-4o-mini (OpenAI) or qwen3-vl-8b (local HF). "
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
        default="_pseudo",
        help="Suffix appended before .json in output filename (default: _pseudo). Verify writes _pseudo_pr_qwen_verify.",
    )
    p.add_argument(
        "--split",
        nargs="+",
        default=None,
        metavar="SPLIT",
        help="Process only these splits (e.g. train). Default: all splits.",
    )
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        choices=sorted(MODEL_CHOICES.keys()),
        help="Generator model: gpt-4o-mini (OpenAI API) or qwen3-vl-8b (local HF).",
    )
    p.add_argument("--openai_api_key_env", type=str, default="OPENAI_API_KEY")
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=sorted(_DTYPE_MAP.keys()),
        help="Qwen model dtype (ignored for gpt-4o-mini). Default: bfloat16.",
    )
    p.add_argument(
        "--n_reasonings",
        type=int,
        default=5,
        help="Number of reasoning variants to generate per sample.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for diversity (must be > 0 when n > 1).",
    )
    p.add_argument(
        "--max_completion_tokens",
        type=int,
        default=512,
        help="Max new tokens per reasoning (OpenAI max_tokens / Qwen max_new_tokens).",
    )
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
    p.add_argument(
        "--use_som_selection",
        action="store_true",
        help="Generate on images showing only the SoM segments pre-selected for the "
        "question, and require the reasoning to cite their numbers. "
        "Run scripts/precompute_som_selection.py first.",
    )
    p.add_argument(
        "--som_dir",
        type=str,
        default=get_som_dir(),
        help="Root of precompute_som.py outputs (default: MEDVQA_SOM_DIR or 02_data/som).",
    )
    p.add_argument(
        "--selection_dir",
        type=str,
        default=get_som_selection_dir(),
        help="Root of selection manifests (default: MEDVQA_SOM_SELECTION_DIR or 02_data/som_selection).",
    )
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.n_reasonings > 1 and args.temperature == 0.0:
        print("Warning: temperature=0 with n>1 produces identical reasonings. Consider temperature>=0.5.")

    backend = MODEL_CHOICES[args.model]["backend"]
    backend_id = MODEL_CHOICES[args.model]["id"]

    openai_client = None
    qwen_bundle = None
    if backend == "openai":
        api_key = os.getenv(args.openai_api_key_env)
        if not api_key:
            raise EnvironmentError(f"Missing API key in env var '{args.openai_api_key_env}'.")
        openai_client = build_openai_client(api_key=api_key, timeout_s=args.request_timeout_s)
    else:
        print(f"Loading Qwen model {backend_id} (dtype={args.dtype})...")
        qwen_bundle = load_qwen_bundle(backend_id, args.dtype)

    print(f"Generating {args.n_reasonings} reasoning(s) per sample with {args.model}")
    print(f"  temperature={args.temperature}  output_suffix={args.output_suffix!r}")
    if args.use_som_selection:
        print(f"  SoM selection: {args.selection_dir} (reasoning cites the marked regions)")
        if args.output_suffix == "_pseudo":
            print(
                "  Warning: --output_suffix is still '_pseudo'; pass e.g. '_pseudo_som' "
                "to keep SoM and non-SoM datasets apart."
            )
    if args.output_dir:
        print(f"  output_dir={args.output_dir}")
    if args.split:
        print(f"  splits={args.split}")

    loader = MedicalVQALoader(data_root=args.data_root, mc_data_root=args.mc_data_root)
    for slug in args.datasets:
        process_dataset(
            args,
            loader,
            slug,
            openai_client=openai_client,
            qwen_bundle=qwen_bundle,
        )
    print("Done.")


if __name__ == "__main__":
    main()
