"""
Build multiple-choice variants with OpenAI-generated distractors.

Rules:
- closed questions (GT == yes/no): options are strictly ["yes", "no"]
- open questions: 4 generated incorrect options + 1 GT option
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from utils.load_data import MedicalVQALoader, VQASample


DATASETS = ["pathvqa", "slake", "vqarad", "medvqa2019"]
CANONICAL_NAME = {
    "pathvqa": "PathVQA",
    "slake": "SLAKE",
    "vqarad": "VQA-Rad",
    "medvqa2019": "Med-VQA-2019",
}


def normalize_answer(text: str, strip_punctuation: bool = True) -> str:
    normalized = (text or "").strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    if strip_punctuation:
        normalized = normalized.translate(str.maketrans("", "", string.punctuation))
        normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def infer_splits(loader: MedicalVQALoader, dataset_slug: str) -> List[str]:
    canonical = CANONICAL_NAME[dataset_slug]
    return list(loader.dataset_configs[canonical]["annotation_files"].keys())


def sample_to_record(sample: VQASample) -> Dict:
    record = {
        "dataset": sample.dataset,
        "split": sample.split,
        "image_path": sample.image_path,
        "question": sample.question,
        "answer": sample.answer,
    }
    if sample.metadata:
        for key, value in sample.metadata.items():
            if key not in record:
                record[key] = value
    return record


def sample_key(sample: VQASample) -> str:
    return "||".join(
        [
            str(sample.dataset),
            str(sample.split),
            str(sample.image_path),
            str(sample.question).strip(),
            str(sample.answer).strip(),
        ]
    )


def detect_question_type(gt_answer: str) -> str:
    norm = normalize_answer(gt_answer, strip_punctuation=True)
    return "closed" if norm in {"yes", "no"} else "open"


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_json_object(raw_text: str) -> Dict:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("OpenAI returned empty content")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in model output: {text[:200]}")
        return json.loads(match.group(0))


def validate_distractors(distractors: List[str], gt_answer: str) -> List[str]:
    gt_norm = normalize_answer(gt_answer, strip_punctuation=True)
    cleaned: List[str] = []
    seen = set()
    for d in distractors:
        if not isinstance(d, str):
            continue
        c = d.strip()
        if not c:
            continue
        norm = normalize_answer(c, strip_punctuation=True)
        if norm == gt_norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        cleaned.append(c)
    if len(cleaned) != 4:
        raise ValueError(f"Expected 4 valid distractors, got {len(cleaned)}")
    return cleaned


def build_openai_prompt(question: str, gt_answer: str) -> str:
    return (
        f"Given this medical VQA question: '{question}'\n"
        f"The correct answer is: '{gt_answer}'\n"
        "Generate 4 plausible but incorrect answer options for a medical professional.\n"
        "They should be:\n"
        "- Clinically plausible (not obviously wrong)\n"
        "- Clearly distinct from the correct answer\n"
        "- Similar in format/length to the correct answer\n"
        "Return strictly valid JSON in this exact schema:\n"
        '{\"distractors\": [\"option1\", \"option2\", \"option3\", \"option4\"]}\n'
        "Return only JSON. No explanation, no markdown."
    )


def build_openai_client(api_key: str, timeout_s: float):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'openai'. Install with: pip install openai"
        ) from exc
    return OpenAI(api_key=api_key, timeout=timeout_s)


def request_openai_distractors(
    client,
    model: str,
    question: str,
    gt_answer: str,
    max_retries: int,
    sleep_between_retries_s: float,
) -> List[str]:
    prompt = build_openai_prompt(question=question, gt_answer=gt_answer)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You generate medical MC distractors in strict JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content if resp.choices else ""
            parsed = parse_json_object(content)
            if "distractors" not in parsed or not isinstance(parsed["distractors"], list):
                raise ValueError("JSON must contain list field 'distractors'")
            return validate_distractors(parsed["distractors"], gt_answer=gt_answer)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < max_retries:
                time.sleep(sleep_between_retries_s)
    raise RuntimeError(f"OpenAI distractor generation failed after {max_retries} retries: {last_err}")


def load_existing_records(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_records(json_path: Path, csv_path: Path, records: List[Dict]) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    df = pd.DataFrame(records)
    if "options" in df.columns:
        df["options"] = df["options"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    df.to_csv(csv_path, index=False, encoding="utf-8")


def save_cache_jsonl(cache_path: Path, cache: Dict[str, List[str]]) -> None:
    ensure_output_dir(cache_path.parent)
    with open(cache_path, "w", encoding="utf-8") as f:
        for key, distractors in cache.items():
            row = {"key": key, "distractors": distractors}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_cache_jsonl(cache_path: Path) -> Dict[str, List[str]]:
    if not cache_path.exists():
        return {}
    out: Dict[str, List[str]] = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = str(row.get("key", ""))
            distractors = row.get("distractors", [])
            if key and isinstance(distractors, list):
                out[key] = [str(x) for x in distractors]
    return out


def build_mc_for_dataset(args, loader: MedicalVQALoader, dataset_slug: str, client) -> None:
    print(f"\n=== Dataset: {dataset_slug} ===")
    splits = infer_splits(loader, dataset_slug)
    dataset_out = Path(args.output_root) / dataset_slug
    ensure_output_dir(dataset_out)

    cache_path = dataset_out / "openai_distractor_cache.jsonl"
    cache = load_cache_jsonl(cache_path)
    print(f"Loaded cache entries: {len(cache)}")

    rng = random.Random(args.seed)

    for split in splits:
        samples = loader.load_dataset(dataset_slug, split)
        if args.n_samples is not None:
            samples = samples[: args.n_samples]

        json_path = dataset_out / f"{dataset_slug}_{split}_mc_openai.json"
        csv_path = dataset_out / f"{dataset_slug}_{split}_mc_openai.csv"
        existing_records = load_existing_records(json_path) if args.resume else []
        records = list(existing_records)
        done_keys = set()
        if args.resume:
            for r in existing_records:
                k = "||".join(
                    [
                        str(r.get("dataset", "")),
                        str(r.get("split", "")),
                        str(r.get("image_path", "")),
                        str(r.get("question", "")).strip(),
                        str(r.get("answer", "")).strip(),
                    ]
                )
                done_keys.add(k)

        open_calls = 0
        closed_count = 0
        open_count = 0
        errors = 0

        pbar = tqdm(samples, total=len(samples), desc=f"{dataset_slug}/{split}", unit="sample")
        for idx, sample in enumerate(pbar, start=1):
            key = sample_key(sample)
            if key in done_keys:
                continue

            base = sample_to_record(sample)
            q_type = detect_question_type(str(sample.answer))
            base["question_type"] = q_type

            try:
                if q_type == "closed":
                    closed_count += 1
                    options = ["yes", "no"]
                    gt_norm = normalize_answer(str(sample.answer))
                    correct_index = 0 if gt_norm == "yes" else 1
                else:
                    open_count += 1
                    cache_key = "||".join(
                        [dataset_slug, split, str(sample.question).strip(), str(sample.answer).strip()]
                    )
                    if cache_key in cache:
                        distractors = cache[cache_key]
                    else:
                        distractors = request_openai_distractors(
                            client=client,
                            model=args.openai_model,
                            question=str(sample.question),
                            gt_answer=str(sample.answer),
                            max_retries=args.max_retries,
                            sleep_between_retries_s=args.sleep_between_retries_s,
                        )
                        cache[cache_key] = distractors
                        open_calls += 1

                    options = list(distractors) + [str(sample.answer)]
                    rng.shuffle(options)
                    gt_norm = normalize_answer(str(sample.answer))
                    matches = [i for i, opt in enumerate(options) if normalize_answer(opt) == gt_norm]
                    if len(matches) != 1:
                        raise ValueError("GT must appear exactly once in options")
                    correct_index = matches[0]

                base["options"] = options
                base["correct_index"] = int(correct_index)
                records.append(base)
                done_keys.add(key)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                err_record = dict(base)
                err_record["options"] = []
                err_record["correct_index"] = -1
                err_record["mc_error"] = str(exc)
                records.append(err_record)
                done_keys.add(key)

            if idx % args.save_every == 0:
                write_records(json_path, csv_path, records)
                save_cache_jsonl(cache_path, cache)

        write_records(json_path, csv_path, records)
        save_cache_jsonl(cache_path, cache)
        print(
            f"[{dataset_slug}/{split}] total={len(samples)} saved={len(records)} "
            f"closed={closed_count} open={open_count} openai_calls={open_calls} errors={errors}"
        )
        print(f"Saved: {json_path}")
        print(f"Saved: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MC datasets with OpenAI-generated distractors.")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_root", type=str, default="data_mc_openai")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openai_api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--request_timeout_s", type=float, default=60.0)
    parser.add_argument("--sleep_between_retries_s", type=float, default=1.5)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--save_every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dir(Path(args.output_root))

    # Load environment variables from .env if present.
    load_dotenv()

    api_key = os.getenv(args.openai_api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Missing API key in env var '{args.openai_api_key_env}'. "
            f"Please export it before running."
        )
    client = build_openai_client(api_key=api_key, timeout_s=args.request_timeout_s)

    print("Building OpenAI MC datasets with settings:")
    print(f"  data_root={args.data_root}")
    print(f"  output_root={args.output_root}")
    print(f"  datasets={args.datasets}")
    print(f"  openai_model={args.openai_model}")
    print(f"  n_samples={args.n_samples}")
    print(f"  resume={args.resume}")
    print(f"  save_every={args.save_every}")

    loader = MedicalVQALoader(data_root=args.data_root)
    for dataset_slug in args.datasets:
        build_mc_for_dataset(args=args, loader=loader, dataset_slug=dataset_slug, client=client)

    print("\nDone.")


if __name__ == "__main__":
    main()
