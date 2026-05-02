"""
Build unique train vocabulary from SLAKE + VQA-Rad + Med-VQA-2019 (free-text),
remove English stopwords, classify each remaining token with OpenAI as
"localizable in a medical image" (visual grounding) or not.

Uses legacy annotations: MedicalVQALoader(..., mc_data_root=None).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from utils.load_data import MedicalVQALoader, VQASample

DATASET_SLUGS = ("slake", "vqarad", "medvqa2019")
SPLIT = "train"

# Words: letters, digits inside word, hyphenated compounds; exclude pure digits.
TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9]*(?:-[a-zA-Z0-9]+)*", re.UNICODE)

MIN_TOKEN_LEN = 2

# Small fallback if NLTK is not installed
_FALLBACK_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "from", "by",
    "for", "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "of", "in", "on", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare", "ought", "used",
    "it", "its", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "what", "which", "who", "whom", "whose", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "as", "also", "there", "here",
}


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


def load_english_stopwords() -> Set[str]:
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except LookupError:
        try:
            import nltk

            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords

            return set(stopwords.words("english"))
        except Exception:  # noqa: BLE001
            return set(_FALLBACK_STOPWORDS)
    except ImportError:
        return set(_FALLBACK_STOPWORDS)


def load_extra_stopwords(path: Path | None) -> Set[str]:
    if path is None or not path.exists():
        return set()
    out: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                out.add(w)
    return out


def extract_tokens(text: str) -> List[str]:
    if not text:
        return []
    return TOKEN_PATTERN.findall(text)


def collect_train_samples(data_root: str) -> List[VQASample]:
    loader = MedicalVQALoader(data_root=data_root, mc_data_root=None)
    samples: List[VQASample] = []
    for slug in DATASET_SLUGS:
        samples.extend(loader.load_dataset(slug, split=SPLIT))
    return samples


def build_vocabulary(samples: List[VQASample]) -> Tuple[List[str], Dict[str, str]]:
    """
    Returns sorted unique tokens (lowercase) and token -> first-seen display string.
    """
    seen: Set[str] = set()
    display: Dict[str, str] = {}
    for s in samples:
        blob = f"{s.question or ''} {s.answer or ''}"
        for raw in extract_tokens(blob):
            t = raw.lower()
            if len(t) < MIN_TOKEN_LEN:
                continue
            if t not in seen:
                seen.add(t)
                display[t] = raw if raw != raw.lower() else t
    ordered = sorted(seen)
    return ordered, display


def filter_stopwords(tokens: List[str], stop: Set[str]) -> List[str]:
    return [t for t in tokens if t not in stop]


def build_openai_client(api_key: str, timeout_s: float):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Missing dependency 'openai'. Install with: pip install openai") from exc
    return OpenAI(api_key=api_key, timeout=timeout_s)


def classification_system_prompt() -> str:
    return (
        "You classify medical vocabulary tokens for visual grounding in radiology / "
        "medical imaging. A token is LOCALIZABLE if a clinician could typically point to "
        "a region, structure, lesion, or pattern in an image (bbox or segmentation). "
        "NOT localizable: abstract concepts, relations, quantities without visual correlate, "
        "pure grammar, modality-only words when not tied to a visible structure in isolation, "
        "or yes/no style discourse. Return only strict JSON as requested."
    )


def classification_user_prompt(batch_tokens: List[str]) -> str:
    joined = ", ".join(json.dumps(t) for t in batch_tokens)
    return (
        "Classify each of the following vocabulary tokens from medical VQA training text.\n"
        f"Tokens: [{joined}]\n\n"
        "Return a single JSON object with this exact shape:\n"
        '{"results": [{"token": "<exact lowercase token from the list>", '
        '"localizable": true or false, "reason": "<short English phrase>"}]}\n'
        "Rules:\n"
        "- One result object per input token, same spelling (lowercase) as in the list.\n"
        "- localizable=true only if the term is typically visually localizable in medical images.\n"
        "- Return only JSON, no markdown."
    )


def validate_batch_response(parsed: Dict, expected: Set[str]) -> Dict[str, Dict]:
    if "results" not in parsed or not isinstance(parsed["results"], list):
        raise ValueError("JSON must contain list field 'results'")
    out: Dict[str, Dict] = {}
    for item in parsed["results"]:
        if not isinstance(item, dict):
            continue
        tok = str(item.get("token", "")).strip().lower()
        if tok not in expected:
            continue
        loc = item.get("localizable")
        if not isinstance(loc, bool):
            raise ValueError(f"localizable must be bool for token {tok!r}")
        reason = str(item.get("reason", "")).strip()
        out[tok] = {"localizable": loc, "reason": reason}
    if set(out.keys()) != expected:
        missing = expected - set(out.keys())
        extra = set(out.keys()) - expected
        raise ValueError(f"Result tokens mismatch. Missing={missing!r} extra={extra!r}")
    return out


def request_batch_classification(
    client,
    model: str,
    batch_tokens: List[str],
    max_retries: int,
    sleep_between_retries_s: float,
) -> Dict[str, Dict]:
    expected = set(batch_tokens)
    user = classification_user_prompt(batch_tokens)
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": classification_system_prompt()},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content if resp.choices else ""
            parsed = parse_json_object(content)
            return validate_batch_response(parsed, expected)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < max_retries:
                time.sleep(sleep_between_retries_s)
    raise RuntimeError(f"OpenAI batch failed after {max_retries} retries: {last_err}")


def load_cache_jsonl(cache_path: Path) -> Dict[str, Dict]:
    if not cache_path.exists():
        return {}
    out: Dict[str, Dict] = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = str(row.get("token", "")).strip().lower()
            if not key:
                continue
            if "localizable" in row and isinstance(row["localizable"], bool):
                out[key] = {"localizable": row["localizable"], "reason": str(row.get("reason", ""))}
    return out


def save_cache_jsonl(cache_path: Path, cache: Dict[str, Dict]) -> None:
    ensure_output_dir(cache_path.parent)
    with open(cache_path, "w", encoding="utf-8") as f:
        for token in sorted(cache.keys()):
            row = {"token": token, **cache[token]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(
    json_path: Path,
    csv_path: Path,
    records: List[Dict],
) -> None:
    ensure_output_dir(json_path.parent)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["token", "token_display", "localizable", "reason"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train vocab (SLAKE+VQA-Rad+Med-VQA-2019) -> stopword filter -> OpenAI localizable classification.",
    )
    p.add_argument("--data_root", type=str, default="data", help="Root with raw dataset folders")
    p.add_argument(
        "--output_dir",
        type=str,
        default="results/reasoning_vocab",
        help="Directory for vocab_localizable_train.* and cache jsonl",
    )
    p.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    p.add_argument("--openai_api_key_env", type=str, default="OPENAI_API_KEY")
    p.add_argument("--batch_size", type=int, default=30, help="Tokens per OpenAI request")
    p.add_argument("--n_tokens", type=int, default=None, help="Only first N tokens after filtering (smoke test)")
    p.add_argument("--resume", action="store_true", help="Resume from cache jsonl")
    p.add_argument("--save_every", type=int, default=5, help="Flush cache every N completed batches")
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--request_timeout_s", type=float, default=120.0)
    p.add_argument("--sleep_between_retries_s", type=float, default=1.5)
    p.add_argument(
        "--extra_stopwords_file",
        type=str,
        default=None,
        help="Optional text file: one stopword per line (lowercase)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.getenv(args.openai_api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Missing API key in env var '{args.openai_api_key_env}'. Export it or use a .env file."
        )

    out_dir = Path(args.output_dir)
    ensure_output_dir(out_dir)
    cache_path = out_dir / "openai_localizable_cache.jsonl"
    json_path = out_dir / "vocab_localizable_train.json"
    csv_path = out_dir / "vocab_localizable_train.csv"

    print("Loading train samples (legacy annotations, mc_data_root=None)...")
    samples = collect_train_samples(args.data_root)
    print(f"  samples={len(samples)}")

    tokens_all, display_map = build_vocabulary(samples)
    print(f"  unique raw tokens={len(tokens_all)}")

    stop = load_english_stopwords()
    stop |= load_extra_stopwords(Path(args.extra_stopwords_file) if args.extra_stopwords_file else None)
    tokens = filter_stopwords(tokens_all, stop)
    print(f"  after stopwords={len(tokens)}")

    if args.n_tokens is not None:
        tokens = tokens[: args.n_tokens]
        print(f"  truncated to n_tokens={len(tokens)}")

    cache: Dict[str, Dict] = {}
    if args.resume:
        cache = load_cache_jsonl(cache_path)
        print(f"  resume: loaded {len(cache)} cached classifications")

    client = build_openai_client(api_key=api_key, timeout_s=args.request_timeout_s)

    pending = [t for t in tokens if t not in cache]
    print(f"  pending OpenAI batches for {len(pending)} tokens")

    batch_size = max(1, args.batch_size)
    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]

    for bi, batch in enumerate(tqdm(batches, desc="OpenAI batches", unit="batch"), start=1):
        classified = request_batch_classification(
            client=client,
            model=args.openai_model,
            batch_tokens=batch,
            max_retries=args.max_retries,
            sleep_between_retries_s=args.sleep_between_retries_s,
        )
        for tok, payload in classified.items():
            cache[tok] = payload

        if bi % args.save_every == 0 or bi == len(batches):
            save_cache_jsonl(cache_path, cache)

    records: List[Dict] = []
    for tok in tokens:
        if tok not in cache:
            continue
        records.append(
            {
                "token": tok,
                "token_display": display_map.get(tok, tok),
                "localizable": cache[tok]["localizable"],
                "reason": cache[tok]["reason"],
            }
        )

    write_outputs(json_path, csv_path, records)
    save_cache_jsonl(cache_path, cache)

    n_loc = sum(1 for r in records if r["localizable"])
    print(f"Done. classified={len(records)} localizable_true={n_loc}")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {cache_path}")


if __name__ == "__main__":
    main()
