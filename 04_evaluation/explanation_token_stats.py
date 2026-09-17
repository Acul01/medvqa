#!/usr/bin/env python3
"""Average token entropy / confidence of the *explanation* (not the MC letter).

Reads ``entropy_tokens_all`` from a pseudo-reasoning SFT (or zeroshot JSON) results
file. The answer-letter token and JSON structural tokens are dropped; remaining
tokens are treated as the rationale.

Per-token max-softmax confidence was never written to the results JSON. This
script reports ``exp(-H)`` as a certainty proxy from the stored Shannon
entropies (nats). The scalar row field ``confidence`` is the *answer* max-prob
and is not used here.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Literal

JsonOrder = Literal["auto", "rationale_first", "answer_first"]

_DUP_SUFFIX = re.compile(r"#(\d+)$")
_JSON_KEYS = {"reasoning", "answer"}
_STRUCTURAL = set("{}[]:,.\"'`\\/") | {"", " "}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        import pandas as pd

        return pd.read_csv(path).to_dict(orient="records")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of rows in {path}")
    return data


def parse_entropy_tokens_all(raw: Any) -> dict[str, float]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return {}
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        return {str(k): float(v) for k, v in json.loads(raw).items()}
    raise TypeError(f"Unexpected entropy_tokens_all type: {type(raw)}")


def _base_token(tok: str) -> str:
    m = _DUP_SUFFIX.search(tok)
    if m:
        tok = tok[: m.start()]
    return tok.strip().strip('"').strip("'")


def _is_structural(base: str) -> bool:
    b = base.strip()
    if b in _STRUCTURAL or not b:
        return True
    # Schema keys are lowercase "reasoning"/"answer"; body uses "REASONING:".
    if b in _JSON_KEYS:
        return True
    if all(ch in '{}[]:,."\'`\\ ' for ch in b):
        return True
    return False


def _fmt_letter(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip().upper()
    return s or None


def detect_json_order(row: dict[str, Any], cli_order: JsonOrder) -> str:
    if cli_order != "auto":
        return cli_order
    flag = row.get("answer_first")
    if flag is True:
        return "answer_first"
    if flag is False and row.get("use_reasoning_json"):
        return "rationale_first"
    raw = str(row.get("predicted_answer") or "")
    a = raw.find('"answer"')
    r = raw.find('"reasoning"')
    if a >= 0 and r >= 0:
        return "answer_first" if a < r else "rationale_first"
    return "rationale_first"


def _answer_token_index(items: list[tuple[str, float]], letter: str | None) -> int | None:
    if not letter:
        return None
    for i, (tok, _) in enumerate(items):
        if _base_token(tok).upper() == letter:
            return i
    return None


def explanation_entropies(
    toks: dict[str, float],
    *,
    letter: str | None,
    json_order: str,
) -> list[float]:
    items = list(toks.items())
    if not items:
        return []
    ans_i = _answer_token_index(items, letter)
    if json_order == "answer_first":
        sliced = items[ans_i + 1 :] if ans_i is not None else items
    else:
        sliced = items[:ans_i] if ans_i is not None else items
    ents: list[float] = []
    for tok, ent in sliced:
        if not isinstance(ent, (int, float)) or not math.isfinite(float(ent)):
            continue
        base = _base_token(tok)
        if _is_structural(base):
            continue
        if letter and base.upper() == letter:
            continue
        ents.append(max(0.0, float(ent)))
    return ents


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _split_means(samples: list[dict[str, float]]) -> dict[str, float]:
    return {
        "mean_entropy": _mean([s["mean_entropy"] for s in samples]),
        "mean_confidence": _mean([s["mean_confidence"] for s in samples]),
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    json_order: JsonOrder = "auto",
) -> dict[str, dict[str, float]]:
    per_sample: list[dict[str, Any]] = []
    for row in rows:
        toks = parse_entropy_tokens_all(row.get("entropy_tokens_all"))
        if not toks:
            continue
        order = detect_json_order(row, json_order)
        letter = _fmt_letter(row.get("predicted_letter"))
        ents = explanation_entropies(toks, letter=letter, json_order=order)
        if not ents:
            continue
        per_sample.append(
            {
                "mean_entropy": _mean(ents),
                "mean_confidence": _mean([math.exp(-h) for h in ents]),
                "question_type": row.get("question_type") or "unknown",
            }
        )
    closed = [s for s in per_sample if s["question_type"] == "closed"]
    open_rows = [s for s in per_sample if s["question_type"] == "open"]
    return {
        "open": _split_means(open_rows),
        "closed": _split_means(closed),
        "overall": _split_means(per_sample),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mean explanation-token entropy and exp(-H) confidence (open / closed / overall)."
    )
    p.add_argument("--results_file", required=True, type=str)
    p.add_argument(
        "--json_order",
        choices=["auto", "rationale_first", "answer_first"],
        default="auto",
        help="JSON field order (default: auto from answer_first / predicted_answer)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON with the six values",
    )
    return p.parse_args()


def _fmt(value: float) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.6f}"


def main() -> None:
    args = parse_args()
    path = Path(args.results_file)
    if not path.exists():
        raise FileNotFoundError(path)
    summary = summarize_rows(_load_rows(path), json_order=args.json_order)
    for split in ("open", "closed", "overall"):
        block = summary[split]
        print(
            f"{split}\tmean_entropy={_fmt(block['mean_entropy'])}\t"
            f"mean_confidence={_fmt(block['mean_confidence'])}"
        )
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
