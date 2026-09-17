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


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    json_order: JsonOrder = "auto",
) -> dict[str, Any]:
    per_sample: list[dict[str, Any]] = []
    all_ents: list[float] = []
    skipped = 0
    for i, row in enumerate(rows):
        toks = parse_entropy_tokens_all(row.get("entropy_tokens_all"))
        if not toks:
            skipped += 1
            continue
        order = detect_json_order(row, json_order)
        letter = _fmt_letter(row.get("predicted_letter"))
        ents = explanation_entropies(toks, letter=letter, json_order=order)
        if not ents:
            skipped += 1
            continue
        confs = [math.exp(-h) for h in ents]
        rec = {
            "index": i,
            "n_explanation_tokens": len(ents),
            "mean_entropy": _mean(ents),
            "mean_confidence_proxy": _mean(confs),
            "json_order": order,
            "question_type": row.get("question_type") or "unknown",
        }
        per_sample.append(rec)
        all_ents.extend(ents)

    n = len(per_sample)
    all_confs = [math.exp(-h) for h in all_ents]
    by_type: dict[str, dict[str, float]] = {}
    for qtype in ("closed", "open"):
        subset = [s for s in per_sample if s.get("question_type") == qtype]
        if not subset:
            continue
        by_type[qtype] = {
            "n_samples": float(len(subset)),
            "macro_mean_entropy": _mean([s["mean_entropy"] for s in subset]),
            "macro_mean_confidence_proxy": _mean([s["mean_confidence_proxy"] for s in subset]),
        }

    return {
        "n_rows": len(rows),
        "n_used": n,
        "n_skipped": skipped,
        "macro_mean_entropy": _mean([s["mean_entropy"] for s in per_sample]),
        "macro_mean_confidence_proxy": _mean([s["mean_confidence_proxy"] for s in per_sample]),
        "micro_mean_entropy": _mean(all_ents),
        "micro_mean_confidence_proxy": _mean(all_confs),
        "mean_explanation_tokens": _mean([float(s["n_explanation_tokens"]) for s in per_sample]),
        "by_question_type": by_type,
        "confidence_note": (
            "confidence_proxy = exp(-token_entropy); per-token max-softmax was not stored. "
            "Row field 'confidence' is the MC-letter max-prob and is excluded."
        ),
        "per_sample": per_sample,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mean explanation-token entropy and exp(-H) confidence from results JSON/CSV."
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
        help="Optional summary JSON (includes per-sample stats)",
    )
    p.add_argument(
        "--no_per_sample",
        action="store_true",
        help="Omit per_sample list in --output",
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
    print(f"file: {path}")
    print(f"used {summary['n_used']} / {summary['n_rows']} rows (skipped {summary['n_skipped']})")
    print(f"mean explanation tokens / sample: {_fmt(summary['mean_explanation_tokens'])}")
    print(f"macro mean token entropy:         {_fmt(summary['macro_mean_entropy'])}")
    print(f"macro mean confidence (exp(-H)):  {_fmt(summary['macro_mean_confidence_proxy'])}")
    print(f"micro mean token entropy:         {_fmt(summary['micro_mean_entropy'])}")
    print(f"micro mean confidence (exp(-H)):  {_fmt(summary['micro_mean_confidence_proxy'])}")
    for qtype, block in summary["by_question_type"].items():
        print(
            f"  {qtype}: n={int(block['n_samples'])}  "
            f"entropy={_fmt(block['macro_mean_entropy'])}  "
            f"conf={_fmt(block['macro_mean_confidence_proxy'])}"
        )
    print(summary["confidence_note"])

    if args.output:
        out = dict(summary)
        out["results_file"] = str(path)
        if args.no_per_sample:
            out.pop("per_sample", None)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
