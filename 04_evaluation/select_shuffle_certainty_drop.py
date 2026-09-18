#!/usr/bin/env python3
"""Pick samples with the largest original→shuffle certainty drop (Qwen2.5 both, AF).

Joins original-image and shuffle-ablation inference JSONs, keeps rows where
entropy rises and confidence falls, ranks by a combined drop score, and writes
original + shuffled images plus metadata for the top-k cases.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

DEFAULT_ORIGINAL = (
    "06_results/FINAL/withX/pseudo_qwen3/both/af/medvqa2019/qwen2.5/"
    "qwen2.5-vl_medvqa2019_all.json"
)
DEFAULT_SHUFFLE = (
    "06_results/FINAL/withX/pseudo_qwen3/shuffle/both/af/medvqa2019/qwen2.5/"
    "qwen2.5-vl_medvqa2019_all.json"
)
DEFAULT_OUT = (
    "06_results/FINAL/withX/pseudo_qwen3/shuffle/certainty_drop/"
    "medvqa2019/qwen2.5_both_af"
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list) or (data and not isinstance(data[0], dict)):
        raise ValueError(f"Expected a list of sample dicts in {path}")
    return data


def sample_key(row: dict[str, Any]) -> tuple[str, str]:
    image = str(row.get("source_image_path") or row.get("image_path") or "").strip()
    question = str(row.get("question") or "").strip()
    if not image or not question:
        raise ValueError("Row missing source_image_path/image_path or question")
    return image, question


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = sample_key(row)
        if key in out:
            raise ValueError(f"Duplicate sample key: {key}")
        out[key] = row
    return out


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _copy_image(src: str, dest: Path) -> str | None:
    path = Path(src)
    if not path.exists():
        print(f"  missing image: {path}")
        return None
    dest = dest.with_suffix(path.suffix or ".jpg")
    shutil.copy2(path, dest)
    return dest.name


def _gt_answer(row: dict[str, Any]) -> str:
    letter = str(row.get("ground_truth_letter") or "").strip()
    text = str(row.get("ground_truth_answer") or "").strip()
    if letter and text:
        return f"{letter}: {text}"
    return letter or text


def _explanation(row: dict[str, Any]) -> str:
    text = row.get("predicted_reasoning_text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    raw = row.get("predicted_answer")
    return raw.strip() if isinstance(raw, str) else ""


def pair_rows(
    original_rows: list[dict[str, Any]],
    shuffle_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    orig_ix = _index_rows(original_rows)
    shuf_ix = _index_rows(shuffle_rows)
    paired: list[dict[str, Any]] = []
    for key, orig in orig_ix.items():
        shuf = shuf_ix.get(key)
        if shuf is None:
            continue
        h0, c0 = _finite(orig.get("entropy")), _finite(orig.get("confidence"))
        h1, c1 = _finite(shuf.get("entropy")), _finite(shuf.get("confidence"))
        if None in (h0, c0, h1, c1):
            continue
        d_h = h1 - h0
        d_c = c0 - c1  # positive = confidence drop
        paired.append(
            {
                "key": key,
                "orig": orig,
                "shuf": shuf,
                "delta_entropy": d_h,
                "delta_confidence": -d_c,  # ablation − original
                "entropy_rise": d_h,
                "confidence_drop": d_c,
                "qualifies": d_h > 0 and d_c > 0,
            }
        )
    return paired


def rank_drops(paired: list[dict[str, Any]], k: int) -> tuple[list[dict[str, Any]], str]:
    qualifying = [p for p in paired if p["qualifies"]]
    pool = qualifying
    note = (
        f"{len(qualifying)} / {len(paired)} paired samples have entropy↑ and confidence↓."
    )
    if len(pool) < k:
        note += (
            f" Only {len(pool)} qualify; ranking those. "
            "Not filling with non-qualifying samples."
        )
    if not pool:
        raise ValueError("No paired samples with entropy increase and confidence decrease.")
    rises = [p["entropy_rise"] for p in pool]
    drops = [p["confidence_drop"] for p in pool]
    z_h = _minmax(rises)
    z_c = _minmax(drops)
    for p, zh, zc in zip(pool, z_h, z_c):
        p["certainty_drop_score"] = zh + zc
    ranked = sorted(pool, key=lambda p: p["certainty_drop_score"], reverse=True)
    return ranked[:k], note


def build_metadata(pair: dict[str, Any], orig_copy: str | None, shuf_copy: str | None) -> dict[str, Any]:
    orig, shuf = pair["orig"], pair["shuf"]
    return {
        "Original_image_path": orig.get("source_image_path") or orig.get("image_path"),
        "Ablation_image_path": shuf.get("image_path"),
        "question": orig.get("question"),
        "options": orig.get("options"),
        "gt_answer": _gt_answer(orig),
        "Prediction_original": orig.get("predicted_letter"),
        "Explanation_original": _explanation(orig),
        "Entropy_original": orig.get("entropy"),
        "Confidence_original": orig.get("confidence"),
        "Prediction_ablation": shuf.get("predicted_letter"),
        "Explanation_ablation": _explanation(shuf),
        "Entropy_ablation": shuf.get("entropy"),
        "Confidence_ablation": shuf.get("confidence"),
        "copied_original": orig_copy,
        "copied_ablation": shuf_copy,
        "delta_entropy": pair["entropy_rise"],
        "delta_confidence": pair["delta_confidence"],
        "certainty_drop_score": pair["certainty_drop_score"],
        "question_type": orig.get("question_type"),
        "is_correct_original": orig.get("is_correct"),
        "is_correct_ablation": shuf.get("is_correct"),
    }


def write_readme(
    dest: Path,
    *,
    original_json: Path,
    shuffle_json: Path,
    n_paired: int,
    note: str,
    selected: list[dict[str, Any]],
) -> None:
    lines = [
        "# Certainty-drop cases: Qwen2.5 both, MedVQA-2019, AF, original vs shuffle",
        "",
        "## What this folder is",
        "",
        "Qualitative dump of the test samples where the **Qwen2.5 dual-order** "
        "(`--both_json_orders`) checkpoint, evaluated **answer-first (AF)**, loses the most "
        "certainty when the **image is shuffled** (question and gold answer stay on the original sample).",
        "",
        "## Models and files",
        "",
        "- Checkpoint: `qwen2.5_medvqa2019_both` (trained on both RF and AF JSON).",
        "- Eval contract: `--use_reasoning_json --answer_first` (AF test).",
        f"- Original images (`ablation_mode=none`): `{original_json}`",
        f"- Shuffle ablation (`ablation_mode=shuffle`): `{shuffle_json}`",
        "",
        "Join key: `(source_image_path, question)`. "
        f"{n_paired} samples appear in both JSONs with finite entropy and confidence.",
        "",
        "## Shuffle ablation",
        "",
        "Inference permutes **image paths** across the test split (seed 42). "
        "`source_image_path` is the original sample image; `image_path` on the shuffle row "
        "is the **other** test image that was shown instead. "
        "The question, options, and GT stay with the original sample.",
        "",
        "## Entropy and confidence",
        "",
        "Values are the **answer-letter** metrics from inference, not explanation-token stats:",
        "",
        "- `entropy`: mean Shannon entropy (nats) at the MC-letter token over `--agreement_samples` (K=5).",
        "- `confidence`: mean max-softmax at that same token over the same K draws.",
        "- `Prediction_*`: greedy-main `predicted_letter`.",
        "- `Explanation_*`: `predicted_reasoning_text` from the AF JSON.",
        "",
        "## Certainty-drop ranking",
        "",
        "A sample qualifies only if **both** hold: `Entropy_ablation > Entropy_original` "
        "and `Confidence_ablation < Confidence_original`.",
        "",
        "Among qualifying samples, `ΔH` and `ΔC = C_original − C_ablation` are min-max "
        "normalized to [0, 1] and summed (`certainty_drop_score`). Top-k by that score.",
        "",
        note,
        "",
        "## Selected samples",
        "",
        "| rank | score | Δ entropy | Δ confidence | pred orig → abl | GT | type |",
        "| ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for i, pair in enumerate(selected, start=1):
        orig, shuf = pair["orig"], pair["shuf"]
        lines.append(
            f"| {i} | {pair['certainty_drop_score']:.3f} | "
            f"{pair['entropy_rise']:+.4f} | {pair['delta_confidence']:+.4f} | "
            f"{orig.get('predicted_letter')} → {shuf.get('predicted_letter')} | "
            f"{_gt_answer(orig)} | {orig.get('question_type')} |"
        )
    lines += [
        "",
        "## Per-sample files",
        "",
        "Each `rankXX_*/` directory contains:",
        "",
        "- `original.<ext>` — image from `Original_image_path`",
        "- `shuffled.<ext>` — image from `Ablation_image_path` (the shuffled-in test image)",
        "- `metadata.json` — the fields listed in the request, plus deltas and correctness flags",
        "",
    ]
    (dest / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Top-k original→shuffle certainty-drop cases for Qwen2.5 both AF."
    )
    p.add_argument("--original_json", type=str, default=DEFAULT_ORIGINAL)
    p.add_argument("--shuffle_json", type=str, default=DEFAULT_SHUFFLE)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUT)
    p.add_argument("--k", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    orig_path = Path(args.original_json)
    shuf_path = Path(args.shuffle_json)
    dest = Path(args.output_dir)
    if not orig_path.exists():
        raise FileNotFoundError(orig_path)
    if not shuf_path.exists():
        raise FileNotFoundError(shuf_path)
    if args.k < 1:
        raise ValueError("--k must be >= 1")

    paired = pair_rows(_load_rows(orig_path), _load_rows(shuf_path))
    selected, note = rank_drops(paired, args.k)
    dest.mkdir(parents=True, exist_ok=True)
    print(note)
    print(f"writing {len(selected)} cases to {dest}")

    index: list[dict[str, Any]] = []
    for rank, pair in enumerate(selected, start=1):
        orig = pair["orig"]
        letter = str(orig.get("predicted_letter") or "x")
        case_dir = dest / f"rank{rank:02d}_{letter}"
        case_dir.mkdir(parents=True, exist_ok=True)
        orig_name = _copy_image(
            str(orig.get("source_image_path") or orig.get("image_path")),
            case_dir / "original",
        )
        shuf_name = _copy_image(str(pair["shuf"].get("image_path")), case_dir / "shuffled")
        meta = build_metadata(pair, orig_name, shuf_name)
        (case_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        index.append({"rank": rank, "dir": case_dir.name, **{k: meta[k] for k in (
            "question", "gt_answer", "Prediction_original", "Prediction_ablation",
            "Entropy_original", "Entropy_ablation", "Confidence_original", "Confidence_ablation",
            "certainty_drop_score",
        )}})
        print(
            f"  {case_dir.name}  ΔH={pair['entropy_rise']:+.4f}  "
            f"ΔC={pair['delta_confidence']:+.4f}  score={pair['certainty_drop_score']:.3f}"
        )

    (dest / "selected.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_readme(
        dest,
        original_json=orig_path,
        shuffle_json=shuf_path,
        n_paired=len(paired),
        note=note,
        selected=selected,
    )
    print(f"wrote {dest / 'README.md'}")


if __name__ == "__main__":
    main()
