"""Checkpoint and resume helpers for MC inference scripts."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

InferenceKey = tuple[str, str]

RESUME_META_FIELDS = (
    "dataset",
    "split",
    "model_path",
    "use_reasoning_json",
    "answer_first",
    "json_order",
    "n_samples",
)


def inference_result_key(row: dict[str, Any]) -> InferenceKey:
    source = row.get("source_image_path") or row.get("image_path") or ""
    question = row.get("question") or ""
    return (str(source), str(question))


def sample_inference_key(source_image_path: str, question: str) -> InferenceKey:
    return (str(source_image_path), str(question))


def is_inference_row_complete(row: dict[str, Any]) -> bool:
    if row.get("predicted_letter"):
        return True
    pred = row.get("predicted_answer")
    return pred is not None and str(pred).strip() != ""


def resolve_partial_path(results_dir: str | Path, base_name: str) -> Path:
    return Path(results_dir) / f"{base_name}.partial.json"


def resolve_final_json_path(results_dir: str | Path, base_name: str) -> Path:
    return Path(results_dir) / f"{base_name}.json"


def resolve_resume_load_path(
    results_dir: str | Path,
    base_name: str,
    *,
    partial_path: str | Path | None = None,
    resume: bool = False,
) -> Path | None:
    if not resume:
        return None
    if partial_path:
        path = Path(partial_path)
        return path if path.exists() else None
    partial = resolve_partial_path(results_dir, base_name)
    if partial.exists():
        return partial
    final = resolve_final_json_path(results_dir, base_name)
    if final.exists():
        return final
    return None


def load_partial_results(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return list(data["results"]), data.get("_meta")
    if isinstance(data, list):
        return data, None
    raise ValueError(f"Unexpected checkpoint format in {path}")


def save_partial_results(
    path: Path,
    results: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"_meta": meta or {}, "results": results}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def validate_resume_meta(saved_meta: dict[str, Any] | None, current_meta: dict[str, Any]) -> None:
    if not saved_meta:
        return
    for field in RESUME_META_FIELDS:
        if field not in saved_meta or field not in current_meta:
            continue
        if saved_meta[field] != current_meta[field]:
            warnings.warn(
                f"Resume meta mismatch for {field!r}: "
                f"checkpoint={saved_meta[field]!r} vs current={current_meta[field]!r}",
                stacklevel=2,
            )


def prepare_resume_state(
    *,
    resume: bool,
    results_dir: str | Path,
    base_name: str,
    partial_path: str | Path | None,
    current_meta: dict[str, Any],
) -> dict[InferenceKey, dict[str, Any]]:
    load_path = resolve_resume_load_path(
        results_dir,
        base_name,
        partial_path=partial_path,
        resume=resume,
    )
    done_by_key: dict[InferenceKey, dict[str, Any]] = {}
    if load_path is None:
        return done_by_key

    rows, saved_meta = load_partial_results(load_path)
    validate_resume_meta(saved_meta, current_meta)
    for row in rows:
        if is_inference_row_complete(row):
            done_by_key[inference_result_key(row)] = row
    print(f"Resume: loaded {len(done_by_key)} completed samples from {load_path}")
    return done_by_key


class InferenceCheckpointManager:
    """Track ordered results and write partial checkpoints during inference."""

    def __init__(
        self,
        *,
        done_by_key: dict[InferenceKey, dict[str, Any]],
        partial_path: Path,
        meta: dict[str, Any],
        checkpoint_every: int,
    ) -> None:
        self.done_by_key = done_by_key
        self.partial_path = partial_path
        self.meta = meta
        self.checkpoint_every = max(0, int(checkpoint_every))
        self.results: list[dict[str, Any]] = []
        self.pending_since_checkpoint = 0
        self.n_skipped = 0
        self.n_new = 0

    def try_skip(self, key: InferenceKey) -> bool:
        if key not in self.done_by_key:
            return False
        self.results.append(self.done_by_key[key])
        self.n_skipped += 1
        return True

    def append_new(self, row: dict[str, Any]) -> None:
        key = inference_result_key(row)
        self.results.append(row)
        self.done_by_key[key] = row
        self.n_new += 1
        self.pending_since_checkpoint += 1
        if self.checkpoint_every > 0 and self.pending_since_checkpoint >= self.checkpoint_every:
            self.save_checkpoint()

    def save_checkpoint(self) -> None:
        save_partial_results(self.partial_path, self.results, self.meta)
        self.pending_since_checkpoint = 0
        print(f"  Checkpoint saved ({len(self.results)} samples) → {self.partial_path}")

    def finalize(self, *, complete: bool = True) -> None:
        if self.pending_since_checkpoint > 0 and self.checkpoint_every > 0:
            self.save_checkpoint()
        if complete and self.partial_path.exists():
            self.partial_path.unlink()
            print(f"  Removed partial checkpoint: {self.partial_path}")

    def print_summary(self) -> None:
        if self.n_skipped or self.n_new:
            print(f"Resume summary: skipped {self.n_skipped}, newly inferred {self.n_new}")


def build_inference_resume_meta(args: Any) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "split": args.split,
        "model_path": args.model_path,
        "use_reasoning_json": bool(getattr(args, "use_reasoning_json", False)),
        "answer_first": bool(getattr(args, "answer_first", False)),
        "json_order": getattr(args, "json_order", None),
        "n_samples": getattr(args, "n_samples", None),
    }


def setup_inference_checkpoint(
    args: Any,
    *,
    resume: bool,
    checkpoint_every: int,
    partial_path: str | None,
    name_suffix: str | None = None,
) -> tuple[InferenceCheckpointManager, str]:
    from utils.save_results import build_results_base_name

    base_name = build_results_base_name(
        args.model_path,
        args.dataset,
        args.n_samples,
        name_suffix=name_suffix,
    )
    resolved_partial = (
        Path(partial_path)
        if partial_path
        else resolve_partial_path(args.results_dir, base_name)
    )
    meta = build_inference_resume_meta(args)
    done_by_key = prepare_resume_state(
        resume=resume,
        results_dir=args.results_dir,
        base_name=base_name,
        partial_path=partial_path,
        current_meta=meta,
    )
    manager = InferenceCheckpointManager(
        done_by_key=done_by_key,
        partial_path=resolved_partial,
        meta=meta,
        checkpoint_every=checkpoint_every,
    )
    return manager, base_name


def add_inference_resume_cli_arguments(parser) -> None:
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load partial/final JSON and skip samples that were already inferred",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=25,
        help="Save partial JSON every N newly inferred samples (0 = only at end; default: 25)",
    )
    parser.add_argument(
        "--partial_path",
        type=str,
        default=None,
        help="Optional explicit path for partial checkpoint JSON (default: <results_dir>/<base>.partial.json)",
    )
