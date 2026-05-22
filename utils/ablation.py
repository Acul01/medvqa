"""Shared image ablation helpers for MC inference scripts."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

AblationLookupKey = Tuple[str, str]
FgrCropLookup = Dict[AblationLookupKey, str]

DEFAULT_FGR_CROP_FIELD = "overall_winner_unsupervised_crop"


@dataclass
class AblationImageResolution:
    """Resolved image path for one inference sample."""

    effective_image_path: str
    fgr_crop_path: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


def normalize_ablation_image_key(image_path: str, data_root: Optional[str] = None) -> str:
    """Stable lookup key for matching loader samples to FGR JSON rows."""
    path = Path(str(image_path).strip()).expanduser()
    if not path.is_absolute() and data_root:
        candidate = Path(data_root).expanduser() / path
        if candidate.exists():
            path = candidate
    if path.exists():
        return str(path.resolve())
    return str(path)


def _resolve_crop_file_path(crop_path: str, *, json_path: Path) -> Optional[Path]:
    raw = Path(str(crop_path).strip()).expanduser()
    if not str(crop_path).strip():
        return None
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(Path.cwd() / raw)
        candidates.append(json_path.parent / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _load_fgr_rows(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        samples = payload.get("samples")
        if isinstance(samples, list):
            return [row for row in samples if isinstance(row, dict)]
    raise ValueError("FGR region comparison JSON must be a list or contain a 'samples' list")


def load_fgr_crop_lookup(
    json_path: str | Path,
    *,
    crop_field: str = DEFAULT_FGR_CROP_FIELD,
    data_root: Optional[str] = None,
) -> FgrCropLookup:
    """Build (image_key, question) -> absolute crop path from FGR comparison JSON."""
    path = Path(json_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FGR region comparison JSON not found: {path}")

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    lookup: FgrCropLookup = {}
    missing_crop = 0
    for row in _load_fgr_rows(payload):
        image_path = row.get("image_path")
        question = row.get("question")
        crop_value = row.get(crop_field)
        if not isinstance(image_path, str) or not isinstance(question, str):
            continue
        if not isinstance(crop_value, str) or not crop_value.strip():
            missing_crop += 1
            continue
        resolved = _resolve_crop_file_path(crop_value, json_path=path)
        if resolved is None:
            missing_crop += 1
            warnings.warn(f"FGR crop file missing for {crop_field!r}: {crop_value!r}", stacklevel=2)
            continue
        key = (normalize_ablation_image_key(image_path, data_root=data_root), question.strip())
        lookup[key] = str(resolved)

    if not lookup:
        raise ValueError(
            f"No usable crop mappings loaded from {path} (field={crop_field!r}, missing={missing_crop})"
        )
    print(f"Loaded {len(lookup)} FGR crop mappings from {path.name} ({crop_field})")
    return lookup


def validate_ablation_args(
    ablation_mode: str,
    fgr_region_comparison_json: Optional[str],
) -> None:
    if ablation_mode == "fgr_crop" and not fgr_region_comparison_json:
        raise ValueError(
            "--fgr_region_comparison_json is required when --ablation_mode fgr_crop"
        )


def prepare_fgr_crop_lookup(
    ablation_mode: str,
    fgr_region_comparison_json: Optional[str],
    *,
    crop_field: str = DEFAULT_FGR_CROP_FIELD,
    data_root: Optional[str] = None,
) -> Optional[FgrCropLookup]:
    validate_ablation_args(ablation_mode, fgr_region_comparison_json)
    if ablation_mode != "fgr_crop":
        return None
    assert fgr_region_comparison_json is not None
    return load_fgr_crop_lookup(
        fgr_region_comparison_json,
        crop_field=crop_field,
        data_root=data_root,
    )


def resolve_ablation_image(
    *,
    ablation_mode: str,
    source_image_path: str,
    question: str,
    sample_idx: int,
    shuffle_assignment: dict[int, str],
    fgr_crop_lookup: Optional[FgrCropLookup] = None,
    data_root: Optional[str] = None,
) -> AblationImageResolution:
    """Resolve which image file to load for the current ablation mode."""
    if ablation_mode == "fgr_crop":
        if fgr_crop_lookup is None:
            raise ValueError("fgr_crop_lookup is required for ablation_mode=fgr_crop")
        key = (
            normalize_ablation_image_key(source_image_path, data_root=data_root),
            question.strip(),
        )
        crop_path = fgr_crop_lookup.get(key)
        if crop_path is None:
            return AblationImageResolution(
                effective_image_path=source_image_path,
                skipped=True,
                skip_reason="no FGR crop mapping",
            )
        return AblationImageResolution(
            effective_image_path=crop_path,
            fgr_crop_path=crop_path,
        )

    effective = shuffle_assignment.get(sample_idx, source_image_path)
    return AblationImageResolution(effective_image_path=effective)
