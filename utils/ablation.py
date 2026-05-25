"""Shared image ablation helpers for MC inference scripts."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

AblationLookupKey = Tuple[str, str]
FgrCropLookup = Dict[AblationLookupKey, str]
FgrBboxLookup = Dict[AblationLookupKey, Optional[List[int]]]

# Sentinel used to distinguish "key not in bbox lookup" from "key maps to None (baseline)"
_MISSING = object()

FGR_CROP_FIELD_BY_MODE = {
    "crop_unsupervised": "overall_winner_unsupervised_crop",
    "crop_supervised": "overall_winner_supervised_crop",
}
FGR_CROP_MODES = frozenset(FGR_CROP_FIELD_BY_MODE)

FGR_BBOX_FIELD_BY_MODE = {
    "bbox_supervised": "overall_winner_supervised_bbox",
}
FGR_BBOX_MODES = frozenset(FGR_BBOX_FIELD_BY_MODE)


@dataclass
class AblationImageResolution:
    """Resolved image path for one inference sample."""

    effective_image_path: str
    fgr_crop_path: Optional[str] = None
    # For bbox_supervised: [x1, y1, x2, y2] to draw on the full image.
    # None means the FGR winner was "baseline" — use the full image as-is.
    fgr_bbox: Optional[List[int]] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


def is_fgr_crop_ablation(ablation_mode: str) -> bool:
    return ablation_mode in FGR_CROP_MODES


def is_fgr_bbox_ablation(ablation_mode: str) -> bool:
    return ablation_mode in FGR_BBOX_MODES


def fgr_crop_field_for_mode(ablation_mode: str) -> str:
    try:
        return FGR_CROP_FIELD_BY_MODE[ablation_mode]
    except KeyError as exc:
        raise ValueError(
            f"Unknown FGR crop ablation mode {ablation_mode!r}; "
            f"expected one of {sorted(FGR_CROP_MODES)}"
        ) from exc


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
    # Always try: filename only, in a *_crops dir next to the JSON.
    # This handles both moved JSONs (absolute paths gone stale) and relative
    # paths recorded from a different working directory.
    fname = raw.name
    stem = json_path.stem  # e.g. fgr_region_comparison_vqarad_all
    candidates = [
        raw,
        Path.cwd() / raw,
        json_path.parent / raw,
        json_path.parent / f"{stem}_crops" / fname,
    ]
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
    crop_field: str,
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


def load_fgr_bbox_lookup(
    json_path: str | Path,
    *,
    bbox_field: str,
    data_root: Optional[str] = None,
) -> FgrBboxLookup:
    """Build (image_key, question) -> bbox ([x1,y1,x2,y2] or None) from FGR comparison JSON.

    None entries correspond to samples where the FGR winner was "baseline"
    (the full image) — no sub-region was identified.  These are *included* in
    the lookup so the caller can distinguish between "sample not found" and
    "sample found but no crop region".
    """
    path = Path(json_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FGR region comparison JSON not found: {path}")

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    lookup: FgrBboxLookup = {}
    for row in _load_fgr_rows(payload):
        image_path = row.get("image_path")
        question = row.get("question")
        if not isinstance(image_path, str) or not isinstance(question, str):
            continue
        bbox_value = row.get(bbox_field)
        if bbox_value is not None:
            if not (isinstance(bbox_value, (list, tuple)) and len(bbox_value) == 4):
                warnings.warn(
                    f"Unexpected bbox value {bbox_value!r} in {path.name}, treating as None",
                    stacklevel=2,
                )
                bbox_value = None
            else:
                bbox_value = [int(v) for v in bbox_value]
        key = (normalize_ablation_image_key(image_path, data_root=data_root), question.strip())
        lookup[key] = bbox_value  # None (baseline) or [x1, y1, x2, y2]

    if not lookup:
        raise ValueError(
            f"No bbox mappings loaded from {path} (field={bbox_field!r})"
        )
    n_baseline = sum(1 for v in lookup.values() if v is None)
    print(
        f"Loaded {len(lookup)} FGR bbox mappings from {path.name} "
        f"({bbox_field}; {n_baseline} baseline/no-crop)"
    )
    return lookup


def prepare_fgr_bbox_lookup(
    ablation_mode: str,
    fgr_region_comparison_json: Optional[str],
    *,
    data_root: Optional[str] = None,
) -> Optional[FgrBboxLookup]:
    if ablation_mode not in FGR_BBOX_MODES:
        return None
    if not fgr_region_comparison_json:
        raise ValueError(
            "--fgr_region_comparison_json is required when --ablation_mode is bbox_supervised"
        )
    bbox_field = FGR_BBOX_FIELD_BY_MODE[ablation_mode]
    return load_fgr_bbox_lookup(
        fgr_region_comparison_json,
        bbox_field=bbox_field,
        data_root=data_root,
    )


def validate_ablation_args(
    ablation_mode: str,
    fgr_region_comparison_json: Optional[str],
) -> None:
    if (is_fgr_crop_ablation(ablation_mode) or is_fgr_bbox_ablation(ablation_mode)) and not fgr_region_comparison_json:
        raise ValueError(
            "--fgr_region_comparison_json is required when --ablation_mode is "
            "crop_unsupervised, crop_supervised, or bbox_supervised"
        )


def prepare_fgr_crop_lookup(
    ablation_mode: str,
    fgr_region_comparison_json: Optional[str],
    *,
    data_root: Optional[str] = None,
) -> Optional[FgrCropLookup]:
    validate_ablation_args(ablation_mode, fgr_region_comparison_json)
    if not is_fgr_crop_ablation(ablation_mode):
        return None
    assert fgr_region_comparison_json is not None
    return load_fgr_crop_lookup(
        fgr_region_comparison_json,
        crop_field=fgr_crop_field_for_mode(ablation_mode),
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
    fgr_bbox_lookup: Optional[FgrBboxLookup] = None,
    data_root: Optional[str] = None,
) -> AblationImageResolution:
    """Resolve which image file to load for the current ablation mode."""
    if is_fgr_crop_ablation(ablation_mode):
        if fgr_crop_lookup is None:
            raise ValueError(
                f"fgr_crop_lookup is required for ablation_mode={ablation_mode!r}"
            )
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

    if is_fgr_bbox_ablation(ablation_mode):
        if fgr_bbox_lookup is None:
            raise ValueError(
                f"fgr_bbox_lookup is required for ablation_mode={ablation_mode!r}"
            )
        key = (
            normalize_ablation_image_key(source_image_path, data_root=data_root),
            question.strip(),
        )
        bbox = fgr_bbox_lookup.get(key, _MISSING)
        if bbox is _MISSING:
            return AblationImageResolution(
                effective_image_path=source_image_path,
                skipped=True,
                skip_reason="no FGR bbox mapping",
            )
        # bbox is None (baseline winner → use full image as-is) or [x1,y1,x2,y2]
        return AblationImageResolution(
            effective_image_path=source_image_path,
            fgr_bbox=bbox,  # type: ignore[arg-type]
        )

    effective = shuffle_assignment.get(sample_idx, source_image_path)
    return AblationImageResolution(effective_image_path=effective)
