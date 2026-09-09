"""
Central path defaults for medvqa.

Override on cluster (e.g. in Slurm or ~/.bashrc) with:

    export MEDVQA_DATA_ROOT=/path/to/02_data
    export MEDVQA_MC_ROOT=/path/to/02_data/data_mc_openai
    export MEDVQA_OUTPUTS_DIR=/path/to/05_outputs
    export MEDVQA_RESULTS_DIR=/path/to/06_results

If unset, fallbacks match local repo layout (02_data, 05_outputs, 06_results).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ENV_DATA = "MEDVQA_DATA_ROOT"
_ENV_MC = "MEDVQA_MC_ROOT"
_ENV_OUTPUTS = "MEDVQA_OUTPUTS_DIR"
_ENV_RESULTS = "MEDVQA_RESULTS_DIR"
_ENV_SOM = "MEDVQA_SOM_DIR"
_ENV_SOM_SELECTION = "MEDVQA_SOM_SELECTION_DIR"
_ENV_LLAVA_MED = "LLAVA_MED_HOME"

_DEFAULT_DATA = "02_data"
_DEFAULT_MC = "02_data/data_mc_openai"
_DEFAULT_OUTPUTS = "05_outputs"
_DEFAULT_RESULTS = "06_results"
_DEFAULT_SOM = "02_data/som"
_DEFAULT_SOM_SELECTION = "02_data/som_selection"


def get_data_root() -> str:
    return os.getenv(_ENV_DATA, _DEFAULT_DATA).strip() or _DEFAULT_DATA


def get_mc_data_root() -> str:
    return os.getenv(_ENV_MC, _DEFAULT_MC).strip() or _DEFAULT_MC


def get_outputs_root() -> str:
    return os.getenv(_ENV_OUTPUTS, _DEFAULT_OUTPUTS).strip() or _DEFAULT_OUTPUTS


def get_results_dir() -> str:
    return os.getenv(_ENV_RESULTS, _DEFAULT_RESULTS).strip() or _DEFAULT_RESULTS


def get_som_dir() -> str:
    return os.getenv(_ENV_SOM, _DEFAULT_SOM).strip() or _DEFAULT_SOM


def get_som_selection_dir() -> str:
    """Root of the per-(image, question) SoM segment selections."""
    return (
        os.getenv(_ENV_SOM_SELECTION, _DEFAULT_SOM_SELECTION).strip()
        or _DEFAULT_SOM_SELECTION
    )


def get_train_output_dir_llava() -> str:
    """Default HF LLaVA checkpoint directory under outputs root."""
    return str(Path(get_outputs_root()) / "llava")


def get_train_output_dir_llava_med() -> str:
    """Default LLaVA-Med checkpoint directory under outputs root."""
    return str(Path(get_outputs_root()) / "llava-med")


def get_train_output_dir_qwen() -> str:
    """Default Qwen2.5-VL checkpoint directory under outputs root."""
    return str(Path(get_outputs_root()) / "qwen")


def get_selector_features_dir() -> str:
    return str(Path(get_results_dir()) / "selector_features")


def get_selector_models_dir() -> str:
    return str(Path(get_results_dir()) / "selector_models")


def get_selector_eval_dir() -> str:
    return str(Path(get_results_dir()) / "selector_eval")


def _llava_med_checkout_candidates() -> list[Path]:
    repo_root = Path(__file__).resolve().parent.parent
    candidates: list[Path] = []
    for env_name in (_ENV_LLAVA_MED, "LLAVA_MED_ROOT"):
        raw = os.getenv(env_name)
        if raw:
            candidates.append(Path(raw).expanduser())
    candidates.extend(
        [
            repo_root / "01_external" / "LLaVA-Med",
            repo_root.parent / "01_external" / "LLaVA-Med",
            repo_root.parent / "LLaVA-Med",
            Path.home() / "LLaVA-Med",
        ]
    )
    work = os.getenv("WORK")
    if work:
        work_root = Path(work)
        candidates.extend(
            [
                work_root / "LLaVA-Med",
                work_root / "medvqa" / "01_external" / "LLaVA-Med",
                work_root / "medvqa" / "LLaVA-Med",
            ]
        )
    conda_prefix = os.getenv("CONDA_PREFIX")
    if conda_prefix:
        envs_root = Path(conda_prefix).resolve().parent
        for env_name in ("llava_med_env", "llava_env", "llava-med"):
            lib = envs_root / env_name / "lib"
            if not lib.is_dir():
                continue
            for site in sorted(lib.glob("python*/site-packages")):
                if _is_llava_med_checkout(site):
                    candidates.append(site)
    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _is_llava_med_checkout(path: Path) -> bool:
    return (path / "llava" / "constants.py").is_file() or (path / "llava" / "__init__.py").is_file()


def find_llava_med_checkout() -> Path | None:
    for path in _llava_med_checkout_candidates():
        if _is_llava_med_checkout(path):
            return path.resolve()
    return None


def prepend_llava_med_checkout() -> Path | None:
    """Put a local LLaVA-Med checkout on sys.path / PYTHONPATH if present."""
    checkout = find_llava_med_checkout()
    if checkout is None:
        return None
    resolved = str(checkout)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    parts = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if resolved not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([resolved, *parts])
    return checkout


def ensure_llava_med_importable() -> None:
    """Import Microsoft's `llava` package, using a local checkout if needed."""
    try:
        import llava  # noqa: F401
        return
    except ImportError:
        pass
    checkout = prepend_llava_med_checkout()
    if checkout is not None:
        try:
            import llava  # noqa: F401
            print(f"Using LLaVA-Med from {checkout}")
            return
        except ImportError:
            pass
    tried = ", ".join(str(p) for p in _llava_med_checkout_candidates())
    raise ModuleNotFoundError(
        "No module named 'llava' (Microsoft LLaVA-Med). "
        "Install with `pip install -e 01_external/LLaVA-Med` or "
        "`export LLAVA_MED_HOME=/path/to/LLaVA-Med`. "
        f"Looked in: {tried}"
    )


# Sentinel for MedicalVQALoader: use env/fallback MC root. ``mc_data_root=None`` = freetext only.
MC_DEFAULT = object()
