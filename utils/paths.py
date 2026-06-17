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
from pathlib import Path

_ENV_DATA = "MEDVQA_DATA_ROOT"
_ENV_MC = "MEDVQA_MC_ROOT"
_ENV_OUTPUTS = "MEDVQA_OUTPUTS_DIR"
_ENV_RESULTS = "MEDVQA_RESULTS_DIR"
_ENV_SOM = "MEDVQA_SOM_DIR"

_DEFAULT_DATA = "02_data"
_DEFAULT_MC = "02_data/data_mc_openai"
_DEFAULT_OUTPUTS = "05_outputs"
_DEFAULT_RESULTS = "06_results"
_DEFAULT_SOM = "02_data/som"


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


# Sentinel for MedicalVQALoader: use env/fallback MC root. ``mc_data_root=None`` = freetext only.
MC_DEFAULT = object()
