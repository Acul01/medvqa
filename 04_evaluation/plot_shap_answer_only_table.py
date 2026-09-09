#!/usr/bin/env python3
"""Backward-compatible wrapper around plot_shap_table.py."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("plot_shap_table.py")), run_name="__main__")
