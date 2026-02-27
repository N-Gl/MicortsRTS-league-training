#!/usr/bin/env python3
"""Wrapper to run the PFSP plot generator from project root."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    root_script = Path(__file__).resolve().parents[1] / "generate_pfsp_latex_plots.py"
    runpy.run_path(str(root_script), run_name="__main__")
