#!/usr/bin/env python3
"""Compatibility wrapper. Canonical path: src/data/prepare_ta20_subset.py."""
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "src" / "data" / "prepare_ta20_subset.py"),
    run_name="__main__",
)
