#!/usr/bin/env python3
"""Compatibility wrapper. Canonical path: src/inspect/render_training_dashboard.py."""
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "src" / "inspect" / "render_training_dashboard.py"),
    run_name="__main__",
)
