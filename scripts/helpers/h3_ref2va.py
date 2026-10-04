"""Compatibility wrapper. Canonical AQ adapter: src/h3_ref2va.py."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.h3_ref2va import evaluate, fit, write_inspect

__all__ = ["evaluate", "fit", "write_inspect"]
