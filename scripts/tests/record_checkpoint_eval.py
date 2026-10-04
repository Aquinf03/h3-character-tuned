#!/usr/bin/env python3
"""Append a benchmark score row for one checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


WEIGHTS = {
    "identity": 0.35,
    "talking_motion": 0.30,
    "audio_sync": 0.05,
    "prompt_following": 0.15,
    "generalization": 0.15,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--identity", type=float, required=True)
    ap.add_argument("--talking-motion", type=float, required=True)
    ap.add_argument("--audio-sync", type=float, required=True)
    ap.add_argument("--prompt-following", type=float, required=True)
    ap.add_argument("--generalization", type=float, required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", type=Path, default=Path("artifacts/h3/checkpoint_evals.jsonl"))
    ap.add_argument("--gate-out", type=Path, default=Path("artifacts/h3/avatar_gate.json"))
    args = ap.parse_args()

    parts = {
        "identity": args.identity,
        "talking_motion": args.talking_motion,
        "audio_sync": args.audio_sync,
        "prompt_following": args.prompt_following,
        "generalization": args.generalization,
    }
    score = sum(parts[k] * WEIGHTS[k] for k in WEIGHTS)
    row = {
        "checkpoint": args.checkpoint,
        "step": args.step,
        "score": round(score, 6),
        "n": args.n,
        "parts": parts,
        "notes": args.notes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    # AQ's eval hook reads avatar_gate.json. Keep it pointed at the most recent
    # reviewed checkpoint so `aq eval` reflects the latest decision point.
    args.gate_out.parent.mkdir(parents=True, exist_ok=True)
    args.gate_out.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    print(f"{args.checkpoint} score={score:.4f}")


if __name__ == "__main__":
    main()
