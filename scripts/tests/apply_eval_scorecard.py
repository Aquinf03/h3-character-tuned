#!/usr/bin/env python3
"""Apply a filled Ref2VA scorecard to AQ benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


METRICS = ["identity", "talking_motion", "audio_sync", "prompt_following", "generalization"]
WEIGHTS = {
    "identity": 0.35,
    "talking_motion": 0.30,
    "audio_sync": 0.05,
    "prompt_following": 0.15,
    "generalization": 0.15,
}


def parse_score(value: str, metric: str, row_number: int) -> float:
    if value.strip() == "":
        raise ValueError(f"missing {metric} score on CSV row {row_number}")
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise ValueError(f"{metric} score on CSV row {row_number} is outside 0..1: {score}")
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("evals/ref2va_eval_batch_001_scorecard.csv"))
    parser.add_argument("--checkpoint", default="ckpt_100")
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--notes", default="5-sample held-out Ref2VA eval batch 001")
    args = parser.parse_args()

    rows = []
    with args.csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, start=2):
            rows.append({metric: parse_score(row.get(metric, ""), metric, row_number) for metric in METRICS})
    if not rows:
        raise RuntimeError(f"no scored rows found in {args.csv}")

    means = {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS}
    weighted = sum(means[metric] * WEIGHTS[metric] for metric in METRICS)
    notes = f"{args.notes}; aggregate={weighted:.3f}; n={len(rows)}"

    cmd = [
        sys.executable,
        "scripts/tests/record_checkpoint_eval.py",
        "--checkpoint",
        args.checkpoint,
        "--step",
        str(args.step),
        "--identity",
        f"{means['identity']:.6f}",
        "--talking-motion",
        f"{means['talking_motion']:.6f}",
        "--audio-sync",
        f"{means['audio_sync']:.6f}",
        "--prompt-following",
        f"{means['prompt_following']:.6f}",
        "--generalization",
        f"{means['generalization']:.6f}",
        "--n",
        str(len(rows)),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, "scripts/helpers/benchmark_report.py"], check=True)

    summary = {
        "checkpoint": args.checkpoint,
        "step": args.step,
        "n": len(rows),
        "means": means,
        "weighted_score": weighted,
        "scorecard": str(args.csv),
    }
    out = Path("artifacts/h3/ref2va_eval_batch_001_score_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
