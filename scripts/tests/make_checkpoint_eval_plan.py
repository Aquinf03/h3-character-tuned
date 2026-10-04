#!/usr/bin/env python3
"""Create sample-generation tasks for every progress checkpoint.

The output is JSONL. Each row says: checkpoint, suite, prompt, ref_id, and
reference/video paths when present in the split manifests.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def by_motion(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("motion_mode") or "unknown")].append(row)
    return grouped


def choose(rows: list[dict[str, Any]], prompt: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    motion = str(prompt.get("motion_mode") or "")
    grouped = by_motion(rows)
    candidates = grouped.get(motion) or rows
    return rng.choice(candidates) if candidates else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=Path, default=Path("src/configs/progress_checkpoints.json"))
    ap.add_argument("--prompts", type=Path, default=Path("evals/progress_prompts.jsonl"))
    ap.add_argument("--train", dest="train_manifest", type=Path, default=Path("data/metadata/train.jsonl"))
    ap.add_argument("--val", dest="val_manifest", type=Path, default=Path("data/metadata/val.jsonl"))
    ap.add_argument("--test", dest="test_manifest", type=Path, default=Path("data/metadata/test.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/h3/checkpoint_eval_plan.jsonl"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    checkpoints = read_json(args.checkpoints)["checkpoints"]
    prompts = read_jsonl(args.prompts)
    splits = {
        "train": read_jsonl(args.train_manifest),
        "val": read_jsonl(args.val_manifest),
        "test": read_jsonl(args.test_manifest),
    }
    fallback = splits["val"] or splits["train"] or splits["test"]
    rng = random.Random(args.seed)

    tasks: list[dict[str, Any]] = []
    for ckpt in checkpoints:
        for i, prompt in enumerate(prompts):
            split = str(prompt.get("split") or "val")
            source = splits.get(split) or fallback
            picked = choose(source, prompt, rng)
            tasks.append(
                {
                    "checkpoint": ckpt["name"],
                    "step": ckpt["step"],
                    "suite": prompt.get("suite"),
                    "prompt_id": f"{prompt.get('suite')}_{i:02d}",
                    "prompt": prompt.get("prompt"),
                    "motion_mode": prompt.get("motion_mode"),
                    "split": split,
                    "ref_id": picked.get("ref_id"),
                    "reference_path": picked.get("reference_path"),
                    "source_video_path": picked.get("video_path"),
                    "expected_output_dir": f"artifacts/h3/generated/{ckpt['name']}/{prompt.get('suite')}_{i:02d}",
                }
            )
    write_jsonl(args.out, tasks)
    print(json.dumps({"tasks": len(tasks), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
