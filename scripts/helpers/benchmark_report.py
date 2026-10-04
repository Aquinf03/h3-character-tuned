#!/usr/bin/env python3
"""Render checkpoint benchmark progress as Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("artifacts/h3/checkpoint_evals.jsonl"))
    ap.add_argument("--checkpoints", type=Path, default=Path("src/configs/progress_checkpoints.json"))
    ap.add_argument("--out", type=Path, default=Path("evals/progress_report.md"))
    args = ap.parse_args()

    cfg = read_json(args.checkpoints)
    rows = read_jsonl(args.scores)
    by_name = {row["checkpoint"]: row for row in rows}

    lines = [
        "# H3 Avatar Checkpoint Progress",
        "",
        "| checkpoint | step | score | target | identity | motion | audio | prompt | generalization | result | notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for ckpt in cfg["checkpoints"]:
        row = by_name.get(ckpt["name"], {})
        parts = row.get("parts") or {}
        score = row.get("score")
        target = ckpt.get("min_score")
        result = ""
        if score is not None and target is not None:
            result = "pass" if float(score) >= float(target) else "fail"
        lines.append(
            "| "
            + " | ".join(
                [
                    ckpt["name"],
                    str(ckpt["step"]),
                    fmt(score),
                    fmt(target),
                    fmt(parts.get("identity")),
                    fmt(parts.get("talking_motion")),
                    fmt(parts.get("audio_sync")),
                    fmt(parts.get("prompt_following")),
                    fmt(parts.get("generalization")),
                    result,
                    str(row.get("notes") or ""),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Benchmark Suites",
            "",
            "Source: `evals/avatar_benchmarks.json` and `evals/progress_prompts.jsonl`.",
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(args.out))


if __name__ == "__main__":
    main()
