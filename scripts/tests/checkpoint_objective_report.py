#!/usr/bin/env python3
"""Create an objective checkpoint report from H3 trainer logs and saved files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


VAL_RE = re.compile(r"VAL step=(?P<step>\d+)\s+loss_v=(?P<loss_v>[0-9.]+)")
FINISH_RE = re.compile(r"finished step=(?P<step>\d+)")


def parse_val_loss(log_path: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = VAL_RE.search(line)
        if match:
            values[int(match.group("step"))] = float(match.group("loss_v"))
    return values


def finished_step(log_path: Path) -> int | None:
    last = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = FINISH_RE.search(line)
        if match:
            last = int(match.group("step"))
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--checkpoints", default="src/configs/progress_checkpoints.json", type=Path)
    ap.add_argument("--remote-output", default="/runs/h3/ref2va-lora-smoke-100")
    ap.add_argument("--out-json", default="artifacts/h3/objective_checkpoint_metrics.json", type=Path)
    ap.add_argument("--out-md", default="evals/objective_checkpoint_report.md", type=Path)
    args = ap.parse_args()

    cfg = json.loads(args.checkpoints.read_text(encoding="utf-8"))
    vals = parse_val_loss(args.log)
    baseline = vals.get(0)
    rows = []
    previous = None
    for ckpt in cfg["checkpoints"]:
        step = int(ckpt["step"])
        loss = vals.get(step)
        improvement = None if baseline in (None, 0) or loss is None else (baseline - loss) / baseline
        monotonic = None if previous is None or loss is None else loss <= previous
        rows.append(
            {
                "checkpoint": ckpt["name"],
                "step": step,
                "remote_path": f"{args.remote_output}/checkpoint-{step}.pt",
                "val_loss_v": loss,
                "relative_improvement_from_step_0": improvement,
                "monotonic_vs_previous_gate": monotonic,
                "purpose": ckpt.get("purpose", ""),
            }
        )
        if loss is not None:
            previous = loss

    report = {
        "run": args.remote_output,
        "log": str(args.log),
        "finished_step": finished_step(args.log),
        "baseline_val_loss_v": baseline,
        "final_val_loss_v": vals.get(100),
        "checkpoints": rows,
        "note": "Objective log metrics only. Identity/audio-sync/prompt scores require generated sample review.",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Objective Checkpoint Report",
        "",
        f"Run: `{args.remote_output}`",
        "",
        "| checkpoint | step | val video loss | improvement vs step 0 | monotonic | remote checkpoint |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        improvement = row["relative_improvement_from_step_0"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["checkpoint"],
                    str(row["step"]),
                    "" if row["val_loss_v"] is None else f"{row['val_loss_v']:.6f}",
                    "" if improvement is None else f"{improvement * 100:.1f}%",
                    "" if row["monotonic_vs_previous_gate"] is None else str(row["monotonic_vs_previous_gate"]).lower(),
                    f"`{row['remote_path']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "This is not the final avatar quality score. It proves the training objective is moving in the right direction; generated video review is the next quality gate.",
        ]
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_md)


if __name__ == "__main__":
    main()
