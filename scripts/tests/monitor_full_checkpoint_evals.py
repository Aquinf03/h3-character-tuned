#!/usr/bin/env python3
"""Monitor full H3 LoRA training and evaluate each 1000-step checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


def run(cmd: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def run_stream(cmd: list[str], *, cwd: Path) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def list_checkpoints(cwd: Path, volume: str, run_dir: str) -> list[int]:
    proc = run(["modal", "volume", "ls", volume, run_dir], cwd=cwd, check=False)
    steps: set[int] = set()
    for match in re.finditer(r"checkpoint-(\d+)\.pt", proc.stdout):
        steps.add(int(match.group(1)))
    return sorted(steps)


def latest_training_status(cwd: Path, app_id: str) -> dict:
    proc = run(["modal", "app", "logs", app_id, "--timestamps"], cwd=cwd, check=False)
    latest_step = None
    latest_val = None
    for line in proc.stdout.splitlines():
        m = re.search(r"step=(\d+) loss=([0-9.]+) loss_v=([0-9.]+) loss_a=([0-9.]+)", line)
        if m:
            latest_step = {
                "step": int(m.group(1)),
                "loss": float(m.group(2)),
                "loss_v": float(m.group(3)),
                "loss_a": float(m.group(4)),
            }
        v = re.search(r"VAL step=(\d+) loss_v=([0-9.]+) u0\.3=([0-9.]+) u0\.6=([0-9.]+) u0\.9=([0-9.]+)", line)
        if v:
            latest_val = {
                "step": int(v.group(1)),
                "loss_v": float(v.group(2)),
                "u0.3": float(v.group(3)),
                "u0.6": float(v.group(4)),
                "u0.9": float(v.group(5)),
            }
    return {"latest_step": latest_step, "latest_val": latest_val}


def ensure_reference_aliases(eval_dir: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["items"]:
        ref_name = Path(item["reference_video"]).name
        ref_path = eval_dir / ref_name
        noaudio = eval_dir / f"{Path(ref_name).stem}_noaudio.mp4"
        if not ref_path.exists() and noaudio.exists():
            ref_path.symlink_to(noaudio.name)


def build_reports(cwd: Path, step: int, local_eval_dir: Path) -> None:
    manifest = local_eval_dir / "eval_batch_manifest.json"
    ensure_reference_aliases(local_eval_dir, manifest)
    gallery = cwd / "evals" / f"ref2va_eval_gallery_full_ckpt_{step}_diverse3.html"
    run(
        [
            "python3",
            "scripts/tests/render_eval_gallery.py",
            "--manifest",
            str(manifest),
            "--generated-dir",
            str(local_eval_dir),
            "--reference-dir",
            str(local_eval_dir),
            "--out",
            str(gallery),
        ],
        cwd=cwd,
    )
    run(
        [
            "python3",
            "scripts/tests/eval_ref2va_batch.py",
            "--manifest",
            str(manifest),
            "--generated-dir",
            str(local_eval_dir),
            "--reference-dir",
            str(local_eval_dir),
            "--json-out",
            f"artifacts/h3/ref2va_full_ckpt{step}_diverse3_objective.json",
            "--md-out",
            f"evals/ref2va_full_ckpt{step}_diverse3_objective.md",
            "--csv-out",
            f"evals/ref2va_full_ckpt{step}_diverse3_scorecard.csv",
        ],
        cwd=cwd,
    )
    review_dir = cwd / "artifacts" / "review" / f"full_ckpt{step}_diverse3"
    review_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for item in data["items"]:
        idx = int(item["index"])
        ref = local_eval_dir / Path(item["reference_video"]).name
        gen = local_eval_dir / Path(item["generated_video"]).name
        for label, path in [("ref", ref), ("gen", gen)]:
            out = review_dir / f"sample{idx}_{label}_sheet.jpg"
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(path),
                    "-vf",
                    "fps=1,scale=256:256,tile=6x1",
                    "-frames:v",
                    "1",
                    str(out),
                ],
                cwd=cwd,
                check=False,
            )
    print(f"gallery=http://127.0.0.1:8765/{gallery.relative_to(cwd).as_posix()}", flush=True)


def eval_checkpoint(cwd: Path, args: argparse.Namespace, step: int) -> None:
    checkpoint = f"{args.run_dir}/checkpoint-{step}.pt"
    remote_out = f"{args.generated_root}/eval_ckpt_{step}_diverse3"
    local_parent = cwd / "artifacts" / "generated" / "ref2va-lora-full-8000"
    local_eval_dir = local_parent / f"eval_ckpt_{step}_diverse3"

    run_stream(
        [
            "modal",
            "run",
            "scripts/helpers/modal_h3_avatar.py::generate_ref2va_eval_batch",
            "--checkpoint",
            checkpoint,
            "--reference-dir",
            args.reference_dir,
            "--output-dir",
            remote_out,
            "--limit",
            "3",
            "--num-inference-steps",
            str(args.inference_steps),
            "--seed",
            str(args.seed_base + step),
            "--no-include-reference-audio",
        ],
        cwd=cwd,
    )
    local_parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "modal",
            "volume",
            "get",
            args.volume,
            remote_out,
            str(local_parent),
            "--force",
        ],
        cwd=cwd,
    )
    build_reports(cwd, step, local_eval_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-id", default="ap-Mk5JAjytLfnaYobhIhrf1e")
    parser.add_argument("--volume", default="h3-avatar-runs")
    parser.add_argument("--volume-run-dir", default="/h3/ref2va-lora-full-8000")
    parser.add_argument("--run-dir", default="/runs/h3/ref2va-lora-full-8000")
    parser.add_argument("--generated-root", default="/runs/generated/ref2va-lora-full-8000")
    parser.add_argument("--reference-dir", default="/runs/samples/ta20_full_test_eval_diverse3")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--start-step", type=int, default=2000)
    parser.add_argument("--max-step", type=int, default=8000)
    parser.add_argument("--inference-steps", type=int, default=6)
    parser.add_argument("--seed-base", type=int, default=200000)
    args = parser.parse_args()

    cwd = Path.cwd()
    state_dir = cwd / "artifacts" / "monitor"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "full_checkpoint_evals_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"evaluated": []}

    print("monitor started", datetime.now().isoformat(timespec="seconds"), flush=True)
    while True:
        status = latest_training_status(cwd, args.app_id)
        checkpoints = list_checkpoints(cwd, args.volume, args.volume_run_dir)
        print(
            json.dumps(
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "status": status,
                    "checkpoints": checkpoints,
                    "evaluated": state["evaluated"],
                },
                indent=2,
            ),
            flush=True,
        )
        pending = [
            s
            for s in checkpoints
            if s >= args.start_step and s <= args.max_step and s % 1000 == 0 and s not in state["evaluated"]
        ]
        for step in pending:
            try:
                eval_checkpoint(cwd, args, step)
                state["evaluated"].append(step)
                state["evaluated"] = sorted(set(state["evaluated"]))
                state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            except Exception as exc:
                print(f"eval failed for checkpoint {step}: {exc}", flush=True)
                break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
