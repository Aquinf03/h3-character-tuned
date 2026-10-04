#!/usr/bin/env python3
"""Objective sanity checks and manual scorecard for a Ref2VA eval batch."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


METRICS = ["identity", "talking_motion", "audio_sync", "prompt_following", "generalization"]


def ffprobe(path: Path) -> dict:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    data = json.loads(proc.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    fmt = data.get("format", {})
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "duration": float(fmt.get("duration", 0.0) or 0.0),
        "video_codec": video.get("codec_name"),
        "width": int(video.get("width", 0) or 0),
        "height": int(video.get("height", 0) or 0),
        "avg_frame_rate": video.get("avg_frame_rate"),
        "nb_frames": int(video.get("nb_frames", 0) or 0),
        "has_audio": bool(audio),
        "audio_codec": audio.get("codec_name"),
        "sample_rate": int(audio.get("sample_rate", 0) or 0) if audio else 0,
        "audio_channels": int(audio.get("channels", 0) or 0) if audio else 0,
    }


def localize(modal_path: str, generated_dir: Path, reference_dir: Path) -> Path:
    name = Path(modal_path).name
    return generated_dir / name if "generated" in modal_path else reference_dir / name


def objective_pass(
    generated: dict,
    expected_width: int,
    expected_height: int,
    *,
    expect_audio: bool,
) -> tuple[bool, list[str]]:
    issues = []
    if generated["bytes"] <= 0:
        issues.append("empty file")
    if generated["width"] != expected_width or generated["height"] != expected_height:
        issues.append(f"unexpected resolution {generated['width']}x{generated['height']}")
    if generated["duration"] < 4.5:
        issues.append(f"short duration {generated['duration']:.2f}s")
    if expect_audio and not generated["has_audio"]:
        issues.append("missing audio stream")
    return not issues, issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=Path("artifacts/h3/ref2va_eval_batch_001_objective.json"))
    parser.add_argument("--md-out", type=Path, default=Path("evals/ref2va_eval_batch_001_objective.md"))
    parser.add_argument("--csv-out", type=Path, default=Path("evals/ref2va_eval_batch_001_scorecard.csv"))
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    for item in manifest["items"]:
        ref_path = localize(item["reference_video"], args.generated_dir, args.reference_dir)
        gen_path = localize(item["generated_video"], args.generated_dir, args.reference_dir)
        reference = ffprobe(ref_path)
        generated = ffprobe(gen_path)
        expect_audio = bool(item.get("generate_audio", True))
        passed, issues = objective_pass(
            generated,
            int(item["width"]),
            int(item["height"]),
            expect_audio=expect_audio,
        )
        rows.append(
            {
                "index": item["index"],
                "seed": item["seed"],
                "reference_path": str(ref_path),
                "generated_path": str(gen_path),
                "reference": reference,
                "generated": generated,
                "objective_pass": passed,
                "issues": issues,
                "prompt": item["prompt"],
            }
        )

    result = {
        "manifest": str(args.manifest),
        "items": rows,
        "passed": sum(1 for row in rows if row["objective_pass"]),
        "total": len(rows),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "seed",
                *METRICS,
                "overall",
                "notes",
                "reference_path",
                "generated_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "index": row["index"],
                    "seed": row["seed"],
                    **{metric: "" for metric in METRICS},
                    "overall": "",
                    "notes": "",
                    "reference_path": row["reference_path"],
                    "generated_path": row["generated_path"],
                }
            )

    table = []
    for row in rows:
        gen = row["generated"]
        ref = row["reference"]
        status = "pass" if row["objective_pass"] else "fail: " + "; ".join(row["issues"])
        table.append(
            "| {index} | {seed} | {ref_dur:.2f}s | {gen_dur:.2f}s | {res} | {audio} | {status} |".format(
                index=row["index"],
                seed=row["seed"],
                ref_dur=ref["duration"],
                gen_dur=gen["duration"],
                res=f"{gen['width']}x{gen['height']}",
                audio="yes" if gen["has_audio"] else "no",
                status=status,
            )
        )

    args.md_out.write_text(
        "\n".join(
            [
                "# Ref2VA Eval Batch 001",
                "",
                f"Objective pass: {result['passed']}/{result['total']}",
                "",
                "Gallery: `evals/index.html`",
                "",
                "| sample | seed | reference duration | generated duration | generated resolution | generated audio | objective status |",
                "| ---: | ---: | ---: | ---: | --- | --- | --- |",
                *table,
                "",
                "## Manual Scorecard",
                "",
                f"CSV: `{args.csv_out}`",
                "",
                "Score each metric from 0.0 to 1.0 after watching the gallery:",
                "",
                "- `identity`: generated subject resembles the reference character.",
                "- `talking_motion`: face/body motion is plausible for talking.",
                "- `audio_sync`: mouth motion and audio feel aligned.",
                "- `prompt_following`: output follows the friendly talking prompt.",
                "- `generalization`: held-out validation reference does not collapse to a memorized training character.",
                "",
                "After manual scoring, record the checkpoint with `scripts/tests/record_checkpoint_eval.py` and run AQ eval.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(args.md_out)
    print(args.csv_out)
    print(args.json_out)


if __name__ == "__main__":
    main()
