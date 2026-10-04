"""AQ custom method for MiniMax H3 Ref2VA LoRA.

This adapter keeps AQ responsible for run identity, artifacts, eval bookkeeping,
and reproducibility while delegating H3-specific cache/training work to an
external MiniMax-H3 fine-tuning repository.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


def _section(rec: dict[str, Any], key: str) -> dict[str, Any]:
    value = rec.get(key)
    return value if isinstance(value, dict) else {}


def _resolve_value(raw: Any) -> str:
    value = "" if raw is None else str(raw)
    if value.startswith("env:"):
        return os.environ.get(value.split(":", 1)[1], "")
    return value


def _train_root(rec: dict[str, Any]) -> Path:
    return Path(str(rec.get("_train") or ".")).resolve()


def _default_trainer_dir() -> Path:
    return Path(__file__).resolve().parent


def _run(cmd: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise SystemExit(f"H3 command failed ({proc.returncode}); see {log_path}")


def _command(script: str, trainer_dir: Path) -> list[str]:
    script_path = trainer_dir / script
    if script_path.is_file():
        return ["python3", str(script_path)]
    return shlex.split(script)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _int_list(raw: Any, fallback: list[int]) -> list[int]:
    if raw is None:
        return fallback
    if isinstance(raw, int):
        return [raw]
    out: list[int] = []
    for part in str(raw).replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or fallback


def _checkpoint_name(step: int) -> str:
    return f"ckpt_{step:03d}"


def fit(src: Path, rec: dict[str, Any]) -> dict[str, Any]:
    train = _train_root(rec)
    h3 = _section(rec, "h3")
    dry_run = bool(rec.get("dry_run", True))

    trainer_raw = _resolve_value(h3.get("trainer_dir"))
    trainer_dir = Path(trainer_raw).expanduser() if trainer_raw else _default_trainer_dir()
    model_dir = Path(_resolve_value(h3.get("model_dir") or rec.get("model"))).expanduser()
    cache_dir = (train / str(h3.get("cache_dir") or "artifacts/h3/cache")).resolve()
    output_dir = (train / str(h3.get("output_dir") or "artifacts/h3/runs/ref2va-lora")).resolve()
    checkpoint_dir = (train / str(h3.get("checkpoint_dir") or output_dir / "checkpoints")).resolve()
    train_manifest = (train / str(h3.get("train_manifest") or src)).resolve()
    val_manifest = (train / str(h3.get("val_manifest") or src)).resolve()
    log_path = train / "artifacts" / "h3" / "logs" / "h3_ref2va_lora.log"
    steps = int(rec.get("steps") or 100)
    checkpoint_steps = _int_list(rec.get("checkpoint_steps"), [max(1, steps // 5) * i for i in range(1, 6)])

    plan = {
        "adapter": "h3_ref2va",
        "dry_run": dry_run,
        "trainer_dir": str(trainer_dir),
        "model_dir": str(model_dir),
        "variant": str(rec.get("variant") or "ref2va"),
        "precision": str(rec.get("precision") or "bf16"),
        "objective": str(rec.get("objective") or "lora"),
        "rank": int(rec.get("rank") or 8),
        "steps": steps,
        "checkpoint_steps": checkpoint_steps,
        "lr": float(rec.get("lr") or 1e-4),
        "seed": int(rec.get("seed") or 7),
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "source_manifest": str(src),
        "started_at": int(time.time()),
    }
    _write_json(train / "artifacts" / "h3" / "plan.json", plan)
    _write_json(
        train / "artifacts" / "h3" / "checkpoint_manifest.json",
        {
            "checkpoints": [
                {
                    "name": _checkpoint_name(step),
                    "step": step,
                    "expected_path": str(checkpoint_dir / _checkpoint_name(step)),
                }
                for step in checkpoint_steps
            ]
        },
    )

    rows = _read_rows(src)
    if dry_run:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for step in checkpoint_steps:
            checkpoint = checkpoint_dir / _checkpoint_name(step)
            checkpoint.mkdir(parents=True, exist_ok=True)
            (checkpoint / "DRY_RUN_LORA_PLACEHOLDER").write_text(
                "Set dry_run: false after configuring H3 paths.\n",
                encoding="utf-8",
            )
        return {
            "kind": "h3_ref2va",
            "backend": "external_h3_trainer",
            "dry_run": True,
            "adapter_path": str(checkpoint_dir / _checkpoint_name(checkpoint_steps[-1])),
            "model_path": str(output_dir),
            "rank": plan["rank"],
            "steps": 0,
            "checkpoint_steps": checkpoint_steps,
            "checkpoint_manifest": "artifacts/h3/checkpoint_manifest.json",
            "train_rows": len(rows),
            "train_loss": None,
            "plan": "artifacts/h3/plan.json",
        }

    if not trainer_dir.is_dir():
        raise SystemExit("H3 trainer missing. Set H3_TRAINER_DIR or h3.trainer_dir.")
    if not model_dir.exists():
        raise SystemExit("MiniMax H3 model dir missing. Set H3_MODEL_DIR or h3.model_dir.")
    if not train_manifest.is_file():
        raise SystemExit(f"train manifest missing: {train_manifest}")

    prepare_cmd = _command(os.environ.get("H3_PREPARE_CMD", "prepare_cache.py"), trainer_dir)
    train_cmd = _command(os.environ.get("H3_TRAIN_CMD", "train.py"), trainer_dir)

    _run(
        prepare_cmd
        + [
            "--metadata",
            str(train_manifest),
            "--output",
            str(cache_dir),
            "--model",
            str(model_dir),
            "--height",
            str(int(h3.get("height") or 256)),
            "--width",
            str(int(h3.get("width") or 256)),
            "--frames",
            str(int(h3.get("frames") or 22)),
            "--encode-text",
            "--encode-audio",
        ],
        cwd=trainer_dir,
        log_path=log_path,
    )
    train_args = train_cmd + [
            "--model",
            str(model_dir),
            "--variant",
            str(plan["variant"]),
            "--cache",
            str(cache_dir),
            "--output",
            str(output_dir),
            "--max-steps",
            str(plan["steps"]),
            "--trainable",
            "lora",
            "--lora-rank",
            str(plan["rank"]),
            "--lr",
            str(plan["lr"]),
    ]
    if os.environ.get("H3_CHECKPOINT_ARGS"):
        train_args += shlex.split(os.environ["H3_CHECKPOINT_ARGS"])
    else:
        interval = checkpoint_steps[0] if len(set(checkpoint_steps)) == 1 else checkpoint_steps[0]
        if len(checkpoint_steps) >= 2:
            diffs = [b - a for a, b in zip(checkpoint_steps, checkpoint_steps[1:])]
            if len(set(diffs)) == 1:
                interval = diffs[0]
        train_args += ["--ckpt-every", str(interval), "--val-every", str(interval)]
    _run(
        train_args,
        cwd=trainer_dir,
        log_path=log_path,
    )

    return {
        "kind": "h3_ref2va",
        "backend": "external_h3_trainer",
        "dry_run": False,
        "adapter_path": str(output_dir),
        "model_path": str(output_dir),
        "rank": plan["rank"],
        "steps": plan["steps"],
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_manifest": "artifacts/h3/checkpoint_manifest.json",
        "train_rows": len(_read_rows(train_manifest)),
        "train_loss": None,
        "plan": "artifacts/h3/plan.json",
        "log": "artifacts/h3/logs/h3_ref2va_lora.log",
    }


def evaluate(model: dict[str, Any], src: Path, rec: dict[str, Any]) -> tuple[float, int]:
    rows = _read_rows(src)
    if model.get("dry_run"):
        return 0.0, max(1, len(rows))
    # Real avatar scoring is intentionally external: write eval metrics to
    # artifacts/h3/avatar_gate.json with {"score": <0..1>, "n": <int>}.
    train = _train_root(rec)
    gate = train / "artifacts" / "h3" / "avatar_gate.json"
    if not gate.is_file():
        raise SystemExit(
            "missing artifacts/h3/avatar_gate.json; run generation/identity/audio probes first"
        )
    data = json.loads(gate.read_text(encoding="utf-8"))
    return float(data.get("score", 0.0)), int(data.get("n") or max(1, len(rows)))


def write_inspect(train: Path, model: dict[str, Any]) -> str:
    path = train / "artifacts" / "inspect.md"
    lines = [
        "# H3 Ref2VA LoRA Inspect",
        "",
        f"- backend: {model.get('backend')}",
        f"- dry_run: {model.get('dry_run')}",
        f"- adapter_path: {model.get('adapter_path')}",
        f"- rank: {model.get('rank')}",
        f"- steps: {model.get('steps')}",
        f"- checkpoint_steps: {model.get('checkpoint_steps')}",
        f"- checkpoint_manifest: {model.get('checkpoint_manifest')}",
        f"- train_rows: {model.get('train_rows')}",
        "",
        "Next AQIT checks:",
        "",
        "```bash",
        "cd /Users/sachin/AQIT",
        "aqit eval custom --prompts /Users/sachin/h3-avatar-aq/evals/probes.jsonl",
        "aqit diff weight --checkpoint /Users/sachin/h3-avatar-aq/artifacts/checkpoints/last.json",
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "artifacts/inspect.md"
