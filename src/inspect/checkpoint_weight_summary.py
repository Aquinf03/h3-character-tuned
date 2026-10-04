#!/usr/bin/env python3
"""Summarize H3 LoRA checkpoint tensors for lightweight AQIT-style inspection."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def tensor_stats(t: torch.Tensor) -> dict[str, float | int | list[int] | str]:
    x = t.detach().float().cpu()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "numel": int(t.numel()),
        "mean": float(x.mean().item()) if x.numel() else 0.0,
        "std": float(x.std(unbiased=False).item()) if x.numel() else 0.0,
        "abs_mean": float(x.abs().mean().item()) if x.numel() else 0.0,
        "abs_max": float(x.abs().max().item()) if x.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(x).item()) if x.numel() else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out-json", type=Path, default=Path("artifacts/h3/weight_summary_ckpt100.json"))
    ap.add_argument("--out-md", type=Path, default=Path("evals/weight_summary_ckpt100.md"))
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = ckpt.get("model", ckpt)
    tensors = {k: v for k, v in model.items() if torch.is_tensor(v)}
    groups: dict[str, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    for name, tensor in tensors.items():
        root = name.split(".", 1)[0]
        groups[root].append((name, tensor))

    group_rows = []
    for root, items in sorted(groups.items()):
        numel = sum(int(t.numel()) for _, t in items)
        l2 = math.sqrt(sum(float(torch.linalg.vector_norm(t.detach().float()).item()) ** 2 for _, t in items))
        group_rows.append({"group": root, "tensors": len(items), "numel": numel, "l2": l2})

    largest = sorted(
        ({"name": name, **tensor_stats(tensor)} for name, tensor in tensors.items()),
        key=lambda row: float(row["l2"]),
        reverse=True,
    )[:25]

    out = {
        "checkpoint": str(args.checkpoint),
        "step": ckpt.get("step"),
        "tensor_count": len(tensors),
        "total_parameters": sum(int(t.numel()) for t in tensors.values()),
        "groups": group_rows,
        "largest_l2_tensors": largest,
        "note": "Local tensor inspection because aqit check weight requires aquin login in this environment.",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# Checkpoint 100 Weight Summary",
        "",
        f"Checkpoint: `{args.checkpoint}`",
        f"Step: `{ckpt.get('step')}`",
        f"LoRA tensors: `{len(tensors)}`",
        f"LoRA parameters: `{out['total_parameters']:,}`",
        "",
        "## Groups",
        "",
        "| group | tensors | params | l2 norm |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in group_rows:
        lines.append(f"| {row['group']} | {row['tensors']} | {row['numel']:,} | {row['l2']:.4f} |")
    lines.extend(["", "## Largest Tensors By L2 Norm", "", "| tensor | shape | l2 | abs max |", "| --- | --- | ---: | ---: |"])
    for row in largest[:12]:
        lines.append(f"| `{row['name']}` | `{row['shape']}` | {row['l2']:.4f} | {row['abs_max']:.4f} |")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_md)


if __name__ == "__main__":
    main()
