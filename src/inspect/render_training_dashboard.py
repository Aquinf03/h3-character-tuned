#!/usr/bin/env python3
"""Render a lightweight HTML dashboard from an H3 trainer log."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


STEP_RE = re.compile(
    r"step=(?P<step>\d+)\s+loss=(?P<loss>[0-9.]+)\s+loss_v=(?P<loss_v>[0-9.]+)\s+loss_a=(?P<loss_a>[0-9.]+)"
)
VAL_RE = re.compile(r"VAL step=(?P<step>\d+)\s+loss_v=(?P<loss_v>[0-9.]+)")


def parse_log(path: Path) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    steps: list[dict[str, float]] = []
    vals: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = STEP_RE.search(line)
        if match:
            steps.append({k: float(v) for k, v in match.groupdict().items()})
            continue
        match = VAL_RE.search(line)
        if match:
            vals.append({k: float(v) for k, v in match.groupdict().items()})
    return steps, vals


def render_svg(series: list[dict[str, float]], vals: list[dict[str, float]], key: str, title: str) -> str:
    width, height = 980, 260
    pad_l, pad_t, pad_r, pad_b = 54, 22, 18, 38
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    if not series:
        return "<p>No data.</p>"
    xs = [row["step"] for row in series]
    ys = [row[key] for row in series]
    if key == "loss_v":
        xs += [row["step"] for row in vals]
        ys += [row["loss_v"] for row in vals]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0.0, max(ys) * 1.08 if ys else 1.0

    def x(step: float) -> float:
        span = max(1.0, x_max - x_min)
        return pad_l + ((step - x_min) / span) * plot_w

    def y(value: float) -> float:
        span = max(1e-9, y_max - y_min)
        return pad_t + plot_h - ((value - y_min) / span) * plot_h

    points = " ".join(f"{x(row['step']):.1f},{y(row[key]):.1f}" for row in series)
    val_points = "".join(
        f"<circle cx='{x(row['step']):.1f}' cy='{y(row['loss_v']):.1f}' r='4' class='val' />"
        for row in vals
    ) if key == "loss_v" else ""
    checkpoints = "".join(
        f"<line x1='{x(step):.1f}' x2='{x(step):.1f}' y1='{pad_t}' y2='{pad_t + plot_h}' class='ckpt' />"
        for step in (20, 40, 60, 80, 100)
        if x_min <= step <= x_max
    )
    return f"""
<section>
  <h2>{html.escape(title)}</h2>
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
    <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" class="axis"/>
    <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" class="axis"/>
    {checkpoints}
    <polyline points="{points}" class="line"/>
    {val_points}
    <text x="{pad_l}" y="{height - 10}" class="tick">step {int(x_min)}</text>
    <text x="{width - 88}" y="{height - 10}" class="tick">step {int(x_max)}</text>
    <text x="8" y="{pad_t + 12}" class="tick">{y_max:.2f}</text>
    <text x="18" y="{pad_t + plot_h}" class="tick">0</text>
  </svg>
</section>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path, default=Path("evals/training_dashboard.html"))
    args = ap.parse_args()

    steps, vals = parse_log(args.log)
    last = steps[-1] if steps else {}
    data_json = json.dumps({"steps": steps, "validation": vals}, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"""<!doctype html>
<meta charset="utf-8">
<title>H3 Avatar LoRA Training</title>
<style>
body {{ margin: 32px; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #101214; color: #eceff3; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
h2 {{ margin: 26px 0 10px; font-size: 16px; color: #cbd5e1; }}
.meta {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 18px 0 8px; }}
.pill {{ border: 1px solid #39414c; background: #171b20; padding: 8px 10px; border-radius: 6px; }}
svg {{ width: 100%; max-width: 980px; height: auto; background: #151a20; border: 1px solid #303743; border-radius: 6px; }}
.axis {{ stroke: #687281; stroke-width: 1; }}
.line {{ fill: none; stroke: #7dd3fc; stroke-width: 2.2; }}
.val {{ fill: #fbbf24; stroke: #111827; stroke-width: 1; }}
.ckpt {{ stroke: #475569; stroke-dasharray: 4 4; }}
.tick {{ fill: #94a3b8; font-size: 11px; }}
details {{ margin-top: 22px; max-width: 980px; }}
pre {{ overflow: auto; background: #151a20; padding: 14px; border-radius: 6px; border: 1px solid #303743; }}
</style>
<h1>H3 Avatar LoRA Training</h1>
<div class="meta">
  <div class="pill">last step: {int(last.get("step", 0))}</div>
  <div class="pill">loss: {last.get("loss", 0):.4f}</div>
  <div class="pill">video loss: {last.get("loss_v", 0):.4f}</div>
  <div class="pill">audio loss: {last.get("loss_a", 0):.4f}</div>
  <div class="pill">validation points: {len(vals)}</div>
</div>
{render_svg(steps, vals, "loss", "Total Loss")}
{render_svg(steps, vals, "loss_v", "Video Loss + Validation Markers")}
{render_svg(steps, vals, "loss_a", "Audio Loss")}
<details><summary>Raw parsed data</summary><pre>{html.escape(data_json)}</pre></details>
""", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
