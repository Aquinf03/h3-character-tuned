#!/usr/bin/env python3
"""Render a local HTML review gallery for Ref2VA eval batches."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


def rel_from_gallery(path: Path, gallery: Path) -> str:
    return Path(os.path.relpath(path.resolve(), gallery.resolve().parent)).as_posix()


def localize_modal_path(modal_path: str, generated_dir: Path, reference_dir: Path) -> Path:
    name = Path(modal_path).name
    if "generated" in modal_path:
        return generated_dir / name
    return reference_dir / name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("evals/ref2va_eval_gallery.html"))
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    cards = []
    for item in manifest["items"]:
        ref = localize_modal_path(item["reference_video"], args.generated_dir, args.reference_dir)
        gen = localize_modal_path(item["generated_video"], args.generated_dir, args.reference_dir)
        poster_dir = args.generated_dir / "posters"
        ref_poster = poster_dir / f"ref_{item['index']:03d}.jpg"
        gen_poster = poster_dir / f"gen_{item['index']:03d}.jpg"
        ref_rel = rel_from_gallery(ref, args.out)
        gen_rel = rel_from_gallery(gen, args.out)
        ref_poster_attr = (
            f' poster="{html.escape(rel_from_gallery(ref_poster, args.out))}"' if ref_poster.is_file() else ""
        )
        gen_poster_attr = (
            f' poster="{html.escape(rel_from_gallery(gen_poster, args.out))}"' if gen_poster.is_file() else ""
        )
        cards.append(
            f"""
<article class="pair">
  <header>
    <h2>Sample {item["index"]}</h2>
    <div class="meta">seed {item["seed"]} · {item["num_frames"]} frames · {item["height"]}x{item["width"]} · {item["num_inference_steps"]} inference steps</div>
  </header>
  <div class="videos">
    <section>
      <h3>Reference</h3>
      <video src="{html.escape(ref_rel)}" controls muted autoplay loop playsinline preload="metadata"{ref_poster_attr}></video>
      <p><a href="{html.escape(ref_rel)}">Open MP4</a> · {html.escape(ref.name)}</p>
    </section>
    <section>
      <h3>Generated</h3>
      <video src="{html.escape(gen_rel)}" controls muted autoplay loop playsinline preload="metadata"{gen_poster_attr}></video>
      <p><a href="{html.escape(gen_rel)}">Open MP4</a> · {html.escape(gen.name)} · {item["generated_bytes"]:,} bytes</p>
    </section>
  </div>
  <p class="prompt">{html.escape(item["prompt"])}</p>
</article>
"""
        )

    args.out.write_text(
        f"""<!doctype html>
<meta charset="utf-8">
<title>H3 Ref2VA Eval Batch</title>
<style>
body {{
  margin: 0;
  background: #101214;
  color: #eef2f6;
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
main {{
  max-width: 1180px;
  margin: 0 auto;
  padding: 28px 22px 48px;
}}
h1 {{
  margin: 0 0 6px;
  font-size: 28px;
  font-weight: 720;
}}
.sub {{
  color: #aab4c0;
  margin: 0 0 22px;
}}
.pair {{
  border-top: 1px solid #2e3641;
  padding: 22px 0 26px;
}}
header {{
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 12px;
}}
h2 {{
  margin: 0;
  font-size: 18px;
}}
.meta {{
  color: #aab4c0;
  font-size: 13px;
}}
.videos {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}}
h3 {{
  margin: 0 0 8px;
  color: #cbd5df;
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0;
}}
video {{
  display: block;
  width: 100%;
  aspect-ratio: 1 / 1;
  object-fit: contain;
  background: #050607;
  border: 1px solid #303843;
  border-radius: 6px;
}}
p {{
  margin: 8px 0 0;
  color: #aab4c0;
  overflow-wrap: anywhere;
}}
a {{
  color: #8ecaff;
}}
.prompt {{
  margin-top: 12px;
  color: #d7dde5;
}}
@media (max-width: 760px) {{
  header, .videos {{
    display: block;
  }}
  section + section {{
    margin-top: 16px;
  }}
}}
</style>
<main>
  <h1>H3 Ref2VA Eval Batch</h1>
  <p class="sub">Checkpoint: {Path(manifest["items"][0]["checkpoint"]).name if manifest.get("items") else "unknown"} · Missing base keys while loading LoRA: {manifest.get("missing_key_count", "unknown")} · Items: {len(manifest["items"])}</p>
  {"".join(cards)}
</main>
""",
        encoding="utf-8",
    )
    print(args.out)


if __name__ == "__main__":
    main()
