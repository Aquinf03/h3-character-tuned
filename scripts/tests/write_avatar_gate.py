#!/usr/bin/env python3
"""Write AQ eval gate score after external avatar review/probes.

Use this after generating validation samples. Scores are 0..1 and averaged:
identity, talking_motion, audio_sync, prompt_following, generalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity", type=float, required=True)
    ap.add_argument("--talking-motion", type=float, required=True)
    ap.add_argument("--audio-sync", type=float, required=True)
    ap.add_argument("--prompt-following", type=float, required=True)
    ap.add_argument("--generalization", type=float, default=None)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("artifacts/h3/avatar_gate.json"))
    args = ap.parse_args()

    parts = {
        "identity": args.identity,
        "talking_motion": args.talking_motion,
        "audio_sync": args.audio_sync,
        "prompt_following": args.prompt_following,
    }
    if args.generalization is not None:
        parts["generalization"] = args.generalization
    score = sum(parts.values()) / len(parts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"score": score, "n": args.n, "parts": parts}, indent=2) + "\n", encoding="utf-8")
    print(f"avatar_gate score={score:.4f} n={args.n}")


if __name__ == "__main__":
    main()
