#!/usr/bin/env python3
"""Create a simple talking-head MP4 from a single reference image."""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
import wave
from pathlib import Path

from PIL import Image, ImageEnhance


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def audio_duration(path: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    return float(proc.stdout.strip())


def synth_voice(text: str, out_wav: Path, voice: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "speech.aiff"
        run(["say", "-v", voice, "-o", str(aiff), text])
        run(["ffmpeg", "-y", "-i", str(aiff), "-ar", "32000", "-ac", "1", str(out_wav)])
    if out_wav.stat().st_size < 8192:
        synth_placeholder_voice(text, out_wav)


def synth_placeholder_voice(text: str, out_wav: Path) -> None:
    """Fallback when macOS say cannot synthesize inside the current environment."""
    import struct

    sample_rate = 32000
    duration = max(2.8, min(7.0, len(text.split()) * 0.32))
    total = int(duration * sample_rate)
    samples = []
    for i in range(total):
        t = i / sample_rate
        syllable = 0.5 + 0.5 * math.sin(2 * math.pi * 5.5 * t)
        gate = 1.0 if syllable > 0.35 else 0.15
        freq = 185 + 35 * math.sin(2 * math.pi * 1.7 * t) + 22 * math.sin(2 * math.pi * 3.1 * t)
        carrier = math.sin(2 * math.pi * freq * t)
        buzz = 0.45 * math.sin(2 * math.pi * (freq * 2.02) * t)
        value = 0.26 * gate * (carrier + buzz)
        fade = min(1.0, i / (0.08 * sample_rate), (total - i) / (0.10 * sample_rate))
        samples.append(int(max(-1, min(1, value * fade)) * 32767))
    with wave.open(str(out_wav), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(struct.pack("<h", s) for s in samples))


def load_audio_envelope(wav_path: Path, fps: int, frames: int) -> list[float]:
    with wave.open(str(wav_path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    if width != 2:
        return [0.5 + 0.5 * math.sin(i * 0.45) for i in range(frames)]

    import array

    data = array.array("h")
    data.frombytes(raw)
    if channels > 1:
        data = array.array("h", data[::channels])
    per_frame = max(1, int(rate / fps))
    env = []
    for i in range(frames):
        chunk = data[i * per_frame : (i + 1) * per_frame]
        if not chunk:
            env.append(0.0)
            continue
        rms = math.sqrt(sum(x * x for x in chunk) / len(chunk)) / 32768.0
        env.append(min(1.0, rms * 9.0))
    # Smooth a little so the puppet does not chatter.
    smoothed = []
    prev = 0.0
    for value in env:
        prev = prev * 0.65 + value * 0.35
        smoothed.append(prev)
    return smoothed


def make_frames(image_path: Path, audio_path: Path, frame_dir: Path, fps: int, duration: float) -> int:
    base = Image.open(image_path).convert("RGB")
    target_w = 512
    target_h = round(base.height * target_w / base.width)
    if target_h % 2:
        target_h += 1
    base = base.resize((target_w, target_h), Image.Resampling.LANCZOS)

    frame_count = int(math.ceil(duration * fps))
    envelope = load_audio_envelope(audio_path, fps, frame_count)
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Tuned for the supplied front-facing image.
    sx = target_w / 274.0
    sy = target_h / 328.0
    mouth_box = tuple(round(v) for v in (63 * sx, 158 * sy, 213 * sx, 230 * sy))
    lower_face_box = tuple(round(v) for v in (55 * sx, 145 * sy, 220 * sx, 247 * sy))

    for i in range(frame_count):
        amp = envelope[i]
        wobble = math.sin(i * 0.18)
        frame = base.copy()

        # Subtle body bob to keep the clip alive.
        canvas = Image.new("RGB", frame.size, "white")
        bob = round(3 * math.sin(i * 0.13))
        canvas.paste(frame, (0, bob))
        frame = canvas

        # Lower-face stretch gives a simple talking motion while preserving the source image.
        lf = frame.crop(lower_face_box)
        stretch = 1.0 + 0.045 * amp
        new_h = max(1, round(lf.height * stretch))
        lf2 = lf.resize((lf.width, new_h), Image.Resampling.BICUBIC)
        y = lower_face_box[1] - round((new_h - lf.height) * 0.25)
        frame.paste(lf2, (lower_face_box[0], y))

        # Mouth emphasis: darker open shape and brighter teeth area moving with audio.
        draw_layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        from PIL import ImageDraw

        draw = ImageDraw.Draw(draw_layer)
        x1, y1, x2, y2 = mouth_box
        open_h = round((10 + 18 * amp) * sy)
        cy = round((190 + 4 * wobble) * sy)
        draw.ellipse((x1 + 24, cy - open_h // 2, x2 - 20, cy + open_h // 2), fill=(18, 8, 7, round(70 * amp)))
        draw.rounded_rectangle(
            (x1 + 42, cy - open_h // 2 - 2, x2 - 42, cy - open_h // 2 + 8),
            radius=5,
            fill=(255, 255, 245, round(42 * amp)),
        )
        frame = Image.alpha_composite(frame.convert("RGBA"), draw_layer).convert("RGB")

        # Tiny contrast pulse, barely visible but helps the face feel less static.
        frame = ImageEnhance.Contrast(frame).enhance(1.0 + 0.015 * amp)
        frame.save(frame_dir / f"frame_{i:05d}.jpg", quality=92)
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--text", default="Hello! I am ready to talk. This is a quick reference animation test.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        wav = tmp_path / "speech.wav"
        frames = tmp_path / "frames"
        synth_voice(args.text, wav, args.voice)
        duration = audio_duration(wav) + 0.25
        make_frames(args.image, wav, frames, args.fps, duration)
        silent = tmp_path / "silent.mp4"
        run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(args.fps),
                "-i",
                str(frames / "frame_%05d.jpg"),
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(args.fps),
                str(silent),
            ]
        )
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(silent),
                "-i",
                str(wav),
                "-shortest",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                str(args.out),
            ]
        )
    print(args.out)


if __name__ == "__main__":
    main()
