"""Encode the H3 training cache: video, (optionally) real audio, and text.

Per sample: target video → H3-VisualVAE latents (patchified rows), target
audio → H3-AudioVAE latents (channel-major rows; zero placeholder when the
clip has no audio track or --encode-audio is off), caption → Qwen3-VL
layer-50 hidden states. One .pt file per sample.

Framing is cover-fit (scale to cover, center-crop) — never stretch: aspect
distortion is exactly the kind of signal a fine-tune will happily learn.
Reference media lists are not yet packed into H3 reference rows; see
examples/README.md before using this cache for a complete multimodal Ref2VA
experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from diffusers.modular_pipelines.minimax_h3.packing import (
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    align_num_frames,
    audio_latent_num_frames,
    patchify_video_latents,
)
from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor


def read_video_cover(path: str, frames: int, height: int, width: int) -> torch.Tensor:
    """Decode up to `frames` frames, cover-fit to (height, width)."""
    container = av.open(path)
    stream = container.streams.video[0]
    decoded = []
    for frame in container.decode(stream):
        img = frame.to_image().convert("RGB")
        w, h = img.size
        scale = max(width / w, height / h)
        rw, rh = math.ceil(w * scale), math.ceil(h * scale)
        img = img.resize((rw, rh), Image.Resampling.BICUBIC)
        left, top = (rw - width) // 2, (rh - height) // 2
        img = img.crop((left, top, left + width, top + height))
        decoded.append(np.asarray(img, dtype=np.float32) / 255.0)
        if len(decoded) >= frames:
            break
    container.close()
    if not decoded:
        raise RuntimeError(f"No video frames: {path}")
    while len(decoded) < frames:
        decoded.append(decoded[-1])
    return torch.from_numpy(np.stack(decoded)).permute(3, 0, 1, 2).contiguous()


def extract_wav(video_path: str, tmpdir: str) -> Path | None:
    """Extract the audio track as 32 kHz stereo wav; None if absent/silent."""
    out = Path(tmpdir) / "audio.wav"
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
                        "-vn", "-ac", "2", "-ar", "32000", str(out)],
                       capture_output=True, timeout=120)
    return out if (r.returncode == 0 and out.exists() and out.stat().st_size > 1000) else None


@torch.no_grad()
def encode_audio_rows(audio_vae, video_path: str, num_frames: int, n_audio: int, device) -> torch.Tensor:
    """Official audio recipe: 32 kHz stereo, the two channels ride as two batch
    items through the mono VAE, latents are normalized then laid out
    channel-major ([ch0 x n_audio, ch1 x n_audio]). 800x downsample aligns the
    waveform to the latent grid. Returns zero rows when there is no track."""
    import torchaudio
    rows = torch.zeros(n_audio * 2, audio_vae.config.latent_channels)
    with tempfile.TemporaryDirectory() as td:
        wav_path = extract_wav(video_path, td)
        if wav_path is None:
            return rows
        wave, sr = torchaudio.load(str(wav_path))
        if wave.shape[0] == 1:
            wave = wave.repeat(2, 1)
        need_samples = int(round(num_frames / 24.0 * 32000))
        if wave.shape[1] < need_samples:
            wave = torch.nn.functional.pad(wave, (0, need_samples - wave.shape[1]))
        wave = wave[:, : n_audio * 800]
        if wave.shape[1] < n_audio * 800:
            wave = torch.nn.functional.pad(wave, (0, n_audio * 800 - wave.shape[1]))
        if wave.abs().sum() == 0:
            return rows
        mean = torch.as_tensor(audio_vae.config.latents_mean).view(1, -1, 1).to(device)
        std = torch.as_tensor(audio_vae.config.latents_std).view(1, -1, 1).to(device)
        post = audio_vae.encode(wave.to(device)[:, None], return_dict=False)[0]
        lat = ((post.mode().float() - mean) / std).transpose(1, 2)  # (2, T_lat, 32)
        if lat.shape[1] < n_audio:
            lat = torch.nn.functional.pad(lat, (0, 0, 0, n_audio - lat.shape[1]))
        return lat[:, :n_audio].reshape(-1, lat.shape[-1]).cpu().contiguous()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--frames", type=int, default=22)
    ap.add_argument("--encode-text", action="store_true")
    ap.add_argument("--encode-audio", action="store_true",
                    help="encode the real audio track via the H3 AudioVAE (zero placeholder otherwise)")
    args = ap.parse_args()
    if args.height % 32 or args.width % 32 or args.frames != align_num_frames(args.frames):
        raise ValueError("H3 cache dimensions require height/width divisible by 32 and frames of form 17*n+5")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    # H3's released VAE keeps its convolution path in float32. The transformer
    # cache is later stored in float32/BF16, but encoding must receive float32.
    vae = AutoencoderKLMiniMaxH3.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32).to(device)
    audio_vae = None
    if args.encode_audio:
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(args.model, subfolder="audio_vae", torch_dtype=torch.float32).to(device).eval()
    text_encoder = tokenizer = processor = None
    if args.encode_text:
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(args.model / "text_encoder", torch_dtype=dtype).to(device)
        tokenizer = Qwen2TokenizerFast.from_pretrained(args.model / "tokenizer")
        processor = Qwen3VLProcessor.from_pretrained(args.model / "processor")
    rows = [json.loads(line) for line in args.metadata.open() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device)[:, None, None, None]
    std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device)[:, None, None, None]
    for index, row in enumerate(rows):
        video_path = row.get("target_video", row["video"])
        video = read_video_cover(video_path, args.frames, args.height, args.width)
        video_latent = vae.encode(((video.to(device) - mean) / std)[None].to(torch.float32)).latent_dist.mode()
        latent_mean = torch.as_tensor(vae.config.latents_mean, device=device, dtype=video_latent.dtype).view(1, -1, 1, 1, 1)
        latent_std = torch.as_tensor(vae.config.latents_std, device=device, dtype=video_latent.dtype).view(1, -1, 1, 1, 1)
        video_latent = (video_latent - latent_mean) / latent_std
        video_rows = patchify_video_latents(video_latent.float(), (1, 2, 2))
        n_audio = audio_latent_num_frames(args.frames)
        if audio_vae is not None:
            audio_rows = encode_audio_rows(audio_vae, video_path, args.frames, n_audio, device)
        else:
            audio_rows = torch.zeros(n_audio * 2, 32)
        prompt = row.get("caption", row.get("prompt", ""))
        if text_encoder is not None:
            ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            mm = processor.create_mm_token_type_ids(ids.tolist())
            out = text_encoder.model(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=torch.tensor(mm, device=device), use_cache=False, output_hidden_states=True)
            prompt_embed = out.hidden_states[50][0].to(torch.bfloat16).cpu()
        else:
            prompt_embed = torch.zeros((min(max(len(prompt), 8), 256), 5120), dtype=torch.bfloat16)
        item = {"video": video_rows.cpu().to(torch.float32), "audio": audio_rows, "prompt": prompt_embed,
                "caption": prompt, "height": args.height, "width": args.width,
                "latent_frames": video_latent.shape[2], "audio_frames": n_audio,
                "num_frames": args.frames}  # num_frames: required by train.py's H3_MAX_LF truncation
        torch.save(item, args.output / f"{index:08d}.pt")
        real_audio = bool(audio_rows.abs().sum() > 0)
        print(f"[{index + 1}/{len(rows)}] {args.output / f'{index:08d}.pt'} video={tuple(video_rows.shape)} audio={'real' if real_audio else 'zero'} text={tuple(prompt_embed.shape)}", flush=True)


if __name__ == "__main__":
    main()
