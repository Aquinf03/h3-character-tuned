"""Modal serving app for the locked H3 Avatar LoRA.

Deploy:

    modal deploy scripts/helpers/modal_h3_avatar.py

Endpoint:

    https://aquin--h3-avatar-warm-generate.modal.run
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal


APP_NAME = os.environ.get("H3_AVATAR_MODAL_APP", "h3-avatar-aq")
LOCKED_CHECKPOINT = "/runs/h3/locked/h3-avatar-audio-v2-step1000.pt"

RUNS_VOL = modal.Volume.from_name("h3-avatar-runs", create_if_missing=True)
CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)

VOLS = {
    "/runs": RUNS_VOL,
    "/cache": CACHE_VOL,
}

REPO_ROOT = Path(__file__).resolve().parents[2]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "accelerate",
        "av",
        "diffusers @ git+https://github.com/huggingface/diffusers.git@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc",
        "einops",
        "fastapi[standard]",
        "huggingface-hub",
        "numpy",
        "opencv-python-headless",
        "peft",
        "pillow",
        "safetensors",
        "sentencepiece",
        "torch",
        "torchaudio",
        "torchcodec",
        "torchvision",
        "transformers",
        "tqdm",
    )
    .add_local_dir(
        str(REPO_ROOT),
        "/root/h3-avatar-aq",
        ignore=[
            ".git/**",
            ".venv/**",
            "__pycache__/**",
            "artifacts/**",
            "data/**",
            "tmp/**",
        ],
    )
)

app = modal.App(APP_NAME, image=image)


def _strip_reference_audio(reference_video: str, output_dir: str) -> str:
    ref = Path(reference_video)
    out = Path(output_dir) / f"{ref.stem}_noaudio.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(ref), "-an", "-c:v", "copy", str(out)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return str(out)


@app.cls(
    image=image,
    volumes=VOLS,
    gpu="H100",
    cpu=16.0,
    memory=262144,
    timeout=4 * 60 * 60,
    scaledown_window=20 * 60,
    max_containers=1,
)
class H3AvatarGenerator:
    @modal.enter()
    def load(self) -> None:
        import torch
        from diffusers import ComponentsManager
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Ref2VABlocks
        from peft import LoraConfig

        self.torch = torch
        self.LoraConfig = LoraConfig
        self.manager = ComponentsManager()
        self.pipe = MiniMaxH3Ref2VABlocks().init_pipeline("/cache/MiniMax-H3", components_manager=self.manager)
        self.pipe.load_components(dtype=torch.bfloat16)
        self.manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")
        self.transformer = getattr(self.pipe, "transformer_ref", None) or getattr(self.pipe, "transformer", None)
        if self.transformer is None:
            raise RuntimeError("No transformer_ref/transformer component found on pipeline")
        self.loaded_checkpoint = None
        self.missing_key_count = 0
        try:
            self.transformer.set_attention_backend("_flash_3_hub")
        except Exception as exc:
            print(f"flash attention backend not enabled: {exc}", flush=True)

    def _load_lora(self, checkpoint: str) -> None:
        ckpt = Path(checkpoint)
        if self.loaded_checkpoint == str(ckpt):
            return
        if not ckpt.is_file():
            raise RuntimeError(f"checkpoint missing: {ckpt}")
        if self.loaded_checkpoint is None:
            self.transformer.add_adapter(
                self.LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.0,
                    target_modules=["to_qkv", "to_out.0", "linear_1", "linear_2"],
                    init_lora_weights="gaussian",
                )
            )
        state = self.torch.load(ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = self.transformer.load_state_dict(state["model"], strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected LoRA keys: {unexpected[:5]}")
        self.transformer.requires_grad_(False)
        self.loaded_checkpoint = str(ckpt)
        self.missing_key_count = len(missing)

    def _generate_payload(self, payload: dict) -> dict:
        import base64
        import uuid

        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
        from diffusers.utils.export_utils import encode_video

        checkpoint = payload.get("checkpoint") or LOCKED_CHECKPOINT
        prompt = payload.get("prompt") or "The same character faces the camera and speaks with an audible natural voice."
        output_path = payload.get("output_path") or f"/tmp/h3_avatar/{uuid.uuid4()}/generated.mp4"
        num_frames = int(payload.get("num_frames", 243))
        height = int(payload.get("height", 512))
        width = int(payload.get("width", 512))
        num_inference_steps = int(payload.get("num_inference_steps", 30))
        seed = int(payload.get("seed", 1500))
        include_reference_audio = bool(payload.get("include_reference_audio", False))
        generate_audio = bool(payload.get("generate_audio", True))

        reference_video = payload.get("reference_video")
        reference_video_b64 = payload.get("reference_video_b64")
        if reference_video_b64:
            ref_dir = Path("/tmp/h3_avatar_refs") / str(uuid.uuid4())
            ref_dir.mkdir(parents=True, exist_ok=True)
            ref_path = ref_dir / "reference.mp4"
            ref_path.write_bytes(base64.b64decode(reference_video_b64))
            reference_video = str(ref_path)
        if not reference_video:
            raise RuntimeError("reference_video or reference_video_b64 is required")

        ref = Path(reference_video)
        out = Path(output_path)
        if not ref.is_file():
            raise RuntimeError(f"reference video missing: {ref}")
        out.parent.mkdir(parents=True, exist_ok=True)

        if str(checkpoint).lower() not in {"", "base", "none", "null"}:
            self._load_lora(str(checkpoint))

        reference_path = str(ref)
        if not include_reference_audio:
            reference_path = _strip_reference_audio(str(ref), str(out.parent))

        reference = MiniMaxH3Reference(video=reference_path)
        generator = self.torch.Generator(device="cpu").manual_seed(seed)
        outputs = ["videos", "audio", "sampling_rate"] if generate_audio else ["videos"]
        results = self.pipe(
            prompt=prompt,
            references=[reference],
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output=outputs,
        )

        if generate_audio:
            encode_video(
                results["videos"][0],
                fps=24,
                output_path=str(out),
                audio=results["audio"][0],
                audio_sample_rate=results["sampling_rate"],
            )
        else:
            encode_video(results["videos"][0], fps=24, output_path=str(out))

        response = {
            "checkpoint": str(checkpoint),
            "reference_video": str(ref),
            "output_path": str(out),
            "bytes": out.stat().st_size,
            "prompt": prompt,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "include_reference_audio": include_reference_audio,
            "generate_audio": generate_audio,
            "missing_key_count": self.missing_key_count,
            "warm": True,
        }
        if payload.get("return_video_b64"):
            response["video_b64"] = base64.b64encode(out.read_bytes()).decode("ascii")
        return response

    @modal.fastapi_endpoint(method="POST", label="h3-avatar-warm-generate", docs=True)
    def generate(self, payload: dict) -> dict:
        try:
            return self._generate_payload(payload)
        except Exception as exc:
            return {
                "state": "error",
                "error": type(exc).__name__,
                "detail": str(exc),
            }

    @modal.method()
    def generate_remote(self, payload: dict) -> dict:
        return self._generate_payload(payload)


@app.function(image=image, volumes=VOLS, cpu=2.0, memory=4096, timeout=4 * 60 * 60)
def generate_ref2va_sample(
    checkpoint: str = LOCKED_CHECKPOINT,
    reference_video: str = "",
    output_path: str = "/runs/generated/ui/generated.mp4",
    prompt: str = "The same character faces the camera and speaks with an audible natural voice.",
    num_frames: int = 243,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 30,
    seed: int = 1500,
    include_reference_audio: bool = False,
    generate_audio: bool = True,
) -> dict:
    payload = {
        "checkpoint": checkpoint,
        "reference_video": reference_video,
        "output_path": output_path,
        "prompt": prompt,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "include_reference_audio": include_reference_audio,
        "generate_audio": generate_audio,
    }
    return H3AvatarGenerator().generate_remote.remote(payload)


@app.function(
    image=image,
    volumes=VOLS,
    gpu="H100",
    cpu=16.0,
    memory=262144,
    timeout=4 * 60 * 60,
)
def generate_ref2va_direct(
    checkpoint: str = LOCKED_CHECKPOINT,
    reference_video: str = "",
    output_path: str = "/runs/generated/direct/generated.mp4",
    prompt: str = "The same character faces the camera and speaks with an audible natural voice.",
    num_frames: int = 124,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 12,
    seed: int = 1500,
    include_reference_audio: bool = False,
    generate_audio: bool = True,
) -> str:
    import torch
    from diffusers import ComponentsManager
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Ref2VABlocks, MiniMaxH3Reference
    from diffusers.utils.export_utils import encode_video
    from peft import LoraConfig

    ref = Path(reference_video)
    out = Path(output_path)
    if not ref.is_file():
        raise RuntimeError(f"reference video missing: {ref}")
    if checkpoint and checkpoint.lower() not in {"base", "none", "null"} and not Path(checkpoint).is_file():
        raise RuntimeError(f"checkpoint missing: {checkpoint}")
    out.parent.mkdir(parents=True, exist_ok=True)

    manager = ComponentsManager()
    pipe = MiniMaxH3Ref2VABlocks().init_pipeline("/cache/MiniMax-H3", components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)
    manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")

    transformer = getattr(pipe, "transformer_ref", None) or getattr(pipe, "transformer", None)
    if transformer is None:
        raise RuntimeError("No transformer_ref/transformer component found on pipeline")
    try:
        transformer.set_attention_backend("_flash_3_hub")
    except Exception as exc:
        print(f"flash attention backend not enabled: {exc}", flush=True)

    missing_key_count = 0
    if checkpoint and checkpoint.lower() not in {"base", "none", "null"}:
        transformer.add_adapter(
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                target_modules=["to_qkv", "to_out.0", "linear_1", "linear_2"],
                init_lora_weights="gaussian",
            )
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = transformer.load_state_dict(state["model"], strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected LoRA keys: {unexpected[:5]}")
        transformer.requires_grad_(False)
        missing_key_count = len(missing)

    reference_path = str(ref)
    if not include_reference_audio:
        reference_path = _strip_reference_audio(str(ref), str(out.parent))

    print(
        {
            "checkpoint": checkpoint,
            "reference_video": str(ref),
            "output_path": str(out),
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "generate_audio": generate_audio,
            "missing_key_count": missing_key_count,
        },
        flush=True,
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    outputs = ["videos", "audio", "sampling_rate"] if generate_audio else ["videos"]
    results = pipe(
        prompt=prompt,
        references=[MiniMaxH3Reference(video=reference_path)],
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        generator=generator,
        output=outputs,
    )

    if generate_audio:
        encode_video(
            results["videos"][0],
            fps=24,
            output_path=str(out),
            audio=results["audio"][0],
            audio_sample_rate=results["sampling_rate"],
        )
    else:
        encode_video(results["videos"][0], fps=24, output_path=str(out))

    size = out.stat().st_size
    print(f"wrote {out} ({size} bytes)", flush=True)
    return str(out)
