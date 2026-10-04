"""MiniMax-H3 supervised rectified-flow fine-tuning over cached latents.

Trains the released H3 transformer (FL2VA or Ref2VA variant) with a
rectified-flow objective that matches the conventions of the official
Diffusers MiniMax-H3 integration. See FIXES.md: fixes 1-5 are the numeric
conventions that silently corrupt weights, fixes 6-9 are the ZeRO-3 /
long-sequence landmines that this version encodes.

On top of the minimal trainer this version adds (all battle-tested on
8xA800, sequences up to ~66k tokens, real stereo audio in the loss):

  * deterministic train/val split — md5(sample_id) % 20 == 0 -> val (~5%) —
    with a lockstep validation loop on a fixed sigma grid {0.3, 0.6, 0.9}
    and per-(file, sigma) seeded noise, so the val curve is comparable
    across steps and runs;
  * LoRA that actually trains under ZeRO-3 ([FIX6]: deepspeed's activation
    checkpointing, not torch's; dropout 0 for deterministic recompute);
  * optional truncation of over-long cached samples via env H3_MAX_LF
    (latent-frame boundary for video rows; audio rows sliced per channel
    because the audio layout is channel-major);
  * a pre-flight, all-reduced sequence-length gate that skips over-long
    samples BEFORE the forward pass ([FIX7]), plus an anti-spin guard that
    aborts if a whole epoch executes zero steps;
  * trainable-only checkpoints gathered from ZeRO-3 shards ([FIX5]) and a
    weight-only --resume loaded BEFORE deepspeed.initialize ([FIX9]).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os

# [FIX8] Must take effect before `import torch`: long-sequence ZeRO-3 training
# fragments the allocator (14-16 GB of unusable slack observed at OOM); the
# expandable-segments allocator reclaims it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from diffusers import MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3.packing import build_packed_sequence, build_row_timesteps

VAL_SIGMAS = (0.3, 0.6, 0.9)


def is_val_file(path: Path) -> bool:
    sid = path.stem.split("_")[-1]
    return int(hashlib.md5(sid.encode()).hexdigest()[:4], 16) % 20 == 0


class H3Cache(Dataset):
    def __init__(self, root: Path, rank: int = 0, world_size: int = 1, split: str = "train"):
        files = sorted(Path(p) for p in glob.glob(str(root / "*.pt")))
        files = [f for f in files if is_val_file(f) == (split == "val")]
        self.files = files[rank::world_size]
        if not self.files and split == "train":
            raise FileNotFoundError(f"No train cache files under {root}; run prepare_cache.py first")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        item = torch.load(self.files[i], map_location="cpu", weights_only=False)
        # Optional truncation for over-long samples: H3_MAX_LF caps the video
        # latent frames (1 latent frame = 4 pixel frames = 1/6 s). Lets one
        # cache serve several sequence-length budgets without re-encoding.
        max_lf = int(os.environ.get("H3_MAX_LF", "0"))
        if max_lf and item["latent_frames"] > max_lf:
            lf_old, n_a_old = int(item["latent_frames"]), int(item["audio_frames"])
            tokens_per_lf = item["video"].shape[0] // lf_old
            nf_new = int(item["num_frames"] * max_lf / lf_old)
            nf_new = max(22, ((nf_new - 5) // 17) * 17 + 5)  # re-align to 17n+5 (22 = shortest valid clip)
            from diffusers.modular_pipelines.minimax_h3.packing import audio_latent_num_frames as _alnf
            n_a_new = min(_alnf(nf_new), n_a_old)
            item["video"] = item["video"][: max_lf * tokens_per_lf]
            # Audio rows are channel-major ([ch0 x n_a_old, ch1 x n_a_old]):
            # slice each channel's block, then re-concatenate.
            a = item["audio"]
            item["audio"] = torch.cat([a[:n_a_new], a[n_a_old : n_a_old + n_a_new]], dim=0)
            item["latent_frames"], item["audio_frames"], item["num_frames"] = max_lf, n_a_new, nf_new
        return item


def layout_for(item, device):
    tags = torch.ones(item["prompt"].shape[0], dtype=torch.long)
    layout = build_packed_sequence(tags, item["latent_frames"], item["height"] // 16, item["width"] // 16, item["audio_frames"], (1, 2, 2))
    return {k: getattr(layout, k).to(device) for k in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices")}


def forward_loss(model, item, layout, device, sigma_v, sigma_a, noise_seed=None):
    video = item["video"].to(device, torch.bfloat16)
    audio = item["audio"].to(device, torch.bfloat16)
    clean_v, clean_a = video.clone(), audio.clone()
    if noise_seed is None:
        noise_v, noise_a = torch.randn_like(video), torch.randn_like(audio)
    else:
        # Validation path: noise depends only on (file, sigma), so the val
        # loss is a fixed function of the weights.
        g = torch.Generator(device="cpu").manual_seed(noise_seed)
        noise_v = torch.randn(video.shape, generator=g, dtype=torch.float32).to(device, torch.bfloat16)
        noise_a = torch.randn(audio.shape, generator=g, dtype=torch.float32).to(device, torch.bfloat16)
    video = (1 - sigma_v) * clean_v + sigma_v * noise_v
    audio = (1 - sigma_a) * clean_a + sigma_a * noise_a
    # [FIX1] Model contract: timestep == 1 - sigma (t=1 clean, t=0 pure noise).
    # Conditioning rows stay pinned at 0.999 (already the 1-sigma convention).
    timesteps, timestep_indices = build_row_timesteps(
        type("L", (), {"sequence_length": int(layout["token_tags"].numel()), "video_indices": layout["video_indices"],
                       "audio_indices": layout["audio_indices"], "num_condition_video_rows": 0, "num_condition_audio_rows": 0})(),
        float(1.0 - sigma_v), float(1.0 - sigma_a), 0.999, 0.999)
    pred_v, pred_a = model(hidden_states=video[None], audio_hidden_states=audio[None],
                           encoder_hidden_states=item["prompt"].to(device, torch.bfloat16)[None],
                           timestep=timesteps.to(device), timestep_indices=timestep_indices.to(device),
                           attention_kwargs=None, return_dict=False, **layout)
    # [FIX2] The model outputs a data-ward velocity v = x0 - noise (the
    # scheduler denoises with x0 = x_t + (1-t)*v, note the +). Regressing the
    # usual (noise - x0) actively corrupts the weights.
    # [FIX3] Zero-placeholder audio: zero the audio loss WEIGHT instead of
    # dropping the term, so gradients still flow (as zeros) through the audio
    # head and DDP/ZeRO stay in sync.
    audio_weight = 1.0 if bool(clean_a.abs().sum() > 0) else 0.0
    audio_loss_scale = float(os.environ.get("H3_AUDIO_LOSS_SCALE", "1.0"))
    loss_v = F.mse_loss(pred_v.float(), (clean_v - noise_v)[None].float())
    loss_a = F.mse_loss(pred_a.float(), (clean_a - noise_a)[None].float())
    return loss_v + audio_loss_scale * audio_weight * loss_a, loss_v.detach(), loss_a.detach(), audio_loss_scale * audio_weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True); ap.add_argument("--cache", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--variant", choices=("fl2va", "ref2va"), default="ref2va")
    ap.add_argument("--strategy", choices=("ddp", "deepspeed"), default="deepspeed")
    ap.add_argument("--deepspeed-config", type=Path, default=Path(__file__).resolve().parent / "deepspeed_zero3.json")
    ap.add_argument("--max-steps", type=int, default=600); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--trainable", choices=("heads", "lora", "all"), default="lora"); ap.add_argument("--resume", type=Path)
    ap.add_argument("--video-shift", type=float, default=12.0); ap.add_argument("--audio-shift", type=float, default=3.0)
    ap.add_argument("--val-every", type=int, default=100); ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--max-seq-tokens", type=int, default=132000,
                    help="pre-flight skip threshold; calibrate to your hardware (see FIXES.md #7/#8)")
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed and args.strategy == "ddp":
        torch.distributed.init_process_group("nccl")
    elif distributed and args.strategy == "deepspeed":
        import deepspeed
        deepspeed.init_distributed("nccl")
    rank = int(os.environ.get("RANK", "0")); world_size = int(os.environ.get("WORLD_SIZE", "1")); local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    log_handle = None
    if rank == 0:
        log_handle = (args.output / "train.log").open("a", buffering=1)
        log_handle.write(f"{datetime.now(timezone.utc).isoformat()} start world_size={world_size} variant={args.variant} trainable={args.trainable} strategy={args.strategy}\n")
    model = MiniMaxH3Transformer3DModel.from_pretrained(args.model, subfolder="transformer_ref" if args.variant == "ref2va" else "transformer", torch_dtype=torch.bfloat16).to(device)
    # [FIX6] Under ZeRO-3 with PEFT-injected LoRA layers, torch's activation
    # checkpointing (reentrant AND non-reentrant) recomputes against already
    # re-partitioned parameters (shape [0]) and crashes. deepspeed's own
    # checkpointing cooperates with the ZeRO gather hooks.
    if args.strategy == "deepspeed":
        import deepspeed
        # Do NOT enable partition_activations / checkpoint_in_cpu here: both
        # slice the 0-d time-embedding scalar and crash the temb linear.
        model.enable_gradient_checkpointing(deepspeed.checkpointing.checkpoint)
    else:
        model.enable_gradient_checkpointing()
    for p in model.parameters(): p.requires_grad_(args.trainable == "all")
    if args.trainable == "heads":
        for name, p in model.named_parameters():
            p.requires_grad_("proj_out" in name or "audio_proj_out" in name)
    elif args.trainable == "lora":
        from peft import LoraConfig
        # lora_dropout=0: the checkpointed recompute must be deterministic
        # (and at a few hundred steps the regularization buys little anyway).
        model.add_adapter(LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["to_qkv", "to_out.0", "linear_1", "linear_2"], init_lora_weights="gaussian"))
    # [FIX9] Resume BEFORE DDP wrapping / deepspeed.initialize, while the
    # parameters are still full tensors. After ZeRO-3 partitioning a plain
    # load_state_dict sees shape-[0] shards and silently loads nothing.
    # ZeRO-3-gathered checkpoints carry no optimizer state, so a deepspeed
    # resume is always weight-only; ddp/single-GPU checkpoints may carry it.
    resume_state = None
    step = 0
    if args.resume:
        if args.trainable == "all" and args.strategy == "deepspeed":
            raise SystemExit("--resume is not supported for --trainable all under deepspeed "
                             "(those checkpoints are DeepSpeed-native directories; restore them with engine.load_checkpoint)")
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(resume_state["model"], strict=False)
        # Missing keys are expected (the frozen base weights); unexpected keys
        # mean the checkpoint does not match this run's --trainable/adapter
        # config — with strict=False that would silently load NOTHING while
        # still advancing `step`, so fail loudly instead.
        if unexpected:
            raise SystemExit(f"resume key mismatch ({len(unexpected)} unexpected keys, e.g. {unexpected[:3]}) — "
                             f"checkpoint was saved with a different --trainable/adapter config")
        step = int(resume_state["step"])
        if rank == 0:
            print(f"resumed {len(resume_state['model'])} tensors at step={step}", flush=True)
    if distributed and args.strategy == "ddp":
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    params = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        print(f"trainable parameters: {sum(p.numel() for p in params):,} world_size={world_size}", flush=True)
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    # Optimizer state only survives a resume on the ddp/single-GPU path:
    # deepspeed.initialize rebuilds the optimizer around fp32 flat partitions,
    # which would orphan any state loaded here (weights still resume fine).
    if resume_state is not None and resume_state.get("optimizer") and args.strategy != "deepspeed":
        optimizer.load_state_dict(resume_state["optimizer"])
    engine = None
    if args.strategy == "deepspeed":
        import deepspeed
        model, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=params, optimizer=optimizer, config=str(args.deepspeed_config))
        engine = model
    train_data = H3Cache(args.cache, rank, world_size, "train")
    val_data = H3Cache(args.cache, rank, world_size, "val")
    # Lockstep: a ZeRO-3 forward is a collective, so every rank must run the
    # same number of val forwards -> use the global minimum of local counts.
    n_val_local = torch.tensor([len(val_data)], device=device)
    if distributed:
        torch.distributed.all_reduce(n_val_local, op=torch.distributed.ReduceOp.MIN)
    n_val = min(int(n_val_local.item()), 2)  # cap per-rank val cost (~2 min/round at 60k tokens)
    if rank == 0:
        print(f"train files(rank0)={len(train_data)} val files(lockstep)={n_val}", flush=True)

    def run_validation(step):
        model.eval()
        tot = torch.zeros(2, device=device)  # sum(loss_v), count
        per_sigma = {s: torch.zeros(2, device=device) for s in VAL_SIGMAS}
        with torch.no_grad():
            for vi in range(n_val):
                item = val_data[vi]
                layout = layout_for(item, device)
                for s in VAL_SIGMAS:
                    sv = args.video_shift * s / (1.0 + (args.video_shift - 1.0) * s)
                    sa = args.audio_shift * s / (1.0 + (args.audio_shift - 1.0) * s)
                    seed = int(hashlib.md5(f"{val_data.files[vi].name}:{s}".encode()).hexdigest()[:8], 16)
                    _, lv, la, aw = forward_loss(model, item, layout, device, sv, sa, noise_seed=seed)
                    tot += torch.stack([lv, torch.ones((), device=device)])
                    per_sigma[s] += torch.stack([lv, torch.ones((), device=device)])
        if distributed:
            torch.distributed.all_reduce(tot)
            for s in VAL_SIGMAS:
                torch.distributed.all_reduce(per_sigma[s])
        if rank == 0:
            parts = " ".join(f"u{s}={float(per_sigma[s][0] / per_sigma[s][1].clamp(min=1)):.4f}" for s in VAL_SIGMAS)
            message = f"VAL step={step} loss_v={float(tot[0] / tot[1].clamp(min=1)):.6f} {parts}"
            print(message, flush=True)
            log_handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")
        model.train()

    n_len_skips = 0
    model.train()
    if n_val > 0:
        run_validation(step)
    while step < args.max_steps:
        steps_at_epoch_start = step
        for item in train_data:
            if step >= args.max_steps: break
            layout = layout_for(item, device)
            # [FIX4] Sample noise levels from BOTH shifted schedules
            # (sigma = shift*u/(1+(shift-1)*u), video 12.0 / audio 3.0, same u),
            # mirroring the lockstep of the two schedulers at inference.
            u = torch.rand((), device=device).clamp(0.02, 0.98)
            sigma_v = args.video_shift * u / (1.0 + (args.video_shift - 1.0) * u)
            sigma_a = args.audio_shift * u / (1.0 + (args.audio_shift - 1.0) * u)
            # [FIX7] OOM protection is a PRE-FLIGHT gate: decide from the packed
            # sequence length, all-reduce so every rank skips together, and only
            # then run forward. Never insert logic between engine.backward() and
            # engine.step() — it breaks ZeRO-3's gradient-accumulation contract
            # (KeyError: averaged_gradients).
            skip_flag = torch.zeros(1, device=device)
            if int(layout["token_tags"].numel()) > args.max_seq_tokens:
                skip_flag.fill_(1)
            if distributed:
                torch.distributed.all_reduce(skip_flag, op=torch.distributed.ReduceOp.MAX)
            if float(skip_flag.item()) > 0:
                n_len_skips += 1
                if rank == 0:
                    msg = (f"LEN_SKIP step={step} seq={int(layout['token_tags'].numel())} "
                           f"f={item.get('num_frames', '?')} total_skips={n_len_skips}")
                    print(msg, flush=True)
                    log_handle.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
                continue
            loss, lv, la, aw = forward_loss(model, item, layout, device, sigma_v, sigma_a)
            # backward and step stay strictly adjacent — see [FIX7].
            if engine is not None:
                engine.backward(loss)
                engine.step()
            else:
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); optimizer.step()
            step += 1
            if rank == 0:
                seq_len = int(layout["token_tags"].numel())
                message = f"step={step} loss={loss.item():.6f} loss_v={float(lv):.6f} loss_a={float(la):.6f} seq={seq_len} f={item.get('num_frames', '?')} sigma_v={float(sigma_v):.3f} audio_w={aw}"
                print(message, flush=True)
                log_handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")
            # Checkpoint BEFORE validation: a validation forward leaves ZeRO-3
            # prefetch state in flight, and a GatheredParameters gather right
            # after it hits "Cannot partition a param in flight". Saving first
            # keeps every save on the same clean post-step state.
            if step % args.ckpt_every == 0 or step == args.max_steps:
                if engine is not None and args.trainable != "all":
                    # [FIX5] Save only trainable tensors, gathered from ZeRO-3
                    # shards (a few MB instead of ~66 GB; full-state saves stall
                    # rank 0 long enough to trip the NCCL watchdog).
                    import deepspeed
                    named = [(n, p) for n, p in engine.module.named_parameters() if p.requires_grad]
                    # The gather can race ZeRO-3's prefetcher ("Cannot partition
                    # a param in flight") and hand rank 0 an EMPTY tensor — a
                    # checkpoint that then crashes every future resume. Sync
                    # first, validate what was read, and write atomically; a
                    # skipped save is strictly better than a corrupt one.
                    torch.cuda.synchronize()
                    with deepspeed.zero.GatheredParameters([p for _, p in named]):
                        if rank == 0:
                            sd = {n: p.detach().float().cpu().clone() for n, p in named}
                            empty = [n for n, t in sd.items() if t.numel() == 0]
                            if empty:
                                msg = f"SAVE_SKIPPED step={step} empty={len(empty)} e.g.{empty[:2]}"
                                print(msg, flush=True)
                                log_handle.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
                            else:
                                tmp = args.output / f".checkpoint-{step}.tmp"
                                torch.save({"model": sd, "step": step}, tmp)
                                os.replace(tmp, args.output / f"checkpoint-{step}.pt")
                    if torch.distributed.is_initialized():
                        torch.distributed.barrier()
                elif engine is not None:
                    engine.save_checkpoint(str(args.output), tag=f"checkpoint-{step}")
                elif rank == 0:
                    core = model.module if distributed else model
                    saved_model = {k: v for k, v in core.state_dict().items() if k in {n for n, p in core.named_parameters() if p.requires_grad}} if args.trainable != "all" else core.state_dict()
                    torch.save({"model": saved_model, "optimizer": optimizer.state_dict() if args.trainable != "all" else None, "step": step}, args.output / f"checkpoint-{step}.pt")
            if n_val > 0 and step % args.val_every == 0:
                run_validation(step)
        if step == steps_at_epoch_start:
            # Anti-spin guard: if the gate skipped every sample, fail loudly
            # instead of looping forever over an epoch that trains nothing.
            raise RuntimeError(f"epoch swept with 0 executed steps (all samples over --max-seq-tokens? skips={n_len_skips})")
    if distributed:
        torch.distributed.destroy_process_group()
    if log_handle is not None:
        log_handle.write(f"{datetime.now(timezone.utc).isoformat()} finished step={step}\n")
        log_handle.close()


if __name__ == "__main__": main()
