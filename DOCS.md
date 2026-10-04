# Docs

Training data notes, trainer documentation, and training recipes for H3 Avatar.

## Dataset

Goal: improve the LoRA by training on cleaner visual motion and judging it on out-of-domain references like the Minion clip, without relying on H3-generated audio.

### Immediate Choice

Use a curated subset of `neosapience/TA2.0_animation_dataset` first.

Why:

- already downloaded on Modal;
- permissive license;
- directly matches animated talking-avatar references;
- metadata includes identity-ish grouping (`ref_id`), motion mode, language, VQA score, C50/SNR, duration/frame counts, and padding fields;
- big enough to filter aggressively instead of training on the whole noisy dump.

Risk:

- synthetic/anime-heavy;
- may not teach enough 3D mascot/cartoon behavior;
- quality scores are only proxies, so visual eval still decides.

### Candidate Datasets

| dataset | license/access | best use | decision |
| --- | --- | --- | --- |
| `neosapience/TA2.0_animation_dataset` | Apache-2.0/open | animated talking-avatar LoRA data | use now, curated only |
| `Shofo/shofo-talking-head-en` | gated/approval | real English talking-head motion and framing | request access; do not block v2 |
| `EngineeringAI-LAB/3DTalkingDataset` | Apache-2.0/open | realistic talking-head/lip/head-pose motion priors and eval references | sample later; too human-centric for primary character LoRA |
| `FreedomIntelligence/TalkVid` | CC-BY-NC-4.0/open | broad multilingual talking-head benchmark/eval | eval/research only unless usage is non-commercial |

### V2 Curation Rules

The new `ta20_visual_v2_1k` manifest does this:

- require local target video files;
- prefer active motion;
- prefer English rows where present;
- prefer higher `vqa_score`, `c50`, and `snr`;
- reject very short clips;
- penalize heavy padding;
- cap each `ref_id` to avoid one character dominating;
- split by `ref_id`, so near-same identity does not leak across train/val/test.

### Next Training Dataset Ladder

| stage | clips | purpose |
| --- | ---: | --- |
| `ta20_visual_v2_1k` | 1000 | quick v2 proof run, 384px, r16 |
| `ta20_visual_v2_2500` | 2500 | better full run if 1k beats base/v1 |
| `mixed_visual_v3` | 2500-5000 | add approved Shofo/3DTalking samples and mascot/cartoon augment refs |

### Acceptance Target

Before scaling beyond 1k clips, the checkpoint must beat the old 8000-step LoRA and base H3 on:

- identity on anime refs;
- motion without face collapse;
- temporal stability;
- out-of-domain Minion/mascot reference usefulness;
- validation loss not worsening across two checkpoints without visual improvement.

## Trainer

Training scripts live in `src/` (`train.py`, `prepare_cache.py`, data and inspect helpers).

English | [中文](README_zh.md)

A minimal, working fine-tuning path for [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) —
the open-weight omni-modal model that jointly generates video **and** synchronized stereo audio.

MiniMax released the H3 weights "to support further development, including fine-tuning", but ships **no
trainer**. The Hugging Face Diffusers integration is inference-only. This repository fills the gap with a
supervised rectified-flow trainer built directly on the official Diffusers integration, plus latent-caching
preprocessors — and documents the nine hard-won fixes behind it: five H3-specific numeric conventions that
silently corrupt weights, and four LoRA × ZeRO-3 × long-sequence landmines (see [FIXES.md](FIXES.md); we
learned every one of them the hard way).

Proven at production scale: LoRA over 2,000 ~30-second clips (448×768, ~65k tokens per sequence, real
stereo audio in the loss) on 8×A800, with lockstep validation and multi-day-run guard rails.

### How it works

**Two-stage design** (so the 33B transformer is the only thing in GPU memory during training):

1. **`prepare_cache.py`** — encodes each training sample offline: target video → H3-VisualVAE latents
   (patchified rows, cover-fit framing — never stretched), target audio → H3-AudioVAE latents
   (`--encode-audio`; zero placeholder otherwise), caption → Qwen3-VL layer-50 hidden states. One `.pt`
   file per sample. **`prepare_cache_pairs.py`** is the long-clip variant: it concatenates two
   consecutively-shot clips into one ~30 s story-continuous sample (17n+5 re-alignment, seam index kept,
   both audio tracks joined).
2. **`train.py`** — loads only the transformer (`transformer` = FL2VA variant or `transformer_ref` = Ref2VA
   variant), rebuilds the packed video/audio/text sequence layout with the official
   `build_packed_sequence` / `build_row_timesteps` utilities, and optimizes a rectified-flow MSE loss.

**Training objective.** H3's released checkpoints are guidance-distilled rectified-flow models with two
peculiarities relative to the common SD3/Wan-style convention:

- the transformer's time input is `t = 1 − σ` (t=1 clean, t=0 pure noise), and
- the transformer predicts a **data-ward velocity** `v = x₀ − ε` (the scheduler reconstructs
  `x₀ = x_t + (1−t)·v` — note the `+`).

The loss therefore noises `x_t = (1−σ)·x₀ + σ·ε` and regresses `pred → (x₀ − ε)`, with **two different σ
per step**: video and audio use separate shifted schedules (`σ = shift·u / (1+(shift−1)·u)`, shift 12.0 for
video and 3.0 for audio) sampled at the same `u`, mirroring how the two schedulers advance in lockstep at
inference. Real cached audio enters the loss at weight 1.0; zero-placeholder audio is trained at weight 0
so the audio head is untouched but gradients stay synchronized ([FIXES.md](FIXES.md) #3).

**Validation.** Samples whose `md5(sample_id) % 20 == 0` (~5%) form a deterministic val split. Every
`--val-every` steps, all ranks run a lockstep evaluation on a fixed σ grid {0.3, 0.6, 0.9} with
per-(file, σ) seeded noise — the val curve is a fixed function of the weights, comparable across steps and
runs. `plot_loss.py` renders train + per-modality + VAL curves from `train.log`.

**Trainable-parameter modes:**

| `--trainable` | What trains | Use case |
|---|---|---|
| `heads` | `proj_out` + `audio_proj_out` (~1M params) | smoke test the whole pipeline on one GPU |
| `lora` (default) | PEFT LoRA r=16 on `to_qkv`, `to_out.0`, `linear_1`, `linear_2` (~10.7M params) | practical single-node fine-tuning |
| `all` | full 33B | requires `--strategy deepspeed` (ZeRO-3 config included) |

Checkpoints store only trainable tensors for `heads`/`lora` (a few MB instead of ~66 GB), gathered from
ZeRO-3 shards; `--resume` loads them weight-only *before* `deepspeed.initialize` (FIXES.md #5/#9).

### Quickstart

```bash
# 1. Environment: Python 3.11, torch >= 2.8. The H3 classes are NOT in a released
#    diffusers wheel yet — install_env.sh installs the pinned integration revision.
bash install_env.sh

# 2. Write a manifest (schema + placeholder samples in examples/)
python validate_manifest.py --manifest path/to/train.jsonl --check-files

# 3. Cache latents (debug scale: 256x256, 22 frames; raise for real runs)
python src/prepare_cache.py \
  --metadata path/to/train.jsonl \
  --output cache/train \
  --model /path/to/MiniMax-H3 \
  --height 256 --width 256 --frames 22 --encode-text --encode-audio

# 4. Train (single GPU smoke)
CUDA_VISIBLE_DEVICES=0 python src/train.py \
  --model /path/to/MiniMax-H3 --variant ref2va \
  --cache cache/train --output runs/smoke \
  --max-steps 10 --trainable heads --strategy ddp

# 4'. Multi-GPU LoRA (8x, DeepSpeed ZeRO-3) — the production path
deepspeed --num_gpus 8 src/train.py \
  --model /path/to/MiniMax-H3 --variant ref2va \
  --cache cache/train --output runs/lora \
  --max-steps 800 --trainable lora --strategy deepspeed
```

Constraints inherited from the model: `num_frames % 17 == 5`, height/width divisible by 32, 24 fps video,
32 kHz stereo audio, `audio_latent_frames = round(num_frames / 24 * 40)`.

**Sequence-length budget (8×A800-80GB, full attention + LoRA + ZeRO-3):** the measured ceiling is ≈**70k
tokens** per sample (65k ⇒ 76 GB steady; 76k and above OOM). Size resolution × duration against it —
448×768 @ 27 s ≈ 65k fits; 576×1024 @ 30 s ≈ 127k does not. Two knobs enforce the budget without
re-encoding caches: `H3_MAX_LF=<n>` (env) truncates cached samples to *n* video latent frames at load
time, and `--max-seq-tokens` (default 132000) is a pre-flight all-reduced skip gate. Details in
[FIXES.md](FIXES.md) #8.

### Does it work?

With the conventions wrong (timestep direction + velocity sign inverted), a heads-only run shows loss
*rising* (7.2 → 9.5 over 10 steps). With the fixes in this repo, the same setup trains at loss ≈ 0.3–1.0,
stable across 1000 steps on 8×A800, and the trained heads produce coherent generations when patched back
into the inference pipeline. The LoRA path runs 30-second 65k-token sequences with real audio at
~7.5–8 min/step on 8×A800 (heads at 33k tokens: ~53 s/step). Details in [FIXES.md](FIXES.md).

### Current limitations

- Reference media are not yet encoded into the conditioning sequence (H3 reference rows); captions and
  target video/audio train, reference images/audio do not condition. This is the main open TODO for
  faithful Ref2VA fine-tuning.
- No sample shuffling between epochs; batch size is fixed at 1 sequence per step.
- Sparse attention (used in H3's final training stage) is not released; training runs full attention, so
  keep the ~70k-token budget above in mind.
- `--resume` restores weights only (trainable-only checkpoints carry no optimizer state).

### Legal

Model weights are governed by the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
(territory and usage restrictions apply; fine-tuned weights are Model Derivatives). This repository contains
code only — no model weights and no training media.

## Recipes

### Locked H3 Avatar Checkpoint

Locked on: 2026-10-03

#### Winner

Use this as the default usable LoRA:

`/runs/h3/locked/h3-avatar-audio-v2-step1000.pt`

Original training checkpoint:

`/runs/h3/ref2va-lora-audio-v2-transcripts-en500-w40-384-56-r16/checkpoint-1000.pt`

#### Why This One

- User-approved as visually good and audio usable.
- Generated speech was clear enough on two different references.
- Lip sync improved versus audio-v1.
- Step 1000 was chosen before further joint video/audio training could drift the already-good visuals.

#### Training Recipe That Worked

- Source manifest: `/runs/manifests/ta20_audio_v1_en_500/ta20_audio_v1_manifest.jsonl`
- ASR/transcript manifest: `/runs/manifests/ta20_audio_v2_transcripts_en_500_w40`
- Accepted clips: 481
- Train/val/test split: 385 / 48 / 48
- Cache: `/runs/cache/ta20_audio_v2_transcripts_en_500_w40_train_384_56`
- Cache shape: 384x384, 56 frames, encoded video/audio/text
- LoRA output: `/runs/h3/ref2va-lora-audio-v2-transcripts-en500-w40-384-56-r16`
- Learning rate: `5e-5`
- Audio loss scale: `2.0`
- Checkpoints saved: 500, 1000

#### Known Good Samples

- `/runs/generated/ref2va-lora-audio-v2-transcripts-en500-w40-384-56-r16/ab_headshot_audio_v2_ckpt1000_5s/generated.mp4`
- `/runs/generated/ref2va-lora-audio-v2-transcripts-en500-w40-384-56-r16/boy_closeup_audio_v2_ckpt1000_5s/generated.mp4`

Local copies:

- `artifacts/generated/ref2va-lora-audio-v2-transcripts-en500-w40-384-56-r16/ab_headshot_audio_v2_ckpt1000_5s/generated.mp4`
- `artifacts/generated/ref2va-lora-audio-v2-transcripts-en500-w40-384-56-r16/boy_closeup_audio_v2_ckpt1000_5s/generated.mp4`

#### Do Not Overwrite

Do not keep training this same run as the main path. Future experiments should write to a new output directory and should preserve this checkpoint as the stable baseline.

Next experimental direction: patch trainer for audio-focused tuning/freeze visual blocks instead of repeating joint visual/audio LoRA training.

### H3 LoRA Visual V2 Recipe

#### End Goal

Build a reusable H3 Ref2VA LoRA that makes a reference character look alive on video.

The model should:

- preserve the input character identity, outfit, colors, and framing;
- create plausible face, head, and mouth motion for a silent/talking-avatar clip;
- keep temporal consistency with no melting, face swaps, or identity drift;
- generalize across at least three reference styles: anime portrait, western/cartoon character, and 3D mascot/object-like character;
- run with external audio added later. H3-generated audio is not part of the success target.

The model should not:

- be selected because training loss alone improved;
- be judged from low-quality `256x256` / `6` step previews;
- rely on generated H3 audio;
- overfit to one repeated character, one crop, or one dataset style.

#### Product Acceptance Bar

A checkpoint is usable only if it passes all of these gates:

| gate | target |
| --- | --- |
| visual identity | at least 8/12 eval clips preserve the reference character clearly |
| motion | at least 8/12 eval clips show visible but stable mouth/head motion |
| temporal stability | no more than 2/12 clips have major melting, flicker, or face collapse |
| generalization | passes at least 2/3 style buckets: anime, illustrated/cartoon, 3D/mascot |
| base comparison | LoRA beats base H3 on at least 7/12 clips |
| validation loss | not worse than the best checkpoint by more than 5% unless visual score is clearly better |
| audio | ignored for checkpoint selection; generated audio can fail without rejecting the visual LoRA |

#### Dataset V2

Do not train on the whole TA2.0 dump directly.

Create a curated set:

| bucket | target clips | rule |
| --- | ---: | --- |
| clean anime talking heads | 600-1000 | centered face, visible mouth, stable camera |
| western/cartoon references | 300-600 | non-anime shapes, broader face/body diversity |
| 3D/mascot/toy-like references | 200-400 | round/simple characters, strong identity cues |
| hard negatives / reject holdout | 100-200 | blurry, duplicate, occluded, bad-mouth clips for eval only |

Filtering rules:

- remove near-duplicates by perceptual hash and CLIP/image embedding clustering;
- reject clips with tiny faces, heavy occlusion, extreme crops, subtitles/watermarks, or unstable camera;
- keep clips where mouth changes over time;
- keep diverse hair/eyes/body/clothing, not just blue-eyed closeups;
- cap each source character cluster so one character cannot dominate.

Split by character/source cluster, not random clip:

- train: 80%
- validation: 10%
- test: 10%

No near-duplicate character should appear in both train and eval.

#### Training

Important correction from upstream docs:

- The current `IAmIronMan42/MiniMax-H3-FineTuning` trainer says reference media are not yet encoded into the conditioning sequence. That means it is not a complete faithful Ref2VA fine-tuning path for our end goal unless we add reference-row conditioning.
- SimpleTuner has MiniMax H3 adapter support and explicitly supports visual-only target mode with `minimax_h3_target_mode: "video"`.
- SimpleTuner also recommends H3 drift distillation to keep the adapter close to the frozen base model instead of drifting away from the distilled H3 behavior.
- The MiniMax integration index notes that FL2VA generally has better raw output quality, while Ref2VA accepts richer reference media but has lower base quality. So if our goal is a polished user-facing demo, we should benchmark FL2VA/I2V and Ref2VA separately instead of assuming Ref2VA is always better.

Recommended first serious run:

| setting | value |
| --- | --- |
| base model | MiniMax-H3 Ref2VA only if reference conditioning is fixed; otherwise SimpleTuner H3 FL2VA/visual adapter |
| target | visual LoRA only (`minimax_h3_target_mode: "video"` equivalent) |
| audio loss | disabled or low-weighted unless target-audio conditioning is verified |
| cache resolution | 384 first, then 512 for final |
| frames | model-compatible 124-frame eval; training cache can stay trainer-compatible |
| LoRA rank | test `r=16` and `r=32` |
| LR | start lower than v1; prefer `5e-5` to `1e-4` sweep |
| checkpoints | every 500 or 1000 steps |
| stop rule | stop when visual score plateaus or validation worsens twice |
| drift control | keep a frozen-base drift/distillation term enabled if using SimpleTuner |

Run two short sweeps before a full train:

1. **Resolution sweep:** same curated subset, `256` vs `384`.
2. **Capacity sweep:** `r=16` vs `r=32`, same steps.

Only run the full training after one sweep clearly wins.

#### Eval

Every checkpoint gets the same eval refs and seeds.

Primary visual eval:

- `512x512`
- `30` inference steps
- `--no-generate-audio`
- same 12 held-out references
- same prompts and seeds

Eval set:

| group | count |
| --- | ---: |
| in-domain anime | 4 |
| out-of-domain cartoon/illustration | 3 |
| 3D/mascot/object-like | 3 |
| user-provided references | 2 |

Manual scorecard:

| metric | weight |
| --- | ---: |
| identity | 0.35 |
| talking/head motion | 0.30 |
| temporal stability | 0.20 |
| prompt/usefulness | 0.10 |
| audio | 0.05 diagnostic only |

Hard reject any checkpoint with:

- repeated face collapse on mascot/cartoon refs;
- identity drift across frames;
- no visible motion in most clips;
- visual score worse than base H3.

#### AQ / AQIT Usage

AQ tracks the experiment as a gate:

- dataset version;
- training config hash;
- checkpoint step;
- validation loss;
- visual score;
- base-vs-LoRA result;
- AQIT weight summary.

AQIT checks:

- LoRA file exists and loads;
- trainable parameter count matches expected rank;
- LoRA norms are not exploding;
- compare checkpoint deltas across steps;
- flag checkpoints where norms spike while validation worsens.

#### Decision Rule

Pick the earliest checkpoint that clears the product acceptance bar.

If none clear it, do not continue training the same recipe. Fix the data or objective first.

For v1, the likely diagnosis is:

- too much noisy/broad data;
- low-resolution training/eval path;
- audio branch distracting evaluation;
- no high-quality base-vs-LoRA gate;
- out-of-domain refs like Minion are not represented in training.

#### Next Run Plan

1. Decide trainer path:
   - Patch the current trainer to include Ref2VA reference conditioning rows; or
   - Switch to SimpleTuner MiniMax H3 adapter training for a visual-only LoRA.
2. Build a curated `visual_v2_1k` set.
3. Generate the 12-reference eval pack, including the Minion image/video as user-provided out-of-domain eval.
4. Train a `384px r=16` smoke for 1000-2000 steps.
5. Run `512px / 30-step / no-audio` eval every 500-1000 steps.
6. Compare against base H3, v1 checkpoints, and an FL2VA baseline.
7. Only then scale to `visual_v2_full`.

### H3 Ref2VA LoRA Audio V1

Goal: make the LoRA useful when H3 itself generates the audio, with clear English-ish speech, audible voice, and mouth motion that follows the generated sound. This run is not using external TTS as the final audio source.

#### Dataset Order

1. Start from TA2.0 only.
2. Keep clips tagged `language=en` and `motion_mode=active`.
3. Require at least 124 frames so the model sees multi-second speech.
4. Prefer clean source audio with `snr >= 8`, `c50 >= 10`, and `vqa_score >= 0.45`.
5. Cap to 2 clips per reference identity so one character cannot dominate the LoRA.

Manifest target: `/runs/manifests/ta20_audio_v1_en_500`.

#### Cache

Use longer cached samples than the visual smoke run:

- resolution: `384x384`
- frames: `56`
- train rows: all curated train rows
- cache: `/runs/cache/ta20_audio_v1_en_500_train_384_56`

#### Training

Start conservative:

- output: `/runs/h3/ref2va-lora-audio-v1-en-384-56-r16`
- max steps: `1500`
- checkpoints: every `500`
- LoRA rank: existing trainer rank 16
- lr: `5e-5`
- audio loss scale: `2.0`

#### Eval

At every checkpoint:

- Generate exactly 3 clips with model audio enabled.
- Use fixed references, seeds, prompts, and 30 inference steps.
- Track audio presence, duration, loudness/RMS, subjective English clarity, lip sync, identity retention, and artifacts.

Pass target for this run:

- voice is audible without external audio replacement
- generated speech sounds closer to English than the previous Korean/non-English outputs
- mouth opens/closes in plausible rhythm with generated voice
- identity is no worse than the visual-v2 1500 checkpoint

If audio is still non-English/noisy by checkpoint 1500, do not brute-force longer training. Next move is a stricter transcript/phoneme-ready dataset pass or H3 audio-head-only/dual-adapter experiment.

### H3 Ref2VA LoRA Audio V2: Transcript Conditioning

Audio-v1 improved lip timing but did not produce reliable English because the dataset only had `language=en`, not the actual spoken words. The next audio run must train on prompts that include transcripts.

#### Goal

The model should generate audible speech where:

- lip motion stays synchronized
- output sounds speech-like
- short English phrases are at least partially intelligible
- identity quality does not regress from visual-v2

#### Required Dataset Upgrade

For each English TA2.0 clip:

1. Extract audio.
2. Run ASR to get transcript and confidence.
3. Keep clips with:
   - English transcript
   - transcript length between 3 and 20 words
   - ASR confidence above threshold
   - SNR/C50 filters from audio-v1
4. Put the actual transcript into the prompt:

`<Video 1> The same character says: "..." with natural voice and synchronized mouth movement.`

#### Training Change

Use the same cache settings as audio-v1 initially:

- 384x384
- 56 frames
- LoRA rank 16
- audio loss scale 2.0

But choose checkpoints by generated transcript intelligibility, not video validation loss alone.

#### UI Behavior

The local UI should expose:

- model audio mode: current H3 generated audio
- transcript-conditioned checkpoint once audio-v2 exists
- no-audio only as a diagnostic option, not the default

Until audio-v2 is trained, UI output audio is experimental and should be treated as placeholder speech, not production narration.
