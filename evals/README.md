# Benchmarks

Combined eval plans, objective reports, weight summaries, eval batches, and scorecards.
HTML galleries and dashboards: `index.html`. Eval prompts and gates in this folder: `avatar_benchmarks.json`, `progress_prompts.jsonl`, `avatar_identity_gate.jsonl`, `probes.jsonl`.

Source definitions:

- `evals/avatar_benchmarks.json`
- `evals/progress_prompts.jsonl`
- `evals/probes.jsonl`
- `src/configs/progress_checkpoints.json`

After a checkpoint review, record scores with `scripts/tests/record_checkpoint_eval.py`, then `scripts/helpers/benchmark_report.py`.

## Eval plan

Run: `/runs/h3/ref2va-lora-full-8000`

Checkpoints: `1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000`

For each checkpoint, record:

| eval | source | pass signal |
| --- | --- | --- |
| validation video loss | trainer log | lower than prior checkpoint or plateau without visual degradation |
| ffprobe sanity | `scripts/tests/eval_ref2va_batch.py` | generated MP4 has video, audio, expected resolution, expected duration |
| reference vs generated gallery | `scripts/tests/render_eval_gallery.py` | identity and motion can be inspected side by side |
| base vs LoRA gallery | same refs, seeds, prompts | LoRA output is visibly different from base in the intended direction |
| manual scorecard | the Scorecards section below | identity, talking motion, audio sync, prompt following, generalization |
| AQ gate row | `scripts/tests/apply_eval_scorecard.py` | weighted checkpoint score is written to `artifacts/h3/avatar_gate.json` |
| AQIT weight summary | `src/inspect/checkpoint_weight_summary.py` / AQIT when available | checkpoint is non-empty and LoRA norms are not exploding |

Manual score weights:

| metric | weight |
| --- | ---: |
| identity | 0.35 |
| talking_motion | 0.30 |
| audio_sync | 0.05 |
| prompt_following | 0.15 |
| generalization | 0.15 |

Selection rule:

Pick the checkpoint with the best combination of validation loss and manual visual score. Do not automatically pick the final step if validation loss worsens or the A/B gallery shows no meaningful improvement over base H3.

Generation policy:

- Generate 3 held-out validation references per checkpoint first.
- Use the same refs, prompts, seeds, frame count, and inference steps across checkpoints.
- Also generate base H3 outputs for those same refs once, so each checkpoint has a base-vs-LoRA comparison.
- Treat generated audio as a separate failure-prone lane. For checkpoint selection, prefer visual identity/talking motion; generated audio should be replaced with external clean English TTS or re-evaluated with a dedicated audio path.
- Use `--no-include-reference-audio` for visual checks, so source Korean audio is not copied into the reference conditioning.

## Checkpoint progress

| checkpoint | step | score | target | identity | motion | audio | prompt | generalization | result | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ckpt_020 | 20 |  | 0.350 |  |  |  |  |  |  |  |
| ckpt_040 | 40 |  | 0.450 |  |  |  |  |  |  |  |
| ckpt_060 | 60 |  | 0.550 |  |  |  |  |  |  |  |
| ckpt_080 | 80 |  | 0.630 |  |  |  |  |  |  |  |
| ckpt_100 | 100 |  | 0.700 |  |  |  |  |  |  |  |

### Benchmark Suites

Source: `evals/avatar_benchmarks.json` and `evals/progress_prompts.jsonl`.

## Objective report · smoke ~100

Run: `/runs/h3/ref2va-lora-smoke-100`

| checkpoint | step | val video loss | improvement vs step 0 | monotonic | remote checkpoint |
| --- | ---: | ---: | ---: | --- | --- |
| ckpt_020 | 20 | 0.598621 | 5.4% |  | `/runs/h3/ref2va-lora-smoke-100/checkpoint-20.pt` |
| ckpt_040 | 40 | 0.498908 | 21.2% | true | `/runs/h3/ref2va-lora-smoke-100/checkpoint-40.pt` |
| ckpt_060 | 60 | 0.415435 | 34.4% | true | `/runs/h3/ref2va-lora-smoke-100/checkpoint-60.pt` |
| ckpt_080 | 80 | 0.323771 | 48.8% | true | `/runs/h3/ref2va-lora-smoke-100/checkpoint-80.pt` |
| ckpt_100 | 100 | 0.293227 | 53.7% | true | `/runs/h3/ref2va-lora-smoke-100/checkpoint-100.pt` |

This is not the final avatar quality score. It proves the training objective is moving in the right direction; generated video review is the next quality gate.

## Objective report · mid ~500

Run: `/runs/h3/ref2va-lora-mid-500`

| step | val video loss | improvement vs step 0 | note |
| ---: | ---: | ---: | --- |
| 0 | 0.632806 | 0.0% | baseline |
| 100 | 0.296290 | 53.2% | improved |
| 200 | 0.263503 | 58.4% | improved |
| 300 | 0.254593 | 59.8% | improved |
| 400 | 0.251537 | 60.3% | best objective checkpoint |
| 500 | 0.255668 | 59.6% | worse than previous |

Best objective checkpoint: **step 400** (`loss_v=0.251537`).

Next gate: visual eval for checkpoint 400 versus 500, because step 500 regressed slightly on validation loss.

## Objective report · long ~1500

Run: `/runs/h3/ref2va-lora-long-1500`

| checkpoint | step | val video loss | improvement vs step 0 | monotonic | remote checkpoint |
| --- | ---: | ---: | ---: | --- | --- |
| ckpt_250 | 250 | 0.261144 | 58.7% |  | `/runs/h3/ref2va-lora-long-1500/checkpoint-250.pt` |
| ckpt_500 | 500 | 0.251213 | 60.3% | true | `/runs/h3/ref2va-lora-long-1500/checkpoint-500.pt` |
| ckpt_750 | 750 | 0.248858 | 60.7% | true | `/runs/h3/ref2va-lora-long-1500/checkpoint-750.pt` |
| ckpt_1000 | 1000 | 0.240677 | 62.0% | true | `/runs/h3/ref2va-lora-long-1500/checkpoint-1000.pt` |
| ckpt_1250 | 1250 | 0.235390 | 62.8% | true | `/runs/h3/ref2va-lora-long-1500/checkpoint-1250.pt` |
| ckpt_1500 | 1500 | 0.241075 | 61.9% | false | `/runs/h3/ref2va-lora-long-1500/checkpoint-1500.pt` |

This is not the final avatar quality score. It proves the training objective is moving in the right direction; generated video review is the next quality gate.

## Weight summary · ckpt 100

Checkpoint: `artifacts/training/ref2va-lora-smoke-100/checkpoints/checkpoint-100.pt`
Step: `100`
LoRA tensors: `108`
LoRA parameters: `10,655,744`

### Groups

| group | tensors | params | l2 norm |
| --- | ---: | ---: | ---: |
| time_embedder | 4 | 219,136 | 18.7126 |
| token_refiner | 4 | 401,408 | 29.9279 |
| transformer_blocks | 100 | 10,035,200 | 149.8501 |

### Largest Tensors By L2 Norm

| tensor | shape | l2 | abs max |
| --- | --- | ---: | ---: |
| `transformer_blocks.22.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2789 | 0.2871 |
| `transformer_blocks.42.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2475 | 0.2988 |
| `transformer_blocks.43.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2462 | 0.2930 |
| `transformer_blocks.9.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2418 | 0.2910 |
| `transformer_blocks.27.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2397 | 0.3145 |
| `transformer_blocks.20.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2392 | 0.2793 |
| `transformer_blocks.16.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2319 | 0.3184 |
| `transformer_blocks.34.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2301 | 0.3418 |
| `transformer_blocks.36.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2247 | 0.2754 |
| `transformer_blocks.4.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2241 | 0.2695 |
| `transformer_blocks.38.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2229 | 0.2930 |
| `transformer_blocks.25.attn.to_out.0.lora_A.default.weight` | `[16, 7168]` | 21.2213 | 0.3066 |

## Eval batch 001 · smoke ckpt 100

Objective pass: 5/5

Gallery: `evals/index.html`

| sample | seed | reference duration | generated duration | generated resolution | generated audio | objective status |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 | 100 | 9.96s | 5.17s | 256x256 | yes | pass |
| 1 | 101 | 9.96s | 5.17s | 256x256 | yes | pass |
| 2 | 102 | 9.96s | 5.17s | 256x256 | yes | pass |
| 3 | 103 | 9.96s | 5.17s | 256x256 | yes | pass |
| 4 | 104 | 9.96s | 5.17s | 256x256 | yes | pass |

### Manual Scorecard

Scores: see Scorecards below

Score each metric from 0.0 to 1.0 after watching the gallery:

- `identity`: generated subject resembles the reference character.
- `talking_motion`: face/body motion is plausible for talking.
- `audio_sync`: mouth motion and audio feel aligned.
- `prompt_following`: output follows the friendly talking prompt.
- `generalization`: held-out validation reference does not collapse to a memorized training character.

After manual scoring, record the checkpoint with `scripts/tests/record_checkpoint_eval.py` and run AQ eval.

## Eval batch · full ckpt 1000 test refs

Objective pass: 3/3

Gallery: `evals/index.html`

| sample | seed | reference duration | generated duration | generated resolution | generated audio | objective status |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 | 1000 | 9.96s | 5.17s | 256x256 | yes | pass |
| 1 | 1001 | 9.96s | 5.17s | 256x256 | yes | pass |
| 2 | 1002 | 9.96s | 5.17s | 256x256 | yes | pass |

### Manual Scorecard

Scores: see Scorecards below

Score each metric from 0.0 to 1.0 after watching the gallery:

- `identity`: generated subject resembles the reference character.
- `talking_motion`: face/body motion is plausible for talking.
- `audio_sync`: mouth motion and audio feel aligned.
- `prompt_following`: output follows the friendly talking prompt.
- `generalization`: held-out validation reference does not collapse to a memorized training character.

After manual scoring, record the checkpoint with `scripts/tests/record_checkpoint_eval.py` and run AQ eval.

## Eval batch · full ckpt 6000 diverse3

Objective pass: 3/3

Gallery: `evals/index.html`

| sample | seed | reference duration | generated duration | generated resolution | generated audio | objective status |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 | 206000 | 9.96s | 5.17s | 256x256 | yes | pass |
| 1 | 206001 | 9.96s | 5.17s | 256x256 | yes | pass |
| 2 | 206002 | 9.96s | 5.17s | 256x256 | yes | pass |

### Manual Scorecard

Scores: see Scorecards below

Score each metric from 0.0 to 1.0 after watching the gallery:

- `identity`: generated subject resembles the reference character.
- `talking_motion`: face/body motion is plausible for talking.
- `audio_sync`: mouth motion and audio feel aligned.
- `prompt_following`: output follows the friendly talking prompt.
- `generalization`: held-out validation reference does not collapse to a memorized training character.

After manual scoring, record the checkpoint with `scripts/tests/record_checkpoint_eval.py` and run AQ eval.

## Eval batch · full ckpt 8000 diverse3

Objective pass: 3/3

Gallery: `evals/index.html`

| sample | seed | reference duration | generated duration | generated resolution | generated audio | objective status |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 | 208000 | 9.96s | 5.17s | 256x256 | yes | pass |
| 1 | 208001 | 9.96s | 5.17s | 256x256 | yes | pass |
| 2 | 208002 | 9.96s | 5.17s | 256x256 | yes | pass |

### Manual Scorecard

Scores: see Scorecards below

Score each metric from 0.0 to 1.0 after watching the gallery:

- `identity`: generated subject resembles the reference character.
- `talking_motion`: face/body motion is plausible for talking.
- `audio_sync`: mouth motion and audio feel aligned.
- `prompt_following`: output follows the friendly talking prompt.
- `generalization`: held-out validation reference does not collapse to a memorized training character.

After manual scoring, record the checkpoint with `scripts/tests/record_checkpoint_eval.py` and run AQ eval.

## Scorecards

Manual Ref2VA eval scorecards, combined from the former `*_scorecard.csv` files.

Metric columns are 0–1: identity, talking motion, audio sync, prompt following, generalization, overall.

### Eval batch 001 (smoke ckpt 100)

Source: `ref2va_eval_batch_001_scorecard.csv`

| idx | seed | identity | motion | audio | prompt | gen. | overall | notes | reference | generated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 100 | — | — | — | — | — | — | — | artifacts/references/ta20_val_eval_clips/000_clip_00338.mp4 | artifacts/generated/ref2va-lora-smoke-100/eval_batch_001/000_000_clip_00338_generated.mp4 |
| 1 | 101 | — | — | — | — | — | — | — | artifacts/references/ta20_val_eval_clips/001_clip_07973.mp4 | artifacts/generated/ref2va-lora-smoke-100/eval_batch_001/001_001_clip_07973_generated.mp4 |
| 2 | 102 | — | — | — | — | — | — | — | artifacts/references/ta20_val_eval_clips/002_clip_02834.mp4 | artifacts/generated/ref2va-lora-smoke-100/eval_batch_001/002_002_clip_02834_generated.mp4 |
| 3 | 103 | — | — | — | — | — | — | — | artifacts/references/ta20_val_eval_clips/003_clip_07127.mp4 | artifacts/generated/ref2va-lora-smoke-100/eval_batch_001/003_003_clip_07127_generated.mp4 |
| 4 | 104 | — | — | — | — | — | — | — | artifacts/references/ta20_val_eval_clips/004_clip_03414.mp4 | artifacts/generated/ref2va-lora-smoke-100/eval_batch_001/004_004_clip_03414_generated.mp4 |

### Full run ckpt 1000 · test refs

Source: `ref2va_full_ckpt1000_testrefs_scorecard.csv`

| idx | seed | identity | motion | audio | prompt | gen. | overall | notes | reference | generated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1000 | 0.72 | 0.55 | 0.20 | 0.65 | 0.70 | 0.57 | Good character/outfit match; face softened and eye detail degraded; visible but muted talking variation; audio quality/sync rated low after user review. | artifacts/generated/ref2va-lora-full-8000/eval_ckpt_1000_testrefs/000_clip_00041.mp4 | artifacts/generated/ref2va-lora-full-8000/eval_ckpt_1000_testrefs/000_000_clip_00041_generated.mp4 |
| 1 | 1001 | 0.68 | 0.25 | 0.15 | 0.45 | 0.55 | 0.44 | Broad clothing preserved but identity is weaker than first pass; generated frames look nearly static; audio bad; this ref is near-duplicate with sample 2. | artifacts/generated/ref2va-lora-full-8000/eval_ckpt_1000_testrefs/001_clip_02048.mp4 | artifacts/generated/ref2va-lora-full-8000/eval_ckpt_1000_testrefs/001_001_clip_02048_generated.mp4 |
| 2 | 1002 | 0.45 | 0.22 | 0.15 | 0.40 | 0.40 | 0.32 | Weak identity and underactive animation; near-duplicate reference with sample 1, so this is not a good independent benchmark sample; audio bad. | artifacts/generated/ref2va-lora-full-8000/eval_ckpt_1000_testrefs/002_clip_10961.mp4 | artifacts/generated/ref2va-lora-full-8000/eval_ckpt_1000_testrefs/002_002_clip_10961_generated.mp4 |

### Full run ckpt 6000 · diverse3

Source: `ref2va_full_ckpt6000_diverse3_scorecard.csv`

| idx | seed | identity | motion | audio | prompt | gen. | overall | notes | reference | generated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 206000 | — | — | — | — | — | — | — | artifacts/generated/ref2va-lora-full-8000/eval_ckpt_6000_diverse3/000_clip_00041.mp4 | artifacts/generated/ref2va-lora-full-8000/eval_ckpt_6000_diverse3/000_000_clip_00041_generated.mp4 |
| 1 | 206001 | — | — | — | — | — | — | — | artifacts/generated/ref2va-lora-full-8000/eval_ckpt_6000_diverse3/001_clip_02048.mp4 | artifacts/generated/ref2va-lora-full-8000/eval_ckpt_6000_diverse3/001_001_clip_02048_generated.mp4 |
| 2 | 206002 | — | — | — | — | — | — | — | artifacts/generated/ref2va-lora-full-8000/eval_ckpt_6000_diverse3/002_clip_04721.mp4 | artifacts/generated/ref2va-lora-full-8000/eval_ckpt_6000_diverse3/002_002_clip_04721_generated.mp4 |

### Full run ckpt 8000 · diverse3

Source: `ref2va_full_ckpt8000_diverse3_scorecard.csv`

| idx | seed | identity | motion | audio | prompt | gen. | overall | notes | reference | generated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 208000 | — | — | — | — | — | — | — | artifacts/generated/ref2va-lora-full-8000/eval_ckpt_8000_diverse3/000_clip_00041.mp4 | artifacts/generated/ref2va-lora-full-8000/eval_ckpt_8000_diverse3/000_000_clip_00041_generated.mp4 |
| 1 | 208001 | — | — | — | — | — | — | — | artifacts/generated/ref2va-lora-full-8000/eval_ckpt_8000_diverse3/001_clip_02048.mp4 | artifacts/generated/ref2va-lora-full-8000/eval_ckpt_8000_diverse3/001_001_clip_02048_generated.mp4 |
| 2 | 208002 | — | — | — | — | — | — | — | artifacts/generated/ref2va-lora-full-8000/eval_ckpt_8000_diverse3/002_clip_04721.mp4 | artifacts/generated/ref2va-lora-full-8000/eval_ckpt_8000_diverse3/002_002_clip_04721_generated.mp4 |
