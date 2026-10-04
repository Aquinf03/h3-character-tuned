# H3 Avatar

Single-image talking-avatar generation powered by a locked MiniMax-H3 Ref2VA LoRA.

![H3 Avatar demo frame](assets/demo.jpg)

## About

H3 Avatar turns one character image plus a short line of dialogue into a talking avatar clip with model-generated speech, synchronized mouth movement, subtle head motion, and controllable scene direction. The repo packages the locked LoRA, local UI, Modal reference serving, exported adapter, training code, eval reports, and AQ recipe handoff.

## Product Shape

The app takes:

- a reference image
- dialogue for the character to say
- direction for expression, motion, and camera behavior
- a background preset or custom background

The model recipe is locked behind the UI so users do not need to choose checkpoints, steps, audio mode, or model variants.

## Model

| Item | Value |
| --- | --- |
| Base | MiniMax-H3 Ref2VA |
| Adapter | LoRA |
| Rank / alpha | `16 / 32` |
| Target modules | `to_qkv`, `to_out.0`, `linear_1`, `linear_2` |
| Trainable params | `10,655,744` |
| Locked checkpoint | `assets/model/h3-avatar-audio-v2-step1000.pt` |
| Partner weights | `results/adapter_model.safetensors` |

Training summary:

- 481 accepted transcript-conditioned clips
- 385 train clips
- 1000 optimizer steps
- about 2.6 train-set passes
- 384x384, 56-frame cache

## Repo Structure

```text
.
├── README.md
├── LICENSE.md
├── DOCS.md
├── assets/
│   ├── demo.jpg
│   └── model/
│       ├── README.md
│       └── h3-avatar-audio-v2-step1000.pt
├── evals/
│   ├── README.md
│   ├── index.html
│   └── prompts and gates
├── recipe.yaml
├── scripts/
│   ├── helpers/
│   └── tests/
├── src/
│   ├── train.py
│   ├── prepare_cache.py
│   ├── h3_ref2va.py
│   ├── data/
│   ├── inspect/
│   └── configs/
└── results/
    ├── README.md
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── inference_config.json
    └── sample_prompt.json
```

## Training And Evaluation

- Training code: `src/` (`train.py`, `prepare_cache.py`, data and inspect scripts)
- Training notes: `DOCS.md`
- Configs: `src/configs/`
- Eval reports, galleries, prompts, and gates: `evals/`
- Eval and reporting scripts: `scripts/tests/` and `scripts/helpers/`
- AQ handoff: `recipe.yaml`

## Run The Local UI

```bash
H3_WARM_ENDPOINT_URL=https://aquin--h3-avatar-warm-generate.modal.run \
python3 scripts/helpers/h3_avatar_ui.py
```

Then open:

```text
http://127.0.0.1:8787
```

## Deploy The Reference Modal Endpoint

```bash
modal deploy scripts/helpers/modal_h3_avatar.py
```

Current deployed endpoint:

```text
https://aquin--h3-avatar-warm-generate.modal.run
```

The Modal path is a reference implementation. Production hosting should use a dedicated inference provider that keeps the base H3 model hot.

## Results

Use `results/` for the exported adapter and hosting configs (Reactor, fal, or another partner).

Expected hosted model:

```text
MiniMax-H3 Ref2VA + h3-avatar-audio-v2 LoRA
```

Recommended endpoint inputs:

```json
{
  "reference_image": "...",
  "dialogue": "Hey Chandan, wassup yaar, how are you doing these days?",
  "direction": "The same character faces the camera with natural voice and synchronized mouth movement.",
  "background": "in a clean studio with soft cinematic lighting"
}
```

The adapter is small; the serving challenge is keeping the H3 base model loaded and optimized.

## License

Copyright 2026 Aquin Labs Private Limited. Licensed under the Apache License, Version 2.0. See `LICENSE.md`.
