# H3 Avatar Results

This folder contains the clean files to hand to a hosting partner for a dedicated `h3-avatar-audio-v2` model endpoint.

## Files

- `adapter_model.safetensors`  
  LoRA-only state dict exported from the locked PyTorch training checkpoint.
- `adapter_config.json`  
  Adapter architecture and compatibility metadata.
- `inference_config.json`  
  Suggested default inference settings and request fields.
- `sample_prompt.json`  
  Example product request.

## Compatibility

This adapter was trained for MiniMax-H3 Ref2VA using Diffusers/PEFT-style module names. A host should validate that the serving model has compatible target modules before baking the adapter into a production endpoint.

The original training checkpoint remains in `assets/model/h3-avatar-audio-v2-step1000.pt`.
