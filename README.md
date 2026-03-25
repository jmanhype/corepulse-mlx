# corepulse-mlx

Attention manipulation toolkit for Stable Diffusion on Apple Silicon (MLX). Implements prompt injection, semantic object replacement, and attention control techniques from the CorePulse V4/DataVoid research.

## Status

**Experimental.** Text-level semantic replacement works. Embedding-level injection is partially implemented -- attention hooks fire during generation but do not yet override semantic content from the text encoder. See "Known limitations" below.

## What it does

| Technique | Mechanism | Status |
|---|---|---|
| Text-level semantic replacement | String substitution before tokenization (apple -> banana) | Working |
| Embedding injection | Blends replacement embeddings into text conditioning tensors | Working (12+ categories tested) |
| Time-windowed prompt injection | Applies secondary prompts during specific diffusion timestep ranges | Working |
| Regional prompt injection | Restricts injection to spatial regions of the latent | Working |
| Token-masked injection | Targets injection at specific token indices | Working |
| Attention chaos/suppress/amplify/invert | Manipulates cross-attention values (V) directly | Working |
| Cross-attention swapping | Swaps attention maps between prompts | Not working |
| Regional spatial control | Fine-grained region-to-prompt mapping | Not working |

## Requirements

- Python >= 3.9
- Apple Silicon Mac (MLX backend) or CUDA GPU (PyTorch backend)
- `diffusers`, `transformers`, `accelerate`, `safetensors`, `Pillow`, `matplotlib`

## Installation

```bash
pip install -e .
```

## Usage

### Text-level replacement

```python
from corpus_mlx import create_semantic_wrapper

wrapper = create_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
wrapper.add_replacement("apple", "banana")
wrapper.add_replacement("cat", "dog")
wrapper.enable()

latents = wrapper.wrapper.generate_latents("a photo of an apple")
# Generates banana instead of apple
```

### Embedding injection

```python
from corpus_mlx import create_true_semantic_wrapper

wrapper = create_true_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
wrapper.add_replacement("cat", "golden retriever dog", weight=0.7)
wrapper.injector.enable_for_prompt("a fluffy cat in a garden")

latents = wrapper.sd.generate_latents("a fluffy cat in a garden")
```

### Attention manipulation

```python
from corpus_mlx import CorePulse
from stable_diffusion import StableDiffusionXL

model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
corepulse = CorePulse(model)

corepulse.chaos(intensity=2.0)
corepulse.suppress(factor=0.05)
corepulse.amplify(strength=10.0)
corepulse.invert()
```

## Known limitations

1. **Semantic prompt override does not work.** Modifying attention values (V) is insufficient to override the text encoder's conditioning. True prompt injection requires replacing conditioning at the encoder level, not blending attention values.
2. **Cross-attention swapping** is stubbed but not functional.
3. **Regional control** is not properly wired to spatial masks.
4. The codebase carries some artifacts from earlier CorePulse iterations (configs, logs, demo scripts) that have not been cleaned up.

## Project structure

```
CorePulse/
  core_pulse/
    models/          # UNet patcher, base model wrappers
    prompt_injection/ # Injection strategies (base, advanced, spatial, attention, masking)
    utils/           # Helpers, logger, tokenizer
  tests/             # Unit tests for injection and model modules
  pyproject.toml
```

## License

See LICENSE file in the repository root.
