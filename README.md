# corpus-mlx

CorePulse V4 DataVoid implementation for MLX/Apple Silicon - Pre-attention KV manipulation and embedding injection for SDXL.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from corpus_mlx import CorePulse
from stable_diffusion import StableDiffusionXL

# Load model
model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)

# Create CorePulse wrapper
corepulse = CorePulse(model)

# Add injection
corepulse.add_injection(
    prompt="ethereal aurora",
    strength=0.3,
    blocks=["mid", "up_0", "up_1"]
)

# Generate
image = corepulse.generate("majestic lion in savanna")
```

## Features

- Pre-attention KV manipulation
- Embedding injection pipeline
- Block-specific control
- MLX optimization for Apple Silicon

## Documentation

See `docs/` for detailed documentation.

## Examples

See `examples/` for usage examples.
