#!/usr/bin/env python3
"""Debug test for prompt injection."""

import sys
sys.path.insert(0, 'src/adapters/mlx/mlx-examples/stable_diffusion')

from corpus_mlx import CorePulse
from stable_diffusion import StableDiffusionXL
import mlx.core as mx

# Load model
print("Loading SDXL-Turbo...")
model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)

# Create CorePulse wrapper
corepulse = CorePulse(model)

# Try injection
print("\nAdding injection: 'yellow banana'")
corepulse.add_injection(
    prompt="yellow banana",
    strength=0.5,
    blocks=["mid", "up_0", "up_1"]
)

# Generate with baseline
print("\nGenerating with injection...")
latents = model.generate_latents(
    "red apple",
    num_steps=1,
    cfg_weight=0.0,
    seed=42
)

# Get final image
for i, x in enumerate(latents):
    print(f"Step {i}")
    final = x

print("\nDone! Check if hook was called above.")