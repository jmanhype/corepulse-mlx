#!/usr/bin/env python3
"""
Example 1: Basic Semantic Object Replacement
Replace objects in prompts to generate different items.
"""

from corpus_mlx import create_semantic_wrapper
from PIL import Image

# Create wrapper with semantic replacement capability
wrapper = create_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")

# Add replacement rules
wrapper.add_replacement("apple", "banana")
wrapper.add_replacement("cat", "dog")
wrapper.add_replacement("car", "bicycle")

# Enable semantic replacement
wrapper.enable()

# Generate with replacements
# "apple" will become "banana" automatically!
print("Generating: 'a photo of an apple' → will produce banana")

latents = None
for step_latents in wrapper.wrapper.generate_latents(
    "a photo of an apple on a table",
    negative_text="blurry, ugly",
    num_steps=20,
    cfg_weight=7.5,
    seed=42
):
    latents = step_latents

# Decode and save
images = wrapper.wrapper.sd.autoencoder.decode(latents)
img = images[0]
img = mx.clip(img, -1, 1)
img = ((img + 1) * 127.5).astype(mx.uint8)
img = np.array(img)

Image.fromarray(img).save("example_apple_to_banana.png")
print("✅ Saved: example_apple_to_banana.png (should show a banana!)")

# Disable for normal generation
wrapper.disable()

import mlx.core as mx
import numpy as np