#!/usr/bin/env python3
"""
Example 3: Multiple Object Replacements
Replace multiple objects in a single prompt.
"""

from corpus_mlx import create_semantic_wrapper
from PIL import Image
import mlx.core as mx
import numpy as np

# Create wrapper
wrapper = create_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")

# Add multiple replacement rules
wrapper.add_replacement("apple", "orange")
wrapper.add_replacement("table", "counter")
wrapper.add_replacement("wooden", "marble")

# Enable replacements
wrapper.enable()

# Original prompt with multiple objects
prompt = "a red apple on a wooden table in a kitchen"
print(f"Original: {prompt}")

# Generate (will replace all matched objects)
latents = None
for step_latents in wrapper.wrapper.generate_latents(
    prompt,
    negative_text="blurry, ugly",
    num_steps=20,
    cfg_weight=7.5,
    seed=42,
    height=256,
    width=256
):
    latents = step_latents

# The prompt was automatically changed to:
# "a red orange on a marble counter in a kitchen"

# Decode and save
images = wrapper.wrapper.sd.autoencoder.decode(latents)
img = images[0]
img = mx.clip(img, -1, 1)
img = ((img + 1) * 127.5).astype(mx.uint8)
img = np.array(img)

Image.fromarray(img).save("example_multiple_replacements.png")
print("✅ Saved: example_multiple_replacements.png")
print("   (should show orange on marble counter instead of apple on wooden table)")

wrapper.disable()