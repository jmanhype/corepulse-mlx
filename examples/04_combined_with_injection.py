#!/usr/bin/env python3
"""
Example 4: Semantic Replacement + Prompt Injection
Combine semantic replacement with corpus-mlx injection features.
"""

from corpus_mlx import create_semantic_wrapper
from PIL import Image
import mlx.core as mx
import numpy as np

# Create wrapper
wrapper = create_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")

# Add semantic replacement
wrapper.add_replacement("apple", "banana")

# Add prompt injection (corpus-mlx feature)
wrapper.wrapper.add_injection(
    prompt="golden shiny metallic",
    weight=0.4  # Moderate injection strength
)

# Enable semantic replacement
wrapper.enable()

# Generate with both features
# - "apple" becomes "banana" (semantic replacement)  
# - "golden shiny metallic" style is injected
print("Generating with semantic replacement + style injection:")
print("  Base: 'a photo of an apple'")
print("  Replacement: apple → banana")
print("  Injection: 'golden shiny metallic' style")

latents = None
for step_latents in wrapper.wrapper.generate_latents(
    "a photo of an apple",
    negative_text="blurry, ugly",
    num_steps=20,
    cfg_weight=7.5,
    seed=42,
    height=256,
    width=256
):
    latents = step_latents

# Decode and save
images = wrapper.wrapper.sd.autoencoder.decode(latents)
img = images[0]
img = mx.clip(img, -1, 1)
img = ((img + 1) * 127.5).astype(mx.uint8)
img = np.array(img)

Image.fromarray(img).save("example_combined_features.png")
print("\n✅ Saved: example_combined_features.png")
print("   (should show a golden/metallic banana!)")

wrapper.disable()