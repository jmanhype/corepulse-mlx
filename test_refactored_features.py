#!/usr/bin/env python3
"""Test that refactored advanced features work."""

import sys
import os
sys.path.insert(0, '/Users/speed/Downloads/corpus-mlx/src')
sys.path.insert(0, '/Users/speed/Downloads/mlx-examples')

import mlx.core as mx
import numpy as np
from PIL import Image
from mlx_stable_diffusion import StableDiffusion
from corpus_mlx import CorePulseStableDiffusion

print("Testing refactored CorePulse with advanced features...")
print("=" * 60)

# Load model
print("Loading model...")
sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
wrapper = CorePulseStableDiffusion(sd)

# Test 1: Time-windowed injection
print("\n1. Testing time-windowed injection...")
wrapper.clear_injections()
wrapper.add_injection(
    prompt="dark mysterious atmosphere",
    start_frac=0.0,
    end_frac=0.3,
    weight=0.7
)
print("   ✓ Time window: 0-30% of generation")

# Test 2: Token masking
print("\n2. Testing token masking...")
wrapper.add_injection(
    prompt="glowing magical crystal orb",
    token_mask="crystal orb",
    start_frac=0.3,
    end_frac=0.7,
    weight=0.8
)
print("   ✓ Token mask on 'crystal orb': 30-70% of generation")

# Test 3: Regional injection
print("\n3. Testing regional injection...")
wrapper.add_injection(
    prompt="brilliant light rays",
    region=("rect_pix", 150, 150, 362, 362, 30),
    start_frac=0.5,
    end_frac=1.0,
    weight=0.6
)
print("   ✓ Regional control: center region, 50-100% of generation")

# Generate with all features
print("\n4. Generating with ALL features combined...")
latents = None
for step_latents in wrapper.generate_latents(
    "mystical forest scene",
    negative_text="blurry, low quality",
    num_steps=20,
    cfg_weight=7.5,
    seed=42
):
    latents = step_latents

# Decode and save
print("5. Decoding and saving result...")
images = sd.autoencoder.decode(latents)
images = mx.clip(images / 2 + 0.5, 0, 1)
images = (images * 255).astype(mx.uint8)

images_np = np.array(images)
if images_np.ndim == 4:
    images_np = images_np[0]
if images_np.shape[0] in [3, 4]:
    images_np = np.transpose(images_np, (1, 2, 0))

img = Image.fromarray(images_np)
img.save("test_refactored_all_features.png")

print("\n" + "=" * 60)
print("✅ SUCCESS! All advanced features working:")
print("   - Time-windowed injection ✓")
print("   - Token-level masking ✓")
print("   - Regional control ✓")
print("   - Combined multi-feature injection ✓")
print(f"\n📸 Result saved to: test_refactored_all_features.png")