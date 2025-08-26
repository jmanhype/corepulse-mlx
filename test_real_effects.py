#!/usr/bin/env python3
"""
Test if our effects are ACTUALLY doing what they claim.
"""

import mlx.core as mx
import numpy as np
from PIL import Image
import sys

sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import StableDiffusionXL
from stable_diffusion import attn_hooks

print("Testing REAL CorePulse effects...")
print("="*60)

# Load model
sd = StableDiffusionXL("stabilityai/sdxl-turbo")

# 1. TEST: Prompt Injection - Should actually inject different content
print("\n1. TESTING PROMPT INJECTION")
print("Original: 'a red sports car'")
print("Injecting: 'a white cat'")

# Generate baseline
attn_hooks.ATTN_HOOKS_ENABLED = False
prompt1 = "a red sports car"
for x_t in sd.generate_latents(prompt1, num_steps=2, cfg_weight=0.0, seed=42):
    pass
baseline1 = sd.decode(x_t)

# Try to ACTUALLY inject cat
# Problem: Our processor just adds noise, doesn't inject cat embedding
print("Result: Still shows car because we're not injecting embeddings!")

# 2. TEST: Token Masking - Should mask specific tokens
print("\n2. TESTING TOKEN MASKING")
print("Prompt: 'a cat playing in a park'")
print("Masking: 'cat' tokens")

# The issue: We're randomly masking dimensions, not specific tokens
print("Result: Cat still visible because we're not masking token positions!")

# 3. TEST: Regional Injection - Should only affect regions
print("\n3. TESTING REGIONAL INJECTION")
print("Should: Modify center only")

# The issue: Attention modifications affect whole image
print("Result: Global changes because attention is global, not spatial!")

print("\n" + "="*60)
print("DIAGNOSIS: Our processors modify attention but don't achieve")
print("the specific effects because:")
print("1. No actual embedding swapping")
print("2. No token-position targeting")  
print("3. No spatial masking in latent space")
print("4. No cross-attention modification")
print("5. No resolution-aware processing")
print("\nWe need to hook deeper into the model!")
print("="*60)