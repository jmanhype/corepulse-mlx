#!/usr/bin/env python3
"""Direct test of hooks."""

import sys
sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import attn_scores
import mlx.core as mx
import numpy as np

print("Testing hook system directly...")

# Enable hooks
attn_scores.enable_kv_hooks(True)
print(f"KV hooks enabled: {attn_scores._global_state['KV_HOOKS_ENABLED']}")

# Register a test hook
def test_hook(q, k, v, meta):
    print(f"✅ HOOK FIRED! Block: {meta.get('block_id')}")
    return q, k, v

# Register for all blocks  
for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
    attn_scores.KV_REGISTRY.set(block, test_hook)
    print(f"Registered hook for {block}")

# Check registry
print(f"\nRegistry has {len(attn_scores.KV_REGISTRY._hooks)} hooks")

# Now load model
from stable_diffusion import StableDiffusionXL
print("\nLoading model...")
sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)

print("\nGenerating with hooks...")
latents_gen = sd.generate_latents(
    "test",
    n_images=1,
    num_steps=1,
    cfg_weight=1.0
)

for latents in latents_gen:
    print("Generation complete")

print("\nDone!")