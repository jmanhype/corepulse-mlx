#!/usr/bin/env python3
"""
Test CorePulse hooks with identity processor to verify zero regression.
"""

import mlx.core as mx
import numpy as np
from pathlib import Path
import sys

# Add stable diffusion to path
sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import StableDiffusionXL
from stable_diffusion import attn_hooks


def identity_processor(*, out=None, meta=None):
    """Identity processor - returns None to keep original."""
    if meta:
        block_id = meta.get('block_id', 'unknown')
        step_idx = meta.get('step_idx', -1)
        if step_idx == 0:  # Only print once per block
            print(f"  ✓ Hook called for {block_id}")
    return None  # Keep original output


def main():
    print("\n🧪 TESTING COREPULSE HOOKS")
    print("="*50)
    
    # Load model
    print("Loading SDXL...")
    sd = StableDiffusionXL("stabilityai/sdxl-turbo")
    
    # Test parameters
    prompt = "a beautiful sunset landscape"
    seed = 42
    steps = 2  # Quick test
    
    # 1. Baseline without hooks
    print("\n1️⃣ Baseline (hooks disabled):")
    print(f"Hooks enabled: {attn_hooks.ATTN_HOOKS_ENABLED}")
    
    for latents_base in sd.generate_latents(prompt, num_steps=steps, seed=seed):
        pass
    
    decoded_base = sd.decode(latents_base)
    img_base = np.array(decoded_base[0])
    img_base = (img_base * 255).astype(np.uint8)
    
    from PIL import Image
    Image.fromarray(img_base).save("test_baseline.png")
    print("Saved: test_baseline.png")
    
    # 2. With identity hooks
    print("\n2️⃣ With identity hooks:")
    attn_hooks.ATTN_HOOKS_ENABLED = True
    print(f"Hooks enabled: {attn_hooks.ATTN_HOOKS_ENABLED}")
    
    # Register identity processor for all blocks
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_hooks.register_processor(block, identity_processor)
    
    print("\nGenerating with hooks...")
    for latents_hook in sd.generate_latents(prompt, num_steps=steps, seed=seed):
        pass
    
    decoded_hook = sd.decode(latents_hook)
    img_hook = np.array(decoded_hook[0])
    img_hook = (img_hook * 255).astype(np.uint8)
    
    Image.fromarray(img_hook).save("test_with_hooks.png")
    print("\nSaved: test_with_hooks.png")
    
    # 3. Verify parity
    print("\n3️⃣ Parity check:")
    is_identical = np.allclose(img_base/255.0, img_hook/255.0, rtol=1e-5)
    print(f"Images identical: {'✅ YES' if is_identical else '❌ NO'}")
    
    if is_identical:
        print("\n🎉 SUCCESS: Zero regression confirmed!")
    else:
        diff = np.abs(img_base.astype(float) - img_hook.astype(float)).mean()
        print(f"\n⚠️ Average difference: {diff:.6f}")
    
    # Clean up
    attn_hooks.ATTN_HOOKS_ENABLED = False
    attn_hooks.attention_registry.clear()


if __name__ == "__main__":
    main()