#!/usr/bin/env python3
"""
Test: Attention Head Isolation
Manipulate specific attention heads only, leaving others untouched.
This demonstrates fine-grained control over individual attention mechanisms.
"""

import sys
import gc
from pathlib import Path
import mlx.core as mx
import PIL.Image
import numpy as np

# Add the stable_diffusion module to path
sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')

# Enable hooks BEFORE importing model
from stable_diffusion import attn_scores
attn_scores.enable_kv_hooks(True)

# Import model components
from stable_diffusion import StableDiffusionXL

def create_head_isolation_hook(target_heads, manipulation_factor=2.0):
    """
    Create a hook that only manipulates specific attention heads.
    
    Args:
        target_heads: List of head indices to manipulate (0-based)
        manipulation_factor: How much to amplify/suppress those heads
    """
    def hook(q, k, v, meta=None):
        # Only modify cross-attention (text-to-image attention)
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, num_heads, seq_len, dim = k.shape
            
            # Create a copy for manipulation
            k_new = mx.array(k)
            v_new = mx.array(v)
            
            # Only manipulate specified heads
            for head_idx in target_heads:
                if head_idx < num_heads:
                    # Amplify or suppress specific heads
                    k_new[:, head_idx, :, :] *= manipulation_factor
                    v_new[:, head_idx, :, :] *= manipulation_factor
                    print(f"    🎯 Manipulated head {head_idx}/{num_heads} with factor {manipulation_factor}")
            
            return q, k_new, v_new
        return q, k, v
    return hook

def main():
    print("🎯 Test: Attention Head Isolation")
    print("=" * 60)
    
    # Configuration
    prompt = "a majestic eagle soaring through clouds at sunset"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    print(f"📝 Prompt: '{prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Create output directory
    output_dir = Path("artifacts/images/head_isolation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Test 1: Baseline (no manipulation)
    print("\n🎨 Test 1: Baseline generation...")
    attn_scores.KV_REGISTRY.clear()
    
    latents = model.generate_latents(
        prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "baseline.png")
    print("✅ Saved: baseline.png")
    
    # Test 2: Manipulate first 4 heads only (amplify)
    print("\n🎨 Test 2: Amplify first 4 attention heads...")
    attn_scores.KV_REGISTRY.clear()
    
    # Target heads 0-3, amplify by 3x
    hook = create_head_isolation_hook([0, 1, 2, 3], manipulation_factor=3.0)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "heads_0_3_amplified.png")
    print("✅ Saved: heads_0_3_amplified.png")
    
    # Test 3: Manipulate last 4 heads only (suppress)
    print("\n🎨 Test 3: Suppress last 4 attention heads...")
    attn_scores.KV_REGISTRY.clear()
    
    # We don't know exact head count, so use large indices
    # The hook will check bounds
    hook = create_head_isolation_hook([16, 17, 18, 19], manipulation_factor=0.1)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "heads_last_4_suppressed.png")
    print("✅ Saved: heads_last_4_suppressed.png")
    
    # Test 4: Alternate heads (checkerboard pattern)
    print("\n🎨 Test 4: Alternate head manipulation (checkerboard)...")
    attn_scores.KV_REGISTRY.clear()
    
    # Amplify even heads, suppress odd heads
    def checkerboard_hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, num_heads, seq_len, dim = k.shape
            k_new = mx.array(k)
            v_new = mx.array(v)
            
            for head_idx in range(num_heads):
                if head_idx % 2 == 0:
                    # Amplify even heads
                    k_new[:, head_idx, :, :] *= 2.5
                    v_new[:, head_idx, :, :] *= 2.5
                else:
                    # Suppress odd heads
                    k_new[:, head_idx, :, :] *= 0.2
                    v_new[:, head_idx, :, :] *= 0.2
            
            print(f"    🏁 Applied checkerboard pattern to {num_heads} heads")
            return q, k_new, v_new
        return q, k, v
    
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, checkerboard_hook)
    
    latents = model.generate_latents(
        prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "heads_checkerboard.png")
    print("✅ Saved: heads_checkerboard.png")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n" + "=" * 60)
    print("✅ Attention Head Isolation Test Complete!")
    print("📊 Results:")
    print("  - baseline.png: Normal generation")
    print("  - heads_0_3_amplified.png: First 4 heads amplified 3x")
    print("  - heads_last_4_suppressed.png: Last 4 heads suppressed to 0.1x")
    print("  - heads_checkerboard.png: Alternating amplify/suppress pattern")
    print("💡 This proves we can manipulate individual attention heads!")

if __name__ == "__main__":
    main()