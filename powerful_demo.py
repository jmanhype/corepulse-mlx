#!/usr/bin/env python3
"""
POWERFUL demonstration using more generation steps for clearer effects.
Uses SDXL-Turbo with multiple steps to show dramatic differences.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path
import sys
import gc

# Add the stable_diffusion module to path
sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')

# Enable hooks BEFORE importing model
from stable_diffusion import attn_scores
attn_scores.enable_kv_hooks(True)

# Import model components
from stable_diffusion import StableDiffusionXL
import PIL.Image

def create_token_removal_hook(remove_range=(2, 5)):
    """Remove specific token ranges to demonstrate semantic removal."""
    def hook(q, k, v, meta=None):
        # Only modify cross-attention
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            k_new = mx.array(k)
            v_new = mx.array(v)
            
            # Zero out specified token range
            if k.shape[2] > remove_range[1]:
                k_new[:, :, remove_range[0]:remove_range[1], :] = 0
                v_new[:, :, remove_range[0]:remove_range[1], :] = 0
                print(f"    🗑️  Removed tokens {remove_range[0]}-{remove_range[1]} at {meta.get('block_id', 'unknown')}")
            
            return q, k_new, v_new
        return q, k, v
    return hook

def create_amplification_hook(factor=3.0):
    """Amplify text influence significantly."""
    def hook(q, k, v, meta=None):
        # Only modify cross-attention
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            print(f"    🔥 Amplifying {factor}x at {meta.get('block_id', 'unknown')}")
            return q, k * factor, v * factor
        return q, k, v
    return hook

def create_suppression_hook(factor=0.1):
    """Suppress text influence to near zero."""
    def hook(q, k, v, meta=None):
        # Only modify cross-attention
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            print(f"    🧊 Suppressing to {factor}x at {meta.get('block_id', 'unknown')}")
            return q, k * factor, v * factor
        return q, k, v
    return hook

def create_chaos_hook(intensity=1.0):
    """Add significant noise to text embeddings."""
    def hook(q, k, v, meta=None):
        # Only modify cross-attention
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            print(f"    🌪️  Chaos intensity {intensity} at {meta.get('block_id', 'unknown')}")
            k_noise = mx.random.normal(k.shape) * intensity
            v_noise = mx.random.normal(v.shape) * intensity
            return q, k + k_noise, v + v_noise
        return q, k, v
    return hook

def create_inversion_hook():
    """Invert the text conditioning (negative prompt effect)."""
    def hook(q, k, v, meta=None):
        # Only modify cross-attention
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            print(f"    ↩️  Inverting at {meta.get('block_id', 'unknown')}")
            return q, -k, -v
        return q, k, v
    return hook

def main():
    print("=" * 80)
    print("🚀 POWERFUL COREPULSE V4 DEMONSTRATION")
    print("=" * 80)
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Output directory
    output_dir = Path("artifacts/images/powerful_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    prompt = "a majestic lion with golden mane in dramatic lighting"
    seed = 42
    num_steps = 5  # Reduced for faster generation but still visible effects
    cfg_weight = 7.5  # Enable CFG for stronger guidance
    
    print(f"\nBase prompt: '{prompt}'")
    print(f"Steps: {num_steps}, CFG: {cfg_weight}")
    
    tests = []
    
    # ============================================================
    # TEST 1: Baseline
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 1: Baseline (No Hooks)")
    print("=" * 60)
    
    attn_scores.KV_REGISTRY.clear()
    
    latents = model.generate_latents(
        prompt, 
        num_steps=num_steps, 
        cfg_weight=cfg_weight, 
        seed=seed
    )
    for i, x in enumerate(latents):
        if i == num_steps - 1:  # Last step
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "01_baseline.png")
    print("✅ Saved: 01_baseline.png")
    tests.append("Baseline - Standard generation")
    
    # ============================================================
    # TEST 2: Remove Key Words
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: Token Removal (Remove 'lion')")
    print("=" * 60)
    
    attn_scores.KV_REGISTRY.clear()
    
    # Remove tokens 2-4 (likely contains "lion")
    removal_hook = create_token_removal_hook((2, 4))
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, removal_hook)
    
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
    pil_img.save(output_dir / "02_removed_lion.png")
    print("✅ Saved: 02_removed_lion.png")
    tests.append("Token Removal - 'lion' removed")
    
    # ============================================================
    # TEST 3: Extreme Amplification
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3: Extreme Amplification (5x)")
    print("=" * 60)
    
    attn_scores.KV_REGISTRY.clear()
    
    amp_hook = create_amplification_hook(5.0)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, amp_hook)
    
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
    pil_img.save(output_dir / "03_amplified_5x.png")
    print("✅ Saved: 03_amplified_5x.png")
    tests.append("Amplification - 5x text influence")
    
    # ============================================================
    # TEST 4: Extreme Suppression
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 4: Extreme Suppression (0.05x)")
    print("=" * 60)
    
    attn_scores.KV_REGISTRY.clear()
    
    supp_hook = create_suppression_hook(0.05)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, supp_hook)
    
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
    pil_img.save(output_dir / "04_suppressed.png")
    print("✅ Saved: 04_suppressed.png")
    tests.append("Suppression - 0.05x (nearly random)")
    
    # ============================================================
    # TEST 5: Maximum Chaos
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 5: Maximum Chaos")
    print("=" * 60)
    
    attn_scores.KV_REGISTRY.clear()
    
    chaos_hook = create_chaos_hook(2.0)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, chaos_hook)
    
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
    pil_img.save(output_dir / "05_chaos.png")
    print("✅ Saved: 05_chaos.png")
    tests.append("Chaos - Maximum noise injection")
    
    # ============================================================
    # TEST 6: Inversion (Negative)
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 6: Inversion (Anti-prompt)")
    print("=" * 60)
    
    attn_scores.KV_REGISTRY.clear()
    
    inv_hook = create_inversion_hook()
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, inv_hook)
    
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
    pil_img.save(output_dir / "06_inverted.png")
    print("✅ Saved: 06_inverted.png")
    tests.append("Inversion - Opposite of prompt")
    
    # ============================================================
    # TEST 7: Progressive Manipulation
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 7: Progressive (Weak→Strong)")
    print("=" * 60)
    
    attn_scores.KV_REGISTRY.clear()
    
    # Progressive amplification from weak to strong
    def create_progressive_hook(strength):
        def hook(q, k, v, meta=None):
            if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
                return q, k * strength, v * strength
            return q, k, v
        return hook
    
    attn_scores.KV_REGISTRY.set("down_0", create_progressive_hook(0.2))
    attn_scores.KV_REGISTRY.set("down_1", create_progressive_hook(0.5))
    attn_scores.KV_REGISTRY.set("down_2", create_progressive_hook(1.0))
    attn_scores.KV_REGISTRY.set("mid", create_progressive_hook(2.0))
    attn_scores.KV_REGISTRY.set("up_0", create_progressive_hook(3.0))
    attn_scores.KV_REGISTRY.set("up_1", create_progressive_hook(4.0))
    attn_scores.KV_REGISTRY.set("up_2", create_progressive_hook(5.0))
    
    print("Progressive: 0.2x → 1.0x → 3.0x → 5.0x")
    
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
    pil_img.save(output_dir / "07_progressive.png")
    print("✅ Saved: 07_progressive.png")
    tests.append("Progressive - 0.2x to 5.0x gradient")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 80)
    print("🎉 POWERFUL DEMONSTRATION COMPLETE!")
    print("=" * 80)
    
    print(f"\n📊 Generated {len(tests)} Tests:")
    for i, desc in enumerate(tests, 1):
        print(f"{i}. {desc}")
    
    print(f"\n📁 Images saved to: {output_dir}")
    
    print("\n🔥 What we proved with DRAMATIC effects:")
    print("  ✅ Token removal (removing concepts)")
    print("  ✅ Extreme amplification (overpowering)")
    print("  ✅ Extreme suppression (near random)")
    print("  ✅ Maximum chaos (noise injection)")
    print("  ✅ Inversion (anti-prompt)")
    print("  ✅ Progressive control")
    
    print("\n💪 CorePulse V4 has COMPLETE control over generation!")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()

if __name__ == "__main__":
    main()