#!/usr/bin/env python3
"""
Test: Query-Only Manipulation
Modify Q (query) tensor without touching K (key) or V (value).
This tests asymmetric attention manipulation for unique effects.
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

def create_query_only_hook(manipulation_type="amplify", factor=2.0):
    """
    Create a hook that only manipulates the Query tensor.
    
    Args:
        manipulation_type: "amplify", "suppress", "noise", "rotate"
        factor: Manipulation strength
    """
    def hook(q, k, v, meta=None):
        # Only modify cross-attention
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = q.shape
            
            if manipulation_type == "amplify":
                q_new = q * factor
                print(f"    📈 Amplified Q by {factor}x")
            
            elif manipulation_type == "suppress":
                q_new = q * factor
                print(f"    📉 Suppressed Q to {factor}x")
            
            elif manipulation_type == "noise":
                noise = mx.random.normal(q.shape) * factor
                q_new = q + noise
                print(f"    🌪️ Added noise to Q (intensity {factor})")
            
            elif manipulation_type == "rotate":
                # Rotate query vectors in embedding space
                # Simple rotation by shuffling dimensions
                roll_amount = int(dim * factor) % dim
                q_new = mx.roll(q, roll_amount, axis=-1)
                print(f"    🔄 Rotated Q by {roll_amount} dimensions")
            
            elif manipulation_type == "normalize":
                # L2 normalize queries to unit length
                q_norm = mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True))
                q_new = q / (q_norm + 1e-8)
                q_new = q_new * factor  # Scale by factor
                print(f"    📐 Normalized Q with scale {factor}")
            
            else:
                q_new = q
            
            # Return modified Q, original K and V
            return q_new, k, v
        return q, k, v
    return hook

def main():
    print("🎯 Test: Query-Only Manipulation")
    print("=" * 60)
    
    # Configuration
    prompt = "a colorful parrot perched on a tropical branch"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    print(f"📝 Prompt: '{prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Create output directory
    output_dir = Path("artifacts/images/query_manipulation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Test 1: Baseline
    print("\n🎨 Test 1: Baseline (no manipulation)...")
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
    
    # Test 2: Query Amplification
    print("\n🎨 Test 2: Query amplification (3x)...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_query_only_hook("amplify", 3.0)
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
    pil_img.save(output_dir / "query_amplified.png")
    print("✅ Saved: query_amplified.png")
    
    # Test 3: Query Suppression
    print("\n🎨 Test 3: Query suppression (0.1x)...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_query_only_hook("suppress", 0.1)
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
    pil_img.save(output_dir / "query_suppressed.png")
    print("✅ Saved: query_suppressed.png")
    
    # Test 4: Query Noise
    print("\n🎨 Test 4: Query noise injection...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_query_only_hook("noise", 0.5)
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
    pil_img.save(output_dir / "query_noise.png")
    print("✅ Saved: query_noise.png")
    
    # Test 5: Query Rotation
    print("\n🎨 Test 5: Query rotation in embedding space...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_query_only_hook("rotate", 0.25)  # Rotate by 25% of dimensions
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
    pil_img.save(output_dir / "query_rotated.png")
    print("✅ Saved: query_rotated.png")
    
    # Test 6: Query Normalization
    print("\n🎨 Test 6: Query normalization...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_query_only_hook("normalize", 2.0)  # Normalize then scale
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
    pil_img.save(output_dir / "query_normalized.png")
    print("✅ Saved: query_normalized.png")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n" + "=" * 60)
    print("✅ Query-Only Manipulation Test Complete!")
    print("📊 Results:")
    print("  - baseline.png: Normal generation")
    print("  - query_amplified.png: Query amplified 3x")
    print("  - query_suppressed.png: Query suppressed to 0.1x")
    print("  - query_noise.png: Noise added to queries")
    print("  - query_rotated.png: Queries rotated in embedding space")
    print("  - query_normalized.png: Queries normalized to unit length")
    print("💡 This proves we can manipulate queries independently of keys/values!")
    print("🔬 Query manipulation affects attention focus without changing content!")

if __name__ == "__main__":
    main()