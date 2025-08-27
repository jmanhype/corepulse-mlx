#!/usr/bin/env python3
"""
Test: Masked Injection Comparison
Creates side-by-side comparison showing masked vs unmasked prompt injection.
Similar to CorePulse's masked_injection_comparison.png
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

def create_masked_injection_hook(model, inject_prompt, mask_pattern="alternating"):
    """Create masked injection with different patterns."""
    inject_cond, _ = model._get_text_conditioning(inject_prompt)
    
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            if seq_len >= inject_cond.shape[1]:
                v_new = mx.array(v)
                embed_dim = min(dim, inject_cond.shape[2])
                embed_len = min(seq_len, inject_cond.shape[1])
                
                # Prepare injection
                inject = inject_cond[:, :embed_len, :embed_dim]
                if len(inject.shape) == 3:
                    inject = inject[None, :, :, :]
                if inject.shape[0] < batch:
                    inject = mx.tile(inject, (batch, 1, 1, 1))
                if inject.shape[1] < heads:
                    # Expand to heads dimension properly
                    inject = mx.repeat(inject[:batch, None, :, :], heads, axis=1)
                    inject = inject.reshape((batch, heads, embed_len, embed_dim))
                
                # Create mask based on pattern
                mask = mx.zeros((embed_len,))
                
                if mask_pattern == "alternating":
                    # Alternating tokens
                    for i in range(0, embed_len, 2):
                        mask = mask.at[i].set(1.0)
                elif mask_pattern == "first_half":
                    # First half only
                    for i in range(embed_len // 2):
                        mask = mask.at[i].set(1.0)
                elif mask_pattern == "sparse":
                    # Sparse random-like pattern
                    for i in range(0, embed_len, 3):
                        mask = mask.at[i].set(1.0)
                elif mask_pattern == "dense":
                    # Dense (most tokens)
                    mask = mx.ones((embed_len,))
                    for i in range(2, embed_len, 5):
                        mask = mask.at[i].set(0.0)
                
                # Expand mask to match dimensions
                mask = mask[None, None, :, None]  # Shape: (1, 1, embed_len, 1)
                
                # Apply masked injection
                v_new[:, :, :embed_len, :embed_dim] = \
                    v[:, :, :embed_len, :embed_dim] * (1 - mask * 0.7) + \
                    inject * mask * 0.7
                
                masked_count = mx.sum(mask)
                print(f"    🎭 Masked injection at {block_id}: {mask_pattern}, {masked_count:.0f}/{embed_len} tokens")
                
                return q, k, v_new
        return q, k, v
    return hook

def main():
    print("🎭 Test: Masked Injection Comparison")
    print("=" * 60)
    
    # Configuration
    base_prompt = "a serene mountain landscape"
    inject_prompt = "cyberpunk neon city"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    print(f"📝 Base Prompt: '{base_prompt}'")
    print(f"💉 Inject Prompt: '{inject_prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Create output directory
    output_dir = Path("artifacts/images/masked_injection_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    comparison_images = []
    
    # Test 1: Baseline (No Injection)
    print("\n🎨 Test 1/4: Baseline (no injection)...")
    attn_scores.KV_REGISTRY.clear()
    
    latents = model.generate_latents(
        base_prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    baseline_img = PIL.Image.fromarray(np.array(img_array))
    baseline_img.save(output_dir / "01_baseline.png")
    comparison_images.append(baseline_img)
    print("✅ Saved: 01_baseline.png")
    
    # Test 2: Full Injection (No Mask)
    print("\n🎨 Test 2/4: Full injection (unmasked)...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_masked_injection_hook(model, inject_prompt, mask_pattern="dense")
    for block in ['mid', 'up_0', 'up_1']:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        base_prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    full_img = PIL.Image.fromarray(np.array(img_array))
    full_img.save(output_dir / "02_full_injection.png")
    comparison_images.append(full_img)
    print("✅ Saved: 02_full_injection.png")
    
    # Test 3: Alternating Mask
    print("\n🎨 Test 3/4: Alternating mask injection...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_masked_injection_hook(model, inject_prompt, mask_pattern="alternating")
    for block in ['mid', 'up_0', 'up_1']:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        base_prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    alternating_img = PIL.Image.fromarray(np.array(img_array))
    alternating_img.save(output_dir / "03_alternating_mask.png")
    comparison_images.append(alternating_img)
    print("✅ Saved: 03_alternating_mask.png")
    
    # Test 4: Sparse Mask
    print("\n🎨 Test 4/4: Sparse mask injection...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_masked_injection_hook(model, inject_prompt, mask_pattern="sparse")
    for block in ['mid', 'up_0', 'up_1']:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        base_prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    sparse_img = PIL.Image.fromarray(np.array(img_array))
    sparse_img.save(output_dir / "04_sparse_mask.png")
    comparison_images.append(sparse_img)
    print("✅ Saved: 04_sparse_mask.png")
    
    # Create comparison grid
    print("\n🔄 Creating comparison grid...")
    width = baseline_img.width
    height = baseline_img.height
    
    grid = PIL.Image.new('RGB', (width * 2, height * 2))
    grid.paste(comparison_images[0], (0, 0))
    grid.paste(comparison_images[1], (width, 0))
    grid.paste(comparison_images[2], (0, height))
    grid.paste(comparison_images[3], (width, height))
    
    grid.save(output_dir / "masked_injection_comparison.png")
    print("✅ Saved: masked_injection_comparison.png")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n" + "=" * 60)
    print("✅ Masked Injection Comparison Complete!")
    print("📊 Results:")
    print("  01_baseline.png: No injection")
    print("  02_full_injection.png: Dense/full injection")
    print("  03_alternating_mask.png: Alternating token injection")
    print("  04_sparse_mask.png: Sparse token injection")
    print("  masked_injection_comparison.png: 2x2 comparison grid")
    print("\n💡 This demonstrates masked vs unmasked injection!")
    print("🎭 Shows how masking patterns affect blending!")

if __name__ == "__main__":
    main()