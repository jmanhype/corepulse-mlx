#!/usr/bin/env python3
"""
Test: Dual Technique (KV + Embedding Combined)
Simultaneously use both KV manipulation AND embedding injection.
This demonstrates the power of combining both techniques for maximum control.
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

def create_kv_manipulation_hook(factor=2.0):
    """Standard KV manipulation hook."""
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            print(f"    🔧 KV Manipulation: {factor}x at {meta.get('block_id', 'unknown')}")
            return q, k * factor, v * factor
        return q, k, v
    return hook

def create_embedding_injection_hook(model, target_prompt):
    """Create embedding and inject it."""
    # Generate target embedding
    target_conditioning = model._get_text_conditioning(target_prompt)
    target_cond, target_pooled = target_conditioning
    
    def hook(q, k, v, meta=None):
        # Only modify cross-attention
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            
            # Inject embedding into V
            if seq_len >= target_cond.shape[1]:
                v_new = mx.array(v)
                # Blend original V with target embedding
                embed_dim = min(dim, target_cond.shape[2])
                embed_len = min(seq_len, target_cond.shape[1])
                
                # 50% original, 50% injected
                v_new[:, :, :embed_len, :embed_dim] = \
                    0.5 * v[:, :, :embed_len, :embed_dim] + \
                    0.5 * target_cond[:, :embed_len, :embed_dim]
                
                print(f"    💉 Embedding Injection: '{target_prompt[:20]}...' at {meta.get('block_id', 'unknown')}")
                return q, k, v_new
        return q, k, v
    return hook

def create_dual_technique_hook(model, kv_factor, inject_prompt, blend_ratio=0.5):
    """
    Combine both techniques in a single hook.
    
    Args:
        model: The SDXL model for generating embeddings
        kv_factor: Factor for KV manipulation
        inject_prompt: Prompt to inject
        blend_ratio: How much injection vs KV manipulation (0=pure KV, 1=pure injection)
    """
    # Pre-generate embedding
    inject_cond, inject_pooled = model._get_text_conditioning(inject_prompt)
    
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            # Step 1: KV Manipulation
            k_manip = k * kv_factor
            v_manip = v * kv_factor
            
            # Step 2: Embedding Injection
            v_inject = mx.array(v)
            if seq_len >= inject_cond.shape[1]:
                embed_dim = min(dim, inject_cond.shape[2])
                embed_len = min(seq_len, inject_cond.shape[1])
                
                v_inject[:, :, :embed_len, :embed_dim] = \
                    inject_cond[:, :embed_len, :embed_dim]
            
            # Step 3: Blend both techniques
            k_final = k_manip  # KV manipulation only affects K directly
            v_final = (1 - blend_ratio) * v_manip + blend_ratio * v_inject
            
            print(f"    🔀 Dual Tech at {block_id}: KV×{kv_factor}, Inject '{inject_prompt[:15]}...', Blend {blend_ratio:.1%}")
            
            return q, k_final, v_final
        return q, k, v
    return hook

def main():
    print("🔀 Test: Dual Technique (KV + Embedding Combined)")
    print("=" * 60)
    
    # Configuration
    base_prompt = "a serene mountain lake at dawn"
    inject_prompt = "cyberpunk neon city"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    print(f"📝 Base Prompt: '{base_prompt}'")
    print(f"💉 Inject Prompt: '{inject_prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Create output directory
    output_dir = Path("artifacts/images/dual_technique")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Test 1: Baseline
    print("\n🎨 Test 1: Baseline (base prompt only)...")
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
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "01_baseline.png")
    print("✅ Saved: 01_baseline.png")
    
    # Test 2: KV Manipulation Only
    print("\n🎨 Test 2: KV manipulation only (amplify 2x)...")
    attn_scores.KV_REGISTRY.clear()
    
    kv_hook = create_kv_manipulation_hook(2.0)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, kv_hook)
    
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
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "02_kv_only.png")
    print("✅ Saved: 02_kv_only.png")
    
    # Test 3: Embedding Injection Only
    print("\n🎨 Test 3: Embedding injection only...")
    attn_scores.KV_REGISTRY.clear()
    
    inject_hook = create_embedding_injection_hook(model, inject_prompt)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, inject_hook)
    
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
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "03_embedding_only.png")
    print("✅ Saved: 03_embedding_only.png")
    
    # Test 4: Dual Technique - Balanced (50/50)
    print("\n🎨 Test 4: Dual technique - balanced blend...")
    attn_scores.KV_REGISTRY.clear()
    
    dual_hook = create_dual_technique_hook(model, 2.0, inject_prompt, 0.5)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, dual_hook)
    
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
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "04_dual_balanced.png")
    print("✅ Saved: 04_dual_balanced.png")
    
    # Test 5: Dual Technique - Progressive
    print("\n🎨 Test 5: Dual technique - progressive (different per block)...")
    attn_scores.KV_REGISTRY.clear()
    
    # Early blocks: More KV manipulation
    # Late blocks: More embedding injection
    block_configs = {
        "down_0": (3.0, 0.2),  # Strong KV, weak injection
        "down_1": (2.5, 0.3),
        "down_2": (2.0, 0.4),
        "mid": (1.5, 0.5),     # Balanced
        "up_0": (1.2, 0.6),
        "up_1": (1.0, 0.7),
        "up_2": (0.8, 0.8),    # Weak KV, strong injection
    }
    
    for block, (kv_factor, blend) in block_configs.items():
        hook = create_dual_technique_hook(model, kv_factor, inject_prompt, blend)
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
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "05_dual_progressive.png")
    print("✅ Saved: 05_dual_progressive.png")
    
    # Test 6: Dual Technique - Inverse (suppress KV, inject different)
    print("\n🎨 Test 6: Dual technique - inverse (suppress + inject)...")
    attn_scores.KV_REGISTRY.clear()
    
    # Suppress original prompt, inject completely different one
    inverse_hook = create_dual_technique_hook(model, 0.1, "abstract geometric patterns", 0.9)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, inverse_hook)
    
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
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "06_dual_inverse.png")
    print("✅ Saved: 06_dual_inverse.png")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n" + "=" * 60)
    print("✅ Dual Technique Test Complete!")
    print("📊 Results:")
    print("  01_baseline.png: Base prompt only")
    print("  02_kv_only.png: KV manipulation only (2x)")
    print("  03_embedding_only.png: Embedding injection only")
    print("  04_dual_balanced.png: 50/50 blend of both techniques")
    print("  05_dual_progressive.png: Progressive blend across blocks")
    print("  06_dual_inverse.png: Suppress original, inject different")
    print("\n💡 This proves we can combine both techniques simultaneously!")
    print("🚀 Dual technique offers unprecedented control over generation!")
    print("🔬 KV manipulation + Embedding injection = Maximum flexibility!")

if __name__ == "__main__":
    main()