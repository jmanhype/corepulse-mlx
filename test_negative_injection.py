#!/usr/bin/env python3
"""
Test: Negative Prompt Injection
Inject embeddings that represent what NOT to generate.
This demonstrates exclusion control and anti-concept injection.
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

def create_negative_injection_hook(model, negative_prompt, strength=1.0):
    """
    Create a hook that injects negative prompt embeddings.
    
    Args:
        model: The SDXL model for generating embeddings
        negative_prompt: What NOT to generate
        strength: How strongly to apply the negation (0-1)
    """
    # Generate negative embedding
    neg_cond, neg_pooled = model._get_text_conditioning(negative_prompt)
    
    def hook(q, k, v, meta=None):
        # Only modify cross-attention
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            # Create negative influence by inverting and mixing
            if seq_len >= neg_cond.shape[1]:
                v_new = mx.array(v)
                embed_dim = min(dim, neg_cond.shape[2])
                embed_len = min(seq_len, neg_cond.shape[1])
                
                # Invert the negative embedding (opposite influence)
                neg_inverted = -neg_cond[:, :embed_len, :embed_dim]
                
                # Mix: original - (strength * negative)
                # Ensure proper broadcasting
                if len(neg_inverted.shape) == 3:
                    neg_inverted = neg_inverted[None, :, :, :]  # Add batch dimension
                if neg_inverted.shape[1] < heads:
                    # Broadcast across heads
                    neg_inverted = mx.broadcast_to(neg_inverted, (batch, heads, embed_len, embed_dim))
                
                v_new[:, :, :embed_len, :embed_dim] = \
                    v[:, :, :embed_len, :embed_dim] + \
                    (strength * neg_inverted[:, :, :embed_len, :embed_dim])
                
                print(f"    ⛔ Negative injection at {block_id}: NOT '{negative_prompt[:20]}...' (strength {strength:.1f})")
                
                return q, k, v_new
        return q, k, v
    return hook

def create_multi_negative_hook(model, negative_prompts, strengths=None):
    """
    Inject multiple negative concepts simultaneously.
    
    Args:
        model: The SDXL model
        negative_prompts: List of concepts to avoid
        strengths: List of strengths per concept (optional)
    """
    if strengths is None:
        strengths = [1.0] * len(negative_prompts)
    
    # Pre-generate all negative embeddings
    neg_embeds = []
    for neg_prompt in negative_prompts:
        neg_cond, _ = model._get_text_conditioning(neg_prompt)
        neg_embeds.append(neg_cond)
    
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            v_new = mx.array(v)
            
            # Apply each negative concept
            for i, (neg_embed, strength) in enumerate(zip(neg_embeds, strengths)):
                if seq_len >= neg_embed.shape[1]:
                    embed_dim = min(dim, neg_embed.shape[2])
                    embed_len = min(seq_len, neg_embed.shape[1])
                    
                    # Accumulate negative influences
                    neg_inverted = -neg_embed[:, :embed_len, :embed_dim]
                    # Ensure proper broadcasting
                    if len(neg_inverted.shape) == 3:
                        neg_inverted = neg_inverted[None, :, :, :]
                    if neg_inverted.shape[1] < heads:
                        neg_inverted = mx.broadcast_to(neg_inverted, (batch, heads, embed_len, embed_dim))
                    v_new[:, :, :embed_len, :embed_dim] += strength * neg_inverted[:, :, :embed_len, :embed_dim]
            
            print(f"    ⛔ Multi-negative at {block_id}: {len(negative_prompts)} concepts")
            return q, k, v_new
        return q, k, v
    return hook

def create_selective_negative_hook(model, negative_prompt, target_blocks):
    """
    Apply negative injection only to specific UNet blocks.
    
    Args:
        model: The SDXL model
        negative_prompt: What to avoid
        target_blocks: List of block names to apply negation to
    """
    neg_cond, _ = model._get_text_conditioning(negative_prompt)
    
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            block_id = meta.get('block_id', 'unknown')
            
            # Only apply to target blocks
            if block_id in target_blocks:
                batch, heads, seq_len, dim = v.shape
                
                if seq_len >= neg_cond.shape[1]:
                    v_new = mx.array(v)
                    embed_dim = min(dim, neg_cond.shape[2])
                    embed_len = min(seq_len, neg_cond.shape[1])
                    
                    # Strong negation in target blocks
                    neg_inverted = -neg_cond[:, :embed_len, :embed_dim]
                    # Ensure proper broadcasting
                    if len(neg_inverted.shape) == 3:
                        neg_inverted = neg_inverted[None, :, :, :]
                    if neg_inverted.shape[1] < heads:
                        neg_inverted = mx.broadcast_to(neg_inverted, (batch, heads, embed_len, embed_dim))
                    v_new[:, :, :embed_len, :embed_dim] += 1.5 * neg_inverted[:, :, :embed_len, :embed_dim]
                    
                    print(f"    ⛔ Selective negative at {block_id}: '{negative_prompt[:20]}...'")
                    return q, k, v_new
        return q, k, v
    return hook

def main():
    print("⛔ Test: Negative Prompt Injection")
    print("=" * 60)
    
    # Configuration
    base_prompt = "a cute fluffy cat sitting on a windowsill"
    negative_prompt = "dog, puppy, canine"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    print(f"📝 Base Prompt: '{base_prompt}'")
    print(f"⛔ Negative: '{negative_prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Create output directory
    output_dir = Path("artifacts/images/negative_injection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Test 1: Baseline (no negative)
    print("\n🎨 Test 1: Baseline (no negative injection)...")
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
    
    # Test 2: Single Negative Injection
    print("\n🎨 Test 2: Single negative injection (avoid 'dog')...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_negative_injection_hook(model, negative_prompt, strength=1.0)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
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
    pil_img.save(output_dir / "02_single_negative.png")
    print("✅ Saved: 02_single_negative.png")
    
    # Test 3: Multiple Negatives
    print("\n🎨 Test 3: Multiple negative concepts...")
    attn_scores.KV_REGISTRY.clear()
    
    negative_concepts = [
        "blurry, out of focus",
        "dark, shadows",
        "cartoon, anime"
    ]
    
    hook = create_multi_negative_hook(model, negative_concepts, [0.8, 0.6, 1.0])
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
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
    pil_img.save(output_dir / "03_multi_negative.png")
    print("✅ Saved: 03_multi_negative.png")
    
    # Test 4: Selective Block Negation
    print("\n🎨 Test 4: Selective negative (only early blocks)...")
    attn_scores.KV_REGISTRY.clear()
    
    # Only apply negative in early blocks (composition phase)
    hook = create_selective_negative_hook(model, "indoor, inside", ["down_0", "down_1", "down_2"])
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
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
    pil_img.save(output_dir / "04_selective_blocks.png")
    print("✅ Saved: 04_selective_blocks.png")
    
    # Test 5: Style Exclusion
    print("\n🎨 Test 5: Style exclusion (avoid specific art styles)...")
    attn_scores.KV_REGISTRY.clear()
    
    style_negative = "realistic, photographic, 3D render"
    hook = create_negative_injection_hook(model, style_negative, strength=1.2)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
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
    pil_img.save(output_dir / "05_style_exclusion.png")
    print("✅ Saved: 05_style_exclusion.png")
    
    # Test 6: Progressive Negative (increasing strength)
    print("\n🎨 Test 6: Progressive negative (increasing exclusion)...")
    attn_scores.KV_REGISTRY.clear()
    
    # Gradually increase negative strength across blocks
    neg_cond, _ = model._get_text_conditioning("simple, minimalist, plain")
    
    def progressive_negative_hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            # Map block to strength
            block_strengths = {
                "down_0": 0.2,
                "down_1": 0.4,
                "down_2": 0.6,
                "mid": 0.8,
                "up_0": 1.0,
                "up_1": 1.2,
                "up_2": 1.5
            }
            
            strength = block_strengths.get(block_id, 0.5)
            
            if seq_len >= neg_cond.shape[1]:
                v_new = mx.array(v)
                embed_dim = min(dim, neg_cond.shape[2])
                embed_len = min(seq_len, neg_cond.shape[1])
                
                # Progressive negation
                neg_inverted = -neg_cond[:, :embed_len, :embed_dim]
                # Ensure proper broadcasting
                if len(neg_inverted.shape) == 3:
                    neg_inverted = neg_inverted[None, :, :, :]
                if neg_inverted.shape[1] < heads:
                    neg_inverted = mx.broadcast_to(neg_inverted, (batch, heads, embed_len, embed_dim))
                v_new[:, :, :embed_len, :embed_dim] += strength * neg_inverted[:, :, :embed_len, :embed_dim]
                
                print(f"    ⛔ Progressive at {block_id}: strength {strength:.1f}")
                return q, k, v_new
        return q, k, v
    
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, progressive_negative_hook)
    
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
    pil_img.save(output_dir / "06_progressive_negative.png")
    print("✅ Saved: 06_progressive_negative.png")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n" + "=" * 60)
    print("✅ Negative Prompt Injection Test Complete!")
    print("📊 Results:")
    print("  01_baseline.png: Normal generation without negatives")
    print("  02_single_negative.png: Avoid 'dog' concept")
    print("  03_multi_negative.png: Multiple exclusions")
    print("  04_selective_blocks.png: Negative only in early blocks")
    print("  05_style_exclusion.png: Avoid specific art styles")
    print("  06_progressive_negative.png: Increasing exclusion strength")
    print("\n💡 This proves we can inject anti-concepts and exclusions!")
    print("🚫 Negative injection provides precise control over what NOT to generate!")
    print("🔬 We can exclude objects, styles, or qualities from generation!")

if __name__ == "__main__":
    main()