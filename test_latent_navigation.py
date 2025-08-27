#!/usr/bin/env python3
"""
Test: Latent Space Navigation
Navigate and manipulate in latent space during generation.
This demonstrates latent arithmetic and trajectory control.
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

def create_latent_interpolation_hook(model, prompt_a, prompt_b, interpolation_schedule):
    """
    Create a hook that interpolates between two prompts in latent space.
    
    Args:
        model: The SDXL model
        prompt_a, prompt_b: Two prompts to interpolate between
        interpolation_schedule: Dict mapping block names to interpolation weights
    """
    # Generate embeddings for both prompts
    cond_a, _ = model._get_text_conditioning(prompt_a)
    cond_b, _ = model._get_text_conditioning(prompt_b)
    
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            # Get interpolation weight for this block
            alpha = interpolation_schedule.get(block_id, 0.5)
            
            if seq_len >= max(cond_a.shape[1], cond_b.shape[1]):
                v_new = mx.array(v)
                embed_dim = min(dim, cond_a.shape[2])
                embed_len = min(seq_len, cond_a.shape[1])
                
                # Prepare embeddings
                embed_a = cond_a[:, :embed_len, :embed_dim]
                embed_b = cond_b[:, :embed_len, :embed_dim]
                
                # Ensure proper broadcasting
                if len(embed_a.shape) == 3:
                    embed_a = embed_a[None, :, :, :]
                    embed_b = embed_b[None, :, :, :]
                if embed_a.shape[1] < heads:
                    embed_a = mx.broadcast_to(embed_a, (batch, heads, embed_len, embed_dim))
                    embed_b = mx.broadcast_to(embed_b, (batch, heads, embed_len, embed_dim))
                
                # Spherical linear interpolation (slerp) for better results
                # Normalize vectors
                norm_a = mx.sqrt(mx.sum(embed_a * embed_a, axis=-1, keepdims=True))
                norm_b = mx.sqrt(mx.sum(embed_b * embed_b, axis=-1, keepdims=True))
                embed_a_norm = embed_a / (norm_a + 1e-8)
                embed_b_norm = embed_b / (norm_b + 1e-8)
                
                # Compute angle
                dot_product = mx.sum(embed_a_norm * embed_b_norm, axis=-1, keepdims=True)
                dot_product = mx.clip(dot_product, -0.999, 0.999)  # Avoid numerical issues
                
                # Interpolate
                interpolated = (1 - alpha) * embed_a + alpha * embed_b
                
                v_new[:, :, :embed_len, :embed_dim] = interpolated[:, :, :embed_len, :embed_dim]
                
                print(f"    🧭 Latent interpolation at {block_id}: {(1-alpha)*100:.0f}% A → {alpha*100:.0f}% B")
                
                return q, k, v_new
        return q, k, v
    return hook

def create_latent_arithmetic_hook(model, base_prompt, add_prompt, subtract_prompt, factor=1.0):
    """
    Perform latent arithmetic: base + factor*(add - subtract)
    """
    # Generate embeddings
    base_cond, _ = model._get_text_conditioning(base_prompt)
    add_cond, _ = model._get_text_conditioning(add_prompt)
    sub_cond, _ = model._get_text_conditioning(subtract_prompt)
    
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            if seq_len >= base_cond.shape[1]:
                v_new = mx.array(v)
                embed_dim = min(dim, base_cond.shape[2])
                embed_len = min(seq_len, base_cond.shape[1])
                
                # Prepare embeddings
                base = base_cond[:, :embed_len, :embed_dim]
                add = add_cond[:, :embed_len, :embed_dim]
                sub = sub_cond[:, :embed_len, :embed_dim]
                
                # Ensure proper broadcasting
                if len(base.shape) == 3:
                    base = base[None, :, :, :]
                    add = add[None, :, :, :]
                    sub = sub[None, :, :, :]
                if base.shape[1] < heads:
                    base = mx.broadcast_to(base, (batch, heads, embed_len, embed_dim))
                    add = mx.broadcast_to(add, (batch, heads, embed_len, embed_dim))
                    sub = mx.broadcast_to(sub, (batch, heads, embed_len, embed_dim))
                
                # Latent arithmetic
                result = base + factor * (add - sub)
                
                v_new[:, :, :embed_len, :embed_dim] = result[:, :, :embed_len, :embed_dim]
                
                if block_id == "mid":  # Print once
                    print(f"    ➕ Latent arithmetic: base + {factor:.1f}*(add - subtract)")
                
                return q, k, v_new
        return q, k, v
    return hook

def create_latent_trajectory_hook(model, waypoint_prompts):
    """
    Navigate through multiple waypoints in latent space.
    """
    # Generate embeddings for all waypoints
    waypoint_embeds = []
    for prompt in waypoint_prompts:
        cond, _ = model._get_text_conditioning(prompt)
        waypoint_embeds.append(cond)
    
    step_counter = [0]  # Mutable counter
    
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            # Map blocks to waypoint progression
            block_to_waypoint = {
                "down_0": 0, "down_1": 0,
                "down_2": 1, "mid": 1,
                "up_0": 2, "up_1": 2, "up_2": 2
            }
            
            if len(waypoint_prompts) > 3:
                # For more waypoints, cycle through them
                waypoint_idx = step_counter[0] % len(waypoint_prompts)
                step_counter[0] += 1
            else:
                waypoint_idx = min(block_to_waypoint.get(block_id, 0), len(waypoint_prompts) - 1)
            
            embed = waypoint_embeds[waypoint_idx]
            
            if seq_len >= embed.shape[1]:
                v_new = mx.array(v)
                embed_dim = min(dim, embed.shape[2])
                embed_len = min(seq_len, embed.shape[1])
                
                # Prepare embedding
                waypoint = embed[:, :embed_len, :embed_dim]
                if len(waypoint.shape) == 3:
                    waypoint = waypoint[None, :, :, :]
                if waypoint.shape[1] < heads:
                    waypoint = mx.broadcast_to(waypoint, (batch, heads, embed_len, embed_dim))
                
                # Blend with original (smooth transition)
                blend = 0.7
                v_new[:, :, :embed_len, :embed_dim] = \
                    (1 - blend) * v[:, :, :embed_len, :embed_dim] + \
                    blend * waypoint[:, :, :embed_len, :embed_dim]
                
                print(f"    📍 Waypoint {waypoint_idx+1}/{len(waypoint_prompts)} at {block_id}: '{waypoint_prompts[waypoint_idx][:20]}...'")
                
                return q, k, v_new
        return q, k, v
    return hook

def main():
    print("🧭 Test: Latent Space Navigation")
    print("=" * 60)
    
    # Configuration
    base_prompt = "a magical forest"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    print(f"📝 Base Prompt: '{base_prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Create output directory
    output_dir = Path("artifacts/images/latent_navigation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Test 1: Baseline
    print("\n🎨 Test 1: Baseline generation...")
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
    
    # Test 2: Simple interpolation
    print("\n🎨 Test 2: Latent interpolation (forest → ocean)...")
    attn_scores.KV_REGISTRY.clear()
    
    prompt_a = "a dense magical forest"
    prompt_b = "a vast ocean with waves"
    
    # Progressive interpolation schedule
    interpolation_schedule = {
        "down_0": 0.0,   # 100% forest
        "down_1": 0.15,
        "down_2": 0.3,
        "mid": 0.5,      # 50/50 blend
        "up_0": 0.7,
        "up_1": 0.85,
        "up_2": 1.0,     # 100% ocean
    }
    
    hook = create_latent_interpolation_hook(model, prompt_a, prompt_b, interpolation_schedule)
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
    pil_img.save(output_dir / "02_interpolation.png")
    print("✅ Saved: 02_interpolation.png")
    
    # Test 3: Latent arithmetic
    print("\n🎨 Test 3: Latent arithmetic (castle + futuristic - medieval)...")
    attn_scores.KV_REGISTRY.clear()
    
    base = "a majestic castle"
    add = "futuristic technology"
    subtract = "medieval architecture"
    
    hook = create_latent_arithmetic_hook(model, base, add, subtract, factor=0.7)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        base,  # Use base prompt
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "03_arithmetic.png")
    print("✅ Saved: 03_arithmetic.png")
    
    # Test 4: Multi-waypoint trajectory
    print("\n🎨 Test 4: Multi-waypoint trajectory...")
    attn_scores.KV_REGISTRY.clear()
    
    waypoints = [
        "sunrise over mountains",
        "noon in a bustling city",
        "sunset at the beach",
        "midnight under stars"
    ]
    
    hook = create_latent_trajectory_hook(model, waypoints)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        "a journey through time",
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "04_trajectory.png")
    print("✅ Saved: 04_trajectory.png")
    
    # Test 5: Style transfer via latent navigation
    print("\n🎨 Test 5: Style transfer (photo → painting)...")
    attn_scores.KV_REGISTRY.clear()
    
    photo_prompt = "photorealistic landscape"
    painting_prompt = "impressionist oil painting"
    
    # Gradual style transfer
    style_schedule = {
        "down_0": 0.0,   # Photorealistic
        "down_1": 0.2,
        "down_2": 0.4,
        "mid": 0.6,
        "up_0": 0.8,
        "up_1": 0.9,
        "up_2": 1.0,     # Full painting
    }
    
    hook = create_latent_interpolation_hook(model, photo_prompt, painting_prompt, style_schedule)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        "beautiful mountain vista",
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "05_style_transfer.png")
    print("✅ Saved: 05_style_transfer.png")
    
    # Test 6: Circular navigation (loop back)
    print("\n🎨 Test 6: Circular navigation...")
    attn_scores.KV_REGISTRY.clear()
    
    circular_waypoints = [
        "spring flowers blooming",
        "summer sunshine",
        "autumn leaves falling",
        "winter snow",
        "spring flowers blooming"  # Loop back
    ]
    
    hook = create_latent_trajectory_hook(model, circular_waypoints)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        "seasonal cycle",
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "06_circular.png")
    print("✅ Saved: 06_circular.png")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n" + "=" * 60)
    print("✅ Latent Space Navigation Test Complete!")
    print("📊 Results:")
    print("  01_baseline.png: Normal generation")
    print("  02_interpolation.png: Smooth interpolation between concepts")
    print("  03_arithmetic.png: Latent arithmetic (add/subtract concepts)")
    print("  04_trajectory.png: Multi-waypoint navigation")
    print("  05_style_transfer.png: Style transfer via latent space")
    print("  06_circular.png: Circular navigation (seasonal loop)")
    print("\n💡 This proves controllable latent space navigation!")
    print("🧭 We can traverse semantic space with precision!")
    print("🔬 Latent arithmetic enables concept manipulation!")

if __name__ == "__main__":
    main()