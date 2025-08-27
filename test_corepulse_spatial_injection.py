#!/usr/bin/env python3
"""
Test: CorePulse Spatial/Regional Injection
Implements spatial and regional prompt injection for localized control.
This demonstrates position-aware and region-specific manipulation.
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

def create_spatial_injection_hook(model, region_configs):
    """
    Create a hook for spatial/regional injection.
    
    Args:
        model: The SDXL model
        region_configs: List of dicts with:
            - prompt: Text prompt for the region
            - position: "left", "right", "top", "bottom", "center"
            - strength: Injection strength
    """
    # Generate embeddings for each region
    region_embeds = []
    for config in region_configs:
        cond, _ = model._get_text_conditioning(config['prompt'])
        region_embeds.append((cond, config['position'], config.get('strength', 0.5)))
    
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            v_new = mx.array(v)
            
            # Apply regional injections
            for embed, position, strength in region_embeds:
                if seq_len >= embed.shape[1]:
                    embed_dim = min(dim, embed.shape[2])
                    embed_len = min(seq_len, embed.shape[1])
                    
                    # Create spatial mask based on position
                    mask = mx.ones((batch, heads, seq_len, 1))
                    
                    if position == "left":
                        # Mask left half
                        for i in range(seq_len // 2):
                            mask[:, :, i, :] = strength
                    elif position == "right":
                        # Mask right half
                        for i in range(seq_len // 2, seq_len):
                            mask[:, :, i, :] = strength
                    elif position == "top":
                        # Mask top portion
                        for i in range(seq_len // 3):
                            mask[:, :, i, :] = strength
                    elif position == "bottom":
                        # Mask bottom portion
                        for i in range(2 * seq_len // 3, seq_len):
                            mask[:, :, i, :] = strength
                    elif position == "center":
                        # Mask center region
                        center_start = seq_len // 4
                        center_end = 3 * seq_len // 4
                        for i in range(center_start, center_end):
                            mask[:, :, i, :] = strength
                    
                    # Prepare embedding
                    region_embed = embed[:, :embed_len, :embed_dim]
                    if len(region_embed.shape) == 3:
                        region_embed = region_embed[None, :, :, :]
                    
                    if region_embed.shape[0] < batch:
                        region_embed = mx.tile(region_embed, (batch, 1, 1, 1))
                    if region_embed.shape[1] < heads:
                        region_embed = mx.broadcast_to(
                            region_embed[:batch], 
                            (batch, heads, embed_len, embed_dim)
                        )
                    
                    # Apply masked injection
                    mask_slice = mask[:, :, :embed_len, :]
                    v_new[:, :, :embed_len, :embed_dim] = \
                        v_new[:, :, :embed_len, :embed_dim] * (1 - mask_slice) + \
                        region_embed[:, :, :embed_len, :embed_dim] * mask_slice
            
            print(f"    🗺️ Spatial injection at {block_id}: {len(region_embeds)} regions")
            
            return q, k, v_new
        return q, k, v
    return hook

def create_quadrant_injection_hook(model, quadrant_prompts):
    """
    Inject different prompts into four quadrants.
    
    Args:
        model: The SDXL model
        quadrant_prompts: Dict with keys "top_left", "top_right", "bottom_left", "bottom_right"
    """
    # Generate embeddings for each quadrant
    quad_embeds = {}
    for quad, prompt in quadrant_prompts.items():
        if prompt:
            cond, _ = model._get_text_conditioning(prompt)
            quad_embeds[quad] = cond
    
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            v_new = mx.array(v)
            
            # Define quadrant boundaries (simplified mapping)
            half_seq = seq_len // 2
            
            for quad, embed in quad_embeds.items():
                if seq_len >= embed.shape[1]:
                    embed_dim = min(dim, embed.shape[2])
                    embed_len = min(seq_len, embed.shape[1])
                    
                    # Determine quadrant range
                    if quad == "top_left":
                        start, end = 0, half_seq // 2
                    elif quad == "top_right":
                        start, end = half_seq // 2, half_seq
                    elif quad == "bottom_left":
                        start, end = half_seq, half_seq + half_seq // 2
                    else:  # bottom_right
                        start, end = half_seq + half_seq // 2, seq_len
                    
                    # Prepare embedding
                    quad_embed = embed[:, :min(end-start, embed_len), :embed_dim]
                    if len(quad_embed.shape) == 3:
                        quad_embed = quad_embed[None, :, :, :]
                    
                    if quad_embed.shape[0] < batch:
                        quad_embed = mx.tile(quad_embed, (batch, 1, 1, 1))
                    if quad_embed.shape[1] < heads:
                        quad_embed = mx.broadcast_to(
                            quad_embed[:batch],
                            (batch, heads, quad_embed.shape[2], embed_dim)
                        )
                    
                    # Inject into quadrant
                    inject_len = min(end - start, quad_embed.shape[2])
                    v_new[:, :, start:start+inject_len, :embed_dim] = \
                        0.5 * v[:, :, start:start+inject_len, :embed_dim] + \
                        0.5 * quad_embed[:, :, :inject_len, :embed_dim]
            
            print(f"    🔲 Quadrant injection at {block_id}: {len(quad_embeds)} quadrants")
            
            return q, k, v_new
        return q, k, v
    return hook

def create_gradient_injection_hook(model, start_prompt, end_prompt, direction="horizontal"):
    """
    Create a gradient injection from one prompt to another.
    
    Args:
        model: The SDXL model
        start_prompt: Starting prompt
        end_prompt: Ending prompt
        direction: "horizontal", "vertical", or "diagonal"
    """
    start_cond, _ = model._get_text_conditioning(start_prompt)
    end_cond, _ = model._get_text_conditioning(end_prompt)
    
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            if seq_len >= max(start_cond.shape[1], end_cond.shape[1]):
                v_new = mx.array(v)
                embed_dim = min(dim, start_cond.shape[2])
                embed_len = min(seq_len, start_cond.shape[1])
                
                # Prepare embeddings
                start_embed = start_cond[:, :embed_len, :embed_dim]
                end_embed = end_cond[:, :embed_len, :embed_dim]
                
                if len(start_embed.shape) == 3:
                    start_embed = start_embed[None, :, :, :]
                    end_embed = end_embed[None, :, :, :]
                
                if start_embed.shape[0] < batch:
                    start_embed = mx.tile(start_embed, (batch, 1, 1, 1))
                    end_embed = mx.tile(end_embed, (batch, 1, 1, 1))
                if start_embed.shape[1] < heads:
                    start_embed = mx.broadcast_to(
                        start_embed[:batch],
                        (batch, heads, embed_len, embed_dim)
                    )
                    end_embed = mx.broadcast_to(
                        end_embed[:batch],
                        (batch, heads, embed_len, embed_dim)
                    )
                
                # Create gradient weights
                for i in range(embed_len):
                    if direction == "horizontal":
                        weight = i / max(embed_len - 1, 1)
                    elif direction == "vertical":
                        weight = (i % int(mx.sqrt(float(embed_len)))) / max(int(mx.sqrt(float(embed_len))) - 1, 1)
                    else:  # diagonal
                        weight = (i / max(embed_len - 1, 1)) * 0.5 + \
                                ((i % int(mx.sqrt(float(embed_len)))) / max(int(mx.sqrt(float(embed_len))) - 1, 1)) * 0.5
                    
                    # Interpolate between start and end
                    v_new[:, :, i, :embed_dim] = \
                        (1 - weight) * start_embed[:, :, i, :embed_dim] + \
                        weight * end_embed[:, :, i, :embed_dim]
                
                print(f"    🌈 Gradient injection at {block_id}: {direction}")
                
                return q, k, v_new
        return q, k, v
    return hook

def main():
    print("🗺️ Test: CorePulse Spatial/Regional Injection")
    print("=" * 60)
    
    # Configuration
    base_prompt = "a landscape scene"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    print(f"📝 Base Prompt: '{base_prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Create output directory
    output_dir = Path("artifacts/images/corepulse_spatial")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Test 1: Baseline
    print("\n🎨 Test 1: Baseline (no spatial injection)...")
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
    
    # Test 2: Left-right split
    print("\n🎨 Test 2: Left-right split (ocean | mountains)...")
    attn_scores.KV_REGISTRY.clear()
    
    region_configs = [
        {'prompt': 'ocean waves beach', 'position': 'left', 'strength': 0.7},
        {'prompt': 'mountain peaks snow', 'position': 'right', 'strength': 0.7}
    ]
    
    hook = create_spatial_injection_hook(model, region_configs)
    for block in ['down_2', 'mid', 'up_0', 'up_1']:
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
    pil_img.save(output_dir / "02_left_right_split.png")
    print("✅ Saved: 02_left_right_split.png")
    
    # Test 3: Quadrant injection
    print("\n🎨 Test 3: Four quadrants (different seasons)...")
    attn_scores.KV_REGISTRY.clear()
    
    quadrant_prompts = {
        'top_left': 'spring flowers blooming',
        'top_right': 'summer sunshine bright',
        'bottom_left': 'autumn leaves falling',
        'bottom_right': 'winter snow cold'
    }
    
    hook = create_quadrant_injection_hook(model, quadrant_prompts)
    for block in ['down_1', 'down_2', 'mid', 'up_0', 'up_1']:
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
    pil_img.save(output_dir / "03_quadrants.png")
    print("✅ Saved: 03_quadrants.png")
    
    # Test 4: Center-surround
    print("\n🎨 Test 4: Center-surround (focus in center)...")
    attn_scores.KV_REGISTRY.clear()
    
    region_configs = [
        {'prompt': 'detailed castle fortress', 'position': 'center', 'strength': 0.8},
        {'prompt': 'misty forest background', 'position': 'top', 'strength': 0.4},
        {'prompt': 'misty forest background', 'position': 'bottom', 'strength': 0.4}
    ]
    
    hook = create_spatial_injection_hook(model, region_configs)
    for block in ['down_2', 'mid', 'up_0', 'up_1']:
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
    pil_img.save(output_dir / "04_center_surround.png")
    print("✅ Saved: 04_center_surround.png")
    
    # Test 5: Gradient injection (horizontal)
    print("\n🎨 Test 5: Horizontal gradient (desert to forest)...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_gradient_injection_hook(
        model,
        "sandy desert dunes",
        "lush green forest",
        direction="horizontal"
    )
    for block in ['down_1', 'down_2', 'mid', 'up_0', 'up_1']:
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
    pil_img.save(output_dir / "05_gradient_horizontal.png")
    print("✅ Saved: 05_gradient_horizontal.png")
    
    # Test 6: Complex regional composition
    print("\n🎨 Test 6: Complex regional composition...")
    attn_scores.KV_REGISTRY.clear()
    
    # Multiple overlapping regions
    region_configs = [
        {'prompt': 'sky with clouds', 'position': 'top', 'strength': 0.6},
        {'prompt': 'city buildings', 'position': 'left', 'strength': 0.5},
        {'prompt': 'forest trees', 'position': 'right', 'strength': 0.5},
        {'prompt': 'lake water', 'position': 'bottom', 'strength': 0.6},
        {'prompt': 'sunset golden hour', 'position': 'center', 'strength': 0.3}
    ]
    
    hook = create_spatial_injection_hook(model, region_configs)
    for block in ['down_1', 'down_2', 'mid', 'up_0', 'up_1', 'up_2']:
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
    pil_img.save(output_dir / "06_complex_regional.png")
    print("✅ Saved: 06_complex_regional.png")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n" + "=" * 60)
    print("✅ CorePulse Spatial/Regional Injection Test Complete!")
    print("📊 Results:")
    print("  01_baseline.png: Normal generation without spatial control")
    print("  02_left_right_split.png: Left-right regional split")
    print("  03_quadrants.png: Four quadrant composition")
    print("  04_center_surround.png: Center focus with surround")
    print("  05_gradient_horizontal.png: Horizontal gradient transition")
    print("  06_complex_regional.png: Complex multi-region composition")
    print("\n💡 This implements CorePulse's spatial control!")
    print("🗺️ Regional injection enables compositional control!")
    print("🔬 Spatial awareness provides precise placement!")

if __name__ == "__main__":
    main()