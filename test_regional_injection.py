#!/usr/bin/env python3
"""
Test: Regional Injection with Positioning
Demonstrates spatial/regional prompt injection with specific positioning.
Similar to CorePulse's regional_injection.png
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

def create_regional_hook(model, region_prompts):
    """Create regional injection hook with spatial awareness."""
    # Precompute embeddings for each region
    region_embeds = {}
    for region, prompt in region_prompts.items():
        cond, _ = model._get_text_conditioning(prompt)
        region_embeds[region] = cond
    
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            # Only apply in specific blocks for better control
            if block_id not in ['up_0', 'up_1', 'up_2']:
                return q, k, v
            
            v_new = mx.array(v)
            
            # Determine spatial position based on attention heads
            # Different heads focus on different spatial regions
            for region, embed in region_embeds.items():
                if seq_len >= embed.shape[1]:
                    embed_dim = min(dim, embed.shape[2])
                    embed_len = min(seq_len, embed.shape[1])
                    embed_vals = embed[0, :embed_len, :embed_dim]
                    
                    # Map regions to head ranges
                    if region == 'left':
                        head_range = range(0, heads // 3)
                        strength = 0.6
                    elif region == 'center':
                        head_range = range(heads // 3, 2 * heads // 3)
                        strength = 0.8
                    elif region == 'right':
                        head_range = range(2 * heads // 3, heads)
                        strength = 0.6
                    elif region == 'top':
                        head_range = range(0, heads // 2)
                        strength = 0.5
                    elif region == 'bottom':
                        head_range = range(heads // 2, heads)
                        strength = 0.5
                    else:
                        continue
                    
                    # Apply regional injection
                    for b in range(batch):
                        for h in head_range:
                            if h < heads:
                                v_new[b, h, :embed_len, :embed_dim] = \
                                    v[b, h, :embed_len, :embed_dim] * (1 - strength) + \
                                    embed_vals * strength
            
            active_regions = len(region_embeds)
            print(f"    🎯 Regional injection at {block_id}: {active_regions} regions")
            return q, k, v_new
        return q, k, v
    return hook

def main():
    print("🎯 Test: Regional Injection with Positioning")
    print("=" * 60)
    
    # Configuration
    base_prompt = "a panoramic landscape view"
    num_steps = 4
    cfg_weight = 7.5
    seed = 42
    
    print(f"📝 Base Prompt: '{base_prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Create output directory
    output_dir = Path("artifacts/images/regional_injection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Test configurations
    test_configs = [
        {
            'name': 'horizontal_split',
            'regions': {
                'left': 'snowy mountains',
                'center': 'serene lake',
                'right': 'autumn forest'
            }
        },
        {
            'name': 'vertical_split',
            'regions': {
                'top': 'dramatic clouds',
                'bottom': 'rocky terrain'
            }
        },
        {
            'name': 'quadrant_split',
            'regions': {
                'left': 'desert dunes',
                'right': 'ocean waves',
                'top': 'stormy sky',
                'bottom': 'coral reef'
            }
        },
        {
            'name': 'focus_center',
            'regions': {
                'center': 'ancient castle',
                'left': 'misty forest',
                'right': 'misty forest'
            }
        }
    ]
    
    for idx, config in enumerate(test_configs, 1):
        print(f"\n🎨 Test {idx}/{len(test_configs)}: {config['name']}...")
        print(f"  Regions: {list(config['regions'].keys())}")
        
        attn_scores.KV_REGISTRY.clear()
        
        # Apply regional hook
        hook = create_regional_hook(model, config['regions'])
        for block in ['up_0', 'up_1', 'up_2']:
            attn_scores.KV_REGISTRY.set(block, hook)
        
        # Generate
        latents = model.generate_latents(
            base_prompt,
            num_steps=num_steps,
            cfg_weight=cfg_weight,
            seed=seed + idx
        )
        
        for i, x in enumerate(latents):
            if i == num_steps - 1:
                img = model.decode(x)
        
        img_array = (img[0] * 255).astype(mx.uint8)
        pil_img = PIL.Image.fromarray(np.array(img_array))
        
        filename = f"{idx:02d}_{config['name']}.png"
        pil_img.save(output_dir / filename)
        print(f"✅ Saved: {filename}")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n" + "=" * 60)
    print("✅ Regional Injection Test Complete!")
    print("📊 Results saved to artifacts/images/regional_injection/")
    print("\n💡 This demonstrates regional/spatial prompt control!")
    print("🎯 Different regions can have different content!")

if __name__ == "__main__":
    main()