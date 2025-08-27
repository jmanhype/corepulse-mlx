#!/usr/bin/env python3
"""
Test: SDXL Composition Landscapes
Demonstrates complex landscape composition using SDXL.
Similar to CorePulse's sdxl_composition.png
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

def create_composition_hook(model, foreground_prompt, background_prompt):
    """Create composition-aware injection hook."""
    fg_cond, _ = model._get_text_conditioning(foreground_prompt)
    bg_cond, _ = model._get_text_conditioning(background_prompt)
    
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            # Foreground in early blocks, background in later blocks
            if block_id in ['down_0', 'down_1']:
                # Foreground emphasis
                inject_cond = fg_cond
                strength = 0.7
                label = "foreground"
            elif block_id in ['up_1', 'up_2']:
                # Background emphasis
                inject_cond = bg_cond
                strength = 0.5
                label = "background"
            else:
                return q, k, v
            
            if seq_len >= inject_cond.shape[1]:
                v_new = mx.array(v)
                embed_dim = min(dim, inject_cond.shape[2])
                embed_len = min(seq_len, inject_cond.shape[1])
                inject_vals = inject_cond[0, :embed_len, :embed_dim]
                
                for b in range(batch):
                    for h in range(heads):
                        v_new[b, h, :embed_len, :embed_dim] = \
                            v[b, h, :embed_len, :embed_dim] * (1 - strength) + \
                            inject_vals * strength
                
                print(f"    🏞️ Composition at {block_id}: {label} (strength={strength})")
                return q, k, v_new
        return q, k, v
    return hook

def main():
    print("🏞️ Test: SDXL Composition Landscapes")
    print("=" * 60)
    
    # Configuration
    base_prompt = "a breathtaking landscape photograph"
    num_steps = 4
    cfg_weight = 7.5
    seed = 42
    
    print(f"📝 Base Prompt: '{base_prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Create output directory
    output_dir = Path("artifacts/images/sdxl_composition")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Test compositions
    compositions = [
        {
            'name': 'mountain_lake',
            'foreground': 'crystal clear alpine lake with reflections',
            'background': 'majestic snow-capped mountains'
        },
        {
            'name': 'desert_oasis',
            'foreground': 'lush palm oasis with water',
            'background': 'vast rolling sand dunes'
        },
        {
            'name': 'city_skyline',
            'foreground': 'illuminated city waterfront',
            'background': 'dramatic sunset sky'
        },
        {
            'name': 'forest_path',
            'foreground': 'winding forest path with sunlight',
            'background': 'dense misty forest canopy'
        }
    ]
    
    for idx, comp in enumerate(compositions, 1):
        print(f"\n🎨 Test {idx}/{len(compositions)}: {comp['name']}...")
        print(f"  Foreground: {comp['foreground']}")
        print(f"  Background: {comp['background']}")
        
        attn_scores.KV_REGISTRY.clear()
        
        # Apply composition hook
        hook = create_composition_hook(model, comp['foreground'], comp['background'])
        for block in ['down_0', 'down_1', 'up_1', 'up_2']:
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
        
        filename = f"{idx:02d}_{comp['name']}.png"
        pil_img.save(output_dir / filename)
        print(f"✅ Saved: {filename}")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n" + "=" * 60)
    print("✅ SDXL Composition Test Complete!")
    print("📊 Results saved to artifacts/images/sdxl_composition/")
    print("\n💡 This demonstrates foreground/background composition!")
    print("🏞️ Complex landscapes with layered elements!")

if __name__ == "__main__":
    main()