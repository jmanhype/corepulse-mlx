#!/usr/bin/env python3
"""
Test: Multi-Scale Architecture Demonstration
Shows prompt injection at different scales/resolutions.
Similar to CorePulse's multi_scale_architecture.png
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

def create_scale_injection_hook(model, inject_prompt, scale_factor=1.0):
    """Create scale-aware injection hook."""
    inject_cond, _ = model._get_text_conditioning(inject_prompt)
    
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            # Different scale factors for different blocks
            block_scales = {
                'down_0': scale_factor * 0.3,  # Low resolution
                'down_1': scale_factor * 0.5,  # Medium-low
                'down_2': scale_factor * 0.7,  # Medium
                'mid': scale_factor * 1.0,     # Full scale
                'up_0': scale_factor * 0.8,    # Medium-high
                'up_1': scale_factor * 0.6,    # Medium
                'up_2': scale_factor * 0.4     # High resolution
            }
            
            current_scale = block_scales.get(block_id, 0.5)
            
            if seq_len >= inject_cond.shape[1] and current_scale > 0:
                v_new = mx.array(v)
                embed_dim = min(dim, inject_cond.shape[2])
                embed_len = min(seq_len, inject_cond.shape[1])
                
                # Simple injection
                inject_vals = inject_cond[0, :embed_len, :embed_dim]
                
                for b in range(batch):
                    for h in range(heads):
                        v_new[b, h, :embed_len, :embed_dim] = \
                            v[b, h, :embed_len, :embed_dim] * (1 - current_scale) + \
                            inject_vals * current_scale
                
                print(f"    🔍 Scale injection at {block_id}: scale={current_scale:.2f}")
                return q, k, v_new
        return q, k, v
    return hook

def main():
    print("🏗️ Test: Multi-Scale Architecture")
    print("=" * 60)
    
    # Configuration
    base_prompt = "a detailed architectural blueprint"
    num_steps = 4
    cfg_weight = 7.5
    seed = 42
    
    print(f"📝 Base Prompt: '{base_prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Create output directory
    output_dir = Path("artifacts/images/multi_scale_architecture")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Test different scale injections
    inject_prompts = [
        ("futuristic skyscraper", 0.3),
        ("gothic cathedral", 0.6),
        ("zen garden temple", 0.9),
        ("cyberpunk megastructure", 1.0)
    ]
    
    for idx, (inject_prompt, scale) in enumerate(inject_prompts, 1):
        print(f"\n🎨 Test {idx}/{len(inject_prompts)}: '{inject_prompt}' at scale {scale}...")
        attn_scores.KV_REGISTRY.clear()
        
        # Apply hook to all blocks
        hook = create_scale_injection_hook(model, inject_prompt, scale)
        for block in ['down_0', 'down_1', 'down_2', 'mid', 'up_0', 'up_1', 'up_2']:
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
        
        filename = f"{idx:02d}_scale_{scale:.1f}_{inject_prompt.replace(' ', '_')}.png"
        pil_img.save(output_dir / filename)
        print(f"✅ Saved: {filename}")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n" + "=" * 60)
    print("✅ Multi-Scale Architecture Test Complete!")
    print("📊 Results saved to artifacts/images/multi_scale_architecture/")
    print("\n💡 This demonstrates scale-aware prompt injection!")
    print("🏗️ Different scales affect different architectural layers!")

if __name__ == "__main__":
    main()