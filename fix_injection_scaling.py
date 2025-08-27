#!/usr/bin/env python3
"""
Fix the injection scaling issues in our attention hooks.
The problem: We're modifying values too aggressively causing corrupted outputs.
The solution: Use proper blending with careful strength control.
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

def create_fixed_injection_hook(model, inject_prompt, strength=0.3):
    """Create a properly scaled injection hook."""
    inject_cond, inject_pooled = model._get_text_conditioning(inject_prompt)
    
    def hook(q, k, v, meta=None):
        # Only modify cross-attention layers
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            # Very conservative strength scaling per block
            block_strengths = {
                'down_0': strength * 0.1,  # Very subtle
                'down_1': strength * 0.2,  # Subtle
                'down_2': strength * 0.3,  # Light
                'mid': strength * 0.4,      # Moderate
                'up_0': strength * 0.3,     # Light
                'up_1': strength * 0.2,     # Subtle
                'up_2': strength * 0.1      # Very subtle
            }
            
            current_strength = block_strengths.get(block_id, 0.1)
            
            if current_strength > 0 and seq_len >= inject_cond.shape[1]:
                # Create properly shaped injection
                embed_len = min(seq_len, inject_cond.shape[1])
                embed_dim = min(dim, inject_cond.shape[2])
                
                # Get injection values
                inject_vals = inject_cond[0, :embed_len, :embed_dim]
                
                # Blend conservatively - FIXED APPROACH
                v_modified = mx.array(v)
                
                # Apply injection with proper broadcasting
                for b in range(batch):
                    for h in range(heads):
                        # Linear interpolation instead of replacement
                        v_modified[b, h, :embed_len, :embed_dim] = \
                            v[b, h, :embed_len, :embed_dim] * (1.0 - current_strength) + \
                            inject_vals * current_strength
                
                return q, k, v_modified
        
        # Don't modify self-attention or if conditions not met
        return q, k, v
    
    return hook

def test_fixed_injection():
    """Test the fixed injection approach."""
    print("🔧 Testing Fixed Injection Scaling")
    print("=" * 60)
    
    # Test prompts
    base_prompt = "a beautiful mountain landscape with lake"
    inject_prompts = [
        ("subtle cyberpunk elements", 0.2),
        ("fantasy magical forest", 0.3),
        ("underwater ocean scene", 0.4),
    ]
    
    # Configuration
    num_steps = 10  # More steps for better quality
    cfg_weight = 7.5
    seed = 42
    
    # Create output directory
    output_dir = Path("artifacts/images/fixed_scaling")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n📦 Loading SDXL-Turbo...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Test 1: Baseline
    print(f"\n🎨 Generating baseline: '{base_prompt}'")
    attn_scores.KV_REGISTRY.clear()
    
    latents = model.generate_latents(
        base_prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    # Get final image
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    baseline_img = PIL.Image.fromarray(np.array(img_array))
    baseline_img.save(output_dir / "00_baseline.png")
    print("✅ Saved: 00_baseline.png")
    
    # Test injections with different strengths
    for idx, (inject_prompt, strength) in enumerate(inject_prompts, 1):
        print(f"\n🎨 Test {idx}: Injecting '{inject_prompt}' at strength {strength}")
        
        # Clear previous hooks
        attn_scores.KV_REGISTRY.clear()
        
        # Apply fixed injection hook to specific blocks only
        hook = create_fixed_injection_hook(model, inject_prompt, strength)
        
        # Only apply to middle and late blocks for better control
        for block in ['mid', 'up_0', 'up_1']:
            attn_scores.KV_REGISTRY.set(block, hook)
        
        # Generate with injection
        latents = model.generate_latents(
            base_prompt,
            num_steps=num_steps,
            cfg_weight=cfg_weight,
            seed=seed
        )
        
        # Get final image
        for i, x in enumerate(latents):
            if i == num_steps - 1:
                img = model.decode(x)
        
        img_array = (img[0] * 255).astype(mx.uint8)
        pil_img = PIL.Image.fromarray(np.array(img_array))
        
        filename = f"{idx:02d}_{inject_prompt.replace(' ', '_')[:20]}_s{strength}.png"
        pil_img.save(output_dir / filename)
        print(f"✅ Saved: {filename}")
    
    # Test 4: Progressive strength increase
    print(f"\n🎨 Test 4: Progressive strength increase")
    inject_prompt = "ethereal dreamlike atmosphere"
    
    for strength_idx, strength in enumerate([0.1, 0.2, 0.3, 0.4, 0.5], 1):
        print(f"  Strength {strength}...")
        
        attn_scores.KV_REGISTRY.clear()
        hook = create_fixed_injection_hook(model, inject_prompt, strength)
        
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
        pil_img = PIL.Image.fromarray(np.array(img_array))
        
        filename = f"progressive_{strength_idx:02d}_s{strength}.png"
        pil_img.save(output_dir / filename)
        print(f"  ✅ Saved: {filename}")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n" + "=" * 60)
    print("✅ Fixed Injection Tests Complete!")
    print("📊 Results saved to artifacts/images/fixed_scaling/")
    print("\n💡 Key improvements:")
    print("  • Conservative strength values (0.1 - 0.5)")
    print("  • Linear interpolation instead of replacement")
    print("  • Block-specific strength scaling")
    print("  • More denoising steps for quality")

if __name__ == "__main__":
    test_fixed_injection()