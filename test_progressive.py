#!/usr/bin/env python3
"""
Individual Test: Progressive Manipulation
Demonstrates gradient control across different attention blocks.
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

def create_progressive_hook(strength):
    """Create hook with specific strength for progressive control"""
    def hook(q, k, v, meta=None):
        # Only modify cross-attention (text-to-image attention)
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            k_new = k * strength
            v_new = v * strength
            return q, k_new, v_new
        return q, k, v
    return hook

def main():
    print("🎯 Individual Test: Progressive Manipulation")
    print("=" * 60)
    
    # Configuration
    prompt = "a majestic lion with golden mane in dramatic lighting"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    # Progressive strengths for different blocks
    strengths = {
        "down_0": 0.2,  # Weak
        "down_1": 0.5,  # Moderate
        "down_2": 1.0,  # Normal
        "mid": 2.0,     # Strong
        "up_0": 3.0,    # Very strong
        "up_1": 4.0,    # Extreme
        "up_2": 5.0     # Maximum
    }
    
    # Create output directory
    output_dir = Path("artifacts/images/individual_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📝 Prompt: '{prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    print("📊 Progressive Strengths:")
    for block, strength in strengths.items():
        print(f"   {block}: {strength}x")
    
    # Hooks already enabled globally before model import
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Clear any existing hooks
    attn_scores.KV_REGISTRY.clear()
    
    # Register progressive hooks with different strengths
    for block, strength in strengths.items():
        hook = create_progressive_hook(strength)
        attn_scores.KV_REGISTRY.set(block, hook)
        print(f"📈 Registered {strength}x hook on {block}")
    
    # Generate with progressive manipulation
    print("\n🎨 Generating image with progressive manipulation...")
    latents = model.generate_latents(
        prompt, 
        num_steps=num_steps, 
        cfg_weight=cfg_weight, 
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    # Save image
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    output_path = output_dir / "progressive.png"
    pil_img.save(output_path)
    
    print(f"✅ Saved progressive image: {output_path}")
    print("📊 Expected: Gradient effects from subtle to extreme manipulation")
    print("💡 This proves we can fine-tune different processing stages!")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n🎉 Progressive test complete!")

if __name__ == "__main__":
    main()