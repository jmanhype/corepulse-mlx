#!/usr/bin/env python3
"""
Individual Test: Baseline Generation (No Hooks)
Generates a clean baseline image without any prompt manipulation.
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

def main():
    print("🎯 Individual Test: Baseline Generation")
    print("=" * 60)
    
    # Configuration
    prompt = "a majestic lion with golden mane in dramatic lighting"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    # Create output directory
    output_dir = Path("artifacts/images/individual_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📝 Prompt: '{prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Hooks already enabled globally before model import
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Clear any existing hooks
    attn_scores.KV_REGISTRY.clear()
    print("🧹 Cleared all hooks - generating clean baseline")
    
    # Generate baseline
    print("\n🎨 Generating baseline image...")
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
    output_path = output_dir / "baseline.png"
    pil_img.save(output_path)
    
    print(f"✅ Saved baseline image: {output_path}")
    print("📊 This is your clean reference - no manipulation applied")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n🎉 Baseline test complete!")

if __name__ == "__main__":
    main()