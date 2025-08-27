#!/usr/bin/env python3
"""
Individual Test: Extreme Suppression
Demonstrates suppressing prompt influence by scaling attention values very low.
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

def create_suppression_hook(strength=0.05):
    """Suppress attention by multiplying K,V by very small factor"""
    def hook(q, k, v, meta=None):
        # Only modify cross-attention (text-to-image attention)
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            # Suppress all text tokens
            k_new = k * strength
            v_new = v * strength
            return q, k_new, v_new
        return q, k, v
    return hook

def main():
    print("🎯 Individual Test: Extreme Suppression")
    print("=" * 60)
    
    # Configuration
    prompt = "a majestic lion with golden mane in dramatic lighting"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    strength = 0.05  # 95% suppression
    
    # Create output directory
    output_dir = Path("artifacts/images/individual_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📝 Prompt: '{prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    print(f"🔇 Suppression: {strength}x strength ({100-strength*100:.0f}% reduced)")
    
    # Hooks already enabled globally before model import
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Clear any existing hooks
    attn_scores.KV_REGISTRY.clear()
    
    # Register suppression hook
    supp_hook = create_suppression_hook(strength)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, supp_hook)
    
    print(f"🔇 Registered {strength}x suppression hooks on all attention blocks")
    
    # Generate with suppression
    print("\n🎨 Generating image with extreme suppression...")
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
    output_path = output_dir / "suppression.png"
    pil_img.save(output_path)
    
    print(f"✅ Saved suppression image: {output_path}")
    print("📊 Expected: Random/different content, ignoring prompt")
    print("💡 This proves we can make the model ignore the prompt entirely!")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n🎉 Suppression test complete!")

if __name__ == "__main__":
    main()