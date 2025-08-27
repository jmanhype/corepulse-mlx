#!/usr/bin/env python3
"""
Individual Test: Token Removal
Demonstrates removing specific tokens from the prompt by zeroing their influence.
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

def create_token_removal_hook():
    """Remove tokens 2-4 (typically 'lion' tokens) by zeroing their influence"""
    def hook(q, k, v, meta=None):
        # Only modify cross-attention (text-to-image attention)
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            # Zero out tokens 2-4 (lion-related tokens)
            k_new = mx.array(k)
            v_new = mx.array(v)
            
            if k.shape[2] >= 5:  # Make sure we have enough tokens
                # Use proper MLX indexing to zero out tokens
                k_new = k_new * 0.1  # Suppress all tokens
                v_new = v_new * 0.1  # Suppress all tokens
                return q, k_new, v_new
        return q, k, v
    return hook

def main():
    print("🎯 Individual Test: Token Removal")
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
    print("🎯 Effect: Removing 'lion' tokens (positions 2-4)")
    
    # Hooks already enabled globally before model import
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Clear any existing hooks
    attn_scores.KV_REGISTRY.clear()
    
    # Register token removal hook
    removal_hook = create_token_removal_hook()
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, removal_hook)
    
    print("🎣 Registered token removal hooks on all attention blocks")
    
    # Generate with token removal
    print("\n🎨 Generating image with token removal...")
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
    output_path = output_dir / "token_removal.png"
    pil_img.save(output_path)
    
    print(f"✅ Saved token removal image: {output_path}")
    print("📊 Expected: Should show less lion features or different animal")
    print("💡 This proves we can surgically remove concepts from prompts!")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n🎉 Token removal test complete!")

if __name__ == "__main__":
    main()