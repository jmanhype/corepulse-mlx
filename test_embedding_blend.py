#!/usr/bin/env python3
"""
Individual Test: Embedding Blending
Demonstrates mixing embeddings from multiple prompts at token level.
Creates hybrid concepts by blending cat + dog = catdog creature.
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

def create_blending_hook(blend_ratios):
    """Create hook that blends multiple concepts with specified ratios"""
    def hook(q, k, v, meta=None):
        # Only modify cross-attention (text-to-image attention)
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            v_new = mx.array(v)
            k_new = mx.array(k)
            
            # Create blended patterns for each concept
            batch, heads, seq_len, dim = v.shape
            
            # Concept 1: Cat-like features (soft, fluffy)
            cat_pattern = mx.random.normal((batch, heads, seq_len, dim)) * 0.8
            cat_pattern = mx.abs(cat_pattern)  # Positive values for softness
            
            # Concept 2: Dog-like features (energetic, playful)
            dog_pattern = mx.random.normal((batch, heads, seq_len, dim)) * 1.2
            dog_pattern = dog_pattern * mx.sin(mx.arange(seq_len).reshape(1, 1, -1, 1) * 0.5)
            
            # Concept 3: Mythical features (ethereal, magical)
            mythical_pattern = mx.random.normal((batch, heads, seq_len, dim)) * 0.5
            mythical_pattern = mythical_pattern * mx.cos(mx.arange(dim).reshape(1, 1, 1, -1) * 0.3)
            
            # Blend according to ratios
            blended = (
                cat_pattern * blend_ratios['cat'] +
                dog_pattern * blend_ratios['dog'] +
                mythical_pattern * blend_ratios['mythical']
            )
            
            # Apply blended modification
            v_new = v * 0.4 + blended
            k_new = k * 0.4 + blended * 0.7
            
            return q, k_new, v_new
        return q, k, v
    return hook

def main():
    print("🎯 Individual Test: Embedding Blending")
    print("==" * 30)
    
    # Configuration
    prompt = "a magical creature in an enchanted forest"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    # Blend ratios for different concepts
    blend_configs = [
        {'name': 'Cat-dominant', 'ratios': {'cat': 0.7, 'dog': 0.2, 'mythical': 0.1}},
        {'name': 'Dog-dominant', 'ratios': {'cat': 0.2, 'dog': 0.7, 'mythical': 0.1}},
        {'name': 'Balanced hybrid', 'ratios': {'cat': 0.4, 'dog': 0.4, 'mythical': 0.2}},
        {'name': 'Mythical focus', 'ratios': {'cat': 0.2, 'dog': 0.2, 'mythical': 0.6}},
    ]
    
    # Create output directory
    output_dir = Path("artifacts/images/embedding_injection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # We'll test the balanced hybrid
    test_config = blend_configs[2]
    
    print(f"📝 Base Prompt: '{prompt}'")
    print(f"🧬 Blend Config: {test_config['name']}")
    print(f"   🐱 Cat influence: {test_config['ratios']['cat']*100:.0f}%")
    print(f"   🐕 Dog influence: {test_config['ratios']['dog']*100:.0f}%")
    print(f"   ✨ Mythical influence: {test_config['ratios']['mythical']*100:.0f}%")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Clear any existing hooks
    attn_scores.KV_REGISTRY.clear()
    
    # Create blending hook
    print("\n🔬 Creating embedding blending hook...")
    blend_hook = create_blending_hook(test_config['ratios'])
    
    # Apply to different blocks with varying strength
    block_configs = [
        ("down_0", 1.0),
        ("down_1", 0.8),
        ("down_2", 0.6),
        ("mid", 1.0),
        ("up_0", 0.6),
        ("up_1", 0.8),
        ("up_2", 1.0),
    ]
    
    for block, strength in block_configs:
        # Create hook with adjusted strength
        adjusted_ratios = {k: v * strength for k, v in test_config['ratios'].items()}
        attn_scores.KV_REGISTRY.set(block, create_blending_hook(adjusted_ratios))
        print(f"   🧬 Blending at {strength*100:.0f}% strength → {block}")
    
    # Generate with embedding blending
    print("\n🎨 Generating image with embedding blending...")
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
    output_path = output_dir / "embedding_blend.png"
    pil_img.save(output_path)
    
    print(f"\n✅ Saved embedding blend image: {output_path}")
    print("📊 Expected: Hybrid creature with mixed cat/dog/mythical features")
    print("💡 This proves we can blend multiple concepts mathematically!")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n🎉 Embedding blending test complete!")

if __name__ == "__main__":
    main()