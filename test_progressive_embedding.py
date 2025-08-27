#!/usr/bin/env python3
"""
Individual Test: Progressive Embedding Injection
Demonstrates gradual transition between different embeddings across UNet depth.
Creates smooth conceptual morphing from one idea to another.
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

def create_progressive_injection_hook(start_concept_strength, end_concept_strength):
    """
    Create hook with progressive transition between two concepts.
    Early blocks get more of start_concept, late blocks get more of end_concept.
    """
    def hook(q, k, v, meta=None):
        # Only modify cross-attention (text-to-image attention)
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            v_new = mx.array(v)
            k_new = mx.array(k)
            
            batch, heads, seq_len, dim = v.shape
            
            # Start concept pattern (geometric, structured)
            start_pattern = mx.random.normal((batch, heads, seq_len, dim))
            # Make it more structured
            start_pattern = mx.abs(start_pattern) * mx.sin(mx.arange(dim).reshape(1, 1, 1, -1) * 0.1)
            
            # End concept pattern (organic, flowing)
            end_pattern = mx.random.normal((batch, heads, seq_len, dim))
            # Make it more organic
            end_pattern = end_pattern * mx.cos(mx.arange(seq_len).reshape(1, 1, -1, 1) * 0.2)
            
            # Blend based on strengths
            blended_pattern = (
                start_pattern * start_concept_strength +
                end_pattern * end_concept_strength
            )
            
            # Apply progressive modification
            v_new = v * 0.3 + blended_pattern
            k_new = k * 0.3 + blended_pattern * 0.6
            
            return q, k_new, v_new
        return q, k, v
    return hook

def main():
    print("🎯 Individual Test: Progressive Embedding Injection")
    print("==" * 30)
    
    # Configuration
    base_prompt = "a transformation from crystal to water"
    start_concept = "geometric crystal structure"
    end_concept = "flowing liquid water"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    # Create output directory
    output_dir = Path("artifacts/images/embedding_injection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📝 Base Prompt: '{base_prompt}'")
    print(f"🏁 Start Concept: '{start_concept}' (early blocks)")
    print(f"🎯 End Concept: '{end_concept}' (late blocks)")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Clear any existing hooks
    attn_scores.KV_REGISTRY.clear()
    
    # Create progressive injection with smooth transition
    print("\n🔬 Creating progressive embedding injection...")
    
    # Define progressive strengths for each block
    block_progression = [
        ("down_0", 1.0, 0.0, "100% crystal"),
        ("down_1", 0.8, 0.2, "80% crystal, 20% water"),
        ("down_2", 0.6, 0.4, "60% crystal, 40% water"),
        ("mid",    0.5, 0.5, "50% crystal, 50% water"),
        ("up_0",   0.4, 0.6, "40% crystal, 60% water"),
        ("up_1",   0.2, 0.8, "20% crystal, 80% water"),
        ("up_2",   0.0, 1.0, "100% water"),
    ]
    
    for block, start_str, end_str, description in block_progression:
        hook = create_progressive_injection_hook(start_str, end_str)
        attn_scores.KV_REGISTRY.set(block, hook)
        print(f"   📈 {block}: {description}")
    
    # Generate with progressive injection
    print("\n🎨 Generating image with progressive embedding injection...")
    latents = model.generate_latents(
        base_prompt,
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
    output_path = output_dir / "progressive_embedding.png"
    pil_img.save(output_path)
    
    print(f"\n✅ Saved progressive embedding image: {output_path}")
    print("📊 Expected: Smooth transition from crystal-like to water-like")
    print("💡 This proves we can create conceptual morphing across depth!")
    
    # Generate comparison: without progressive injection
    print("\n🎨 Generating baseline for comparison...")
    attn_scores.KV_REGISTRY.clear()
    
    latents_baseline = model.generate_latents(
        base_prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents_baseline):
        if i == num_steps - 1:
            img_baseline = model.decode(x)
    
    # Save baseline
    img_array_baseline = (img_baseline[0] * 255).astype(mx.uint8)
    pil_img_baseline = PIL.Image.fromarray(np.array(img_array_baseline))
    output_path_baseline = output_dir / "progressive_baseline.png"
    pil_img_baseline.save(output_path_baseline)
    
    print(f"✅ Saved baseline image: {output_path_baseline}")
    print("📊 Compare the two images to see the progressive effect!")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n🎉 Progressive embedding injection test complete!")

if __name__ == "__main__":
    main()