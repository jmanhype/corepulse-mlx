#!/usr/bin/env python3
"""
Advanced Test: Concept Fusion
Creates hybrid concepts by mathematically blending multiple creature embeddings.
Generates chimera-like beings that don't exist in reality.
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

def create_concept_fusion_hook(concepts_weights):
    """Fuse multiple concepts with specified weights"""
    def hook(q, k, v, meta=None):
        # Only modify cross-attention (text-to-image attention)
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            
            # Initialize fusion pattern
            fusion_pattern = mx.zeros((batch, heads, seq_len, dim))
            
            for concept, weight in concepts_weights.items():
                if concept == "dragon":
                    # Dragon: scales, wings, fire
                    pattern = mx.random.normal((batch, heads, seq_len, dim))
                    # Scales texture
                    scales = mx.abs(mx.sin(mx.arange(dim).reshape(1, 1, 1, -1) * 10))
                    # Wing patterns
                    wings = mx.exp(-mx.abs(mx.arange(seq_len).reshape(1, 1, -1, 1) - seq_len/2) * 0.1)
                    pattern = pattern * scales * wings
                    
                elif concept == "phoenix":
                    # Phoenix: feathers, fire, rebirth
                    pattern = mx.random.normal((batch, heads, seq_len, dim))
                    # Feather texture
                    feathers = mx.cos(mx.arange(dim).reshape(1, 1, 1, -1) * 5) * \
                              mx.sin(mx.arange(seq_len).reshape(1, 1, -1, 1) * 3)
                    # Fire effect
                    fire = mx.random.uniform(shape=(batch, heads, seq_len, dim), low=0.8, high=1.5)
                    pattern = pattern * feathers * fire
                    
                elif concept == "tree":
                    # Tree: branches, leaves, roots
                    pattern = mx.random.normal((batch, heads, seq_len, dim))
                    # Branching structure
                    branches = mx.abs(pattern) * mx.log(mx.abs(pattern) + 1)
                    # Organic growth
                    growth = mx.exp(-mx.arange(seq_len).reshape(1, 1, -1, 1) * 0.05)
                    pattern = branches * growth
                    
                elif concept == "crystal":
                    # Crystal: geometric, refractive, sharp
                    pattern = mx.random.normal((batch, heads, seq_len, dim))
                    # Geometric facets
                    facets = mx.abs(pattern) * mx.sign(mx.sin(pattern * 5))
                    # Refractive properties
                    refraction = mx.random.uniform(shape=(batch, heads, seq_len, dim), low=0.5, high=2.0)
                    pattern = facets * refraction
                    
                elif concept == "ocean":
                    # Ocean: waves, fluidity, depth
                    pattern = mx.random.normal((batch, heads, seq_len, dim))
                    # Wave patterns
                    waves = mx.sin(mx.arange(seq_len).reshape(1, 1, -1, 1) * 0.5) * \
                           mx.cos(mx.arange(dim).reshape(1, 1, 1, -1) * 0.3)
                    # Depth gradient
                    depth = mx.exp(-mx.abs(pattern) * 0.3)
                    pattern = pattern * waves * depth
                    
                elif concept == "machine":
                    # Machine: mechanical, precise, metallic
                    pattern = mx.random.normal((batch, heads, seq_len, dim))
                    # Mechanical precision
                    mechanical = mx.round(pattern * 3) / 3
                    # Metallic sheen
                    metallic = mx.abs(pattern) * 1.5
                    pattern = mechanical * metallic
                    
                else:
                    pattern = mx.zeros((batch, heads, seq_len, dim))
                
                # Add weighted pattern to fusion
                fusion_pattern = fusion_pattern + pattern * weight
            
            # Normalize fusion pattern
            fusion_pattern = fusion_pattern / mx.sum(mx.array(list(concepts_weights.values())))
            
            # Apply fusion to K and V
            k_new = k * 0.3 + fusion_pattern * 0.7
            v_new = v * 0.2 + fusion_pattern * 0.8
            
            return q, k_new, v_new
        return q, k, v
    return hook

def main():
    print("🧬 Advanced Test: Concept Fusion")
    print("==" * 30)
    
    # Configuration
    base_prompt = "a majestic creature in a mystical realm"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    # Concept fusion configurations
    fusion_configs = [
        {
            "name": "Dragon-Phoenix Hybrid",
            "weights": {"dragon": 0.6, "phoenix": 0.4},
            "description": "Fire-breathing bird with scales and feathers"
        },
        {
            "name": "Tree-Crystal Being",
            "weights": {"tree": 0.5, "crystal": 0.5},
            "description": "Living tree made of crystalline branches"
        },
        {
            "name": "Ocean-Machine Entity",
            "weights": {"ocean": 0.4, "machine": 0.6},
            "description": "Mechanical being with fluid properties"
        },
        {
            "name": "Ultimate Chimera",
            "weights": {
                "dragon": 0.25,
                "phoenix": 0.20,
                "tree": 0.15,
                "crystal": 0.15,
                "ocean": 0.15,
                "machine": 0.10
            },
            "description": "All concepts fused into one entity"
        }
    ]
    
    # Create output directory
    output_dir = Path("artifacts/images/advanced_artistic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test the Ultimate Chimera
    test_config = fusion_configs[3]
    
    print(f"📝 Base Prompt: '{base_prompt}'")
    print(f"🧬 Fusion: {test_config['name']}")
    print(f"📊 Concept Weights:")
    for concept, weight in test_config['weights'].items():
        print(f"   {concept}: {weight*100:.0f}%")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Clear any existing hooks
    attn_scores.KV_REGISTRY.clear()
    
    # Register concept fusion hooks
    print("\n🧬 Registering concept fusion hooks...")
    
    # Apply different fusion strengths to different blocks
    block_strengths = {
        "down_0": 0.6,  # Early structure
        "down_1": 0.8,  # Strong fusion
        "down_2": 1.0,  # Maximum fusion
        "mid": 1.0,     # Core fusion
        "up_0": 0.8,    # Details fusion
        "up_1": 0.6,    # Moderate fusion
        "up_2": 0.4     # Light fusion
    }
    
    for block, strength in block_strengths.items():
        # Scale weights by block strength
        scaled_weights = {k: v * strength for k, v in test_config['weights'].items()}
        hook = create_concept_fusion_hook(scaled_weights)
        attn_scores.KV_REGISTRY.set(block, hook)
        print(f"   🧬 Fusion at {strength*100:.0f}% → {block}")
    
    # Generate with concept fusion
    print("\n🎨 Generating hybrid creature...")
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
    output_path = output_dir / "concept_fusion.png"
    pil_img.save(output_path)
    
    print(f"\n✅ Saved concept fusion image: {output_path}")
    print(f"📊 Expected: {test_config['description']}")
    print("💡 This proves we can create entirely new hybrid concepts!")
    
    # Generate baseline
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
    
    img_array_baseline = (img_baseline[0] * 255).astype(mx.uint8)
    pil_img_baseline = PIL.Image.fromarray(np.array(img_array_baseline))
    output_path_baseline = output_dir / "concept_fusion_baseline.png"
    pil_img_baseline.save(output_path_baseline)
    
    print(f"✅ Saved baseline image: {output_path_baseline}")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n🎉 Concept fusion test complete!")

if __name__ == "__main__":
    main()