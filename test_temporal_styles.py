#!/usr/bin/env python3
"""
Advanced Test: Temporal Styles
Generates images in specific historical time period styles.
Transitions from ancient to futuristic across UNet blocks.
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

def create_temporal_style_hook(era, intensity):
    """Create time period specific style patterns"""
    def hook(q, k, v, meta=None):
        # Only modify cross-attention (text-to-image attention)
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            
            if era == "ancient":
                # Ancient (3000 BCE - 500 CE): Stone, hieroglyphs, weathered
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Weathered, aged texture
                weathering = mx.exp(-mx.abs(pattern) * 0.3)
                # Stone-like granular texture
                stone = mx.abs(mx.sin(pattern * 20)) * 0.3 + 0.7
                # Hieroglyphic patterns
                symbols = mx.where(
                    mx.random.uniform(shape=(batch, heads, seq_len, dim)) > 0.8,
                    mx.full((batch, heads, seq_len, dim), 2.0),
                    mx.ones((batch, heads, seq_len, dim))
                )
                pattern = pattern * weathering * stone * symbols * intensity
                
            elif era == "medieval":
                # Medieval (500 - 1500): Illuminated manuscripts, gothic, tapestry
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Gothic architectural patterns
                gothic = mx.abs(mx.sin(mx.arange(seq_len).reshape(1, 1, -1, 1) * 0.5)) * \
                        mx.abs(mx.cos(mx.arange(dim).reshape(1, 1, 1, -1) * 0.3))
                # Manuscript gold leaf effect
                illumination = mx.where(
                    mx.random.uniform(shape=(batch, heads, seq_len, dim)) > 0.9,
                    mx.full((batch, heads, seq_len, dim), 3.0),
                    mx.ones((batch, heads, seq_len, dim))
                )
                pattern = pattern * gothic * illumination * intensity
                
            elif era == "renaissance":
                # Renaissance (1400 - 1600): Classical proportions, sfumato, chiaroscuro
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Golden ratio proportions
                golden = 1.618
                proportion = mx.exp(-mx.abs(pattern - golden) * 0.5)
                # Sfumato (soft edges)
                # Approximate gradient with difference between neighboring elements
                grad_approx = mx.abs(mx.roll(pattern, 1, axis=2) - pattern) + \
                             mx.abs(mx.roll(pattern, 1, axis=3) - pattern)
                sfumato = mx.exp(-grad_approx * 0.2)
                # Chiaroscuro (strong light/dark contrast)
                chiaroscuro = mx.abs(mx.sin(pattern * 2)) * 2
                pattern = pattern * proportion * sfumato * chiaroscuro * intensity
                
            elif era == "industrial":
                # Industrial (1760 - 1900): Steam, metal, mechanical
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Mechanical gears pattern
                gears = mx.abs(mx.sin(pattern * 10)) * mx.abs(mx.cos(pattern * 10))
                # Steam and smoke
                steam = mx.random.normal((batch, heads, seq_len, dim)) * 0.3
                # Metal texture
                metal = mx.abs(pattern) * 1.5
                pattern = (gears + steam) * metal * intensity
                
            elif era == "art_deco":
                # Art Deco (1920s - 1930s): Geometric, luxurious, streamlined
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Geometric patterns
                geometric = mx.abs(mx.round(pattern * 3) / 3)
                # Luxurious metallic
                luxury = mx.abs(pattern) * mx.random.uniform(shape=(batch, heads, seq_len, dim), low=1.0, high=2.0)
                # Streamlined curves
                streamline = mx.sin(mx.arange(seq_len).reshape(1, 1, -1, 1) * 0.2)
                pattern = geometric * luxury * streamline * intensity
                
            elif era == "digital":
                # Digital Age (1980s - 2000s): Pixels, neon, glitch
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Pixelated effect
                pixels = mx.round(pattern * 8) / 8
                # Neon colors
                neon = mx.abs(pattern) * mx.random.uniform(shape=(batch, heads, seq_len, dim), low=0.5, high=3.0)
                # Digital glitch
                glitch = mx.where(
                    mx.random.uniform(shape=(batch, heads, seq_len, dim)) > 0.95,
                    mx.random.normal((batch, heads, seq_len, dim)) * 5,
                    mx.ones((batch, heads, seq_len, dim))
                )
                pattern = pixels * neon * glitch * intensity
                
            elif era == "cyberpunk":
                # Cyberpunk Future (2050+): Holographic, neural, quantum
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Holographic interference
                hologram = mx.sin(pattern * 15) * mx.cos(pattern * 20)
                # Neural network patterns
                neural = mx.tanh(pattern) * mx.sigmoid(pattern * 2)
                # Quantum uncertainty
                quantum = mx.random.normal((batch, heads, seq_len, dim)) * \
                         mx.exp(-mx.abs(pattern) * 0.1)
                pattern = (hologram + neural + quantum) * intensity
                
            else:
                pattern = mx.zeros((batch, heads, seq_len, dim))
            
            # Apply temporal style to K and V
            k_new = k + pattern * 0.4
            v_new = v + pattern * 0.6
            
            return q, k_new, v_new
        return q, k, v
    return hook

def main():
    print("⏰ Advanced Test: Temporal Styles")
    print("==" * 30)
    
    # Configuration
    base_prompt = "a portrait of a noble person in their environment"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    # Temporal progression across blocks (ancient to future)
    temporal_config = {
        "down_0": ("ancient", 0.9, "Ancient civilizations"),
        "down_1": ("medieval", 0.8, "Medieval period"),
        "down_2": ("renaissance", 0.7, "Renaissance era"),
        "mid": ("industrial", 0.8, "Industrial revolution"),
        "up_0": ("art_deco", 0.7, "Art Deco period"),
        "up_1": ("digital", 0.8, "Digital age"),
        "up_2": ("cyberpunk", 0.9, "Cyberpunk future")
    }
    
    # Create output directory
    output_dir = Path("artifacts/images/advanced_artistic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📝 Base Prompt: '{base_prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    print("\n⏰ Temporal Progression (Ancient → Future):")
    for block, (era, intensity, desc) in temporal_config.items():
        print(f"   {block}: {desc} @ {intensity*100:.0f}%")
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Clear any existing hooks
    attn_scores.KV_REGISTRY.clear()
    
    # Register temporal style hooks
    print("\n⏰ Registering temporal style hooks...")
    for block, (era, intensity, desc) in temporal_config.items():
        hook = create_temporal_style_hook(era, intensity)
        attn_scores.KV_REGISTRY.set(block, hook)
        print(f"   ⏰ Applied {era} era to {block}")
    
    # Generate with temporal styles
    print("\n🎨 Generating time-traveling artwork...")
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
    output_path = output_dir / "temporal_styles.png"
    pil_img.save(output_path)
    
    print(f"\n✅ Saved temporal styles image: {output_path}")
    print("📊 Expected: Portrait blending multiple time periods")
    print("   - Ancient weathering and stone textures")
    print("   - Medieval gothic elements")
    print("   - Renaissance proportions")
    print("   - Industrial mechanical aspects")
    print("   - Art Deco geometry")
    print("   - Digital pixelation")
    print("   - Cyberpunk holographics")
    print("💡 This proves we can travel through time in a single image!")
    
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
    output_path_baseline = output_dir / "temporal_styles_baseline.png"
    pil_img_baseline.save(output_path_baseline)
    
    print(f"✅ Saved baseline image: {output_path_baseline}")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n🎉 Temporal styles test complete!")

if __name__ == "__main__":
    main()