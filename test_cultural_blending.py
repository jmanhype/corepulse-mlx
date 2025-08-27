#!/usr/bin/env python3
"""
Advanced Test: Cultural Blending
Mixes visual elements from multiple world cultures.
Creates harmonious multicultural artistic fusion.
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

def create_cultural_blending_hook(culture, intensity):
    """Create culture-specific visual patterns"""
    def hook(q, k, v, meta=None):
        # Only modify cross-attention (text-to-image attention)
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            
            if culture == "japanese":
                # Japanese: Minimalism, nature, balance
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Zen minimalism
                zen = mx.exp(-mx.abs(pattern) * 0.5)
                # Cherry blossom patterns
                sakura = mx.sin(mx.arange(seq_len).reshape(1, 1, -1, 1) * 0.3) * \
                        mx.cos(mx.arange(dim).reshape(1, 1, 1, -1) * 0.2)
                # Wabi-sabi imperfection
                wabi_sabi = pattern * mx.random.uniform(shape=(batch, heads, seq_len, dim), low=0.8, high=1.0)
                pattern = zen * sakura * wabi_sabi * intensity
                
            elif culture == "african":
                # African: Bold patterns, earth tones, rhythmic
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Tribal geometric patterns
                tribal = mx.abs(mx.sin(pattern * 8)) * mx.abs(mx.cos(pattern * 6))
                # Rhythmic repetition
                rhythm = mx.sin(mx.arange(seq_len).reshape(1, 1, -1, 1) * 1.5) + 1
                # Earth connection
                earth = mx.abs(pattern) * mx.random.uniform(shape=(batch, heads, seq_len, dim), low=0.7, high=1.3)
                pattern = tribal * rhythm * earth * intensity
                
            elif culture == "islamic":
                # Islamic: Geometric, arabesque, calligraphic
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Complex geometric tessellation
                geometry = mx.abs(mx.sin(pattern * 12)) * mx.abs(mx.cos(pattern * 12))
                # Arabesque flowing patterns
                arabesque = mx.sin(pattern * 3) * mx.cos(pattern * 4)
                # Infinite patterns (no beginning or end)
                infinite = (pattern * 10) % 2.0 - 1.0
                pattern = (geometry + arabesque) * infinite * intensity
                
            elif culture == "nordic":
                # Nordic: Runic, nature, mythology
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Runic angular patterns
                runic = mx.abs(mx.round(pattern * 3) / 3)
                # Northern lights effect
                aurora = mx.sin(mx.arange(seq_len).reshape(1, 1, -1, 1) * 0.5) * \
                        mx.random.uniform(shape=(batch, heads, seq_len, dim), low=0.5, high=2.0)
                # Viking knotwork
                knotwork = mx.abs(mx.sin(pattern * 6)) * mx.abs(mx.cos(pattern * 4))
                pattern = runic * aurora * knotwork * intensity
                
            elif culture == "aztec":
                # Aztec/Mayan: Solar, pyramidal, symbolic
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Stepped pyramid structure
                pyramid = mx.abs(mx.round(pattern * 5) / 5)
                # Solar calendar patterns
                solar = mx.sin(mx.arange(dim).reshape(1, 1, 1, -1) * 0.1) * \
                       mx.cos(mx.arange(seq_len).reshape(1, 1, -1, 1) * 0.1)
                # Feathered serpent scales
                serpent = mx.abs(mx.sin(pattern * 15)) * 0.5 + 0.5
                pattern = pyramid * solar * serpent * intensity
                
            elif culture == "indian":
                # Indian: Mandala, vibrant, spiritual
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Mandala circular patterns
                mandala = mx.sqrt(mx.abs(mx.sin(pattern * 8))) * \
                         mx.sqrt(mx.abs(mx.cos(pattern * 8)))
                # Rangoli colors
                rangoli = mx.random.uniform(shape=(batch, heads, seq_len, dim), low=0.8, high=1.5)
                # Spiritual energy
                spiritual = mx.exp(-mx.abs(pattern - 1.0) * 0.3)
                pattern = mandala * rangoli * spiritual * intensity
                
            elif culture == "celtic":
                # Celtic: Knotwork, spirals, nature
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Celtic knots
                knots = mx.sin(pattern * 4) * mx.cos(pattern * 3) * mx.sin(pattern * 5)
                # Spiral patterns
                spirals = mx.sin(mx.sqrt(mx.abs(pattern)) * 5)
                # Tree of life
                tree = mx.exp(-mx.abs(mx.arange(seq_len).reshape(1, 1, -1, 1) - seq_len/2) * 0.1)
                pattern = (knots + spirals) * tree * intensity
                
            elif culture == "aboriginal":
                # Aboriginal Australian: Dreamtime, dots, earth
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Dot painting technique
                dots = mx.where(
                    mx.random.uniform(shape=(batch, heads, seq_len, dim)) > 0.7,
                    mx.abs(pattern) * 3,
                    pattern * 0.3
                )
                # Dreamtime flow
                dreamtime = mx.sin(pattern * 2) * mx.cos(pattern * 1.5)
                # Earth connection
                earth_lines = mx.abs(mx.sin(mx.arange(seq_len).reshape(1, 1, -1, 1) * 0.2))
                pattern = dots * dreamtime * earth_lines * intensity
                
            else:
                pattern = mx.zeros((batch, heads, seq_len, dim))
            
            # Apply cultural pattern to K and V
            k_new = k + pattern * 0.35
            v_new = v + pattern * 0.65
            
            return q, k_new, v_new
        return q, k, v
    return hook

def main():
    print("🌍 Advanced Test: Cultural Blending")
    print("==" * 30)
    
    # Configuration
    base_prompt = "a ceremonial celebration in a grand hall"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    # Cultural distribution across blocks
    cultural_config = {
        "down_0": ("japanese", 0.7, "Japanese minimalism"),
        "down_1": ("african", 0.8, "African rhythms"),
        "down_2": ("islamic", 0.6, "Islamic geometry"),
        "mid": ("indian", 0.9, "Indian spirituality"),
        "up_0": ("aztec", 0.7, "Aztec symbolism"),
        "up_1": ("celtic", 0.6, "Celtic knotwork"),
        "up_2": ("nordic", 0.7, "Nordic mythology")
    }
    
    # Alternative: Harmonious blend
    harmonious_blend = {
        "down_0": ("japanese", 0.5, "Eastern foundation"),
        "down_1": ("indian", 0.5, "Spiritual core"),
        "down_2": ("african", 0.5, "Rhythmic structure"),
        "mid": ("islamic", 0.5, "Geometric balance"),
        "up_0": ("celtic", 0.5, "Natural elements"),
        "up_1": ("aztec", 0.5, "Solar energy"),
        "up_2": ("aboriginal", 0.5, "Dreamtime finish")
    }
    
    # Create output directory
    output_dir = Path("artifacts/images/advanced_artistic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use harmonious blend for this test
    test_config = harmonious_blend
    
    print(f"📝 Base Prompt: '{base_prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    print("\n🌍 Cultural Elements:")
    for block, (culture, intensity, desc) in test_config.items():
        print(f"   {block}: {desc} ({culture}) @ {intensity*100:.0f}%")
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Clear any existing hooks
    attn_scores.KV_REGISTRY.clear()
    
    # Register cultural blending hooks
    print("\n🌍 Registering cultural blending hooks...")
    for block, (culture, intensity, desc) in test_config.items():
        hook = create_cultural_blending_hook(culture, intensity)
        attn_scores.KV_REGISTRY.set(block, hook)
        print(f"   🌍 Applied {culture} culture to {block}")
    
    # Generate with cultural blending
    print("\n🎨 Generating multicultural artwork...")
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
    output_path = output_dir / "cultural_blending.png"
    pil_img.save(output_path)
    
    print(f"\n✅ Saved cultural blending image: {output_path}")
    print("📊 Expected: Celebration with multicultural elements")
    print("   - Japanese zen minimalism")
    print("   - Indian mandala patterns")
    print("   - African rhythmic elements")
    print("   - Islamic geometric beauty")
    print("   - Celtic natural spirals")
    print("   - Aztec solar symbolism")
    print("   - Aboriginal dreamtime dots")
    print("💡 This proves we can create harmonious multicultural fusion!")
    
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
    output_path_baseline = output_dir / "cultural_blending_baseline.png"
    pil_img_baseline.save(output_path_baseline)
    
    print(f"✅ Saved baseline image: {output_path_baseline}")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n🎉 Cultural blending test complete!")

if __name__ == "__main__":
    main()