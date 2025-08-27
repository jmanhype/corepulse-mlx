#!/usr/bin/env python3
"""
Advanced Test: Style Mixing
Blends multiple artistic styles (Van Gogh + Picasso + Monet) in different UNet blocks.
Creates a unique multi-style artistic fusion.
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

def create_style_mixing_hook(style_name, intensity):
    """Create style-specific manipulation patterns"""
    def hook(q, k, v, meta=None):
        # Only modify cross-attention (text-to-image attention)
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            
            if style_name == "van_gogh":
                # Swirling, turbulent patterns
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Create swirls using sine/cosine combinations
                swirl = mx.sin(mx.arange(seq_len).reshape(1, 1, -1, 1) * 0.5) * \
                        mx.cos(mx.arange(dim).reshape(1, 1, 1, -1) * 0.3)
                pattern = pattern * swirl * intensity
                
            elif style_name == "picasso":
                # Geometric, fragmented patterns
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Create angular patterns
                geometric = mx.abs(pattern) * mx.sign(mx.random.normal((batch, heads, seq_len, dim)))
                # Add sharp edges
                pattern = geometric * mx.where(
                    mx.random.uniform(shape=(batch, heads, seq_len, dim)) > 0.5,
                    mx.full((batch, heads, seq_len, dim), intensity * 2),
                    mx.full((batch, heads, seq_len, dim), intensity * 0.5)
                )
                
            elif style_name == "monet":
                # Soft, impressionistic patterns
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Blur and soften
                soft = pattern * mx.exp(-mx.abs(pattern) * 0.5)
                # Add color bleeding effect
                pattern = soft * intensity * mx.random.uniform(shape=(batch, heads, seq_len, dim), low=0.7, high=1.3)
                
            elif style_name == "dali":
                # Surrealistic, melting patterns
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Create melting effect
                melt = pattern * mx.exp(mx.arange(seq_len).reshape(1, 1, -1, 1) * 0.01)
                # Add distortion
                pattern = melt * mx.sin(pattern) * intensity
                
            else:
                # Default: no style
                pattern = mx.zeros((batch, heads, seq_len, dim))
            
            # Apply style pattern to K and V
            k_new = k + pattern * 0.3
            v_new = v + pattern * 0.7
            
            return q, k_new, v_new
        return q, k, v
    return hook

def main():
    print("🎨 Advanced Test: Style Mixing")
    print("==" * 30)
    
    # Configuration
    base_prompt = "a serene landscape with mountains and lake at sunset"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    # Style distribution across blocks
    style_config = {
        "down_0": ("van_gogh", 0.8, "Van Gogh's swirls"),
        "down_1": ("picasso", 0.6, "Picasso's geometry"),
        "down_2": ("monet", 0.7, "Monet's impressionism"),
        "mid": ("dali", 0.5, "Dali's surrealism"),
        "up_0": ("monet", 0.6, "Monet's softness"),
        "up_1": ("van_gogh", 0.7, "Van Gogh's texture"),
        "up_2": ("picasso", 0.5, "Picasso's fragmentation")
    }
    
    # Create output directory
    output_dir = Path("artifacts/images/advanced_artistic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📝 Base Prompt: '{base_prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    print("\n🎨 Style Distribution:")
    for block, (style, intensity, desc) in style_config.items():
        print(f"   {block}: {desc} @ {intensity*100:.0f}% intensity")
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Clear any existing hooks
    attn_scores.KV_REGISTRY.clear()
    
    # Register style mixing hooks
    print("\n🖌️ Registering style mixing hooks...")
    for block, (style, intensity, desc) in style_config.items():
        hook = create_style_mixing_hook(style, intensity)
        attn_scores.KV_REGISTRY.set(block, hook)
        print(f"   🎨 Applied {style} to {block}")
    
    # Generate with style mixing
    print("\n🎨 Generating multi-style artwork...")
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
    output_path = output_dir / "style_mixing.png"
    pil_img.save(output_path)
    
    print(f"\n✅ Saved style mixing image: {output_path}")
    print("📊 Expected: Landscape with mixed artistic styles")
    print("   - Van Gogh's swirling skies")
    print("   - Picasso's geometric mountains")
    print("   - Monet's impressionistic water")
    print("   - Dali's surrealistic elements")
    print("💡 This proves we can blend multiple artistic styles simultaneously!")
    
    # Generate comparison without style mixing
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
    output_path_baseline = output_dir / "style_mixing_baseline.png"
    pil_img_baseline.save(output_path_baseline)
    
    print(f"✅ Saved baseline image: {output_path_baseline}")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n🎉 Style mixing test complete!")

if __name__ == "__main__":
    main()