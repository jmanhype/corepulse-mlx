#!/usr/bin/env python3
"""
Generate 5 images of a boy and a cat using different CorePulse techniques.
"""

import mlx.core as mx
import sys
from pathlib import Path
import numpy as np
import PIL.Image

# Add paths
sys.path.append('src')
sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')

# Enable hooks BEFORE importing model
from stable_diffusion import attn_scores
attn_scores.enable_kv_hooks(True)

# Import model and CorePulse
from stable_diffusion import StableDiffusionXL
from corpus_mlx import CorePulse


def generate_boy_cat_images():
    """Generate 5 variations of a boy and cat."""
    
    print("=" * 60)
    print("🎨 Generating 5 Images: Boy and Cat")
    print("=" * 60)
    
    # Load model
    print("\nLoading SDXL-Turbo...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Create CorePulse wrapper
    corepulse = CorePulse(model)
    
    # Base prompt
    prompt = "a happy young boy playing with a fluffy orange cat in a sunny garden, photorealistic"
    
    # Configuration
    num_steps = 4  # SDXL-Turbo optimized for 4 steps
    cfg_weight = 0.0  # SDXL-Turbo doesn't use CFG
    seed = 42
    
    # Output directory
    output_dir = Path("boy_cat_images")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Output: {output_dir}/\n")
    
    # ========== Image 1: Standard (Baseline) ==========
    print("1. Standard generation...")
    corepulse.clear()
    
    latents = model.generate_latents(
        prompt, 
        num_steps=num_steps, 
        cfg_weight=cfg_weight, 
        seed=seed
    )
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    save_image(img[0], output_dir / "1_standard.png")
    print("   ✅ Standard boy and cat\n")
    
    # ========== Image 2: Dreamlike (Amplified) ==========
    print("2. Dreamlike (amplified attention)...")
    corepulse.clear()
    corepulse.amplify(strength=2.0, blocks=["mid", "up_0", "up_1"])
    
    latents = model.generate_latents(
        prompt, 
        num_steps=num_steps, 
        cfg_weight=cfg_weight, 
        seed=seed + 1  # Different seed for variation
    )
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    save_image(img[0], output_dir / "2_dreamlike.png")
    print("   ✅ Dreamlike enhanced version\n")
    
    # ========== Image 3: Cartoon Style (Injection) ==========
    print("3. Cartoon style (with injection)...")
    corepulse.clear()
    corepulse.add_injection(
        prompt="cartoon illustration, anime style, vibrant colors",
        strength=0.4,
        blocks=["mid", "up_0", "up_1", "up_2"]
    )
    
    latents = model.generate_latents(
        prompt, 
        num_steps=num_steps, 
        cfg_weight=cfg_weight, 
        seed=seed + 2
    )
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    save_image(img[0], output_dir / "3_cartoon.png")
    print("   ✅ Cartoon/anime style blend\n")
    
    # ========== Image 4: Focus on Cat (Token manipulation) ==========
    print("4. Cat focus (suppressing 'boy' tokens)...")
    corepulse.clear()
    # Amplify cat, suppress boy
    corepulse.progressive_strength({
        "down_0": 1.0,
        "down_1": 0.5,   # Suppress early layers (structure)
        "down_2": 0.5,
        "mid": 1.5,      # Amplify mid (content)
        "up_0": 2.0,     # Amplify late layers (details)
        "up_1": 2.0,
        "up_2": 1.5
    })
    
    latents = model.generate_latents(
        "a fluffy orange cat playing in a sunny garden with a small boy in background", 
        num_steps=num_steps, 
        cfg_weight=cfg_weight, 
        seed=seed + 3
    )
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    save_image(img[0], output_dir / "4_cat_focus.png")
    print("   ✅ Cat-focused composition\n")
    
    # ========== Image 5: Magical/Fantasy (Multiple techniques) ==========
    print("5. Magical fantasy (combined techniques)...")
    corepulse.clear()
    
    # Add magical elements
    corepulse.add_injection(
        prompt="magical sparkles, fairy tale, ethereal glow, fantasy art",
        strength=0.3,
        blocks=["up_0", "up_1", "up_2"]  # Late blocks for style
    )
    
    # Slight amplification for drama
    corepulse.amplify(strength=1.3, blocks=["mid"])
    
    latents = model.generate_latents(
        prompt + ", magical atmosphere", 
        num_steps=num_steps, 
        cfg_weight=cfg_weight, 
        seed=seed + 4
    )
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    save_image(img[0], output_dir / "5_magical.png")
    print("   ✅ Magical fantasy version\n")
    
    # ========== Summary ==========
    print("=" * 60)
    print("✨ Successfully Generated 5 Images!")
    print("=" * 60)
    print("\n📁 Images saved to:", output_dir.absolute())
    print("\nGenerated variations:")
    print("  1. Standard     - Normal boy and cat")
    print("  2. Dreamlike    - Enhanced/amplified for dreamy effect")
    print("  3. Cartoon      - Anime/illustration style injection")
    print("  4. Cat Focus    - Emphasized cat, de-emphasized boy")
    print("  5. Magical      - Fantasy atmosphere with sparkles")
    print("\n🎨 Each image uses different CorePulse techniques!")
    
    # Clean up
    del model
    del corepulse
    mx.metal.clear_cache()


def save_image(img_array, path):
    """Save image from array."""
    img_array = (img_array * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(path)


if __name__ == "__main__":
    generate_boy_cat_images()