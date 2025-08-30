#!/usr/bin/env python3
"""Test the fixed showcase generation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from corpus_mlx import create_semantic_wrapper
from PIL import Image
import mlx.core as mx
import numpy as np

def generate_images_for_prompt(model, prompt, negative="blurry, ugly", seed=42, num_steps=15):
    """Generate image for a given prompt."""
    latents = None
    for step in model.generate_latents(
        prompt,
        negative_text=negative,
        num_steps=num_steps,
        cfg_weight=7.5,
        seed=seed
    ):
        latents = step
    
    # Decode
    if hasattr(model, 'autoencoder'):
        images = model.autoencoder.decode(latents)
    elif hasattr(model, 'sd'):
        images = model.sd.autoencoder.decode(latents)
    else:
        raise AttributeError(f"Cannot find autoencoder in {type(model)}")
        
    img = images[0]
    img = mx.clip(img, -1, 1)
    img = ((img + 1) * 127.5).astype(mx.uint8)
    return np.array(img)

def test_apple_banana():
    """Test apple to banana replacement."""
    print("🍎➡️🍌 Testing Apple → Banana replacement")
    
    # Create wrapper
    wrapper = create_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
    prompt = "a red apple on a wooden table"
    
    print("1. Generating BEFORE (apple)...")
    before_img = generate_images_for_prompt(wrapper.wrapper, prompt)
    Image.fromarray(before_img).save("test_before.png")
    print("✅ Saved test_before.png")
    
    print("2. Adding replacement and generating AFTER (banana)...")
    wrapper.add_replacement("apple", "banana")
    wrapper.enable()
    
    after_img = generate_images_for_prompt(wrapper.wrapper, prompt)
    Image.fromarray(after_img).save("test_after.png")
    print("✅ Saved test_after.png")
    
    # Compare
    diff = np.mean(np.abs(before_img.astype(float) - after_img.astype(float)))
    print(f"\n📊 Pixel difference: {diff}")
    
    if diff > 5.0:  # Allow some threshold for actual changes
        print("✅ SUCCESS: Images are different - replacement worked!")
    else:
        print("❌ FAILURE: Images are too similar - replacement failed!")

if __name__ == "__main__":
    test_apple_banana()