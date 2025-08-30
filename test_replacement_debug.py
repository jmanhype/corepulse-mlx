#!/usr/bin/env python3
"""Debug why semantic replacement isn't working."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from corpus_mlx import create_semantic_wrapper
from PIL import Image
import mlx.core as mx
import numpy as np

def generate_and_save(model, prompt, filename):
    """Generate image and save it."""
    print(f"Generating: '{prompt}' -> {filename}")
    
    latents = None
    for step in model.generate_latents(
        prompt,
        negative_text="blurry, ugly",
        num_steps=15,
        cfg_weight=7.5,
        seed=42
    ):
        latents = step
    
    # Decode
    if hasattr(model, 'autoencoder'):
        images = model.autoencoder.decode(latents)
    elif hasattr(model, 'sd'):
        images = model.sd.autoencoder.decode(latents)
    else:
        raise AttributeError("Cannot find autoencoder")
        
    img = images[0]
    img = mx.clip(img, -1, 1)
    img = ((img + 1) * 127.5).astype(mx.uint8)
    img_array = np.array(img)
    
    Image.fromarray(img_array).save(filename)
    print(f"✅ Saved: {filename}")
    return img_array

def test_text_replacement():
    """Test text replacement step by step."""
    print("🔍 DEBUGGING TEXT REPLACEMENT")
    print("=" * 50)
    
    # Create wrapper
    print("Creating semantic wrapper...")
    wrapper = create_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
    
    # Test prompt
    test_prompt = "a red apple on a wooden table"
    print(f"Original prompt: '{test_prompt}'")
    
    # Test 1: Generate without replacement
    print("\n1. Testing WITHOUT replacement:")
    before_img = generate_and_save(wrapper.wrapper, test_prompt, "debug_before.png")
    
    # Test 2: Add replacement but don't enable
    print("\n2. Adding replacement but NOT enabling:")
    wrapper.add_replacement("apple", "banana")
    print(f"Replacements: {wrapper.replacements}")
    print(f"Active: {wrapper.active}")
    
    without_enable_img = generate_and_save(wrapper.wrapper, test_prompt, "debug_without_enable.png")
    
    # Test 3: Enable replacement
    print("\n3. Enabling replacement:")
    wrapper.enable()
    print(f"Active: {wrapper.active}")
    
    # Test the patched method directly
    print("\n4. Testing prompt modification:")
    modified_prompt, was_replaced = wrapper._apply_replacements(test_prompt)
    print(f"Original: '{test_prompt}'")
    print(f"Modified: '{modified_prompt}'")
    print(f"Was replaced: {was_replaced}")
    
    # Test 5: Generate with replacement enabled
    print("\n5. Generating with replacement ENABLED:")
    after_img = generate_and_save(wrapper.wrapper, test_prompt, "debug_after.png")
    
    # Test 6: Check if images are actually different
    print("\n6. Comparing images:")
    diff = np.mean(np.abs(before_img.astype(float) - after_img.astype(float)))
    print(f"Mean pixel difference: {diff}")
    
    if diff < 1.0:
        print("❌ Images are nearly identical!")
        print("🔍 The replacement is NOT working!")
    else:
        print("✅ Images are different - replacement working!")

if __name__ == "__main__":
    test_text_replacement()