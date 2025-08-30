#!/usr/bin/env python3
"""Test the TRUE embedding injection functionality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from corpus_mlx import create_true_semantic_wrapper
from PIL import Image
import mlx.core as mx
import numpy as np

def generate_and_save(sd_model, prompt, filename):
    """Generate image and save it."""
    print(f"Generating: '{prompt}' -> {filename}")
    
    latents = None
    for step in sd_model.generate_latents(
        prompt,
        negative_text="blurry, ugly",
        num_steps=15,
        cfg_weight=7.5,
        seed=42
    ):
        latents = step
    
    # Decode
    images = sd_model.autoencoder.decode(latents)
    img = images[0]
    img = mx.clip(img, -1, 1)
    img = ((img + 1) * 127.5).astype(mx.uint8)
    img_array = np.array(img)
    
    Image.fromarray(img_array).save(filename)
    print(f"✅ Saved: {filename}")
    return img_array

def test_embedding_injection():
    """Test TRUE embedding injection."""
    print("🧠 TESTING TRUE EMBEDDING INJECTION")
    print("=" * 50)
    
    # Create wrapper
    print("Creating TRUE semantic wrapper...")
    wrapper = create_true_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
    
    test_prompt = "a orange cat playing in a garden"
    print(f"Test prompt: '{test_prompt}'")
    
    # Test 1: Generate without injection
    print("\n1. Testing WITHOUT injection:")
    before_img = generate_and_save(wrapper.sd, test_prompt, "embedding_before.png")
    
    # Test 2: Add injection and test
    print("\n2. Adding injection...")
    wrapper.add_replacement("cat", "golden retriever dog", weight=1.0)
    wrapper.injector.enable_for_prompt(test_prompt)
    
    print("\n3. Testing WITH injection:")
    after_img = generate_and_save(wrapper.sd, test_prompt, "embedding_after.png")
    
    # Compare
    diff = np.mean(np.abs(before_img.astype(float) - after_img.astype(float)))
    print(f"\n📊 Pixel difference: {diff}")
    
    if diff > 5.0:
        print("✅ SUCCESS: Images are different - embedding injection worked!")
    else:
        print("❌ FAILURE: Images are too similar - embedding injection failed!")
        print("🔍 The TRUE embedding injection is NOT working!")

if __name__ == "__main__":
    test_embedding_injection()