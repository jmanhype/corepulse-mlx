#!/usr/bin/env python3
"""Test simplified embedding injection by modifying text embeddings directly."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adapters.stable_diffusion import StableDiffusion
from PIL import Image
import mlx.core as mx
import numpy as np

def generate_and_save(sd_model, prompt, filename, modified_embeddings=None):
    """Generate image with optional embedding modification."""
    print(f"Generating: '{prompt}' -> {filename}")
    
    # Get original embeddings
    original_embeddings = sd_model._get_text_conditioning(prompt)
    
    # Use modified embeddings if provided
    if modified_embeddings is not None:
        print("🧠 Using modified embeddings for generation")
        # Monkey-patch the embeddings
        sd_model._cached_text_conditioning = modified_embeddings
    else:
        sd_model._cached_text_conditioning = original_embeddings
    
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

def test_direct_embedding_replacement():
    """Test direct embedding replacement."""
    print("🧠 TESTING DIRECT EMBEDDING REPLACEMENT")
    print("=" * 50)
    
    # Create model
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base")
    
    original_prompt = "a orange cat playing in a garden"
    replacement_prompt = "a golden retriever dog playing in a garden"
    
    print(f"Original prompt: '{original_prompt}'")
    print(f"Replacement prompt: '{replacement_prompt}'")
    
    # Generate with original embeddings
    print("\n1. Generating with ORIGINAL embeddings...")
    before_img = generate_and_save(sd, original_prompt, "direct_embedding_before.png")
    
    # Get replacement embeddings
    print("\n2. Getting replacement embeddings...")
    replacement_embeddings = sd._get_text_conditioning(replacement_prompt)
    print(f"Original embeddings shape: {sd._get_text_conditioning(original_prompt).shape}")
    print(f"Replacement embeddings shape: {replacement_embeddings.shape}")
    
    # Generate with replacement embeddings but original prompt
    print("\n3. Generating with REPLACEMENT embeddings...")
    after_img = generate_and_save(sd, original_prompt, "direct_embedding_after.png", 
                                modified_embeddings=replacement_embeddings)
    
    # Compare
    diff = np.mean(np.abs(before_img.astype(float) - after_img.astype(float)))
    print(f"\n📊 Pixel difference: {diff}")
    
    if diff > 5.0:
        print("✅ SUCCESS: Direct embedding injection worked!")
        print("🎯 This proves the concept - we can inject embeddings directly")
    else:
        print("❌ FAILURE: Images are too similar")

if __name__ == "__main__":
    test_direct_embedding_replacement()