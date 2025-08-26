#!/usr/bin/env python3
"""Simple test to verify we can hook into the generation process."""

import sys
import os
sys.path.append('/Users/speed/Downloads/corpus-mlx/src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import StableDiffusion
from tqdm import tqdm
import mlx.core as mx


def test_basic_generation():
    """Test basic generation to verify everything works."""
    print("\n=== BASIC GENERATION TEST ===")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    prompt = "a cute dog"
    print(f"Generating: {prompt}")
    
    latents = sd.generate_latents(prompt, n_images=1, num_steps=20, cfg_weight=7.5, seed=42)
    for x_t in tqdm(latents, total=20):
        final_latents = x_t
    
    image = sd.decode(final_latents)[0]
    
    # Save the image
    import numpy as np
    from PIL import Image
    
    if not isinstance(image, Image.Image):
        image = Image.fromarray((np.array(image) * 255).astype(np.uint8))
    
    save_dir = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme"
    os.makedirs(save_dir, exist_ok=True)
    
    image.save(f"{save_dir}/basic_test.png")
    print("✅ Basic generation complete")


def test_different_prompts():
    """Test generating two different images to verify they're different."""
    print("\n=== DIFFERENT PROMPTS TEST ===")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    # Generate dog
    print("Generating dog...")
    latents = sd.generate_latents("a cute dog", n_images=1, num_steps=20, cfg_weight=7.5, seed=42)
    for x_t in tqdm(latents, total=20):
        dog_latents = x_t
    dog_image = sd.decode(dog_latents)[0]
    
    # Generate cat  
    print("Generating cat...")
    latents = sd.generate_latents("a white fluffy cat", n_images=1, num_steps=20, cfg_weight=7.5, seed=42)
    for x_t in tqdm(latents, total=20):
        cat_latents = x_t
    cat_image = sd.decode(cat_latents)[0]
    
    # Save both
    import numpy as np
    from PIL import Image
    
    if not isinstance(dog_image, Image.Image):
        dog_image = Image.fromarray((np.array(dog_image) * 255).astype(np.uint8))
    if not isinstance(cat_image, Image.Image):
        cat_image = Image.fromarray((np.array(cat_image) * 255).astype(np.uint8))
    
    save_dir = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme"
    os.makedirs(save_dir, exist_ok=True)
    
    dog_image.save(f"{save_dir}/test_dog.png")
    cat_image.save(f"{save_dir}/test_cat.png")
    
    print("✅ Generated dog and cat images")
    print("If they look different, text conditioning is working!")


def main():
    print("\n" + "="*50)
    print("SIMPLE INJECTION TESTS")
    print("="*50)
    
    # Test 1: Basic generation
    test_basic_generation()
    
    # Test 2: Different prompts
    test_different_prompts()
    
    print("\n" + "="*50)
    print("TESTS COMPLETE!")
    print("="*50)


if __name__ == "__main__":
    main()