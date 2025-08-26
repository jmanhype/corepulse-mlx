#!/usr/bin/env python3
"""Test if the seed is causing the issue with prompts."""

import sys
import os
sys.path.append('/Users/speed/Downloads/corpus-mlx/src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import StableDiffusion
from tqdm import tqdm
import mlx.core as mx
import numpy as np
from PIL import Image


def test_seed_hypothesis():
    """Test if different seeds fix the prompt issue."""
    print("\n=== TESTING SEED HYPOTHESIS ===")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    prompt = "a cute dog"
    seeds = [None, 1, 2, 100, 999, 12345]
    
    save_dir = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme"
    os.makedirs(save_dir, exist_ok=True)
    
    for i, seed in enumerate(seeds):
        print(f"\nGenerating with seed={seed}...")
        
        latents = sd.generate_latents(prompt, n_images=1, num_steps=20, cfg_weight=7.5, seed=seed)
        for x_t in tqdm(latents, total=20):
            final_latents = x_t
        
        image = sd.decode(final_latents)[0]
        
        if not isinstance(image, Image.Image):
            image = Image.fromarray((np.array(image) * 255).astype(np.uint8))
        
        seed_name = "random" if seed is None else f"seed_{seed}"
        image.save(f"{save_dir}/dog_{seed_name}.png")
        print(f"✅ Saved dog_{seed_name}.png")


def test_unconditioned_generation():
    """Test generating without text conditioning."""
    print("\n=== TESTING UNCONDITIONED GENERATION ===")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    # Try empty prompt
    print("Generating with empty prompt...")
    latents = sd.generate_latents("", n_images=1, num_steps=20, cfg_weight=1.0, seed=42)
    for x_t in tqdm(latents, total=20):
        final_latents = x_t
    
    image = sd.decode(final_latents)[0]
    
    if not isinstance(image, Image.Image):
        image = Image.fromarray((np.array(image) * 255).astype(np.uint8))
    
    save_dir = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme"
    os.makedirs(save_dir, exist_ok=True)
    image.save(f"{save_dir}/unconditioned.png")
    print("✅ Saved unconditioned.png")


def main():
    print("\n" + "="*60)
    print("TESTING SEED AND CONDITIONING HYPOTHESIS")
    print("="*60)
    
    # Test seeds
    test_seed_hypothesis()
    
    # Test unconditioned
    test_unconditioned_generation()
    
    print("\n" + "="*60)
    print("HYPOTHESIS TESTS COMPLETE!")
    print("Check the generated images to see if different seeds help")
    print("="*60)


if __name__ == "__main__":
    main()