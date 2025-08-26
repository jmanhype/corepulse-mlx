#!/usr/bin/env python3
"""FIXED generation - adding mx.eval() calls like the official script."""

import sys
import os
sys.path.append('/Users/speed/Downloads/corpus-mlx/src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import StableDiffusion
from tqdm import tqdm
import mlx.core as mx
import numpy as np
from PIL import Image


def test_fixed_generation():
    """Test generation with proper mx.eval() calls."""
    print("\n=== TESTING FIXED GENERATION ===")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    prompts = [
        "a cute dog",
        "a white fluffy cat",
        "a red sports car",
        "a beautiful landscape"
    ]
    
    save_dir = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme"
    os.makedirs(save_dir, exist_ok=True)
    
    for prompt in prompts:
        print(f"\nGenerating: '{prompt}'")
        
        # Generate latents - with proper mx.eval() calls like the official script
        latents = sd.generate_latents(
            prompt, 
            n_images=1, 
            num_steps=20, 
            cfg_weight=7.5, 
            seed=42
        )
        
        # THIS IS THE KEY: evaluate each step!
        for x_t in tqdm(latents, total=20):
            mx.eval(x_t)  # ← THE MISSING PIECE!
        
        # Decode
        image = sd.decode(x_t)[0]
        mx.eval(image)  # ← Also evaluate decode
        
        if not isinstance(image, Image.Image):
            image = Image.fromarray((np.array(image) * 255).astype(np.uint8))
        
        safe_name = prompt.replace(" ", "_").replace(",", "")
        image.save(f"{save_dir}/FIXED_{safe_name}.png")
        print(f"✅ Saved FIXED_{safe_name}.png")


def main():
    print("\n" + "="*60)
    print("TESTING FIXED GENERATION WITH MX.EVAL() CALLS")
    print("="*60)
    
    test_fixed_generation()
    
    print("\n" + "="*60)
    print("FIXED GENERATION COMPLETE!")
    print("This should generate ACTUAL dogs and cats!")
    print("="*60)


if __name__ == "__main__":
    main()