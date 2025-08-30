#!/usr/bin/env python3
"""Test with MUCH stronger injection."""

import sys
sys.path.insert(0, 'src/adapters/mlx/mlx-examples/stable_diffusion')

# CRITICAL: Import and enable hooks BEFORE importing model!
from stable_diffusion import attn_scores
attn_scores.enable_kv_hooks(True)

from corpus_mlx import CorePulse
from stable_diffusion import StableDiffusionXL
from pathlib import Path
from PIL import Image
import mlx.core as mx
import numpy as np

def save_image(img_array, path):
    """Save MLX array as image."""
    img = np.array(img_array)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    

def test_stronger():
    """Test with much stronger injection."""
    
    print("Loading SDXL-Turbo...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    corepulse = CorePulse(model)
    
    output_dir = Path("examples/output/stronger_injection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prompts = [
        ("red apple", None),
        ("red apple", "yellow banana"),  
        ("red apple", "orange citrus fruit"),
        ("red apple", "purple grape cluster")
    ]
    
    for i, (base_prompt, inject_prompt) in enumerate(prompts):
        corepulse.clear()
        
        if inject_prompt:
            print(f"\n{i}. Injecting '{inject_prompt}' with strength 0.9...")
            corepulse.add_injection(
                prompt=inject_prompt,
                strength=0.9,  # MUCH stronger!
                blocks=["down_0", "down_1", "mid", "up_0", "up_1"]  # ALL blocks
            )
        else:
            print(f"\n{i}. Baseline: '{base_prompt}'")
        
        latents = model.generate_latents(
            base_prompt,
            num_steps=4,  # More steps
            cfg_weight=0.0,
            seed=42
        )
        
        for j, x in enumerate(latents):
            if j == 3:  # Final
                img = model.decode(x)
                filename = f"{i}_{inject_prompt.replace(' ', '_') if inject_prompt else 'baseline'}.png"
                save_image(img[0], output_dir / filename)
                print(f"   ✅ Saved: {filename}")
    
    print("\n✨ Done! Check examples/output/stronger_injection/")
    

if __name__ == "__main__":
    test_stronger()