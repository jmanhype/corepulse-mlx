#!/usr/bin/env python3
"""
GET SCHWIFTY - Show the Cromulons what we got!
Demonstrates progressive chaos injection for psychedelic effects.
"""

import mlx.core as mx
import sys
from pathlib import Path
import numpy as np
import PIL.Image

sys.path.append('src')
sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import attn_scores
attn_scores.enable_kv_hooks(True)

from stable_diffusion import StableDiffusionXL
from corpus_mlx import CorePulse


def get_schwifty():
    """Time to get schwifty in here!"""
    
    print("=" * 60)
    print("🎵 SHOW ME WHAT YOU GOT! 🎵")
    print("Getting Schwifty with Progressive Chaos...")
    print("=" * 60)
    
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    corepulse = CorePulse(model)
    
    prompt = "giant floating head in the sky judging Earth, cosmic horror, surreal"
    
    output_dir = Path("examples/output/cromulons")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Progressive chaos levels - getting schwiftier
    chaos_levels = [0, 0.5, 1.0, 2.0, 5.0]
    
    for i, chaos in enumerate(chaos_levels):
        print(f"\n🎸 Chaos Level {chaos}...")
        
        corepulse.clear()
        if chaos > 0:
            corepulse.chaos(intensity=chaos, blocks=["mid", "up_0", "up_1"])
            corepulse.progressive_strength({
                "down_0": 1.0,
                "down_1": 1.0 + chaos * 0.2,
                "mid": 1.0 + chaos * 0.5,
                "up_0": 1.0 + chaos,
                "up_1": 1.0 + chaos * 1.5,
                "up_2": 1.0 + chaos * 2.0
            })
        
        latents = model.generate_latents(
            prompt,
            num_steps=4,
            cfg_weight=0.0,
            seed=42
        )
        
        for j, x in enumerate(latents):
            if j == 3:
                img = model.decode(x)
                save_image(img[0], output_dir / f"schwifty_{i}_chaos_{chaos}.png")
                print(f"   ✅ Saved with chaos {chaos}")
    
    print("\n🎵 I LIKE WHAT YOU GOT! GOOD JOB! 🎵")
    

def save_image(img_array, path):
    img_array = (img_array * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(path)


if __name__ == "__main__":
    get_schwifty()