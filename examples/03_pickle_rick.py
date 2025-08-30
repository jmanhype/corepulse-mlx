#!/usr/bin/env python3
"""
PICKLE RICK - Progressive transformation demonstration!
Watch as we progressively transform from human to pickle!
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


def pickle_rick_transformation():
    """I turned myself into a pickle! I'm Pickle Rick!"""
    
    print("=" * 60)
    print("🥒 PICKLE RICK TRANSFORMATION 🥒")
    print("Progressive transformation from scientist to pickle...")
    print("=" * 60)
    
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    corepulse = CorePulse(model)
    
    base_prompt = "mad scientist in laboratory"
    pickle_prompt = "green pickle with face, vegetable"
    
    output_dir = Path("examples/output/pickle_rick")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Progressive transformation steps
    transformations = [
        (0.0, "Human form"),
        (0.2, "Starting transformation"),
        (0.4, "Half transformed"),
        (0.6, "Mostly pickle"),
        (0.8, "Almost complete"),
        (1.0, "PICKLE RICK!")
    ]
    
    for strength, stage in transformations:
        print(f"\n🥒 {stage} (strength: {strength})...")
        
        corepulse.clear()
        
        if strength > 0:
            # Progressive injection of pickle characteristics
            corepulse.add_injection(
                prompt=pickle_prompt,
                strength=strength * 0.5,
                blocks=["down_1", "down_2", "mid"]
            )
            
            # Progressively suppress human features
            corepulse.progressive_strength({
                "down_0": 1.0 - (strength * 0.3),
                "down_1": 1.0 - (strength * 0.4),
                "down_2": 1.0 - (strength * 0.5),
                "mid": 1.0,
                "up_0": 1.0 + (strength * 0.3),
                "up_1": 1.0 + (strength * 0.5),
                "up_2": 1.0 + (strength * 0.7)
            })
        
        # Mix prompts based on transformation level
        mixed_prompt = f"{base_prompt}, {pickle_prompt}" if strength > 0.5 else base_prompt
        
        latents = model.generate_latents(
            mixed_prompt,
            num_steps=4,
            cfg_weight=0.0,
            seed=3327  # Random pickle seed
        )
        
        for i, x in enumerate(latents):
            if i == 3:
                img = model.decode(x)
                save_image(img[0], output_dir / f"transformation_{int(strength*100):03d}.png")
                print(f"   ✅ {stage} saved!")
    
    print("\n🥒 I'M PICKLE RICK! 🥒")
    print("Boom! Big reveal! I turned myself into a pickle!")
    

def save_image(img_array, path):
    img_array = (img_array * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(path)


if __name__ == "__main__":
    pickle_rick_transformation()