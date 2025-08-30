#!/usr/bin/env python3
"""
PORTAL GUN - Create interdimensional portals with injection!
Blend multiple realities using prompt injection.
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


def portal_gun():
    """Create portals between dimensions!"""
    
    print("=" * 60)
    print("🌀 PORTAL GUN ACTIVATED 🌀")
    print("Opening portals to alternate dimensions...")
    print("=" * 60)
    
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    corepulse = CorePulse(model)
    
    base_reality = "a scientist in a garage laboratory with inventions"
    
    # Different dimensions to blend
    dimensions = [
        ("C-137", "normal reality"),
        ("Cronenberg", "body horror, flesh mutations, grotesque"),
        ("Doofus", "candy kingdom, pastel colors, cute"),
        ("Evil", "dark dystopian, sinister atmosphere"),
        ("Blender", "mixed realities colliding, dimensional rift")
    ]
    
    output_dir = Path("examples/output/portal_gun")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (dimension, injection) in enumerate(dimensions):
        print(f"\n🌀 Dimension {dimension}...")
        
        corepulse.clear()
        
        if i == 0:  # C-137 (normal)
            pass
        elif i == 4:  # Blender dimension - mix everything
            corepulse.add_injection(
                prompt="body horror grotesque",
                strength=0.2,
                blocks=["down_1", "down_2"]
            )
            corepulse.add_injection(
                prompt="candy kingdom pastel cute",
                strength=0.3,
                blocks=["mid", "up_0"]
            )
            corepulse.add_injection(
                prompt="dark dystopian sinister",
                strength=0.25,
                blocks=["up_1", "up_2"]
            )
            corepulse.chaos(intensity=0.5)  # Add some chaos
        else:
            corepulse.add_injection(
                prompt=injection,
                strength=0.4,
                blocks=["mid", "up_0", "up_1", "up_2"]
            )
        
        latents = model.generate_latents(
            base_reality,
            num_steps=4,
            cfg_weight=0.0,
            seed=137  # C-137 reference
        )
        
        for j, x in enumerate(latents):
            if j == 3:
                img = model.decode(x)
                save_image(img[0], output_dir / f"dimension_{dimension}.png")
                print(f"   ✅ Portal to {dimension} opened!")
    
    print("\n🌀 Wubba lubba dub dub! 🌀")
    

def save_image(img_array, path):
    img_array = (img_array * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(path)


if __name__ == "__main__":
    portal_gun()