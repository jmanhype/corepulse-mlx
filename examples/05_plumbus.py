#!/usr/bin/env python3
"""
PLUMBUS FACTORY - How they make plumbuses with token manipulation!
Everyone has a plumbus in their home!
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


def plumbus_factory():
    """First they take the dinglebop..."""
    
    print("=" * 60)
    print("🔧 PLUMBUS FACTORY 🔧")
    print("How they do it: The definitive manufacturing process")
    print("=" * 60)
    
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    corepulse = CorePulse(model)
    
    # The famous plumbus manufacturing steps
    base_prompt = "strange alien household tool device, pink organic mechanical"
    
    output_dir = Path("examples/output/plumbus")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manufacturing_steps = [
        ("Step 1: Take the dinglebop", 
         {"amplify": 2.0, "blocks": ["down_0", "down_1"]},
         "pink blob with protrusions"),
        
        ("Step 2: Smooth it out with schleem", 
         {"suppress": 0.3, "smooth": True},
         "wet glossy surface, smooth"),
        
        ("Step 3: Push through the grumbo", 
         {"chaos": 0.8, "distort": True},
         "twisted organic shape, warped"),
        
        ("Step 4: Rub with fleeb juice", 
         {"inject": "shiny metallic juice coating", "strength": 0.4},
         "metallic sheen, wet surface"),
        
        ("Step 5: Cut the fleeb", 
         {"remove_tokens": (2, 4), "invert": True},
         "segmented parts, cut sections"),
        
        ("Step 6: Final plumbus", 
         {"progressive": True, "perfect": True},
         "finished plumbus product, household item")
    ]
    
    for i, (step_name, config, injection_prompt) in enumerate(manufacturing_steps):
        print(f"\n🔧 {step_name}...")
        
        corepulse.clear()
        
        # Apply different techniques for each manufacturing step
        if "amplify" in config:
            corepulse.amplify(
                strength=config["amplify"],
                blocks=config.get("blocks", ["mid", "up_0"])
            )
        
        if "suppress" in config:
            corepulse.suppress(factor=config["suppress"])
        
        if "chaos" in config:
            corepulse.chaos(intensity=config["chaos"])
        
        if "inject" in config:
            corepulse.add_injection(
                prompt=config["inject"],
                strength=config.get("strength", 0.3),
                blocks=["mid", "up_0", "up_1"]
            )
        
        if "remove_tokens" in config:
            corepulse.remove_tokens(token_range=config["remove_tokens"])
        
        if "invert" in config:
            corepulse.invert(blocks=["up_1", "up_2"])
        
        if "progressive" in config:
            # Final assembly with progressive strength
            corepulse.progressive_strength({
                "down_0": 0.8,
                "down_1": 0.9,
                "down_2": 1.0,
                "mid": 1.2,
                "up_0": 1.5,
                "up_1": 1.8,
                "up_2": 2.0
            })
        
        # Combine base prompt with step-specific details
        full_prompt = f"{base_prompt}, {injection_prompt}"
        
        latents = model.generate_latents(
            full_prompt,
            num_steps=4,
            cfg_weight=0.0,
            seed=6669  # Plumbus seed
        )
        
        for j, x in enumerate(latents):
            if j == 3:
                img = model.decode(x)
                save_image(img[0], output_dir / f"plumbus_step_{i+1}.png")
                print(f"   ✅ Manufacturing step complete!")
    
    print("\n🔧 PLUMBUS COMPLETE! 🔧")
    print("Everyone has a plumbus in their home.")
    print("First they take the dinglebop, and they smooth it out with a bunch of schleem...")
    

def save_image(img_array, path):
    img_array = (img_array * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(path)


if __name__ == "__main__":
    plumbus_factory()