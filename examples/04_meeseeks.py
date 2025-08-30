#!/usr/bin/env python3
"""
MEESEEKS BOX - Summon helpers with different attention heads!
Each Meeseeks uses different attention head isolation.
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


def summon_meeseeks():
    """I'm Mr. Meeseeks! Look at me!"""
    
    print("=" * 60)
    print("📦 MEESEEKS BOX ACTIVATED 📦")
    print("Summoning different Meeseeks with attention head isolation...")
    print("=" * 60)
    
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    corepulse = CorePulse(model)
    
    prompt = "blue humanoid creature helper, simple character, look at me"
    
    output_dir = Path("examples/output/meeseeks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Different Meeseeks with different attention configurations
    meeseeks_configs = [
        ("Original", None, "Normal Meeseeks"),
        ("Stickler", [0, 1], "Only first 2 attention heads - detail focused"),
        ("Kirkland", [4, 5, 6, 7], "Middle attention heads - generic brand"),
        ("Fancy", [8, 9, 10, 11, 12, 13, 14, 15], "Upper attention heads - sophisticated"),
        ("Chaotic", [0, 2, 4, 6, 8, 10, 12, 14], "Every other head - existence is pain!")
    ]
    
    for i, (name, heads, description) in enumerate(meeseeks_configs):
        print(f"\n📦 Summoning {name} Meeseeks...")
        print(f"   {description}")
        
        corepulse.clear()
        
        if heads:
            # Isolate specific attention heads
            corepulse.isolate_attention_heads(
                head_indices=heads,
                blocks=["mid", "up_0", "up_1"]
            )
            
            # Add some variation based on the type
            if name == "Chaotic":
                corepulse.chaos(intensity=1.0, blocks=["up_1", "up_2"])
            elif name == "Fancy":
                corepulse.amplify(strength=1.5, blocks=["up_0", "up_1"])
            elif name == "Stickler":
                corepulse.amplify(strength=2.0, blocks=["down_1", "down_2"])
        
        latents = model.generate_latents(
            prompt,
            num_steps=4,
            cfg_weight=0.0,
            seed=1996 + i  # Different seed per Meeseeks
        )
        
        for j, x in enumerate(latents):
            if j == 3:
                img = model.decode(x)
                save_image(img[0], output_dir / f"meeseeks_{name.lower()}.png")
                print(f"   ✅ {name} Meeseeks: 'I'm Mr. Meeseeks! Look at me!'")
    
    print("\n📦 CAN DO! 📦")
    print("All Meeseeks summoned! Remember: Existence is pain!")
    

def save_image(img_array, path):
    img_array = (img_array * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(path)


if __name__ == "__main__":
    summon_meeseeks()