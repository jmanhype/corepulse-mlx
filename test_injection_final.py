#!/usr/bin/env python3
"""Final test with improved injection matching the working version."""

import sys
sys.path.insert(0, 'src/adapters/mlx/mlx-examples/stable_diffusion')

# CRITICAL: Import and enable hooks BEFORE importing model!
from stable_diffusion import attn_scores
attn_scores.enable_kv_hooks(True)
print("✅ Enabled KV hooks")

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
    

def test_injection_final():
    """Test with direct replacement approach from working version."""
    
    print("\nLoading SDXL-Turbo...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    corepulse = CorePulse(model)
    
    output_dir = Path("examples/output/injection_final")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test cases with very high strength for replacement
    test_cases = [
        ("a red apple on a table", None, "baseline"),
        ("a red apple on a table", "yellow banana fruit", "banana"),
        ("a red apple on a table", "orange citrus fruit", "orange"),
        ("a red apple on a table", "purple grapes cluster", "grapes"),
        ("a red apple on a table", "green watermelon", "watermelon")
    ]
    
    for i, (prompt, inject, name) in enumerate(test_cases):
        print(f"\n{i+1}. Testing: {name}")
        print(f"   Prompt: '{prompt}'")
        if inject:
            print(f"   Injection: '{inject}'")
        
        corepulse.clear()
        
        if inject:
            # Use very high strength for direct replacement
            corepulse.add_injection(
                prompt=inject,
                strength=0.95,  # Almost complete replacement
                blocks=["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]
            )
        
        latents = model.generate_latents(
            prompt,
            num_steps=4,
            cfg_weight=0.0,
            seed=42
        )
        
        # Get final latent
        for j, x in enumerate(latents):
            if j == 3:
                img = model.decode(x)
                save_image(img[0], output_dir / f"{i:02d}_{name}.png")
                print(f"   ✅ Saved: {i:02d}_{name}.png")
    
    print("\n✨ Complete! Check examples/output/injection_final/")
    print("If injection is working, you should see different fruits, not all apples.")
    

if __name__ == "__main__":
    test_injection_final()