#!/usr/bin/env python3
"""
Ruby-style DataVoid Fusion for MLX.
Uses monkey patching to inject different prompts at different denoising phases.
"""

import mlx.core as mx
import numpy as np
from pathlib import Path
import time
import sys

# Add path for stable diffusion
sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')
from stable_diffusion import StableDiffusionXL

# Import our Ruby-style monkey patcher
from ruby_monkey_patch import monkey_patch_sdxl, set_injection, disable_injection


def main():
    print("\n" + "="*80)
    print("💎 RUBY-STYLE DATAVOID FUSION FOR MLX 💎")
    print("="*80)
    
    # Load SDXL model
    model = "stabilityai/sdxl-turbo"
    sd = StableDiffusionXL(model)
    print(f"\n✅ Loaded: {model}")
    
    # Apply Ruby-style monkey patch
    sd = monkey_patch_sdxl(sd)
    
    # Test cases
    test_cases = [
        {
            "name": "CAT_DOG_FUSION",
            "structure": "a white fluffy cat sitting",
            "content": "a golden retriever dog playing",
            "style": "vibrant oil painting style with thick brushstrokes",
            "output": "artifacts/images/readme/RUBY_cat_dog_fusion.png"
        },
        {
            "name": "CAR_PLANE_HYBRID",
            "structure": "a sleek sports car on ground",
            "content": "a military fighter jet in flight",
            "style": "futuristic cyberpunk neon aesthetic",
            "output": "artifacts/images/readme/RUBY_car_plane_fusion.png"
        },
        {
            "name": "NATURE_TECH_BLEND",
            "structure": "serene mountain landscape",
            "content": "futuristic cityscape with skyscrapers",
            "style": "dramatic sunset with golden hour lighting",
            "output": "artifacts/images/readme/RUBY_nature_tech_fusion.png"
        }
    ]
    
    # Generation parameters
    num_steps = 30  # More steps for proper phased injection
    cfg_weight = 7.5
    seed = 42
    
    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"🎨 Generating: {test['name']}")
        print(f"{'='*60}")
        
        # Configure injection for this test
        set_injection(
            test['structure'],
            test['content'], 
            test['style'],
            sd,
            total_steps=num_steps
        )
        
        # Generate with injection
        print(f"\n🚀 Generating with {num_steps} steps...")
        start_time = time.time()
        
        # Generate latents (injection happens automatically via monkey patch)
        for latents in sd.generate_latents(
            test['content'],  # Base prompt (will be overridden by injection)
            n_images=1,
            num_steps=num_steps,
            cfg_weight=cfg_weight,
            seed=seed,
        ):
            pass  # Get final latents
            
        # Decode to image
        decoded = sd.decode(latents)
        image = np.array(decoded[0])
        image = (image * 255).astype(np.uint8)
        
        # Save injected image
        from PIL import Image
        output_path = Path(test['output'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(output_path)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Generated in {elapsed:.2f}s")
        print(f"💾 Saved: {output_path}")
        
        # Generate baseline without injection
        print("\n🎯 Generating baseline (no injection)...")
        disable_injection()
        
        for latents_base in sd.generate_latents(
            test['content'],
            n_images=1,
            num_steps=num_steps,
            cfg_weight=cfg_weight,
            seed=seed,
        ):
            pass
            
        decoded_base = sd.decode(latents_base)
        image_base = np.array(decoded_base[0])
        image_base = (image_base * 255).astype(np.uint8)
        
        baseline_path = output_path.parent / f"RUBY_BASELINE_{output_path.name}"
        Image.fromarray(image_base).save(baseline_path)
        print(f"💾 Baseline saved: {baseline_path}")
    
    print("\n" + "="*80)
    print("💎 RUBY-STYLE DATAVOID FUSION COMPLETE! 💎")
    print("="*80)


if __name__ == "__main__":
    main()