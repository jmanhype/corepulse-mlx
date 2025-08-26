#!/usr/bin/env python3
"""SDXL-only injection - no more broken SD 2.1!"""

import sys
import os
sys.path.append('/Users/speed/Downloads/corpus-mlx/src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import StableDiffusionXL
from tqdm import tqdm
import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sdxl_dramatic_injection():
    """Use ONLY SDXL with more extreme prompt combinations."""
    print("🚀 SDXL-ONLY dramatic injection (no broken SD 2.1)...")
    
    sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Test extreme prompt combinations
    tests = [
        {
            "name": "DOG_CAR_FUSION", 
            "base": "a golden retriever puppy",
            "inject": "a bright red ferrari car",
            "step": 0
        },
        {
            "name": "CAT_DRAGON_FUSION",
            "base": "a white fluffy cat", 
            "inject": "a fierce dragon with wings",
            "step": 1
        },
        {
            "name": "FOREST_CITY_FUSION",
            "base": "a peaceful green forest",
            "inject": "a cyberpunk neon cityscape", 
            "step": 0
        }
    ]
    
    results = {}
    
    for test in tests:
        print(f"\n🎯 {test['name']}: {test['base']} + {test['inject']}")
        
        # Generate baseline
        baseline_latents = sd.generate_latents(test['base'], n_images=1, num_steps=2, cfg_weight=0.0, seed=42)
        for x_t in tqdm(baseline_latents, total=2):
            mx.eval(x_t)
        baseline = sd.decode(x_t)[0]
        mx.eval(baseline)
        
        # Generate injection
        base_cond = sd._get_text_conditioning(test['base'], n_images=1, cfg_weight=0.0)
        inject_cond = sd._get_text_conditioning(test['inject'], n_images=1, cfg_weight=0.0)
        
        original_call = sd.unet.__call__
        step_count = 0
        
        def injection_hook(x, t, encoder_x, encoder_x2=None, text_time=None, y=None):
            nonlocal step_count
            
            if step_count == test['step']:
                print(f"    💥 INJECTING: {test['inject']}")
                encoder_x = inject_cond
            else:
                print(f"    📝 Using: {test['base']}")
                
            result = original_call(x, t, encoder_x, encoder_x2, text_time, y)
            step_count += 1
            return result
        
        sd.unet.__call__ = injection_hook
        
        inject_latents = sd.generate_latents(test['base'], n_images=1, num_steps=2, cfg_weight=0.0, seed=42)
        for x_t in tqdm(inject_latents, total=2):
            mx.eval(x_t)
        injected = sd.decode(x_t)[0]
        mx.eval(injected)
        
        # Restore
        sd.unet.__call__ = original_call
        step_count = 0
        
        results[test['name']] = (baseline, injected)
    
    return results


def create_sdxl_comparison(baseline, injected, title):
    """Create SDXL comparison."""
    size = (400, 400)
    
    # Ensure PIL
    if not isinstance(baseline, Image.Image):
        baseline = Image.fromarray((np.array(baseline) * 255).astype(np.uint8))
    if not isinstance(injected, Image.Image):
        injected = Image.fromarray((np.array(injected) * 255).astype(np.uint8))
    
    baseline = baseline.resize(size, Image.LANCZOS)
    injected = injected.resize(size, Image.LANCZOS)
    
    # Canvas
    width = size[0] * 2 + 60
    height = size[1] + 120
    canvas = Image.new('RGB', (width, height), '#111111')
    draw = ImageDraw.Draw(canvas)
    
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        title_font = label_font = None
    
    # Title
    draw.text((width//2, 20), title, fill='#00aaff', font=title_font, anchor='mm')
    
    # Images
    canvas.paste(baseline, (20, 50))
    canvas.paste(injected, (size[0] + 40, 50))
    
    # Labels
    draw.text((20 + size[0]//2, size[1] + 60), "BASELINE", fill='#888', font=label_font, anchor='mt')
    draw.text((size[0] + 40 + size[0]//2, size[1] + 60), "INJECTED", fill='#ff6600', font=label_font, anchor='mt')
    
    return canvas


def main():
    print("\n" + "="*70)
    print("SDXL-ONLY INJECTION - NO MORE BROKEN SD 2.1!")
    print("="*70)
    
    save_dir = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme"
    os.makedirs(save_dir, exist_ok=True)
    
    # Run SDXL-only tests
    results = sdxl_dramatic_injection()
    
    # Create comparisons
    for name, (baseline, injected) in results.items():
        comparison = create_sdxl_comparison(baseline, injected, f"SDXL {name}")
        comparison.save(f"{save_dir}/SDXL_{name.lower()}.png")
        print(f"✅ Saved SDXL_{name.lower()}.png")
    
    print("\n" + "="*70)
    print("SDXL-ONLY INJECTION COMPLETE!")
    print("Using only the working model - no more abstract art!")
    print("="*70)


if __name__ == "__main__":
    main()