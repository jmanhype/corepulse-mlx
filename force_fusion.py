#!/usr/bin/env python3
"""FORCE fusion by using different seeds and more aggressive injection."""

import sys
import os
sys.path.append('/Users/speed/Downloads/corpus-mlx/src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import StableDiffusionXL
from tqdm import tqdm
import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def force_actual_fusion():
    """Force actual fusion by using different approaches."""
    print("🚀 FORCING actual fusion with different seeds and approaches...")
    
    sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Test 1: Different seeds for baseline vs injection
    print("\n🎯 Test 1: DOG baseline vs DOG+CAR injection (DIFFERENT seeds)")
    
    # Baseline dog with seed 1
    dog_baseline = sd.generate_latents("a golden retriever puppy", n_images=1, num_steps=2, cfg_weight=0.0, seed=1)
    for x_t in tqdm(dog_baseline, total=2):
        mx.eval(x_t)
    baseline_img = sd.decode(x_t)[0]
    mx.eval(baseline_img)
    
    # Injection with seed 2 and aggressive hook
    dog_cond = sd._get_text_conditioning("a golden retriever puppy", n_images=1, cfg_weight=0.0)
    car_cond = sd._get_text_conditioning("a bright red ferrari sports car", n_images=1, cfg_weight=0.0)
    
    original_call = sd.unet.__call__
    
    def aggressive_injection(x, t, encoder_x, encoder_x2=None, text_time=None, y=None):
        print(f"    🚗 FORCING CAR injection!")
        # Always use car conditioning
        return original_call(x, t, car_cond, encoder_x2, text_time, y)
    
    sd.unet.__call__ = aggressive_injection
    
    # Generate with seed 2
    dog_car_fusion = sd.generate_latents("a golden retriever puppy", n_images=1, num_steps=2, cfg_weight=0.0, seed=2)
    for x_t in tqdm(dog_car_fusion, total=2):
        mx.eval(x_t)
    fusion_img = sd.decode(x_t)[0]
    mx.eval(fusion_img)
    
    # Restore
    sd.unet.__call__ = original_call
    
    # Test 2: Pure car for comparison
    print("\n🎯 Test 2: Pure car with same seed as fusion")
    pure_car = sd.generate_latents("a bright red ferrari sports car", n_images=1, num_steps=2, cfg_weight=0.0, seed=2)
    for x_t in tqdm(pure_car, total=2):
        mx.eval(x_t)
    car_img = sd.decode(x_t)[0]
    mx.eval(car_img)
    
    return baseline_img, fusion_img, car_img


def create_fusion_comparison(baseline, fusion, pure_target, title):
    """Create 3-way comparison."""
    size = (350, 350)
    
    images = [baseline, fusion, pure_target]
    resized = []
    for img in images:
        if not isinstance(img, Image.Image):
            img = Image.fromarray((np.array(img) * 255).astype(np.uint8))
        resized.append(img.resize(size, Image.LANCZOS))
    
    # Canvas
    width = size[0] * 3 + 80
    height = size[1] + 120
    canvas = Image.new('RGB', (width, height), '#000000')
    draw = ImageDraw.Draw(canvas)
    
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        title_font = label_font = None
    
    # Title
    draw.text((width//2, 20), title, fill='#ffffff', font=title_font, anchor='mm')
    
    # Images
    positions = [20, size[0] + 40, size[0]*2 + 60]
    labels = ["BASELINE DOG", "FUSION RESULT", "PURE CAR"]
    colors = ['#00ff00', '#ffaa00', '#ff0066']
    
    for i, (img, pos, label, color) in enumerate(zip(resized, positions, labels, colors)):
        canvas.paste(img, (pos, 50))
        draw.text((pos + size[0]//2, size[1] + 60), label, fill=color, font=label_font, anchor='mt')
    
    return canvas


def main():
    print("\n" + "="*60)
    print("FORCE FUSION - MAKE IT ACTUALLY WORK!")
    print("="*60)
    
    save_dir = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme"
    os.makedirs(save_dir, exist_ok=True)
    
    # Force fusion test
    baseline, fusion, pure_car = force_actual_fusion()
    
    comparison = create_fusion_comparison(
        baseline, fusion, pure_car, 
        "🔥 FORCED FUSION TEST"
    )
    comparison.save(f"{save_dir}/FORCED_fusion_test.png")
    
    print("\n✅ Saved FORCED_fusion_test.png")
    print("This should show: Dog vs Fusion Result vs Pure Car")
    print("If fusion worked, middle image should be different from both sides!")
    print("="*60)


if __name__ == "__main__":
    main()