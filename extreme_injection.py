#!/usr/bin/env python3
"""EXTREME injection - force dramatic changes with longer steps and mixed conditioning!"""

import sys
import os
sys.path.append('/Users/speed/Downloads/corpus-mlx/src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import StableDiffusion, StableDiffusionXL
from tqdm import tqdm
import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def extreme_mixed_injection():
    """Use SD 2.1 with MORE steps for stronger injection."""
    print("🚀 Loading SD 2.1 for EXTREME injection (more steps = stronger effect)...")
    
    # Use SD 2.1 but with MIXED conditioning to force dramatic changes
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    base_prompt = "a simple image"
    inject1_prompt = "a fierce red dragon breathing fire"
    inject2_prompt = "a sleek red sports car"
    
    print(f"\n🎯 EXTREME MIXED INJECTION:")
    print(f"  Base: '{base_prompt}'")
    print(f"  Inject 1: '{inject1_prompt}'")
    print(f"  Inject 2: '{inject2_prompt}'")
    
    # Get embeddings
    base_cond = sd._get_text_conditioning(base_prompt, n_images=1, cfg_weight=7.5)
    dragon_cond = sd._get_text_conditioning(inject1_prompt, n_images=1, cfg_weight=7.5) 
    car_cond = sd._get_text_conditioning(inject2_prompt, n_images=1, cfg_weight=7.5)
    
    # Store original
    original_call = sd.unet.__call__
    step_count = 0
    
    def extreme_injection_hook(x, t, encoder_x, text_time=None):
        nonlocal step_count
        
        # Use different conditioning at different steps
        if step_count < 15:  # Early steps - dragon
            print(f"    🐉 DRAGON INJECTION at step {step_count}")
            encoder_x = dragon_cond
        elif step_count < 30:  # Mid steps - car
            print(f"    🚗 CAR INJECTION at step {step_count}")
            encoder_x = car_cond
        else:  # Late steps - mix
            print(f"    🌀 MIXED INJECTION at step {step_count}")
            # Mix dragon and car conditioning
            encoder_x = 0.5 * dragon_cond + 0.5 * car_cond
            
        result = original_call(x, t, encoder_x, text_time)
        step_count += 1
        return result
    
    # Apply hook
    sd.unet.__call__ = extreme_injection_hook
    
    # Generate with MORE steps for stronger effect
    print("\nGenerating with 50 steps for EXTREME effect...")
    latents = sd.generate_latents(base_prompt, n_images=1, num_steps=50, cfg_weight=7.5, seed=42)
    for x_t in tqdm(latents, total=50):
        mx.eval(x_t)
    
    # Decode
    image = sd.decode(x_t)[0]
    mx.eval(image)
    
    # Restore
    sd.unet.__call__ = original_call
    
    return image


def pure_conditioning_swap():
    """Try pure conditioning swap - replace ENTIRE text encoding."""
    print("\n🔥 PURE CONDITIONING SWAP TEST")
    
    sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Generate pure dog
    print("Generating pure dog...")
    dog_latents = sd.generate_latents("a cute golden retriever puppy", n_images=1, num_steps=2, cfg_weight=0.0, seed=123)
    for x_t in tqdm(dog_latents, total=2):
        mx.eval(x_t)
    pure_dog = sd.decode(x_t)[0]
    mx.eval(pure_dog)
    
    # Generate pure car
    print("Generating pure car...")  
    car_latents = sd.generate_latents("a red ferrari sports car", n_images=1, num_steps=2, cfg_weight=0.0, seed=123)
    for x_t in tqdm(car_latents, total=2):
        mx.eval(x_t)
    pure_car = sd.decode(x_t)[0]
    mx.eval(pure_car)
    
    # Now try FORCED mixed conditioning
    print("Generating FORCED mix...")
    dog_cond = sd._get_text_conditioning("a cute golden retriever puppy", n_images=1, cfg_weight=0.0)
    car_cond = sd._get_text_conditioning("a red ferrari sports car", n_images=1, cfg_weight=0.0)
    
    # Mix the embeddings directly
    mixed_cond = 0.7 * dog_cond + 0.3 * car_cond
    
    original_call = sd.unet.__call__
    
    def forced_mix_hook(x, t, encoder_x, encoder_x2=None, text_time=None, y=None):
        print(f"    🌀 FORCED MIX: 70% dog + 30% car")
        return original_call(x, t, mixed_cond, encoder_x2, text_time, y)
    
    sd.unet.__call__ = forced_mix_hook
    
    mixed_latents = sd.generate_latents("base prompt", n_images=1, num_steps=2, cfg_weight=0.0, seed=123)
    for x_t in tqdm(mixed_latents, total=2):
        mx.eval(x_t)
    forced_mix = sd.decode(x_t)[0]
    mx.eval(forced_mix)
    
    # Restore
    sd.unet.__call__ = original_call
    
    return pure_dog, pure_car, forced_mix


def create_extreme_showcase(images, titles, main_title):
    """Create showcase for extreme results."""
    size = (350, 350)
    resized = []
    
    for img in images:
        if not isinstance(img, Image.Image):
            img = Image.fromarray((np.array(img) * 255).astype(np.uint8))
        resized.append(img.resize(size, Image.LANCZOS))
    
    # Canvas
    cols = len(images)
    width = size[0] * cols + 30 * (cols + 1)
    height = size[1] + 140
    canvas = Image.new('RGB', (width, height), '#000000')
    draw = ImageDraw.Draw(canvas)
    
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        title_font = label_font = None
    
    # Main title
    draw.text((width//2, 25), main_title, fill='#ff0000', font=title_font, anchor='mm')
    
    # Images
    colors = ['#00ff00', '#ff6600', '#ff0066']
    for i, (img, title, color) in enumerate(zip(resized, titles, colors)):
        x = 30 + i * (size[0] + 30)
        y = 60
        canvas.paste(img, (x, y))
        
        draw.text((x + size[0]//2, y + size[1] + 10), title, 
                  fill=color, font=label_font, anchor='mt')
    
    return canvas


def main():
    print("\n" + "="*80)
    print("EXTREME INJECTION - FORCING DRAMATIC TRANSFORMATIONS!")
    print("="*80)
    
    save_dir = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme"
    os.makedirs(save_dir, exist_ok=True)
    
    # Test 1: Extreme mixed injection with SD 2.1
    extreme_result = extreme_mixed_injection()
    if not isinstance(extreme_result, Image.Image):
        extreme_result = Image.fromarray((np.array(extreme_result) * 255).astype(np.uint8))
    extreme_result.save(f"{save_dir}/EXTREME_mixed_injection.png")
    print("✅ Saved EXTREME mixed injection")
    
    # Test 2: Pure conditioning swap
    pure_dog, pure_car, forced_mix = pure_conditioning_swap()
    
    showcase = create_extreme_showcase(
        [pure_dog, pure_car, forced_mix],
        ["PURE DOG", "PURE CAR", "FORCED MIX"],
        "🔥 EXTREME CONDITIONING SWAP"
    )
    showcase.save(f"{save_dir}/EXTREME_conditioning_swap.png")
    print("✅ Saved EXTREME conditioning swap")
    
    print("\n" + "="*80)
    print("EXTREME INJECTION COMPLETE!")
    print("Using longer steps and mixed conditioning for maximum effect!")
    print("="*80)


if __name__ == "__main__":
    main()