#!/usr/bin/env python3
"""REAL dramatic injection like DataVoid - actual subject swapping!"""

import sys
import os
sys.path.append('/Users/speed/Downloads/corpus-mlx/src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import StableDiffusionXL
from tqdm import tqdm
import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class DramaticInjection:
    """Create DRAMATIC prompt injection with actual subject swapping."""
    
    def __init__(self):
        print("🚀 Loading SDXL for DRAMATIC injection...")
        self.sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
        self.original_call = None
        
    def dramatic_swap(self, base_prompt, inject_prompt, inject_at_step=0, steps=2, seed=42):
        """Swap prompts dramatically at specific step."""
        print(f"\n🎯 DRAMATIC SWAP:")
        print(f"  Base: '{base_prompt}'")
        print(f"  Inject: '{inject_prompt}' at step {inject_at_step}")
        
        # Get embeddings
        base_conditioning = self.sd._get_text_conditioning(base_prompt, n_images=1, cfg_weight=0.0)
        inject_conditioning = self.sd._get_text_conditioning(inject_prompt, n_images=1, cfg_weight=0.0)
        
        # Store original
        if not self.original_call:
            self.original_call = self.sd.unet.__call__
            
        step_count = 0
        
        def dramatic_injection_hook(x, t, encoder_x, encoder_x2=None, text_time=None, y=None):
            nonlocal step_count
            
            if step_count == inject_at_step:
                print(f"    💥 DRAMATIC INJECTION at step {step_count}!")
                encoder_x = inject_conditioning
            else:
                print(f"    📝 Using base prompt at step {step_count}")
                
            result = self.original_call(x, t, encoder_x, encoder_x2, text_time, y)
            step_count += 1
            return result
        
        # Apply hook
        self.sd.unet.__call__ = dramatic_injection_hook
        
        # Generate
        latents = self.sd.generate_latents(base_prompt, n_images=1, num_steps=steps, cfg_weight=0.0, seed=seed)
        for x_t in tqdm(latents, total=steps):
            mx.eval(x_t)
        
        # Decode
        image = self.sd.decode(x_t)[0]
        mx.eval(image)
        
        # Restore
        self.sd.unet.__call__ = self.original_call
        step_count = 0
        
        return image


def create_dramatic_comparison(images, titles, main_title):
    """Create dramatic comparison grid."""
    size = (400, 400)
    resized = []
    
    for img in images:
        if not isinstance(img, Image.Image):
            img = Image.fromarray((np.array(img) * 255).astype(np.uint8))
        resized.append(img.resize(size, Image.LANCZOS))
    
    # Canvas
    cols = len(images)
    width = size[0] * cols + 40 * (cols + 1)
    height = size[1] + 160
    canvas = Image.new('RGB', (width, height), '#0a0a0a')
    draw = ImageDraw.Draw(canvas)
    
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        title_font = label_font = None
    
    # Main title
    draw.text((width//2, 30), main_title, fill='#ff6b35', font=title_font, anchor='mm')
    
    # Images and labels
    colors = ['#666', '#f39c12', '#e74c3c', '#9b59b6']
    for i, (img, title, color) in enumerate(zip(resized, titles, colors)):
        x = 40 + i * (size[0] + 40)
        y = 80
        canvas.paste(img, (x, y))
        
        # Label
        draw.text((x + size[0]//2, y + size[1] + 15), title, 
                  fill=color, font=label_font, anchor='mt')
    
    return canvas


def test_dramatic_swaps():
    """Test DRAMATIC subject swapping."""
    print("\n=== TESTING DRAMATIC SWAPS ===")
    
    injector = DramaticInjection()
    
    # Test 1: Dog → Car swap
    print("\n🚗 DOG → CAR SWAP")
    dog = injector.dramatic_swap("a cute golden retriever puppy", "a cute golden retriever puppy", inject_at_step=-1, seed=123)  # Baseline
    dog_car_step0 = injector.dramatic_swap("a cute golden retriever puppy", "a red ferrari sports car", inject_at_step=0, seed=123)
    dog_car_step1 = injector.dramatic_swap("a cute golden retriever puppy", "a red ferrari sports car", inject_at_step=1, seed=123)
    
    comparison1 = create_dramatic_comparison(
        [dog, dog_car_step0, dog_car_step1],
        ["PUPPY BASELINE", "INJECT STEP 0", "INJECT STEP 1"], 
        "🚗 DRAMATIC DOG → CAR INJECTION"
    )
    
    # Test 2: Cat → Dragon swap  
    print("\n🐉 CAT → DRAGON SWAP")
    cat = injector.dramatic_swap("a fluffy white persian cat", "a fluffy white persian cat", inject_at_step=-1, seed=456)
    cat_dragon_step0 = injector.dramatic_swap("a fluffy white persian cat", "a fierce red dragon breathing fire", inject_at_step=0, seed=456)
    cat_dragon_step1 = injector.dramatic_swap("a fluffy white persian cat", "a fierce red dragon breathing fire", inject_at_step=1, seed=456)
    
    comparison2 = create_dramatic_comparison(
        [cat, cat_dragon_step0, cat_dragon_step1],
        ["CAT BASELINE", "INJECT STEP 0", "INJECT STEP 1"],
        "🐉 DRAMATIC CAT → DRAGON INJECTION"
    )
    
    # Test 3: Forest → Cyberpunk swap
    print("\n🌆 FOREST → CYBERPUNK SWAP") 
    forest = injector.dramatic_swap("a peaceful forest with tall trees", "a peaceful forest with tall trees", inject_at_step=-1, seed=789)
    forest_cyber_step0 = injector.dramatic_swap("a peaceful forest with tall trees", "cyberpunk neon city with skyscrapers", inject_at_step=0, seed=789)
    forest_cyber_step1 = injector.dramatic_swap("a peaceful forest with tall trees", "cyberpunk neon city with skyscrapers", inject_at_step=1, seed=789)
    
    comparison3 = create_dramatic_comparison(
        [forest, forest_cyber_step0, forest_cyber_step1],
        ["FOREST BASELINE", "INJECT STEP 0", "INJECT STEP 1"],
        "🌆 DRAMATIC FOREST → CYBERPUNK INJECTION"  
    )
    
    return comparison1, comparison2, comparison3


def main():
    print("\n" + "="*80)
    print("REAL DRAMATIC INJECTION - ACTUAL SUBJECT SWAPPING!")
    print("="*80)
    
    save_dir = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme"
    os.makedirs(save_dir, exist_ok=True)
    
    comparisons = test_dramatic_swaps()
    
    comparisons[0].save(f"{save_dir}/DRAMATIC_dog_car_swap.png")
    comparisons[1].save(f"{save_dir}/DRAMATIC_cat_dragon_swap.png") 
    comparisons[2].save(f"{save_dir}/DRAMATIC_forest_cyber_swap.png")
    
    print("\n✅ Saved all dramatic injection comparisons!")
    print("\n" + "="*80)
    print("NOW THIS IS REAL DATAVOID-STYLE INJECTION!")
    print("Dogs becoming cars, cats becoming dragons, forests becoming cities!")
    print("="*80)


if __name__ == "__main__":
    main()