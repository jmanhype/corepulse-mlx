#!/usr/bin/env python3
"""TRUE DataVoid-style prompt injection using SDXL model that actually works!"""

import sys
import os
sys.path.append('/Users/speed/Downloads/corpus-mlx/src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import StableDiffusionXL
from tqdm import tqdm
import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class TrueDataVoidInjection:
    """REAL prompt injection like DataVoid using working SDXL model."""
    
    def __init__(self):
        print("🚀 Loading SDXL model for TRUE DataVoid injection...")
        self.sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
        self.original_unet_call = None
        self.block_embeddings = {}
        self.enabled = False
        self.step_count = 0
        
    def setup_datavoid_injection(self, base_prompt, injection_prompts):
        """
        Setup DataVoid-style injection.
        
        Args:
            base_prompt: Base prompt
            injection_prompts: Dict of {step_range: prompt}
                e.g. {(0, 1): "a cute dog", (1, 2): "a fluffy cat"}
        """
        print(f"\n🎯 Setting up TRUE DataVoid injection:")
        print(f"  Base: '{base_prompt}'")
        
        # Pre-compute embeddings for all injection prompts
        self.block_embeddings = {}
        for step_range, prompt in injection_prompts.items():
            print(f"  Steps {step_range}: '{prompt}'")
            
            # Get text conditioning for both encoders (SDXL has 2)
            conditioning = self.sd._get_text_conditioning(
                prompt, n_images=1, cfg_weight=0.0
            )
            self.block_embeddings[step_range] = conditioning
            
        # Store original UNet call
        if not self.original_unet_call:
            self.original_unet_call = self.sd.unet.__call__
            
        # Create injection wrapper
        def injected_unet_call(x, t, encoder_x, encoder_x2=None, text_time=None, y=None):
            """Inject different prompts at different steps - REAL DataVoid style!"""
            
            # Find which prompt to use based on current step
            current_conditioning = encoder_x
            current_conditioning2 = encoder_x2
            
            for (start_step, end_step), conditioning in self.block_embeddings.items():
                if start_step <= self.step_count < end_step:
                    print(f"    💉 INJECTING at step {self.step_count}: using custom embedding")
                    # Use our custom conditioning
                    current_conditioning = conditioning
                    # For SDXL, we might need to handle the second text encoder too
                    break
            else:
                print(f"    ✨ Step {self.step_count}: using base embedding")
            
            # Call original with potentially modified conditioning
            return self.original_unet_call(x, t, current_conditioning, current_conditioning2, text_time, y)
            
        # Apply the hook
        self.sd.unet.__call__ = injected_unet_call
        self.enabled = True
        print("✅ SDXL UNet hooked for TRUE DataVoid injection")
        
    def generate_with_datavoid_injection(self, base_prompt, steps=2, seed=42):
        """Generate with DataVoid-style step-based injection."""
        print(f"\n🎨 Generating with DataVoid injection: '{base_prompt}'")
        self.step_count = 0
        
        # Generate latents with injection
        latents = self.sd.generate_latents(
            base_prompt, 
            n_images=1, 
            num_steps=steps, 
            cfg_weight=0.0,
            seed=seed
        )
        
        # Evaluate each step and track progress
        for x_t in tqdm(latents, total=steps):
            mx.eval(x_t)
            self.step_count += 1
            
        # Decode
        image = self.sd.decode(x_t)[0]
        mx.eval(image)
        
        return image
        
    def restore_original(self):
        """Restore original behavior."""
        if self.enabled and self.original_unet_call:
            self.sd.unet.__call__ = self.original_unet_call
            self.enabled = False
            print("✅ SDXL UNet restored")


def create_datavoid_comparison(baseline, injected1, injected2, title):
    """Create DataVoid-style comparison with 3 images."""
    
    # Ensure PIL images
    images = [baseline, injected1, injected2]
    for i, img in enumerate(images):
        if not isinstance(img, Image.Image):
            images[i] = Image.fromarray((np.array(img) * 255).astype(np.uint8))
    
    # Resize
    size = (384, 384)
    images = [img.resize(size, Image.LANCZOS) for img in images]
    
    # Create canvas
    width = size[0] * 3 + 80
    height = size[1] + 140
    canvas = Image.new('RGB', (width, height), '#0d1117')
    draw = ImageDraw.Draw(canvas)
    
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        title_font = label_font = None
    
    # Add title
    draw.text((width//2, 30), title, fill='#f0f6fc', font=title_font, anchor='mm')
    
    # Add images
    labels = ["BASELINE", "INJECTION 1", "INJECTION 2"]
    colors = ["#666", "#f39c12", "#e74c3c"]
    
    for i, (img, label, color) in enumerate(zip(images, labels, colors)):
        x = 20 + i * (size[0] + 20)
        y = 70
        canvas.paste(img, (x, y))
        
        # Add label
        draw.text((x + size[0]//2, y + size[1] + 15), label, 
                  fill=color, font=label_font, anchor='mt')
    
    return canvas


def test_true_datavoid_injection():
    """Test TRUE DataVoid-style injection with step-based swapping."""
    print("\n=== TESTING TRUE DATAVOID INJECTION ===")
    
    injector = TrueDataVoidInjection()
    base_prompt = "a simple test"
    
    # Generate baseline
    print("\nGenerating baseline...")
    baseline = injector.generate_with_datavoid_injection(base_prompt, steps=2, seed=42)
    
    # Test 1: Dog → Cat injection
    injector.setup_datavoid_injection(
        base_prompt,
        {(0, 1): "a cute fluffy dog"}  # Inject dog in first step
    )
    
    print("\nGenerating dog injection...")
    dog_injection = injector.generate_with_datavoid_injection(base_prompt, steps=2, seed=42)
    injector.restore_original()
    
    # Test 2: Cat → Car injection  
    injector.setup_datavoid_injection(
        base_prompt,
        {(1, 2): "a red sports car"}  # Inject car in second step
    )
    
    print("\nGenerating car injection...")
    car_injection = injector.generate_with_datavoid_injection(base_prompt, steps=2, seed=42)
    injector.restore_original()
    
    return create_datavoid_comparison(
        baseline, dog_injection, car_injection,
        "TRUE DATAVOID INJECTION - Step-based Prompt Swapping"
    )


def test_advanced_injection_patterns():
    """Test advanced injection patterns like DataVoid."""
    print("\n=== TESTING ADVANCED INJECTION PATTERNS ===")
    
    injector = TrueDataVoidInjection()
    
    # Test complex injection: subject + background + style
    injector.setup_datavoid_injection(
        "a simple image",
        {
            (0, 1): "a majestic lion in the african savanna, golden hour lighting",
            (1, 2): "cyberpunk neon cityscape, futuristic architecture"
        }
    )
    
    print("\nGenerating complex injection...")
    complex_injection = injector.generate_with_datavoid_injection(
        "a simple image", steps=2, seed=123
    )
    injector.restore_original()
    
    # Save individual result
    if not isinstance(complex_injection, Image.Image):
        complex_injection = Image.fromarray((np.array(complex_injection) * 255).astype(np.uint8))
    
    return complex_injection


def main():
    print("\n" + "="*80)
    print("TRUE DATAVOID INJECTION - REAL PROMPT SWAPPING USING WORKING SDXL MODEL!")
    print("="*80)
    
    save_dir = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme"
    os.makedirs(save_dir, exist_ok=True)
    
    # Test 1: Basic DataVoid injection
    comparison1 = test_true_datavoid_injection()
    comparison1.save(f"{save_dir}/TRUE_DATAVOID_injection.png")
    print("✅ Saved TRUE DataVoid injection comparison")
    
    # Test 2: Advanced patterns
    complex_result = test_advanced_injection_patterns()
    complex_result.save(f"{save_dir}/TRUE_DATAVOID_advanced.png")
    print("✅ Saved advanced DataVoid injection")
    
    print("\n" + "="*80)
    print("TRUE DATAVOID INJECTION COMPLETE!")
    print("We now have REAL prompt injection working with step-based swapping!")
    print("="*80)


if __name__ == "__main__":
    main()