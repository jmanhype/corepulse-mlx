#!/usr/bin/env python3
"""Use FULL SDXL (not turbo) with proper DataVoid approach - 30+ steps for real injection."""

import sys
import os
sys.path.append('/Users/speed/Downloads/corpus-mlx/src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import StableDiffusionXL
from tqdm import tqdm
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class FullSDXLDataVoid:
    """Full SDXL with proper multi-step injection like DataVoid."""
    
    def __init__(self):
        print("🚀 Loading FULL SDXL (not turbo) for proper DataVoid injection...")
        # Load regular SDXL, NOT turbo - we need more steps!
        self.sd = StableDiffusionXL("stabilityai/stable-diffusion-xl-base-1.0", float16=True)
        
        # Change sampler to support more steps
        from stable_diffusion.sampler import SimpleEulerSampler
        self.sd.sampler = SimpleEulerSampler(self.sd.diffusion_config)
        
        self.original_unet_call = None
        self.content_cond = None
        self.style_cond = None
        self.base_cond = None
        self.current_step = 0
        self.total_steps = 0
        
    def setup_injection(self, base_prompt, content_prompt, style_prompt):
        """Setup DataVoid-style content/style injection."""
        print(f"\n🎯 DataVoid Injection Setup:")
        print(f"  Base: '{base_prompt}'")
        print(f"  Content: '{content_prompt}'")  
        print(f"  Style: '{style_prompt}'")
        
        # Get conditioning for all prompts
        self.base_cond = self.sd._get_text_conditioning(base_prompt, n_images=1, cfg_weight=7.5)
        self.content_cond = self.sd._get_text_conditioning(content_prompt, n_images=1, cfg_weight=7.5)
        self.style_cond = self.sd._get_text_conditioning(style_prompt, n_images=1, cfg_weight=7.5)
        
        # Store original
        if not self.original_unet_call:
            self.original_unet_call = self.sd.unet.__call__
            
    def create_phased_injection(self, steps=30):
        """Create DataVoid-style phased injection."""
        self.total_steps = steps
        self.current_step = 0
        
        def datavoid_injection(x, t, encoder_x, encoder_x2=None, text_time=None, y=None):
            """Inject based on denoising phase like DataVoid."""
            
            progress = self.current_step / self.total_steps
            
            # DataVoid phases:
            if progress < 0.3:  # Structure phase
                print(f"    📐 Step {self.current_step}: STRUCTURE")
                cond = self.base_cond
            elif progress < 0.7:  # Content phase  
                print(f"    🎨 Step {self.current_step}: CONTENT")
                cond = self.content_cond
            else:  # Style phase
                print(f"    ✨ Step {self.current_step}: STYLE")
                cond = self.style_cond
                
            result = self.original_unet_call(x, t, cond, encoder_x2, text_time, y)
            self.current_step += 1
            return result
        
        self.sd.unet.__call__ = datavoid_injection
        print(f"✅ Phased injection ready for {steps} steps")
        
    def generate(self, prompt, steps=30, seed=42):
        """Generate with proper multi-step process."""
        print(f"\n🎨 Generating with {steps} steps...")
        self.current_step = 0
        
        latents = self.sd.generate_latents(
            prompt,
            n_images=1,
            num_steps=steps,  # Many steps for gradual injection
            cfg_weight=7.5,    # Standard CFG
            seed=seed
        )
        
        for x_t in tqdm(latents, total=steps):
            mx.eval(x_t)
            
        image = self.sd.decode(x_t)[0]
        mx.eval(image)
        
        return image
        
    def restore(self):
        """Restore original UNet."""
        if self.original_unet_call:
            self.sd.unet.__call__ = self.original_unet_call
            self.current_step = 0


def test_full_sdxl_datavoid():
    """Test with full SDXL and proper steps."""
    print("\n=== FULL SDXL DATAVOID TEST ===")
    
    injector = FullSDXLDataVoid()
    
    # Test case: Dog + Car fusion with style
    base = "a simple scene"
    content = "a golden retriever puppy playing"
    style = "cyberpunk neon aesthetic, futuristic lighting"
    
    # Setup injection
    injector.setup_injection(base, content, style)
    
    # Generate baseline (no injection)
    print("\n1️⃣ Baseline generation...")
    baseline = injector.generate(base, steps=30, seed=123)
    
    # Setup phased injection
    injector.create_phased_injection(steps=30)
    
    # Generate with injection
    print("\n2️⃣ DataVoid injection...")
    injected = injector.generate(base, steps=30, seed=123)
    
    injector.restore()
    
    return baseline, injected


def create_comparison(baseline, injected):
    """Create comparison image."""
    size = (400, 400)
    
    if not isinstance(baseline, Image.Image):
        baseline = Image.fromarray((np.array(baseline) * 255).astype(np.uint8))
    if not isinstance(injected, Image.Image):
        injected = Image.fromarray((np.array(injected) * 255).astype(np.uint8))
        
    baseline = baseline.resize(size, Image.LANCZOS)
    injected = injected.resize(size, Image.LANCZOS)
    
    # Canvas
    width = size[0] * 2 + 60
    height = size[1] + 140
    canvas = Image.new('RGB', (width, height), '#000')
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = small_font = None
    
    # Title
    draw.text((width//2, 20), "FULL SDXL DATAVOID INJECTION (30 STEPS)", 
              fill='#ff6b35', font=font, anchor='mm')
    
    # Images
    canvas.paste(baseline, (20, 50))
    canvas.paste(injected, (size[0] + 40, 50))
    
    # Labels
    draw.text((20 + size[0]//2, size[1] + 60), "BASELINE", 
              fill='#888', font=font, anchor='mt')
    draw.text((size[0] + 40 + size[0]//2, size[1] + 60), "DATAVOID INJECTED",
              fill='#00ff88', font=font, anchor='mt')
              
    # Description
    draw.text((width//2, size[1] + 100), 
              "Structure (0-30%) → Content: 'golden retriever' (30-70%) → Style: 'cyberpunk' (70-100%)",
              fill='#666', font=small_font, anchor='mm')
    
    return canvas


def main():
    print("\n" + "="*70)
    print("FULL SDXL WITH DATAVOID APPROACH - 30 STEPS")
    print("="*70)
    
    save_dir = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme"
    os.makedirs(save_dir, exist_ok=True)
    
    # Test full SDXL
    baseline, injected = test_full_sdxl_datavoid()
    
    # Create comparison
    comparison = create_comparison(baseline, injected)
    comparison.save(f"{save_dir}/FULL_SDXL_DATAVOID.png")
    
    print("\n✅ Saved FULL_SDXL_DATAVOID.png")
    print("\n" + "="*70)
    print("FULL SDXL DATAVOID COMPLETE!")
    print("Used 30 steps with phased injection like the real DataVoid")
    print("="*70)


if __name__ == "__main__":
    main()