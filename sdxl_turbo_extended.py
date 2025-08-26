#!/usr/bin/env python3
"""Use SDXL-turbo but FORCE it to run 30 steps like DataVoid."""

import sys
import os
sys.path.append('/Users/speed/Downloads/corpus-mlx/src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import StableDiffusionXL
from stable_diffusion.sampler import SimpleEulerSampler
from tqdm import tqdm
import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class SDXLTurboExtended:
    """SDXL-turbo forced to run many steps for DataVoid-style injection."""
    
    def __init__(self):
        print("🚀 Loading SDXL-turbo and forcing extended steps...")
        self.sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
        
        # Replace sampler to allow more steps
        print("🔧 Replacing sampler for extended steps...")
        self.sd.sampler = SimpleEulerSampler(self.sd.diffusion_config)
        
        self.original_unet_call = None
        self.injections = {}
        self.current_step = 0
        self.total_steps = 0
        
    def setup_datavoid_injection(self, base, content, style):
        """Setup DataVoid-style injection."""
        print(f"\n🎯 DataVoid Setup:")
        print(f"  Base: '{base}'")
        print(f"  Content: '{content}'")
        print(f"  Style: '{style}'")
        
        # Pre-compute conditioning
        self.injections = {
            'base': self.sd._get_text_conditioning(base, n_images=1, cfg_weight=7.5),
            'content': self.sd._get_text_conditioning(content, n_images=1, cfg_weight=7.5),
            'style': self.sd._get_text_conditioning(style, n_images=1, cfg_weight=7.5)
        }
        
        if not self.original_unet_call:
            self.original_unet_call = self.sd.unet.__call__
            
    def apply_phased_injection(self, steps=30):
        """Apply DataVoid phased injection."""
        self.total_steps = steps
        self.current_step = 0
        
        def phased_hook(x, t, encoder_x, encoder_x2=None, text_time=None, y=None):
            progress = self.current_step / self.total_steps
            
            if progress < 0.3:
                cond = self.injections['base']
                phase = "STRUCTURE"
            elif progress < 0.7:
                cond = self.injections['content']
                phase = "CONTENT"
            else:
                cond = self.injections['style']
                phase = "STYLE"
                
            if self.current_step % 5 == 0:  # Print every 5 steps
                print(f"    Step {self.current_step}: {phase}")
                
            result = self.original_unet_call(x, t, cond, encoder_x2, text_time, y)
            self.current_step += 1
            return result
        
        self.sd.unet.__call__ = phased_hook
        print(f"✅ Phased injection configured for {steps} steps")
        
    def generate_extended(self, prompt, steps=30, seed=42):
        """Generate with extended steps."""
        print(f"\n🎨 Generating with {steps} extended steps...")
        self.current_step = 0
        
        # Force extended generation
        latents = self.sd.generate_latents(
            prompt,
            n_images=1, 
            num_steps=steps,  # Force many steps
            cfg_weight=7.5,    # Use proper CFG
            seed=seed
        )
        
        for x_t in tqdm(latents, total=steps):
            mx.eval(x_t)
            
        image = self.sd.decode(x_t)[0]
        mx.eval(image)
        
        return image
        
    def restore(self):
        if self.original_unet_call:
            self.sd.unet.__call__ = self.original_unet_call
            self.current_step = 0


def test_extended_injection():
    """Test DataVoid approach with extended SDXL-turbo."""
    print("\n=== SDXL-TURBO EXTENDED (30 STEPS) ===")
    
    injector = SDXLTurboExtended()
    
    # Test: Mix dog, car, and cyberpunk
    base = "a scene"
    content = "a golden retriever puppy and a red ferrari"
    style = "cyberpunk neon city, blade runner aesthetic"
    
    injector.setup_datavoid_injection(base, content, style)
    
    # Baseline
    print("\n1️⃣ Generating baseline...")
    baseline = injector.generate_extended(base, steps=30, seed=999)
    
    # With injection
    injector.apply_phased_injection(steps=30)
    print("\n2️⃣ Generating with DataVoid injection...")
    injected = injector.generate_extended(base, steps=30, seed=999)
    
    injector.restore()
    
    return baseline, injected


def create_extended_comparison(baseline, injected):
    """Create comparison."""
    size = (450, 450)
    
    if not isinstance(baseline, Image.Image):
        baseline = Image.fromarray((np.array(baseline) * 255).astype(np.uint8))
    if not isinstance(injected, Image.Image):
        injected = Image.fromarray((np.array(injected) * 255).astype(np.uint8))
        
    baseline = baseline.resize(size, Image.LANCZOS)
    injected = injected.resize(size, Image.LANCZOS)
    
    width = size[0] * 2 + 60
    height = size[1] + 140
    canvas = Image.new('RGB', (width, height), '#111')
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = small = None
    
    draw.text((width//2, 25), "SDXL-TURBO EXTENDED: 30 STEPS DATAVOID", 
              fill='#ff6b35', font=font, anchor='mm')
    
    canvas.paste(baseline, (20, 60))
    canvas.paste(injected, (size[0] + 40, 60))
    
    draw.text((20 + size[0]//2, size[1] + 70), "BASELINE",
              fill='#888', font=font, anchor='mt')
    draw.text((size[0] + 40 + size[0]//2, size[1] + 70), "DATAVOID INJECTED",
              fill='#00ff88', font=font, anchor='mt')
              
    phases = "0-30%: Structure | 30-70%: 'puppy + ferrari' | 70-100%: 'cyberpunk neon'"
    draw.text((width//2, size[1] + 110), phases, fill='#666', font=small, anchor='mm')
    
    return canvas


def main():
    print("\n" + "="*70)
    print("SDXL-TURBO EXTENDED TO 30 STEPS - DATAVOID APPROACH")
    print("="*70)
    
    save_dir = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme"
    os.makedirs(save_dir, exist_ok=True)
    
    baseline, injected = test_extended_injection()
    
    comparison = create_extended_comparison(baseline, injected)
    comparison.save(f"{save_dir}/SDXL_TURBO_EXTENDED.png")
    
    print("\n✅ Saved SDXL_TURBO_EXTENDED.png")
    print("="*70)


if __name__ == "__main__":
    main()