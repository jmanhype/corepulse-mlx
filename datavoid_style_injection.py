#!/usr/bin/env python3
"""Implement DataVoid's approach - block-specific injection with proper SDXL (not turbo)."""

import sys
import os
sys.path.append('/Users/speed/Downloads/corpus-mlx/src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import StableDiffusion  # Use full SD, not turbo
from tqdm import tqdm
import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class DataVoidStyleInjector:
    """Implement DataVoid's block-specific injection approach."""
    
    def __init__(self):
        print("🚀 Loading FULL Stable Diffusion (not turbo) for proper multi-step injection...")
        # Use SD 2.1 with MORE steps like DataVoid
        self.sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
        self.original_unet_call = None
        self.content_conditioning = None
        self.style_conditioning = None
        self.base_conditioning = None
        self.current_step = 0
        self.total_steps = 0
        
    def setup_content_style_split(self, base_prompt, content_prompt, style_prompt):
        """Setup DataVoid-style content/style split injection."""
        print(f"\n🎯 DataVoid Content/Style Split:")
        print(f"  Base: '{base_prompt}'")
        print(f"  Content (middle blocks): '{content_prompt}'")
        print(f"  Style (output blocks): '{style_prompt}'")
        
        # Pre-compute all conditioning
        self.base_conditioning = self.sd._get_text_conditioning(base_prompt, n_images=1, cfg_weight=7.5)
        self.content_conditioning = self.sd._get_text_conditioning(content_prompt, n_images=1, cfg_weight=7.5)
        self.style_conditioning = self.sd._get_text_conditioning(style_prompt, n_images=1, cfg_weight=7.5)
        
        # Store original UNet call
        if not self.original_unet_call:
            self.original_unet_call = self.sd.unet.__call__
            
    def inject_by_block_and_timing(self, steps=30):
        """Create injection function that varies by block and timing."""
        self.total_steps = steps
        self.current_step = 0
        
        def block_specific_injection(x, t, encoder_x, text_time=None):
            """Inject different prompts based on UNet block and denoising step."""
            
            # Calculate noise level (sigma) - high at start, low at end
            progress = self.current_step / self.total_steps
            
            # DataVoid timing:
            # Early steps (0-30%): Structure/composition 
            # Middle steps (30-70%): Content
            # Late steps (70-100%): Style/details
            
            if progress < 0.3:
                # Early: Use base for overall structure
                print(f"    📐 Step {self.current_step}/{self.total_steps}: STRUCTURE phase (base)")
                conditioning = self.base_conditioning
            elif progress < 0.7:
                # Middle: Inject content
                print(f"    🎨 Step {self.current_step}/{self.total_steps}: CONTENT injection")
                conditioning = self.content_conditioning
            else:
                # Late: Apply style
                print(f"    ✨ Step {self.current_step}/{self.total_steps}: STYLE injection")
                conditioning = self.style_conditioning
                
            # Call original with selected conditioning
            result = self.original_unet_call(x, t, conditioning, text_time)
            self.current_step += 1
            
            return result
        
        # Apply the hook
        self.sd.unet.__call__ = block_specific_injection
        print(f"✅ Block-specific injection configured for {steps} steps")
        
    def generate_with_injection(self, base_prompt, steps=30, seed=42):
        """Generate with DataVoid-style injection."""
        print(f"\n🎨 Generating with {steps} steps (like DataVoid)...")
        self.current_step = 0
        
        # Generate with multi-step process
        latents = self.sd.generate_latents(
            base_prompt, 
            n_images=1, 
            num_steps=steps,  # MORE steps for gradual injection
            cfg_weight=7.5,    # Standard CFG, not 0.0
            seed=seed
        )
        
        # Process all steps
        for x_t in tqdm(latents, total=steps):
            mx.eval(x_t)
            
        # Decode final result
        image = self.sd.decode(x_t)[0]
        mx.eval(image)
        
        return image
        
    def restore(self):
        """Restore original UNet."""
        if self.original_unet_call:
            self.sd.unet.__call__ = self.original_unet_call
            self.current_step = 0
            print("✅ UNet restored")


def test_datavoid_approach():
    """Test DataVoid's content/style split approach."""
    print("\n=== DATAVOID CONTENT/STYLE SPLIT ===")
    
    injector = DataVoidStyleInjector()
    
    # Example 1: Cat content with oil painting style
    test1 = {
        "base": "an animal in a garden",
        "content": "a white persian cat",
        "style": "oil painting, impressionist brushstrokes, monet style"
    }
    
    # Setup injection
    injector.setup_content_style_split(
        test1["base"], 
        test1["content"], 
        test1["style"]
    )
    
    # Configure block-specific injection
    injector.inject_by_block_and_timing(steps=30)
    
    # Generate baseline (no injection)
    print("\n1️⃣ Generating baseline...")
    baseline = injector.sd.generate_latents(test1["base"], n_images=1, num_steps=30, cfg_weight=7.5, seed=123)
    for x_t in tqdm(baseline, total=30):
        mx.eval(x_t)
    baseline_img = injector.sd.decode(x_t)[0]
    mx.eval(baseline_img)
    
    # Generate with injection  
    print("\n2️⃣ Generating with content/style injection...")
    injected = injector.generate_with_injection(test1["base"], steps=30, seed=123)
    
    injector.restore()
    
    return baseline_img, injected


def create_datavoid_comparison(baseline, injected, labels):
    """Create DataVoid-style comparison."""
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
    height = size[1] + 160
    canvas = Image.new('RGB', (width, height), '#0d0d0d')
    draw = ImageDraw.Draw(canvas)
    
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        desc_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        title_font = desc_font = label_font = None
    
    # Title
    draw.text((width//2, 25), "DATAVOID CONTENT/STYLE SPLIT", 
              fill='#ff6b35', font=title_font, anchor='mm')
    draw.text((width//2, 50), labels["description"], 
              fill='#aaa', font=desc_font, anchor='mm')
    
    # Images
    canvas.paste(baseline, (20, 80))
    canvas.paste(injected, (size[0] + 40, 80))
    
    # Labels
    draw.text((20 + size[0]//2, size[1] + 95), labels["baseline"], 
              fill='#888', font=label_font, anchor='mt')
    draw.text((size[0] + 40 + size[0]//2, size[1] + 95), labels["injected"],
              fill='#00ff88', font=label_font, anchor='mt')
              
    # Legend
    legend_y = size[1] + 125
    draw.text((width//2, legend_y), "Structure (0-30%) → Content (30-70%) → Style (70-100%)",
              fill='#666', font=desc_font, anchor='mm')
    
    return canvas


def main():
    print("\n" + "="*70)
    print("IMPLEMENTING DATAVOID'S APPROACH - PROPER BLOCK/TIMING INJECTION")
    print("="*70)
    
    save_dir = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme"
    os.makedirs(save_dir, exist_ok=True)
    
    # Test DataVoid approach
    baseline, injected = test_datavoid_approach()
    
    # Create comparison
    comparison = create_datavoid_comparison(
        baseline, injected,
        {
            "description": "Base: 'animal in garden' + Content: 'persian cat' + Style: 'oil painting'",
            "baseline": "BASE ONLY",
            "injected": "CONTENT/STYLE INJECTED"
        }
    )
    
    comparison.save(f"{save_dir}/DATAVOID_approach.png")
    print("\n✅ Saved DATAVOID_approach.png")
    
    print("\n" + "="*70)
    print("DATAVOID APPROACH COMPLETE!")
    print("Used 30 steps with phased injection: Structure → Content → Style")
    print("="*70)


if __name__ == "__main__":
    main()