#!/usr/bin/env python3
"""
MLX DataVoid Fusion - TRUE prompt injection with block-specific conditioning.
Uses custom UNet modifications for phased injection like the original DataVoid.
"""

import mlx.core as mx
from pathlib import Path
import time
import argparse
from datavoid_unet import inject_datavoid_unet, InjectionConfig

# Add path for local imports
import sys
sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')
from stable_diffusion import StableDiffusion

def create_fusion_examples():
    """Generate DataVoid-style fusion examples with MLX."""
    
    print("\n" + "="*80)
    print("🔥 MLX DATAVOID FUSION - TRUE BLOCK-SPECIFIC INJECTION 🔥")
    print("="*80)
    
    # Initialize SDXL-turbo
    model = "stabilityai/sdxl-turbo"
    from stable_diffusion import StableDiffusionXL
    sd = StableDiffusionXL(model)
    print(f"\n✅ Loaded: {model}")
    
    # Inject our custom DataVoid UNet
    datavoid_unet = inject_datavoid_unet(sd)
    
    # Test cases for fusion
    test_cases = [
        {
            "name": "CAT_DOG_FUSION",
            "structure": "a fluffy cat",      # Early steps - basic shape
            "content": "a playful dog",        # Middle steps - main subject
            "style": "oil painting style",     # Late steps - artistic style
            "output": "artifacts/images/readme/MLX_DATAVOID_cat_dog_fusion.png"
        },
        {
            "name": "CAR_PLANE_FUSION", 
            "structure": "a sports car",       # Ground vehicle structure
            "content": "a fighter jet",        # Aircraft content
            "style": "cyberpunk neon style",   # Futuristic style
            "output": "artifacts/images/readme/MLX_DATAVOID_car_plane_fusion.png"
        },
        {
            "name": "LANDSCAPE_CITY_FUSION",
            "structure": "mountain landscape", # Natural structure
            "content": "futuristic city",      # Urban content
            "style": "sunset golden hour",     # Lighting style
            "output": "artifacts/images/readme/MLX_DATAVOID_landscape_city_fusion.png"
        }
    ]
    
    # Extended steps for proper phased injection
    num_steps = 30
    cfg_weight = 7.5
    seed = 42
    
    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"🎨 {test['name']}")
        print(f"{'='*60}")
        
        # Encode prompts for each phase
        print("\n📝 Encoding prompts for each phase:")
        print(f"  Structure (0-30%): {test['structure']}")
        print(f"  Content (30-70%): {test['content']}")  
        print(f"  Style (70-100%): {test['style']}")
        
        # Create conditioning for each phase
        cond_structure, pool_structure = sd._get_text_conditioning(test['structure'], n_images=1)
        cond_content, pool_content = sd._get_text_conditioning(test['content'], n_images=1)
        cond_style, pool_style = sd._get_text_conditioning(test['style'], n_images=1)
        
        # Configure injection for UNet blocks
        injection_config = InjectionConfig(
            down_blocks=cond_structure,  # Structure in down blocks
            mid_blocks=cond_content,      # Content in middle blocks
            up_blocks=cond_style          # Style in up blocks
        )
        
        # Set injection configuration
        datavoid_unet.set_injection(injection_config, total_steps=num_steps)
        
        # Generate with phased injection
        print(f"\n🚀 Generating with {num_steps} steps...")
        start_time = time.time()
        
        # Generate latents with injection
        # Use the base prompt but injection will override it
        for latents in sd.generate_latents(
            test['content'],  # Base prompt (will be overridden by injection)
            n_images=1,
            cfg_weight=cfg_weight,
            num_steps=num_steps,
            seed=seed,
        ):
            pass  # Get final latents
        
        # Decode to image
        decoded = sd.decode(latents)
        # Convert MLX array to numpy
        import numpy as np
        image = np.array(decoded[0])
        image = (image * 255).astype(np.uint8)
        
        # Save image
        from PIL import Image
        output_path = Path(test['output'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(output_path)
        
        elapsed = time.time() - start_time
        print(f"✅ Generated in {elapsed:.2f}s")
        print(f"💾 Saved: {output_path}")
        
        # Also generate baseline for comparison
        print("\n🎯 Generating baseline with content prompt only...")
        datavoid_unet.set_injection(None)  # Disable injection
        
        for latents_baseline in sd.generate_latents(
            test['content'],
            n_images=1,
            cfg_weight=cfg_weight,
            num_steps=num_steps,
            seed=seed,
        ):
            pass  # Get final latents
        
        decoded_baseline = sd.decode(latents_baseline)
        image_baseline = np.array(decoded_baseline[0])
        image_baseline = (image_baseline * 255).astype(np.uint8)
        
        baseline_path = output_path.parent / f"MLX_BASELINE_{output_path.name}"
        Image.fromarray(image_baseline).save(baseline_path)
        print(f"💾 Baseline saved: {baseline_path}")
        
    print("\n" + "="*80)
    print("🎉 MLX DATAVOID FUSION COMPLETE!")
    print("="*80)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLX DataVoid Fusion")
    parser.add_argument("--test", action="store_true", help="Run test examples")
    args = parser.parse_args()
    
    create_fusion_examples()