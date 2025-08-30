#!/usr/bin/env python3
"""
Direct test of conditioning replacement for semantic object replacement.
This directly manipulates the UNet's text conditioning input.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
from PIL import Image
import numpy as np


def test_direct_conditioning_swap():
    """Test replacing text conditioning directly in the model."""
    print("Testing Direct Conditioning Replacement")
    print("=" * 50)
    
    # Import base model - use SD2.1 as SDXL not available
    from adapters.stable_diffusion import StableDiffusion
    model = StableDiffusion("stabilityai/stable-diffusion-2-1-base")
    
    # Test case: apple -> banana
    original_prompt = "a photo of an apple on a wooden table"
    replacement_prompt = "a photo of a banana on a wooden table"
    
    print(f"\nOriginal: {original_prompt}")
    print(f"Replacement: {replacement_prompt}")
    
    # Method 1: Generate with original (baseline)
    print("\n1. Generating baseline...")
    baseline = model.generate(
        prompt=original_prompt,
        num_steps=20,
        cfg_weight=7.5,
        seed=42,
        width=512,
        height=512
    )
    Image.fromarray((baseline * 255).astype(np.uint8)).save("direct_baseline_apple.png")
    print("   Saved: direct_baseline_apple.png")
    
    # Method 2: Generate with replacement (reference)
    print("\n2. Generating reference...")
    reference = model.generate(
        prompt=replacement_prompt,
        num_steps=20,
        cfg_weight=7.5,
        seed=42,
        width=512,
        height=512
    )
    Image.fromarray((reference * 255).astype(np.uint8)).save("direct_reference_banana.png")
    print("   Saved: direct_reference_banana.png")
    
    # Method 3: Conditioning replacement during generation
    print("\n3. Testing conditioning replacement...")
    
    # Store original methods
    original_get_cond = model._get_text_conditioning
    original_denoise = model._denoising_step
    
    # Get both conditionings
    orig_cond = model._get_text_conditioning(original_prompt, cfg_weight=7.5)
    repl_cond = model._get_text_conditioning(replacement_prompt, cfg_weight=7.5)
    
    # Track denoising progress
    step_counter = [0]
    total_steps = 20
    
    def patched_denoise(x_t, t, t_prev, conditioning, cfg_weight=7.5, text_time=None):
        """Patched denoising that swaps conditioning."""
        step_counter[0] += 1
        progress = step_counter[0] / total_steps
        
        # Replace conditioning for first 70% of denoising
        if progress <= 0.7:
            # Use replacement conditioning
            print(f"   Step {step_counter[0]}/{total_steps}: Using replacement conditioning")
            return original_denoise(x_t, t, t_prev, repl_cond, cfg_weight, text_time)
        else:
            # Use original for final details
            return original_denoise(x_t, t, t_prev, conditioning, cfg_weight, text_time)
    
    # Monkey-patch the method
    model._denoising_step = patched_denoise
    step_counter[0] = 0
    
    # Generate with original prompt but swapped conditioning
    swapped = model.generate(
        prompt=original_prompt,  # Original prompt but will use replacement conditioning
        num_steps=20,
        cfg_weight=7.5,
        seed=42,
        width=512,
        height=512
    )
    
    # Restore original method
    model._denoising_step = original_denoise
    
    Image.fromarray((swapped * 255).astype(np.uint8)).save("direct_swapped_apple_to_banana.png")
    print("\n   Saved: direct_swapped_apple_to_banana.png")
    
    print("\n" + "=" * 50)
    print("Test Complete!")
    print("\nCheck the images:")
    print("  - direct_baseline_apple.png: Should show APPLE")
    print("  - direct_reference_banana.png: Should show BANANA")
    print("  - direct_swapped_apple_to_banana.png: Should show BANANA if swap worked!")
    print("\nIf swapped image shows banana, semantic replacement is working!")


if __name__ == "__main__":
    test_direct_conditioning_swap()