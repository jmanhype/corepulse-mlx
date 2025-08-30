#!/usr/bin/env python3
"""
Minimal test of semantic replacement by swapping conditioning.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
from PIL import Image
import numpy as np


def test_minimal_semantic():
    """Minimal test of conditioning swap for semantic replacement."""
    print("Minimal Semantic Replacement Test")
    print("=" * 50)
    
    # Use corpus-mlx wrapper which has generate method
    from corpus_mlx import CorePulseStableDiffusion
    
    # Create wrapper with SD2.1
    from adapters.stable_diffusion import StableDiffusion
    base_sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base")
    model = CorePulseStableDiffusion(base_sd)
    
    # Test case
    original_prompt = "a photo of an apple on a wooden table"
    replacement_prompt = "a photo of a banana on a wooden table"
    
    print(f"\nOriginal: {original_prompt}")
    print(f"Target: {replacement_prompt}")
    
    # Monkey-patch to intercept and replace conditioning
    original_get_cond = base_sd._get_text_conditioning
    
    # State tracking
    state = {'use_replacement': False, 'step': 0, 'total_steps': 20}
    
    def patched_get_cond(text, n_images=1, cfg_weight=7.5, negative_text=""):
        """Patched method that can return replacement conditioning."""
        if state['use_replacement'] and text == original_prompt:
            print(f"   🔄 Replacing conditioning: apple -> banana")
            # Return conditioning for replacement prompt instead
            return original_get_cond(replacement_prompt, n_images, cfg_weight, negative_text)
        return original_get_cond(text, n_images, cfg_weight, negative_text)
    
    # Apply patch
    base_sd._get_text_conditioning = patched_get_cond
    
    # Test 1: Generate with original (baseline)
    print("\n1. Baseline generation (should show APPLE):")
    state['use_replacement'] = False
    latents = None
    for step_latents in model.generate_latents(
        original_prompt,
        negative_text="blurry, ugly",
        num_steps=20,
        cfg_weight=7.5,
        seed=42,
        height=512,
        width=512
    ):
        latents = step_latents
    
    # Decode latents to image
    images = base_sd.autoencoder.decode(latents)
    baseline = mx.concatenate(images, axis=0)[0]  # Get first image
    baseline = ((baseline + 1) * 127.5).astype(mx.uint8)
    baseline = np.array(baseline)
    Image.fromarray(baseline).save("minimal_baseline_apple.png")
    print("   Saved: minimal_baseline_apple.png")
    
    # Test 2: Generate with conditioning replacement
    print("\n2. With conditioning replacement (should show BANANA):")
    state['use_replacement'] = True
    latents = None
    for step_latents in model.generate_latents(
        original_prompt,  # Same prompt!
        negative_text="blurry, ugly",
        num_steps=20,
        cfg_weight=7.5,
        seed=42,
        height=512,
        width=512
    ):
        latents = step_latents
    
    images = base_sd.autoencoder.decode(latents)
    replaced = mx.concatenate(images, axis=0)[0]
    replaced = ((replaced + 1) * 127.5).astype(mx.uint8)
    replaced = np.array(replaced)
    Image.fromarray(replaced).save("minimal_replaced_banana.png")
    print("   Saved: minimal_replaced_banana.png")
    
    # Test 3: Direct generation with banana prompt (reference)
    print("\n3. Reference generation (definitely shows BANANA):")
    state['use_replacement'] = False
    latents = None
    for step_latents in model.generate_latents(
        replacement_prompt,
        negative_text="blurry, ugly",
        num_steps=20,
        cfg_weight=7.5,
        seed=42,
        height=512,
        width=512
    ):
        latents = step_latents
    
    images = base_sd.autoencoder.decode(latents)
    reference = mx.concatenate(images, axis=0)[0]
    reference = ((reference + 1) * 127.5).astype(mx.uint8)
    reference = np.array(reference)
    Image.fromarray(reference).save("minimal_reference_banana.png")
    print("   Saved: minimal_reference_banana.png")
    
    # Restore original
    base_sd._get_text_conditioning = original_get_cond
    
    print("\n" + "=" * 50)
    print("Test Complete!")
    print("\n✅ Success if:")
    print("  - minimal_baseline_apple.png shows APPLE")
    print("  - minimal_replaced_banana.png shows BANANA")
    print("  - minimal_reference_banana.png shows BANANA")
    print("\n❌ Failed if:")
    print("  - minimal_replaced_banana.png still shows APPLE")
    print("\nThis proves whether conditioning replacement can achieve semantic object replacement.")


if __name__ == "__main__":
    test_minimal_semantic()