#!/usr/bin/env python3
"""
Test PROPER semantic replacement that replaces prompts before tokenization.
This should actually achieve apple->banana replacement!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from corpus_mlx.semantic_proper import create_semantic_wrapper
from PIL import Image


def test_proper_semantic():
    """Test the proper semantic replacement implementation."""
    print("=" * 60)
    print("PROPER Semantic Object Replacement Test")
    print("=" * 60)
    
    # Create wrapper with semantic replacement
    print("\nInitializing semantic wrapper...")
    wrapper = create_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
    
    # Test cases
    test_cases = [
        ("a photo of an apple on a wooden table", "apple", "banana"),
        ("a fluffy cat sitting on a sofa", "cat", "dog"),
        ("a red car parked on the street", "car", "bicycle")
    ]
    
    # Generation settings
    gen_kwargs = {
        'negative_text': "blurry, ugly, distorted",
        'num_steps': 20,
        'cfg_weight': 7.5,
        'seed': 42,
        'height': 256,
        'width': 256
    }
    
    for prompt, original, replacement in test_cases:
        print(f"\n{'='*50}")
        print(f"Test: {original} -> {replacement}")
        print(f"Prompt: {prompt}")
        
        # Generate comparison
        baseline, replaced = wrapper.generate_comparison(
            prompt, original, replacement, **gen_kwargs
        )
        
        # Save images
        baseline_path = f"proper_{original}_baseline.png"
        replaced_path = f"proper_{original}_to_{replacement}.png"
        
        Image.fromarray(baseline).save(baseline_path)
        Image.fromarray(replaced).save(replaced_path)
        
        print(f"\n📸 Saved:")
        print(f"   Baseline: {baseline_path}")
        print(f"   Replaced: {replaced_path}")
    
    # Also test with injections
    print(f"\n{'='*50}")
    print("Testing with prompt injection + semantic replacement:")
    
    # Clear and setup for injection test
    wrapper.wrapper.clear_injections()
    wrapper.replacements.clear()
    
    # Add an injection that will also be replaced
    wrapper.wrapper.add_injection(
        prompt="shiny metal apple",
        weight=0.5
    )
    
    # Add replacement
    wrapper.add_replacement("apple", "orange")
    wrapper.enable()
    
    print("\nGenerating with injection + replacement...")
    print("  Base: 'a photo of an apple'")
    print("  Injection: 'shiny metal apple' (will become 'shiny metal orange')")
    
    latents = None
    for step_latents in wrapper.wrapper.generate_latents(
        "a photo of an apple",
        **gen_kwargs
    ):
        latents = step_latents
    
    # Decode
    images = wrapper.wrapper.sd.autoencoder.decode(latents)
    img = images[0]
    img = mx.clip(img, -1, 1)
    img = ((img + 1) * 127.5).astype(mx.uint8)
    img = np.array(img)
    
    Image.fromarray(img).save("proper_injection_replaced.png")
    print("  Saved: proper_injection_replaced.png")
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE!")
    print("=" * 60)
    print("\nExpected Results:")
    print("  • Baseline images should show original objects (apple, cat, car)")
    print("  • Replaced images should show replacements (banana, dog, bicycle)")
    print("  • Injection test should show orange (not apple)")
    print("\nIf replacements worked, semantic object replacement is SUCCESS!")


if __name__ == "__main__":
    import mlx.core as mx
    import numpy as np
    test_proper_semantic()