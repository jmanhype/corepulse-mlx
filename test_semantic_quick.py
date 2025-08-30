#!/usr/bin/env python3
"""
Quick test of semantic replacement with fixed image conversion.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from corpus_mlx import create_semantic_wrapper
from PIL import Image


def test_quick():
    """Quick test with proper image conversion."""
    print("Quick Semantic Test - Apple to Banana")
    print("=" * 40)
    
    # Create wrapper
    wrapper = create_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
    
    # Quick generation settings (low quality for speed)
    gen_kwargs = {
        'negative_text': "blurry",
        'num_steps': 10,  # Very few steps
        'cfg_weight': 7.5,
        'seed': 42,
        'height': 256,
        'width': 256
    }
    
    # Test apple -> banana
    prompt = "a photo of an apple on a wooden table"
    
    print(f"\nPrompt: {prompt}")
    print("Generating baseline and replacement...")
    
    baseline, replaced = wrapper.generate_comparison(
        prompt, "apple", "banana", **gen_kwargs
    )
    
    # Save
    Image.fromarray(baseline).save("quick_apple_baseline.png")
    Image.fromarray(replaced).save("quick_apple_to_banana.png")
    
    print("\n✅ Saved:")
    print("  - quick_apple_baseline.png (should show apple)")
    print("  - quick_apple_to_banana.png (should show banana)")
    
    # Check if images are valid
    print(f"\nImage stats:")
    print(f"  Baseline: shape {baseline.shape}, range [{baseline.min()}-{baseline.max()}]")
    print(f"  Replaced: shape {replaced.shape}, range [{replaced.min()}-{replaced.max()}]")
    
    if baseline.min() == baseline.max():
        print("  ❌ Baseline is uniform color!")
    else:
        print("  ✅ Baseline has variation")
        
    if replaced.min() == replaced.max():
        print("  ❌ Replaced is uniform color!")
    else:
        print("  ✅ Replaced has variation")


if __name__ == "__main__":
    import mlx.core as mx
    import numpy as np
    test_quick()