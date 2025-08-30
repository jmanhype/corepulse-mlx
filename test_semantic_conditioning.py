#!/usr/bin/env python3
"""
Test semantic object replacement using conditioning replacement.
This replaces text embeddings like CorePulse, not attention values.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from corpus_mlx import CorePulseStableDiffusion
from corpus_mlx.injection_semantic import create_semantic_replacement
import mlx.core as mx
from PIL import Image
import numpy as np


def test_semantic_replacement():
    """Test semantic object replacement with conditioning replacement."""
    print("Initializing CorePulse × MLX...")
    model = CorePulseStableDiffusion("stable-diffusion-xl-base-1.0")
    
    # Test cases: original -> replacement
    test_cases = [
        ("apple", "banana", "a photo of a {} on a wooden table"),
        ("cat", "dog", "a fluffy {} sitting on a sofa"),
        ("car", "bicycle", "a red {} parked on the street"),
    ]
    
    for original, replacement, prompt_template in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing: {original} -> {replacement}")
        print(f"Prompt template: {prompt_template}")
        
        # Generate with original object (baseline)
        original_prompt = prompt_template.format(original)
        print(f"\nGenerating baseline with: {original_prompt}")
        
        baseline_image = model.generate(
            prompt=original_prompt,
            num_steps=30,
            cfg_weight=7.5,
            negative_prompt="blurry, ugly, distorted",
            seed=42,
            width=512,
            height=512
        )
        
        # Save baseline
        baseline_path = f"semantic_baseline_{original}.png"
        Image.fromarray((baseline_image * 255).astype(np.uint8)).save(baseline_path)
        print(f"Baseline saved to {baseline_path}")
        
        # Now test with semantic replacement
        # We need to modify the model to use conditioning replacement
        
        # The key insight from CorePulse: they encode the replacement prompt
        # and inject those embeddings at specific UNet blocks
        
        # For corpus-mlx, we need to:
        # 1. Get embeddings for replacement prompt  
        # 2. Replace original embeddings during cross-attention
        
        replacement_prompt = prompt_template.format(replacement)
        print(f"\nAttempting semantic replacement to: {replacement_prompt}")
        
        # Method 1: High-strength injection (our current approach)
        print("Method 1: High-strength attention injection...")
        model.reset_injections()
        model.add_injection(
            prompt=replacement_prompt,
            strength=2.0,  # Maximum strength
            blocks=['mid', 'down_0', 'down_1', 'up_0', 'up_1'],  # All blocks
        )
        
        method1_image = model.generate(
            prompt=original_prompt,  # Still use original prompt
            num_steps=30,
            cfg_weight=7.5,
            negative_prompt="blurry, ugly, distorted",
            seed=42,
            width=512,
            height=512
        )
        
        method1_path = f"semantic_method1_{original}_to_{replacement}.png"
        Image.fromarray((method1_image * 255).astype(np.uint8)).save(method1_path)
        print(f"Method 1 saved to {method1_path}")
        
        # Method 2: Direct prompt replacement (sanity check)
        print("\nMethod 2: Direct prompt replacement (sanity check)...")
        model.reset_injections()
        
        direct_image = model.generate(
            prompt=replacement_prompt,  # Use replacement directly
            num_steps=30,
            cfg_weight=7.5,
            negative_prompt="blurry, ugly, distorted",
            seed=42,
            width=512,
            height=512
        )
        
        direct_path = f"semantic_direct_{replacement}.png"
        Image.fromarray((direct_image * 255).astype(np.uint8)).save(direct_path)
        print(f"Direct generation saved to {direct_path}")
        
        # Method 3: Conditioning replacement (new approach)
        # This would require modifying the model internals to replace
        # text embeddings before they enter cross-attention
        print("\nMethod 3: Conditioning replacement (CorePulse-style)...")
        print("NOTE: This requires deeper model modifications")
        
        # For now, let's try an extreme injection approach
        # We'll inject at ALL blocks with maximum strength
        # and try to override the original conditioning
        
        model.reset_injections()
        
        # Add multiple injections to overwhelm original
        for i in range(3):  # Multiple injection passes
            model.add_injection(
                prompt=replacement_prompt,
                strength=2.0,
                blocks=['mid', 'down_0', 'down_1', 'down_2', 'up_0', 'up_1', 'up_2'],
            )
        
        method3_image = model.generate(
            prompt=original_prompt,
            num_steps=30,
            cfg_weight=7.5,
            negative_prompt="blurry, ugly, distorted",
            seed=42,
            width=512,
            height=512
        )
        
        method3_path = f"semantic_method3_{original}_to_{replacement}.png"
        Image.fromarray((method3_image * 255).astype(np.uint8)).save(method3_path)
        print(f"Method 3 saved to {method3_path}")
        
        print(f"\nCompleted {original} -> {replacement} test")
        print(f"Check images to see if semantic replacement worked:")
        print(f"  - Baseline: {baseline_path}")
        print(f"  - Method 1 (high strength): {method1_path}")
        print(f"  - Method 2 (direct): {direct_path}")
        print(f"  - Method 3 (multiple): {method3_path}")


if __name__ == "__main__":
    test_semantic_replacement()
    print("\n" + "="*50)
    print("All tests complete! Check generated images for results.")
    print("\nExpected results for successful semantic replacement:")
    print("  - apple -> banana: Should show banana instead of apple")
    print("  - cat -> dog: Should show dog instead of cat")
    print("  - car -> bicycle: Should show bicycle instead of car")
    print("\nIf objects didn't change, semantic replacement isn't working.")