#!/usr/bin/env python3
"""
Test semantic object replacement with conditioning replacement.
This should actually replace objects by swapping text embeddings.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from corpus_mlx.semantic_replacement import create_semantic_sd
from PIL import Image
import numpy as np


def test_semantic_replacement():
    """Test true semantic object replacement."""
    print("=" * 60)
    print("Testing Semantic Object Replacement with Conditioning Swap")
    print("=" * 60)
    
    # Create model with semantic replacement capability
    print("\nInitializing model with semantic replacement...")
    base_model, wrapper = create_semantic_sd("stable-diffusion-xl-base-1.0")
    
    # Test cases
    test_cases = [
        ("apple", "banana", "a photo of a {} on a wooden table"),
        ("cat", "dog", "a fluffy {} sitting on a sofa"),
        ("car", "bicycle", "a red {} parked on the street"),
    ]
    
    for original, replacement, prompt_template in test_cases:
        print(f"\n{'-'*50}")
        print(f"Test: {original} -> {replacement}")
        print(f"Template: {prompt_template}")
        
        # Generate baseline (what original looks like)
        original_prompt = prompt_template.format(original)
        print(f"\n1. Generating baseline: {original_prompt}")
        
        wrapper.reset()
        baseline_img = wrapper.generate(
            prompt=original_prompt,
            num_steps=30,
            cfg_weight=7.5,
            negative_prompt="blurry, ugly, distorted",
            seed=42,
            width=512,
            height=512
        )
        
        baseline_path = f"semantic_baseline_{original}.png"
        Image.fromarray((baseline_img * 255).astype(np.uint8)).save(baseline_path)
        print(f"   Saved: {baseline_path}")
        
        # Generate with semantic replacement
        print(f"\n2. Testing semantic replacement: {original} -> {replacement}")
        
        # Add replacement configuration
        wrapper.reset()
        wrapper.add_semantic_replacement(
            original_object=original,
            replacement_object=replacement,
            strength=1.0,  # Full replacement
            start_frac=0.0,
            end_frac=0.7   # Replace early in generation
        )
        
        # Generate with original prompt - should produce replacement object!
        replaced_img = wrapper.generate(
            prompt=original_prompt,  # Still use original prompt!
            num_steps=30,
            cfg_weight=7.5,
            negative_prompt="blurry, ugly, distorted",
            seed=42,
            width=512,
            height=512
        )
        
        replaced_path = f"semantic_replaced_{original}_to_{replacement}.png"
        Image.fromarray((replaced_img * 255).astype(np.uint8)).save(replaced_path)
        print(f"   Saved: {replaced_path}")
        
        # Also generate direct reference (what replacement should look like)
        replacement_prompt = prompt_template.format(replacement)
        print(f"\n3. Generating reference: {replacement_prompt}")
        
        wrapper.reset()
        reference_img = wrapper.generate(
            prompt=replacement_prompt,
            num_steps=30,
            cfg_weight=7.5,
            negative_prompt="blurry, ugly, distorted",
            seed=42,
            width=512,
            height=512
        )
        
        reference_path = f"semantic_reference_{replacement}.png"
        Image.fromarray((reference_img * 255).astype(np.uint8)).save(reference_path)
        print(f"   Saved: {reference_path}")
        
        print(f"\nCompleted {original} -> {replacement}")
        print(f"Check these images:")
        print(f"  - Baseline (should show {original}): {baseline_path}")
        print(f"  - Replaced (should show {replacement}): {replaced_path}")
        print(f"  - Reference (definitely shows {replacement}): {reference_path}")
    
    print("\n" + "=" * 60)
    print("Semantic Replacement Test Complete!")
    print("=" * 60)
    print("\nSuccess Criteria:")
    print("  ✅ Replaced images should show the replacement object")
    print("  ✅ Should match reference images closely")
    print("  ❌ If still showing original object, replacement failed")


if __name__ == "__main__":
    test_semantic_replacement()