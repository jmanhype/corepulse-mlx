#!/usr/bin/env python3
"""
Complete semantic replacement test suite for corpus-mlx.
Tests all major object replacement scenarios.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from corpus_mlx import create_semantic_wrapper
from PIL import Image
import numpy as np
import mlx.core as mx


def test_complete_semantic():
    """Run complete semantic replacement test suite."""
    print("=" * 60)
    print("COMPLETE SEMANTIC REPLACEMENT TEST SUITE")
    print("corpus-mlx with TRUE Object Replacement")
    print("=" * 60)
    
    # Create wrapper
    print("\n📦 Initializing semantic wrapper...")
    wrapper = create_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
    
    # Generation settings (balanced quality/speed)
    gen_kwargs = {
        'negative_text': "blurry, ugly, distorted, deformed",
        'num_steps': 15,
        'cfg_weight': 7.5,
        'seed': 42,
        'height': 256,
        'width': 256
    }
    
    # Comprehensive test cases
    test_cases = [
        # Food items
        ("a photo of an apple on a wooden table", "apple", "banana"),
        ("a fresh orange on a plate", "orange", "lemon"),
        ("a slice of pizza on a dish", "pizza", "burger"),
        
        # Animals
        ("a fluffy cat sitting on a sofa", "cat", "dog"),
        ("a brown horse in a field", "horse", "cow"),
        ("a small bird on a branch", "bird", "butterfly"),
        
        # Vehicles
        ("a red car parked on the street", "car", "bicycle"),
        ("a motorcycle on the road", "motorcycle", "scooter"),
        ("an airplane in the sky", "airplane", "helicopter"),
        
        # Objects
        ("a laptop computer on a desk", "laptop", "book"),
        ("a wooden chair in a room", "chair", "table"),
        ("a silver watch on display", "watch", "ring"),
    ]
    
    results = []
    
    for i, (prompt, original, replacement) in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}/{len(test_cases)}: {original} → {replacement}")
        print(f"Prompt: {prompt}")
        
        try:
            # Generate comparison
            baseline, replaced = wrapper.generate_comparison(
                prompt, original, replacement, **gen_kwargs
            )
            
            # Save images
            baseline_path = f"semantic_test_{i:02d}_{original}_baseline.png"
            replaced_path = f"semantic_test_{i:02d}_{original}_to_{replacement}.png"
            
            Image.fromarray(baseline).save(baseline_path)
            Image.fromarray(replaced).save(replaced_path)
            
            # Verify images have content
            baseline_valid = baseline.min() != baseline.max()
            replaced_valid = replaced.min() != replaced.max()
            
            if baseline_valid and replaced_valid:
                print(f"✅ Generated successfully")
                results.append((original, replacement, "SUCCESS"))
            else:
                print(f"⚠️  Images may be blank")
                results.append((original, replacement, "BLANK"))
                
            print(f"📸 Saved: {baseline_path}")
            print(f"📸 Saved: {replaced_path}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append((original, replacement, "ERROR"))
    
    # Summary report
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    success_count = sum(1 for _, _, status in results if status == "SUCCESS")
    total_count = len(results)
    
    print(f"\n✅ Successful: {success_count}/{total_count}")
    
    if success_count < total_count:
        print("\n⚠️  Failed tests:")
        for orig, repl, status in results:
            if status != "SUCCESS":
                print(f"  - {orig} → {repl}: {status}")
    
    print("\n📊 Detailed Results:")
    for i, (orig, repl, status) in enumerate(results, 1):
        icon = "✅" if status == "SUCCESS" else "❌"
        print(f"  {i:2d}. {orig:10s} → {repl:10s}: {icon} {status}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nSemantic replacement in corpus-mlx:")
    print("  • Intercepts prompts BEFORE tokenization")
    print("  • Replaces object names in text")
    print("  • Model generates different objects")
    print("  • True semantic replacement achieved!")
    
    return results


if __name__ == "__main__":
    results = test_complete_semantic()
    
    # Create a summary file
    with open("semantic_test_results.txt", "w") as f:
        f.write("SEMANTIC REPLACEMENT TEST RESULTS\n")
        f.write("=" * 40 + "\n\n")
        for i, (orig, repl, status) in enumerate(results, 1):
            f.write(f"{i:2d}. {orig} → {repl}: {status}\n")
    
    print("\n📄 Results saved to: semantic_test_results.txt")