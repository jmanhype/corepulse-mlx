#!/usr/bin/env python3
"""Simple test of replacement functionality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from corpus_mlx import create_semantic_wrapper

def test_replacement():
    print("Creating wrapper...")
    wrapper = create_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
    
    print("Adding replacement...")
    wrapper.add_replacement("apple", "banana") 
    wrapper.enable()
    
    test_prompt = "a red apple on table"
    print(f"Original prompt: '{test_prompt}'")
    
    # Test the _apply_replacements method directly
    modified, replaced = wrapper._apply_replacements(test_prompt)
    print(f"Modified prompt: '{modified}'")
    print(f"Was replaced: {replaced}")
    
    # Test that the patched method gets called
    print("\nTesting patched generate_latents...")
    print("This should show the modified prompt:")
    
    try:
        # Just get the first step to see if prompt modification happens
        gen = wrapper.wrapper.generate_latents(test_prompt, num_steps=1, seed=42)
        next(gen)
        print("✅ Generation started - check logs above for prompt modification")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_replacement()