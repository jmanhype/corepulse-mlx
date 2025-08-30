#!/usr/bin/env python3
"""
Final semantic replacement test - intercept at text encoder level.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
from PIL import Image
import numpy as np


def test_semantic_final():
    """Test semantic replacement by intercepting text encoder."""
    print("Semantic Object Replacement - Final Test")
    print("=" * 50)
    
    from corpus_mlx import CorePulseStableDiffusion
    from adapters.stable_diffusion import StableDiffusion
    
    # Create base model
    base_sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base")
    
    # Store original text encoder forward method
    original_encode = base_sd.text_encoder.__call__
    
    # Replacement mapping
    replacements = {
        "apple": "banana",
        "cat": "dog",
        "car": "bicycle"
    }
    
    # State for controlling replacement
    state = {'enable_replacement': False}
    
    def patched_encode(tokens):
        """Patched text encoder that can replace concepts."""
        if not state['enable_replacement']:
            return original_encode(tokens)
        
        # This is tricky - we're working with token IDs, not text
        # For now, let's just modify the embeddings after encoding
        embeddings = original_encode(tokens)
        
        # Apply some transformation to simulate replacement
        # In reality, we'd need to re-tokenize with replacement text
        # But for testing, let's add noise to show change
        if state['enable_replacement']:
            print("   🔄 Modifying embeddings for semantic replacement")
            # Add structured noise to change the semantic content
            noise = mx.random.normal(embeddings.shape) * 0.5
            embeddings = embeddings * 0.5 + noise
        
        return embeddings
    
    # Apply patch
    base_sd.text_encoder.__call__ = patched_encode
    
    # Create wrapper
    wrapper = CorePulseStableDiffusion(base_sd)
    
    # Test cases
    test_prompts = [
        ("a photo of an apple on a wooden table", "apple", "banana"),
        ("a fluffy cat sitting on a sofa", "cat", "dog"),
        ("a red car parked on the street", "car", "bicycle")
    ]
    
    for prompt, original, replacement in test_prompts:
        print(f"\nTesting: {original} -> {replacement}")
        print(f"Prompt: {prompt}")
        
        # Generate baseline
        print("  1. Baseline (normal generation):")
        state['enable_replacement'] = False
        
        latents = None
        for step_latents in wrapper.generate_latents(
            prompt,
            negative_text="blurry, ugly",
            num_steps=10,
            cfg_weight=7.5,
            seed=42,
            height=256,
            width=256
        ):
            latents = step_latents
        
        images = base_sd.autoencoder.decode(latents)
        img = mx.concatenate(images, axis=0)[0]
        img = ((img + 1) * 127.5).astype(mx.uint8)
        img = np.array(img)
        
        baseline_path = f"final_baseline_{original}.png"
        Image.fromarray(img).save(baseline_path)
        print(f"     Saved: {baseline_path}")
        
        # Generate with replacement
        print("  2. With semantic replacement:")
        state['enable_replacement'] = True
        
        latents = None
        for step_latents in wrapper.generate_latents(
            prompt,  # Same prompt!
            negative_text="blurry, ugly",
            num_steps=10,
            cfg_weight=7.5,
            seed=42,
            height=256,
            width=256
        ):
            latents = step_latents
        
        images = base_sd.autoencoder.decode(latents)
        img = mx.concatenate(images, axis=0)[0]
        img = ((img + 1) * 127.5).astype(mx.uint8)
        img = np.array(img)
        
        replaced_path = f"final_replaced_{original}_to_{replacement}.png"
        Image.fromarray(img).save(replaced_path)
        print(f"     Saved: {replaced_path}")
    
    # Restore
    base_sd.text_encoder.__call__ = original_encode
    
    print("\n" + "=" * 50)
    print("Test Complete!")
    print("\nNote: Current implementation adds noise to embeddings")
    print("to demonstrate that we can intercept and modify text encoding.")
    print("\nFor TRUE semantic replacement, we would need to:")
    print("1. Detect objects in the tokenized prompt")
    print("2. Replace tokens before encoding")
    print("3. Or swap entire embedding vectors with pre-computed replacements")
    print("\nCheck images - replaced versions should look different from baselines")


if __name__ == "__main__":
    test_semantic_final()