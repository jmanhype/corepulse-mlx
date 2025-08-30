#!/usr/bin/env python3
"""
Debug test for semantic replacement - trace the conditioning flow.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
from PIL import Image
import numpy as np


def test_semantic_debug():
    """Debug semantic replacement to see where conditioning flows."""
    print("Semantic Replacement Debug Test")
    print("=" * 50)
    
    from corpus_mlx import CorePulseStableDiffusion
    from adapters.stable_diffusion import StableDiffusion
    
    # Create base model
    base_sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base")
    
    # Store original methods
    original_get_cond = base_sd._get_text_conditioning
    original_denoising = base_sd._denoising_step
    
    # Debug tracking
    debug_info = {
        'get_cond_calls': [],
        'denoising_calls': [],
        'step': 0
    }
    
    def debug_get_cond(text, n_images=1, cfg_weight=7.5, negative_text=""):
        """Track conditioning calls."""
        debug_info['get_cond_calls'].append(text)
        print(f"   _get_text_conditioning called with: '{text[:30]}...'")
        
        # For testing: replace apple with banana
        if "apple" in text:
            print(f"   🔄 REPLACING: apple -> banana")
            text = text.replace("apple", "banana")
        
        return original_get_cond(text, n_images, cfg_weight, negative_text)
    
    def debug_denoising(x_t, t, t_prev, conditioning, cfg_weight=7.5, text_time=None):
        """Track denoising calls."""
        debug_info['step'] += 1
        debug_info['denoising_calls'].append(debug_info['step'])
        
        # Only print first few steps
        if debug_info['step'] <= 3:
            print(f"   Step {debug_info['step']}: denoising with conditioning shape {conditioning.shape}")
        
        return original_denoising(x_t, t, t_prev, conditioning, cfg_weight, text_time)
    
    # Apply debug patches
    base_sd._get_text_conditioning = debug_get_cond
    base_sd._denoising_step = debug_denoising
    
    # Create wrapper
    wrapper = CorePulseStableDiffusion(base_sd)
    
    print("\n1. Testing with 'apple' prompt:")
    print("   Should see replacement to 'banana'")
    
    # Reset debug info
    debug_info['get_cond_calls'] = []
    debug_info['denoising_calls'] = []
    debug_info['step'] = 0
    
    # Generate
    latents = None
    for step_latents in wrapper.generate_latents(
        "a photo of an apple on a wooden table",
        negative_text="blurry",
        num_steps=5,  # Just a few steps for debug
        cfg_weight=7.5,
        seed=42
    ):
        latents = step_latents
    
    # Decode
    print("\n   Decoding latents...")
    images = base_sd.autoencoder.decode(latents)
    img = mx.concatenate(images, axis=0)[0]
    img = ((img + 1) * 127.5).astype(mx.uint8)
    img = np.array(img)
    Image.fromarray(img).save("debug_semantic_test.png")
    print("   Saved: debug_semantic_test.png")
    
    # Report
    print("\n" + "=" * 50)
    print("Debug Report:")
    print(f"  - get_text_conditioning called {len(debug_info['get_cond_calls'])} times")
    print(f"  - Prompts seen: {debug_info['get_cond_calls']}")
    print(f"  - Denoising steps: {len(debug_info['denoising_calls'])}")
    
    # Restore
    base_sd._get_text_conditioning = original_get_cond
    base_sd._denoising_step = original_denoising
    
    print("\n✅ If you saw '🔄 REPLACING' messages, conditioning replacement is working")
    print("❌ If no replacement messages, the flow is different than expected")


if __name__ == "__main__":
    test_semantic_debug()