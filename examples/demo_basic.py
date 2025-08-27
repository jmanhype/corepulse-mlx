#!/usr/bin/env python3
"""
Basic CorePulse demo showing simple prompt injection.
Uses the proven approach from our test files.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sys
import gc
from pathlib import Path

# Add paths
sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')

# Enable hooks BEFORE importing model
from stable_diffusion import attn_scores
attn_scores.enable_kv_hooks(True)

# Import model components  
from stable_diffusion import StableDiffusionXL
import PIL.Image


def create_injection_hook(inject_strength=0.3):
    """Create a simple injection hook for demonstration."""
    def hook(q, k, v, meta=None):
        # Only modify cross-attention
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            # Simple amplification to show the effect
            print(f"    ✨ Injecting at {meta.get('block_id', 'unknown')} with strength {inject_strength}")
            return q, k * (1 + inject_strength), v * (1 + inject_strength)
        return q, k, v
    return hook


def main():
    """Run basic injection demo."""
    
    # Load SDXL model
    print("Loading SDXL-Turbo model...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Output directory
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    prompt = "majestic lion in african savanna, sunset"
    seed = 42
    num_steps = 4  # SDXL-Turbo optimized for 4 steps
    cfg_weight = 0.0  # SDXL-Turbo doesn't use CFG
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Steps: {num_steps}, Seed: {seed}")
    
    # Generate baseline first
    print("\n=== Generating Baseline (No Injection) ===")
    attn_scores.KV_REGISTRY.clear()
    
    latents = model.generate_latents(
        prompt, 
        num_steps=num_steps, 
        cfg_weight=cfg_weight, 
        seed=seed
    )
    for i, x in enumerate(latents):
        if i == num_steps - 1:  # Last step
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "demo_basic_baseline.png")
    print("✅ Saved: demo_basic_baseline.png")
    
    # Generate with injection
    print("\n=== Generating with Injection (Enhanced) ===")
    attn_scores.KV_REGISTRY.clear()
    
    # Apply injection to mid and upper blocks for style influence
    injection_hook = create_injection_hook(0.5)
    for block in ["mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, injection_hook)
    
    latents = model.generate_latents(
        prompt, 
        num_steps=num_steps, 
        cfg_weight=cfg_weight, 
        seed=seed
    )
    for i, x in enumerate(latents):
        if i == num_steps - 1:  # Last step
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "demo_basic_injected.png")
    print("✅ Saved: demo_basic_injected.png")
    
    print("\n✨ Demo complete!")
    print(f"Images saved to: {output_dir}")
    print("\nCompare the two images to see the injection effect:")
    print("  - demo_basic_baseline.png: Standard generation")
    print("  - demo_basic_injected.png: With prompt injection (50% amplification)")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()


if __name__ == "__main__":
    main()