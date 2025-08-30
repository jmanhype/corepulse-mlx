#!/usr/bin/env python3
"""Fixed test for prompt injection - enable hooks BEFORE creating model."""

import sys
sys.path.insert(0, 'src/adapters/mlx/mlx-examples/stable_diffusion')

# CRITICAL: Import and enable hooks BEFORE importing model!
from stable_diffusion import attn_scores
attn_scores.enable_kv_hooks(True)
print("✅ Enabled KV hooks BEFORE model creation")

from corpus_mlx import CorePulse
from stable_diffusion import StableDiffusionXL
from pathlib import Path
from PIL import Image
import mlx.core as mx
import numpy as np

def save_image(img_array, path):
    """Save MLX array as image."""
    # Convert to numpy and proper format
    img = np.array(img_array)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    

def test_injection():
    """Test prompt injection with proper hook setup."""
    
    # NOW create model (with PatchedMHA enabled)
    print("\nLoading SDXL-Turbo...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Create CorePulse wrapper
    corepulse = CorePulse(model)
    
    output_dir = Path("examples/output/injection_test_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test 1: Baseline (no injection)
    print("\n1. Generating baseline red apple...")
    corepulse.clear()
    
    latents = model.generate_latents(
        "red apple",
        num_steps=2,
        cfg_weight=0.0,
        seed=42
    )
    
    for i, x in enumerate(latents):
        if i == 1:  # Get final
            img = model.decode(x)
            save_image(img[0], output_dir / "1_baseline_apple.png")
            print("   ✅ Saved baseline")
    
    # Test 2: With banana injection
    print("\n2. Injecting 'yellow banana'...")
    corepulse.clear()
    corepulse.add_injection(
        prompt="yellow banana",
        strength=0.5,
        blocks=["mid", "up_0", "up_1"]
    )
    
    latents = model.generate_latents(
        "red apple",  # Still asking for apple
        num_steps=2,
        cfg_weight=0.0,
        seed=42
    )
    
    for i, x in enumerate(latents):
        if i == 1:
            img = model.decode(x)
            save_image(img[0], output_dir / "2_banana_injection.png")
            print("   ✅ Saved with banana injection")
    
    # Test 3: With orange injection
    print("\n3. Injecting 'orange fruit'...")
    corepulse.clear()
    corepulse.add_injection(
        prompt="orange fruit",
        strength=0.5,
        blocks=["mid", "up_0", "up_1"]
    )
    
    latents = model.generate_latents(
        "red apple",
        num_steps=2,
        cfg_weight=0.0,
        seed=42
    )
    
    for i, x in enumerate(latents):
        if i == 1:
            img = model.decode(x)
            save_image(img[0], output_dir / "3_orange_injection.png")
            print("   ✅ Saved with orange injection")
    
    print("\n✨ Test complete! Check examples/output/injection_test_fixed/")
    

if __name__ == "__main__":
    test_injection()