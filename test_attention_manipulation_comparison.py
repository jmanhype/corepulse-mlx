#!/usr/bin/env python3
"""
Test: Attention Manipulation Comparison
Creates side-by-side comparison of different attention manipulation techniques.
Similar to CorePulse's attention_manipulation_image.png
"""

import sys
import gc
from pathlib import Path
import mlx.core as mx
import PIL.Image
import numpy as np

# Add the stable_diffusion module to path
sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')

# Enable hooks BEFORE importing model
from stable_diffusion import attn_scores
attn_scores.enable_kv_hooks(True)

# Import model components
from stable_diffusion import StableDiffusionXL

def create_attention_suppression_hook(strength=0.3):
    """Suppress attention scores globally."""
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            # Suppress attention by scaling K
            k_suppressed = k * (1.0 - strength)
            return q, k_suppressed, v
        return q, k, v
    return hook

def create_attention_amplification_hook(strength=2.0):
    """Amplify attention scores globally."""
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            # Amplify attention by scaling K
            k_amplified = k * strength
            return q, k_amplified, v
        return q, k, v
    return hook

def create_attention_shift_hook(shift_amount=0.5):
    """Shift attention patterns."""
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = k.shape
            
            # Shift attention by rotating K
            shift_idx = int(seq_len * shift_amount)
            k_shifted = mx.roll(k, shift_idx, axis=2)
            
            return q, k_shifted, v
        return q, k, v
    return hook

def create_attention_focus_hook(focus_center=0.5, focus_width=0.3):
    """Focus attention on a specific region."""
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = k.shape
            
            # Create focus mask
            center_idx = int(seq_len * focus_center)
            width = int(seq_len * focus_width)
            
            mask = mx.zeros((1, 1, seq_len, 1))
            start = max(0, center_idx - width // 2)
            end = min(seq_len, center_idx + width // 2)
            
            for i in range(start, end):
                mask[:, :, i, :] = 1.0
            
            # Apply mask to K
            k_focused = k * mask
            
            return q, k_focused, v
        return q, k, v
    return hook

def create_comparison_grid(images, labels, output_path):
    """Create a 2x2 grid comparison image."""
    # Assuming images is a list of 4 PIL images
    width = images[0].width
    height = images[0].height
    
    # Create a new image for the grid
    grid_img = PIL.Image.new('RGB', (width * 2, height * 2))
    
    # Paste images in grid
    grid_img.paste(images[0], (0, 0))
    grid_img.paste(images[1], (width, 0))
    grid_img.paste(images[2], (0, height))
    grid_img.paste(images[3], (width, height))
    
    # Save grid
    grid_img.save(output_path)
    print(f"✅ Saved comparison grid: {output_path}")

def main():
    print("🎯 Test: Attention Manipulation Comparison")
    print("=" * 60)
    
    # Configuration
    prompt = "a majestic castle on a hilltop at sunset"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    print(f"📝 Prompt: '{prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Create output directory
    output_dir = Path("artifacts/images/attention_manipulation_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    comparison_images = []
    comparison_labels = []
    
    # Test 1: Baseline (No Manipulation)
    print("\n🎨 Test 1/4: Baseline (No Manipulation)...")
    attn_scores.KV_REGISTRY.clear()
    
    latents = model.generate_latents(
        prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "01_baseline.png")
    comparison_images.append(pil_img)
    comparison_labels.append("Baseline")
    print("✅ Saved: 01_baseline.png")
    
    # Test 2: Attention Suppression
    print("\n🎨 Test 2/4: Attention Suppression...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_attention_suppression_hook(strength=0.5)
    for block in ['down_1', 'down_2', 'mid', 'up_0', 'up_1']:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "02_suppressed.png")
    comparison_images.append(pil_img)
    comparison_labels.append("Suppressed")
    print("✅ Saved: 02_suppressed.png")
    
    # Test 3: Attention Amplification
    print("\n🎨 Test 3/4: Attention Amplification...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_attention_amplification_hook(strength=2.0)
    for block in ['down_1', 'down_2', 'mid', 'up_0', 'up_1']:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "03_amplified.png")
    comparison_images.append(pil_img)
    comparison_labels.append("Amplified")
    print("✅ Saved: 03_amplified.png")
    
    # Test 4: Attention Focus
    print("\n🎨 Test 4/4: Attention Focus...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_attention_focus_hook(focus_center=0.5, focus_width=0.3)
    for block in ['down_1', 'down_2', 'mid', 'up_0', 'up_1']:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "04_focused.png")
    comparison_images.append(pil_img)
    comparison_labels.append("Focused")
    print("✅ Saved: 04_focused.png")
    
    # Create comparison grid
    print("\n🔄 Creating comparison grid...")
    create_comparison_grid(
        comparison_images,
        comparison_labels,
        output_dir / "attention_manipulation_comparison.png"
    )
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n" + "=" * 60)
    print("✅ Attention Manipulation Comparison Complete!")
    print("📊 Results:")
    print("  01_baseline.png: Normal generation")
    print("  02_suppressed.png: Suppressed attention")
    print("  03_amplified.png: Amplified attention")
    print("  04_focused.png: Focused attention")
    print("  attention_manipulation_comparison.png: 2x2 comparison grid")
    print("\n💡 This demonstrates different attention manipulation techniques!")
    print("🎯 Shows the impact of attention control on generation!")

if __name__ == "__main__":
    main()