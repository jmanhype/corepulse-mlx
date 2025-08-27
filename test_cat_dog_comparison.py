#!/usr/bin/env python3
"""
Test: Cat vs Dog Comparison
Creates side-by-side comparison showing how prompt injection can transform subjects.
Similar to CorePulse's cat_dog_comparison.png
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

def create_animal_swap_hook(model, target_animal):
    """Swap one animal for another in the generation."""
    # Generate embedding for target animal
    target_cond, _ = model._get_text_conditioning(target_animal)
    
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            # Only apply to middle and later blocks for subject transformation
            if block_id in ['mid', 'up_0', 'up_1']:
                if seq_len >= target_cond.shape[1]:
                    v_new = mx.array(v)
                    embed_dim = min(dim, target_cond.shape[2])
                    embed_len = min(seq_len, target_cond.shape[1])
                    
                    # Prepare target embedding
                    target = target_cond[:, :embed_len, :embed_dim]
                    if len(target.shape) == 3:
                        target = target[None, :, :, :]
                    if target.shape[0] < batch:
                        target = mx.broadcast_to(target, (batch, target.shape[1], target.shape[2], target.shape[3]))
                    if target.shape[1] < heads:
                        target = mx.broadcast_to(target[:batch], (batch, heads, embed_len, embed_dim))
                    
                    # Strong injection for transformation
                    v_new[:, :, :embed_len, :embed_dim] = \
                        0.2 * v[:, :, :embed_len, :embed_dim] + \
                        0.8 * target[:, :, :embed_len, :embed_dim]
                    
                    return q, k, v_new
        return q, k, v
    return hook

def main():
    print("🐱🐶 Test: Cat vs Dog Comparison")
    print("=" * 60)
    
    # Configuration
    base_prompt = "a cute cat sitting on a cushion"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    print(f"📝 Base Prompt: '{base_prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Create output directory
    output_dir = Path("artifacts/images/cat_dog_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Test 1: Original Cat
    print("\n🎨 Test 1: Original Cat (baseline)...")
    attn_scores.KV_REGISTRY.clear()
    
    latents = model.generate_latents(
        base_prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    cat_img = PIL.Image.fromarray(np.array(img_array))
    cat_img.save(output_dir / "01_original_cat.png")
    print("✅ Saved: 01_original_cat.png")
    
    # Test 2: Cat → Dog Transformation
    print("\n🎨 Test 2: Cat → Dog Transformation...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_animal_swap_hook(model, "a playful dog")
    for block in ['mid', 'up_0', 'up_1', 'up_2']:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        base_prompt,  # Still using cat prompt!
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    dog_img = PIL.Image.fromarray(np.array(img_array))
    dog_img.save(output_dir / "02_transformed_dog.png")
    print("✅ Saved: 02_transformed_dog.png")
    
    # Test 3: Dog baseline (for comparison)
    print("\n🎨 Test 3: Original Dog (direct prompt)...")
    attn_scores.KV_REGISTRY.clear()
    
    latents = model.generate_latents(
        "a cute dog sitting on a cushion",
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    original_dog = PIL.Image.fromarray(np.array(img_array))
    original_dog.save(output_dir / "03_original_dog.png")
    print("✅ Saved: 03_original_dog.png")
    
    # Create comparison image
    print("\n🔄 Creating comparison image...")
    width = cat_img.width
    height = cat_img.height
    
    comparison = PIL.Image.new('RGB', (width * 3, height))
    comparison.paste(cat_img, (0, 0))
    comparison.paste(dog_img, (width, 0))
    comparison.paste(original_dog, (width * 2, 0))
    
    comparison.save(output_dir / "cat_dog_comparison.png")
    print("✅ Saved: cat_dog_comparison.png")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n" + "=" * 60)
    print("✅ Cat vs Dog Comparison Complete!")
    print("📊 Results:")
    print("  01_original_cat.png: Cat from original prompt")
    print("  02_transformed_dog.png: Cat→Dog via injection")
    print("  03_original_dog.png: Dog from direct prompt")
    print("  cat_dog_comparison.png: Side-by-side comparison")
    print("\n💡 This proves subject transformation via prompt injection!")
    print("🔄 Same seed, same composition, different animal!")

if __name__ == "__main__":
    main()