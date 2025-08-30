#!/usr/bin/env python3
"""Test injection using EXACT approach from working old tests."""

import sys
sys.path.insert(0, 'src/adapters/mlx/mlx-examples/stable_diffusion')

# CRITICAL: Enable hooks BEFORE importing model!
from stable_diffusion import attn_scores
attn_scores.enable_kv_hooks(True)
print("✅ Enabled KV hooks")

from stable_diffusion import StableDiffusionXL
from pathlib import Path
from PIL import Image
import mlx.core as mx
import numpy as np

def create_injection_hook_old_style(target_embedding):
    """Create hook using EXACT approach from old working test."""
    def hook(q, k, v, meta=None):
        # Only modify cross-attention (text conditioning)
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            # Get block_id from meta
            block_id = meta.get('block_id', 'unknown') if meta else 'unknown'
            
            # Direct replacement like in working test
            if v.shape[2] >= 77:  # Has text tokens
                v_new = v.copy()
                
                # Extract conditioning 
                cond, pooled = target_embedding
                
                # Replace text tokens (0-77) in value tensor
                text_tokens = min(77, cond.shape[1])
                
                # The working test did: v_new[:, :, :text_tokens, :new_emb.shape[-1]] = new_emb[:, :text_tokens, :]
                # But v shape is (batch, heads, seq, dim) and cond is (batch, seq, 2048)
                # We need to broadcast properly
                
                # Take only the dim that fits
                inject_dim = min(v.shape[-1], cond.shape[-1])
                
                # Direct replacement for all heads
                for h in range(v.shape[1]):
                    v_new[:, h, :text_tokens, :inject_dim] = cond[:, :text_tokens, :inject_dim]
                
                print(f"💉 Injected at {block_id}")
                return q, k, v_new
        return q, k, v
    return hook


def main():
    print("\n🧪 Testing injection with OLD WORKING approach")
    print("=" * 60)
    
    # Load model
    print("\nLoading SDXL-Turbo...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    output_dir = Path("examples/output/injection_old_style")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate embeddings for different prompts
    print("\nGenerating embeddings...")
    cat_embedding = model._get_text_conditioning("a cute fluffy cat")
    dog_embedding = model._get_text_conditioning("a playful golden retriever") 
    banana_embedding = model._get_text_conditioning("yellow banana fruit")
    
    # Test 1: Baseline (cat)
    print("\n1. Baseline: generating cat...")
    attn_scores.KV_REGISTRY.clear()
    
    latents = model.generate_latents(
        "a cute fluffy cat",
        num_steps=2,
        cfg_weight=0.0,
        seed=42
    )
    for i, x in enumerate(latents):
        if i == 1:
            img = model.decode(x)
            img_array = np.array(img[0])
            img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_array).save(output_dir / "1_baseline_cat.png")
            print("   ✅ Saved baseline cat")
    
    # Test 2: Inject dog while prompting for cat
    print("\n2. Injecting dog while prompting for cat...")
    attn_scores.KV_REGISTRY.clear()
    
    # Create and register injection hook
    dog_hook = create_injection_hook_old_style(dog_embedding)
    
    # Apply to ALL blocks like in working test
    blocks = ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]
    for block in blocks:
        attn_scores.KV_REGISTRY.set(block, dog_hook)
    
    latents = model.generate_latents(
        "a cute fluffy cat",  # Still prompting for cat
        num_steps=2,
        cfg_weight=0.0,
        seed=42
    )
    for i, x in enumerate(latents):
        if i == 1:
            img = model.decode(x)
            img_array = np.array(img[0])
            img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_array).save(output_dir / "2_cat_with_dog_injection.png")
            print("   ✅ Saved with dog injection")
    
    # Test 3: Inject banana while prompting for apple
    print("\n3. Injecting banana while prompting for apple...")
    attn_scores.KV_REGISTRY.clear()
    
    banana_hook = create_injection_hook_old_style(banana_embedding)
    for block in blocks:
        attn_scores.KV_REGISTRY.set(block, banana_hook)
    
    latents = model.generate_latents(
        "a red apple on a table",
        num_steps=2,
        cfg_weight=0.0,
        seed=42
    )
    for i, x in enumerate(latents):
        if i == 1:
            img = model.decode(x)
            img_array = np.array(img[0])
            img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_array).save(output_dir / "3_apple_with_banana_injection.png")
            print("   ✅ Saved with banana injection")
    
    print("\n✨ Complete! Check examples/output/injection_old_style/")
    print("If this works, you should see:")
    print("  1. A cat (baseline)")
    print("  2. A dog (not a cat)")
    print("  3. A banana (not an apple)")
    

if __name__ == "__main__":
    main()