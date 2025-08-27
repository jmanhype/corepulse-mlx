#!/usr/bin/env python3
"""
Test: CorePulse-style Prompt Injection
Implements CorePulse's block-specific prompt injection technique.
This demonstrates injection into different UNet architectural blocks.
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

def create_corepulse_injection_hook(model, injection_config):
    """
    Create a CorePulse-style prompt injection hook.
    
    Args:
        model: The SDXL model
        injection_config: Dict with keys:
            - block: Target block(s) - "content", "style", "composition"
            - prompt: Injection prompt
            - weight: Injection strength
            - sigma_range: (start, end) noise levels
    """
    # Generate embedding for injection prompt
    inject_cond, _ = model._get_text_conditioning(injection_config['prompt'])
    
    # Map CorePulse block names to our UNet blocks
    block_mapping = {
        'composition': ['down_0', 'down_1'],  # Input layers
        'content': ['mid'],                    # Middle layers
        'style': ['up_1', 'up_2']              # Output layers
    }
    
    target_blocks = block_mapping.get(injection_config['block'], [])
    weight = injection_config.get('weight', 1.0)
    
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            # Check if this block is targeted
            if block_id in target_blocks:
                if seq_len >= inject_cond.shape[1]:
                    v_new = mx.array(v)
                    embed_dim = min(dim, inject_cond.shape[2])
                    embed_len = min(seq_len, inject_cond.shape[1])
                    
                    # Prepare injection embedding
                    injection = inject_cond[:, :embed_len, :embed_dim]
                    if len(injection.shape) == 3:
                        injection = injection[None, :, :, :]
                    if injection.shape[1] < heads:
                        injection = mx.broadcast_to(injection, (batch, heads, embed_len, embed_dim))
                    
                    # Apply weighted injection
                    v_new[:, :, :embed_len, :embed_dim] = \
                        (1 - weight) * v[:, :, :embed_len, :embed_dim] + \
                        weight * injection[:, :, :embed_len, :embed_dim]
                    
                    print(f"    💉 CorePulse injection at {block_id} ({injection_config['block']}): weight {weight:.1f}")
                    
                    return q, k, v_new
        return q, k, v
    return hook

def create_multi_block_injection(model, injections):
    """
    Create hooks for multiple block injections simultaneously.
    
    Args:
        model: The SDXL model
        injections: List of injection configs
    """
    hooks = {}
    
    block_mapping = {
        'composition': ['down_0', 'down_1'],
        'content': ['mid'],
        'style': ['up_1', 'up_2'],
        'early': ['down_0'],
        'middle': ['down_2', 'mid'],
        'late': ['up_0', 'up_1', 'up_2']
    }
    
    for config in injections:
        inject_cond, _ = model._get_text_conditioning(config['prompt'])
        target_blocks = block_mapping.get(config['block'], [])
        weight = config.get('weight', 1.0)
        
        for block in target_blocks:
            if block not in hooks:
                hooks[block] = []
            hooks[block].append((inject_cond, weight, config['prompt'][:20]))
    
    def create_block_hook(block_hooks):
        def hook(q, k, v, meta=None):
            if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
                batch, heads, seq_len, dim = v.shape
                block_id = meta.get('block_id', 'unknown')
                
                v_new = mx.array(v)
                
                # Apply all injections for this block
                for inject_cond, weight, prompt_snippet in block_hooks:
                    if seq_len >= inject_cond.shape[1]:
                        embed_dim = min(dim, inject_cond.shape[2])
                        embed_len = min(seq_len, inject_cond.shape[1])
                        
                        injection = inject_cond[:, :embed_len, :embed_dim]
                        if len(injection.shape) == 3:
                            injection = injection[None, :, :, :]
                        if injection.shape[1] < heads:
                            injection = mx.broadcast_to(injection, (batch, heads, embed_len, embed_dim))
                        
                        v_new[:, :, :embed_len, :embed_dim] = \
                            (1 - weight) * v_new[:, :, :embed_len, :embed_dim] + \
                            weight * injection[:, :, :embed_len, :embed_dim]
                        
                        print(f"      • Injecting '{prompt_snippet}...' weight {weight:.1f}")
                
                return q, k, v_new
            return q, k, v
        return hook
    
    return {block: create_block_hook(hooks[block]) for block in hooks}

def main():
    print("💉 Test: CorePulse-style Prompt Injection")
    print("=" * 60)
    
    # Configuration
    base_prompt = "a photo of a dog in a garden"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    print(f"📝 Base Prompt: '{base_prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Create output directory
    output_dir = Path("artifacts/images/corepulse_injection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Test 1: Baseline
    print("\n🎨 Test 1: Baseline (no injection)...")
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
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "01_baseline.png")
    print("✅ Saved: 01_baseline.png")
    
    # Test 2: Content injection (middle blocks)
    print("\n🎨 Test 2: Content injection (replace dog with cat)...")
    attn_scores.KV_REGISTRY.clear()
    
    config = {
        'block': 'content',
        'prompt': 'a white cat',
        'weight': 0.8
    }
    
    hook = create_corepulse_injection_hook(model, config)
    for block in ['down_0', 'down_1', 'down_2', 'mid', 'up_0', 'up_1', 'up_2']:
        attn_scores.KV_REGISTRY.set(block, hook)
    
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
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "02_content_injection.png")
    print("✅ Saved: 02_content_injection.png")
    
    # Test 3: Style injection (output blocks)
    print("\n🎨 Test 3: Style injection (oil painting style)...")
    attn_scores.KV_REGISTRY.clear()
    
    config = {
        'block': 'style',
        'prompt': 'oil painting, impressionist, Van Gogh style',
        'weight': 0.7
    }
    
    hook = create_corepulse_injection_hook(model, config)
    for block in ['down_0', 'down_1', 'down_2', 'mid', 'up_0', 'up_1', 'up_2']:
        attn_scores.KV_REGISTRY.set(block, hook)
    
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
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "03_style_injection.png")
    print("✅ Saved: 03_style_injection.png")
    
    # Test 4: Composition injection (input blocks)
    print("\n🎨 Test 4: Composition injection (beach setting)...")
    attn_scores.KV_REGISTRY.clear()
    
    config = {
        'block': 'composition',
        'prompt': 'on a tropical beach with palm trees',
        'weight': 0.6
    }
    
    hook = create_corepulse_injection_hook(model, config)
    for block in ['down_0', 'down_1', 'down_2', 'mid', 'up_0', 'up_1', 'up_2']:
        attn_scores.KV_REGISTRY.set(block, hook)
    
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
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "04_composition_injection.png")
    print("✅ Saved: 04_composition_injection.png")
    
    # Test 5: Multi-block injection
    print("\n🎨 Test 5: Multi-block injection (complex transformation)...")
    attn_scores.KV_REGISTRY.clear()
    
    injections = [
        {'block': 'composition', 'prompt': 'in a magical forest', 'weight': 0.5},
        {'block': 'content', 'prompt': 'a majestic wolf', 'weight': 0.7},
        {'block': 'style', 'prompt': 'fantasy art, ethereal glow', 'weight': 0.6}
    ]
    
    print("  Injections:")
    for inj in injections:
        print(f"    • {inj['block']}: '{inj['prompt']}' (weight {inj['weight']})")
    
    hooks = create_multi_block_injection(model, injections)
    for block, hook in hooks.items():
        attn_scores.KV_REGISTRY.set(block, hook)
    
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
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "05_multi_block.png")
    print("✅ Saved: 05_multi_block.png")
    
    # Test 6: Progressive weight injection
    print("\n🎨 Test 6: Progressive weight injection...")
    attn_scores.KV_REGISTRY.clear()
    
    # Create progressive injection (increasing weight through blocks)
    def create_progressive_hook(model, prompt, base_weight=0.3):
        inject_cond, _ = model._get_text_conditioning(prompt)
        
        def hook(q, k, v, meta=None):
            if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
                batch, heads, seq_len, dim = v.shape
                block_id = meta.get('block_id', 'unknown')
                
                # Progressive weight based on block
                weight_map = {
                    'down_0': base_weight * 0.5,
                    'down_1': base_weight * 0.7,
                    'down_2': base_weight * 0.9,
                    'mid': base_weight * 1.0,
                    'up_0': base_weight * 1.1,
                    'up_1': base_weight * 1.2,
                    'up_2': base_weight * 1.3
                }
                
                weight = weight_map.get(block_id, base_weight)
                
                if seq_len >= inject_cond.shape[1]:
                    v_new = mx.array(v)
                    embed_dim = min(dim, inject_cond.shape[2])
                    embed_len = min(seq_len, inject_cond.shape[1])
                    
                    injection = inject_cond[:, :embed_len, :embed_dim]
                    if len(injection.shape) == 3:
                        injection = injection[None, :, :, :]
                    if injection.shape[1] < heads:
                        injection = mx.broadcast_to(injection, (batch, heads, embed_len, embed_dim))
                    
                    v_new[:, :, :embed_len, :embed_dim] = \
                        (1 - weight) * v[:, :, :embed_len, :embed_dim] + \
                        weight * injection[:, :, :embed_len, :embed_dim]
                    
                    print(f"    📈 Progressive injection at {block_id}: weight {weight:.2f}")
                    
                    return q, k, v_new
            return q, k, v
        return hook
    
    hook = create_progressive_hook(model, "cyberpunk neon city", base_weight=0.4)
    for block in ['down_0', 'down_1', 'down_2', 'mid', 'up_0', 'up_1', 'up_2']:
        attn_scores.KV_REGISTRY.set(block, hook)
    
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
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "06_progressive_weight.png")
    print("✅ Saved: 06_progressive_weight.png")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n" + "=" * 60)
    print("✅ CorePulse-style Prompt Injection Test Complete!")
    print("📊 Results:")
    print("  01_baseline.png: Original prompt without injection")
    print("  02_content_injection.png: Content block injection (dog → cat)")
    print("  03_style_injection.png: Style block injection (oil painting)")
    print("  04_composition_injection.png: Composition block injection (beach)")
    print("  05_multi_block.png: Multiple simultaneous injections")
    print("  06_progressive_weight.png: Progressive weight through blocks")
    print("\n💡 This implements CorePulse's block-specific injection!")
    print("🎯 Different blocks control different aspects of generation!")
    print("🔬 Composition → Content → Style pipeline demonstrated!")

if __name__ == "__main__":
    main()