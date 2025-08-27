#!/usr/bin/env python3
"""
Test: CorePulse Token-Level Attention Masking
Implements selective token masking for fine-grained control.
This demonstrates masking specific tokens to control their influence.
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

def create_token_masking_hook(model, mask_config):
    """
    Create a hook that masks specific tokens in attention.
    
    Args:
        model: The SDXL model
        mask_config: Dict with keys:
            - tokens_to_mask: List of token indices or keywords
            - mask_strength: How strongly to suppress (0=no effect, 1=complete suppression)
            - mask_mode: "suppress", "amplify", or "isolate"
    """
    tokens = mask_config.get('tokens_to_mask', [])
    strength = mask_config.get('mask_strength', 0.8)
    mode = mask_config.get('mask_mode', 'suppress')
    
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = k.shape
            block_id = meta.get('block_id', 'unknown')
            
            # Create attention mask
            mask = mx.ones((batch, heads, seq_len, 1))
            
            # Apply masking based on token positions
            for token_idx in tokens:
                if isinstance(token_idx, int) and 0 <= token_idx < seq_len:
                    if mode == "suppress":
                        mask[:, :, token_idx, :] = 1.0 - strength
                    elif mode == "amplify":
                        mask[:, :, token_idx, :] = 1.0 + strength
                    elif mode == "isolate":
                        # Suppress all except specified tokens
                        mask = mx.full_like(mask, 1.0 - strength)
                        mask[:, :, token_idx, :] = 1.0
            
            # Apply mask to K and V
            k_masked = k * mask
            v_masked = v * mask
            
            print(f"    🎭 Token masking at {block_id}: {mode} {len(tokens)} tokens, strength {strength:.1f}")
            
            return q, k_masked, v_masked
        return q, k, v
    return hook

def create_semantic_masking_hook(model, positive_words, negative_words):
    """
    Create a hook that masks based on semantic content.
    
    Args:
        model: The SDXL model
        positive_words: Words/concepts to amplify
        negative_words: Words/concepts to suppress
    """
    # Generate embeddings for positive and negative concepts
    pos_embeds = []
    neg_embeds = []
    
    for word in positive_words:
        cond, _ = model._get_text_conditioning(word)
        pos_embeds.append(cond)
    
    for word in negative_words:
        cond, _ = model._get_text_conditioning(word)
        neg_embeds.append(cond)
    
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            v_new = mx.array(v)
            
            # Apply positive amplification
            for pos_embed in pos_embeds:
                if seq_len >= pos_embed.shape[1]:
                    embed_dim = min(dim, pos_embed.shape[2])
                    embed_len = min(seq_len, pos_embed.shape[1])
                    
                    pos = pos_embed[:, :embed_len, :embed_dim]
                    if len(pos.shape) == 3:
                        pos = pos[None, :, :, :]
                    
                    # Handle different batch sizes properly
                    if pos.shape[0] < batch:
                        pos = mx.broadcast_to(pos, (batch, pos.shape[1], pos.shape[2], pos.shape[3]))
                    if pos.shape[1] < heads:
                        pos = mx.broadcast_to(pos[:batch], (batch, heads, embed_len, embed_dim))
                    
                    # Amplify positive concepts
                    v_new[:, :, :embed_len, :embed_dim] += 0.3 * pos[:, :, :embed_len, :embed_dim]
            
            # Apply negative suppression
            for neg_embed in neg_embeds:
                if seq_len >= neg_embed.shape[1]:
                    embed_dim = min(dim, neg_embed.shape[2])
                    embed_len = min(seq_len, neg_embed.shape[1])
                    
                    neg = neg_embed[:, :embed_len, :embed_dim]
                    if len(neg.shape) == 3:
                        neg = neg[None, :, :, :]
                    
                    # Handle different batch sizes properly
                    if neg.shape[0] < batch:
                        neg = mx.broadcast_to(neg, (batch, neg.shape[1], neg.shape[2], neg.shape[3]))
                    if neg.shape[1] < heads:
                        neg = mx.broadcast_to(neg[:batch], (batch, heads, embed_len, embed_dim))
                    
                    # Suppress negative concepts
                    v_new[:, :, :embed_len, :embed_dim] -= 0.3 * neg[:, :, :embed_len, :embed_dim]
            
            print(f"    🎯 Semantic masking at {block_id}: +{len(positive_words)} -{len(negative_words)}")
            
            return q, k, v_new
        return q, k, v
    return hook

def create_attention_gate_hook(threshold=0.5):
    """
    Create a hook that gates attention based on magnitude.
    Only allows through attention values above threshold.
    """
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = k.shape
            block_id = meta.get('block_id', 'unknown')
            
            # Compute attention scores (simplified)
            scores = mx.sum(k * k, axis=-1, keepdims=True)
            scores_norm = scores / (mx.max(scores) + 1e-8)
            
            # Create gate based on threshold
            gate = mx.where(scores_norm > threshold, 1.0, 0.1)
            
            # Apply gate to K and V
            k_gated = k * gate
            v_gated = v * gate
            
            active_tokens = mx.sum(gate > 0.5) / (batch * heads * seq_len)
            print(f"    🚪 Attention gate at {block_id}: {active_tokens*100:.1f}% tokens active")
            
            return q, k_gated, v_gated
        return q, k, v
    return hook

def create_progressive_masking_hook(model, prompt, start_ratio=0.0, end_ratio=1.0):
    """
    Progressively reveal tokens through the generation process.
    """
    cond, _ = model._get_text_conditioning(prompt)
    
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            # Map blocks to progression
            block_progress = {
                'down_0': 0.0, 'down_1': 0.15, 'down_2': 0.3,
                'mid': 0.5,
                'up_0': 0.7, 'up_1': 0.85, 'up_2': 1.0
            }
            
            progress = block_progress.get(block_id, 0.5)
            current_ratio = start_ratio + (end_ratio - start_ratio) * progress
            
            if seq_len >= cond.shape[1]:
                v_new = mx.array(v)
                embed_dim = min(dim, cond.shape[2])
                embed_len = min(seq_len, cond.shape[1])
                
                # Calculate how many tokens to reveal
                tokens_to_reveal = int(embed_len * current_ratio)
                
                if tokens_to_reveal > 0:
                    inject = cond[:, :tokens_to_reveal, :embed_dim]
                    if len(inject.shape) == 3:
                        inject = inject[None, :, :, :]
                    
                    # Handle different batch sizes properly
                    if inject.shape[0] < batch:
                        inject = mx.broadcast_to(inject, (batch, inject.shape[1], inject.shape[2], inject.shape[3]))
                    if inject.shape[1] < heads:
                        inject = mx.broadcast_to(inject[:batch], (batch, heads, tokens_to_reveal, embed_dim))
                    
                    # Progressive injection
                    v_new[:, :, :tokens_to_reveal, :embed_dim] = \
                        0.3 * v[:, :, :tokens_to_reveal, :embed_dim] + \
                        0.7 * inject[:, :, :tokens_to_reveal, :embed_dim]
                    
                    print(f"    📈 Progressive reveal at {block_id}: {tokens_to_reveal}/{embed_len} tokens ({current_ratio*100:.0f}%)")
                
                return q, k, v_new
        return q, k, v
    return hook

def main():
    print("🎭 Test: CorePulse Token-Level Attention Masking")
    print("=" * 60)
    
    # Configuration
    base_prompt = "a majestic lion in the jungle"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    print(f"📝 Base Prompt: '{base_prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Create output directory
    output_dir = Path("artifacts/images/corepulse_token_masking")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Test 1: Baseline
    print("\n🎨 Test 1: Baseline (no masking)...")
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
    
    # Test 2: Token suppression
    print("\n🎨 Test 2: Token suppression (mask specific positions)...")
    attn_scores.KV_REGISTRY.clear()
    
    mask_config = {
        'tokens_to_mask': [2, 3, 4],  # Mask middle tokens
        'mask_strength': 0.8,
        'mask_mode': 'suppress'
    }
    
    hook = create_token_masking_hook(model, mask_config)
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
    pil_img.save(output_dir / "02_token_suppression.png")
    print("✅ Saved: 02_token_suppression.png")
    
    # Test 3: Token amplification
    print("\n🎨 Test 3: Token amplification...")
    attn_scores.KV_REGISTRY.clear()
    
    mask_config = {
        'tokens_to_mask': [0, 1],  # Amplify first tokens
        'mask_strength': 0.5,
        'mask_mode': 'amplify'
    }
    
    hook = create_token_masking_hook(model, mask_config)
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
    pil_img.save(output_dir / "03_token_amplification.png")
    print("✅ Saved: 03_token_amplification.png")
    
    # Test 4: Semantic masking
    print("\n🎨 Test 4: Semantic masking (amplify/suppress concepts)...")
    attn_scores.KV_REGISTRY.clear()
    
    positive_words = ["majestic", "powerful", "golden"]
    negative_words = ["dark", "scary", "aggressive"]
    
    hook = create_semantic_masking_hook(model, positive_words, negative_words)
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
    pil_img.save(output_dir / "04_semantic_masking.png")
    print("✅ Saved: 04_semantic_masking.png")
    
    # Test 5: Attention gating
    print("\n🎨 Test 5: Attention gating (threshold-based)...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_attention_gate_hook(threshold=0.6)
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
    pil_img.save(output_dir / "05_attention_gate.png")
    print("✅ Saved: 05_attention_gate.png")
    
    # Test 6: Progressive masking
    print("\n🎨 Test 6: Progressive token reveal...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_progressive_masking_hook(
        model,
        "mystical ethereal glow",
        start_ratio=0.2,
        end_ratio=1.0
    )
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
    pil_img.save(output_dir / "06_progressive_reveal.png")
    print("✅ Saved: 06_progressive_reveal.png")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n" + "=" * 60)
    print("✅ CorePulse Token-Level Masking Test Complete!")
    print("📊 Results:")
    print("  01_baseline.png: Original without masking")
    print("  02_token_suppression.png: Specific tokens suppressed")
    print("  03_token_amplification.png: Specific tokens amplified")
    print("  04_semantic_masking.png: Concept-based masking")
    print("  05_attention_gate.png: Threshold-based gating")
    print("  06_progressive_reveal.png: Progressive token reveal")
    print("\n💡 This implements CorePulse's token-level control!")
    print("🎭 Fine-grained masking enables precise manipulation!")
    print("🔬 Token-level control provides surgical precision!")

if __name__ == "__main__":
    main()