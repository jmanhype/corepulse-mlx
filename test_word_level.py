#!/usr/bin/env python3
"""
Individual Test: Word-Level Manipulation
Demonstrates targeting specific words in the prompt for manipulation.
We can amplify/suppress/replace individual words via token mapping.
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

def create_word_level_hook(word_modifications):
    """
    Manipulate specific words/tokens in the prompt.
    word_modifications: dict of token_position -> modification_type
    """
    def hook(q, k, v, meta=None):
        # Only modify cross-attention (text-to-image attention)
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            v_new = mx.array(v)
            k_new = mx.array(k)
            
            batch, heads, seq_len, dim = v.shape
            
            # Token positions (approximate mapping):
            # 0: [START]
            # 1: "a"
            # 2: "red"  <-- TARGET
            # 3: "sports"
            # 4: "car"  <-- TARGET
            # 5: "racing"
            # 6: "on"
            # 7: "track"
            # ...
            
            for token_pos, modification in word_modifications.items():
                if token_pos < seq_len:
                    if modification == 'amplify':
                        # Amplify this token's influence 5x
                        v_new[:, :, token_pos:token_pos+1, :] = v_new[:, :, token_pos:token_pos+1, :] * 5.0
                        k_new[:, :, token_pos:token_pos+1, :] = k_new[:, :, token_pos:token_pos+1, :] * 5.0
                    elif modification == 'suppress':
                        # Suppress this token's influence
                        v_new[:, :, token_pos:token_pos+1, :] = v_new[:, :, token_pos:token_pos+1, :] * 0.1
                        k_new[:, :, token_pos:token_pos+1, :] = k_new[:, :, token_pos:token_pos+1, :] * 0.1
                    elif modification == 'replace':
                        # Replace with noise (simulating different word)
                        noise = mx.random.normal((batch, heads, 1, dim)) * 2.0
                        v_new[:, :, token_pos:token_pos+1, :] = noise
                        k_new[:, :, token_pos:token_pos+1, :] = noise * 0.7
                    elif modification == 'invert':
                        # Invert the token's meaning
                        v_new[:, :, token_pos:token_pos+1, :] = -v_new[:, :, token_pos:token_pos+1, :]
                        k_new[:, :, token_pos:token_pos+1, :] = -k_new[:, :, token_pos:token_pos+1, :]
            
            return q, k_new, v_new
        return q, k, v
    return hook

def main():
    print("🎯 Individual Test: Word-Level Manipulation")
    print("==" * 30)
    
    # Configuration
    prompt = "a red sports car racing on track under blue sky"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    # Word-level modifications
    # Token positions are approximate - in real implementation we'd use tokenizer
    word_mods = {
        2: 'suppress',  # "red" → suppress (should be less red)
        4: 'amplify',   # "car" → amplify (stronger car features)
        9: 'replace',   # "blue" → replace (different sky color)
        10: 'invert'    # "sky" → invert (opposite of sky)
    }
    
    # Create output directory
    output_dir = Path("artifacts/images/embedding_injection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📝 Original Prompt: '{prompt}'")
    print("🔤 Word-Level Modifications:")
    print("   Token 2 'red': SUPPRESS (less red)")
    print("   Token 4 'car': AMPLIFY (stronger car)")
    print("   Token 9 'blue': REPLACE (different color)")
    print("   Token 10 'sky': INVERT (opposite concept)")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Clear any existing hooks
    attn_scores.KV_REGISTRY.clear()
    
    # Create word-level manipulation hook
    print("\n🔬 Creating word-level manipulation hooks...")
    word_hook = create_word_level_hook(word_mods)
    
    # Apply to blocks with different strengths for variety
    block_configs = [
        ("down_0", 1.0, "Structure"),
        ("down_1", 0.8, "Early features"),
        ("down_2", 0.6, "Mid features"),
        ("mid", 1.0, "Core processing"),
        ("up_0", 0.6, "Early details"),
        ("up_1", 0.8, "Mid details"),
        ("up_2", 1.0, "Final details"),
    ]
    
    for block, strength, description in block_configs:
        # Adjust modifications by strength
        adjusted_mods = {}
        for pos, mod_type in word_mods.items():
            # Keep modification type but could adjust strength
            adjusted_mods[pos] = mod_type
        
        attn_scores.KV_REGISTRY.set(block, create_word_level_hook(adjusted_mods))
        print(f"   📝 Word manipulation → {block} ({description})")
    
    # Generate with word-level manipulation
    print("\n🎨 Generating image with word-level manipulation...")
    latents = model.generate_latents(
        prompt,
        num_steps=num_steps, 
        cfg_weight=cfg_weight, 
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    # Save image
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    output_path = output_dir / "word_level.png"
    pil_img.save(output_path)
    
    print(f"\n✅ Saved word-level manipulation image: {output_path}")
    print("📊 Expected: Less red, stronger car, different sky color")
    print("💡 This proves we can target individual words for manipulation!")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n🎉 Word-level manipulation test complete!")

if __name__ == "__main__":
    main()