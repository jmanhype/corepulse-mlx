#!/usr/bin/env python3
"""Generate REAL CorePulse comparison images showing before and after effects."""

import mlx.core as mx
from pathlib import Path
import sys
sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')
from stable_diffusion import StableDiffusionXL
from stable_diffusion import cond_ctx
from stable_diffusion import cond_policy
from stable_diffusion import attn_scores
from stable_diffusion import regional
import numpy as np
from PIL import Image

def generate_base_image(sd, prompt, seed=42):
    """Generate base image without any effects."""
    # Clear all hooks
    attn_scores.KV_REGISTRY.clear()
    attn_scores.SCORES_REGISTRY.clear()
    attn_scores.KV_HOOKS_ENABLED = False
    attn_scores.SCORES_HOOKS_ENABLED = False
    cond_policy.COND_POLICY_ENABLED = False
    regional.REGIONAL_POLICY_ENABLED = False
    
    # Generate
    latents_gen = sd.generate_latents(
        prompt,
        n_images=1,
        num_steps=2,
        cfg_weight=0.0,
        seed=seed
    )
    
    for latents in latents_gen:
        pass
    
    x = sd.decode(latents)
    x = (x * 255).astype(mx.uint8)
    return Image.fromarray(np.array(x[0]))

def test_token_masking(sd, prompt, mask_positions, seed=42):
    """Test token masking effect."""
    # Setup token masking
    def kv_hook(q, k, v, meta):
        """Mask specific token positions in KV."""
        seq_len = k.shape[2]
        for pos in mask_positions:
            if pos < seq_len:
                # Zero out the key at this position
                k = mx.array(k)  # Make copy
                k[:, :, pos, :] = k[:, :, pos, :] * 0.1  # Reduce influence
        return q, k, v
    
    attn_scores.KV_REGISTRY.set("down_2", kv_hook)
    attn_scores.KV_REGISTRY.set("mid", kv_hook)
    attn_scores.KV_HOOKS_ENABLED = True
    
    # Generate
    latents_gen = sd.generate_latents(
        prompt,
        n_images=1,
        num_steps=2,
        cfg_weight=0.0,
        seed=seed
    )
    
    for latents in latents_gen:
        pass
    
    x = sd.decode(latents)
    x = (x * 255).astype(mx.uint8)
    
    # Clear hooks
    attn_scores.KV_REGISTRY.clear()
    attn_scores.KV_HOOKS_ENABLED = False
    
    return Image.fromarray(np.array(x[0]))

def test_regional_control(sd, prompt, amplify_region, seed=42):
    """Test regional/spatial control."""
    def apply_regional(x, meta):
        """Apply spatial mask to latent."""
        B, H, W, C = x.shape
        y1, y2, x1, x2 = amplify_region
        
        # Convert normalized coords to pixel coords
        y1_px = int(y1 * H)
        y2_px = int(y2 * H)
        x1_px = int(x1 * W)
        x2_px = int(x2 * W)
        
        # Create mask that amplifies center region
        mask = mx.ones_like(x) * 0.7  # Reduce everywhere
        mask[:, y1_px:y2_px, x1_px:x2_px, :] = 1.5  # Amplify region
        
        return x * mask
    
    regional.REGIONAL_POLICY.set_apply(apply_regional)
    regional.REGIONAL_POLICY_ENABLED = True
    
    # Generate
    latents_gen = sd.generate_latents(
        prompt,
        n_images=1,
        num_steps=2,
        cfg_weight=0.0,
        seed=seed
    )
    
    for latents in latents_gen:
        pass
    
    x = sd.decode(latents)
    x = (x * 255).astype(mx.uint8)
    
    # Clear hooks
    regional.REGIONAL_POLICY_ENABLED = False
    
    return Image.fromarray(np.array(x[0]))

def test_attention_manipulation(sd, prompt, seed=42):
    """Test attention score manipulation."""
    def scores_hook(scores, meta):
        """Manipulate attention scores pre-softmax."""
        B, H, Tq, Tk = scores.shape
        
        # Increase diagonal (self-attention)
        eye = mx.eye(min(Tq, Tk))
        if Tq != Tk:
            # Pad or trim to match dimensions
            eye_full = mx.zeros((Tq, Tk))
            min_dim = min(Tq, Tk)
            eye_full[:min_dim, :min_dim] = eye
            eye = eye_full
            
        eye = mx.broadcast_to(eye[None, None, :, :], (B, H, Tq, Tk))
        
        # Strongly enhance diagonal
        scores = scores + eye * 2.0
        
        return scores
    
    # Register for multiple blocks
    attn_scores.SCORES_REGISTRY.set("down_1", scores_hook)
    attn_scores.SCORES_REGISTRY.set("down_2", scores_hook)
    attn_scores.SCORES_REGISTRY.set("mid", scores_hook)
    attn_scores.SCORES_HOOKS_ENABLED = True
    
    # Generate
    latents_gen = sd.generate_latents(
        prompt,
        n_images=1,
        num_steps=2,
        cfg_weight=0.0,
        seed=seed
    )
    
    for latents in latents_gen:
        pass
    
    x = sd.decode(latents)
    x = (x * 255).astype(mx.uint8)
    
    # Clear hooks
    attn_scores.SCORES_REGISTRY.clear()
    attn_scores.SCORES_HOOKS_ENABLED = False
    
    return Image.fromarray(np.array(x[0]))

def create_comparison(before, after, title):
    """Create side-by-side comparison image."""
    width = before.width + after.width + 20
    height = max(before.height, after.height) + 60
    
    comparison = Image.new('RGB', (width, height), color='white')
    
    # Paste images
    comparison.paste(before, (0, 40))
    comparison.paste(after, (before.width + 20, 40))
    
    # Add labels (requires PIL ImageDraw)
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comparison)
    
    # Try to use a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Title
    draw.text((width//2, 10), title, fill='black', font=font, anchor='mt')
    
    # Labels
    draw.text((before.width//2, height-15), "BEFORE", fill='black', font=small_font, anchor='mt')
    draw.text((before.width + 20 + after.width//2, height-15), "AFTER", fill='red', font=small_font, anchor='mt')
    
    return comparison

def main():
    # Load model
    print("Loading SDXL-turbo model...")
    sd = StableDiffusionXL(
        "stabilityai/sdxl-turbo",
        float16=True
    )
    
    # Test 1: Token Masking
    print("\n=== Test 1: Token Masking Comparison ===")
    prompt = "a red sports car on a sunny beach"
    
    before = generate_base_image(sd, prompt, seed=42)
    after = test_token_masking(sd, prompt, [1, 2], seed=42)  # Mask tokens 1 and 2
    
    comparison = create_comparison(before, after, "TOKEN MASKING EFFECT")
    comparison.save("artifacts/images/readme/REAL_token_masking.png")
    print("Saved REAL_token_masking.png")
    
    # Test 2: Regional Control
    print("\n=== Test 2: Regional Control Comparison ===")
    prompt = "a landscape with mountains and lake"
    
    before = generate_base_image(sd, prompt, seed=43)
    after = test_regional_control(sd, prompt, (0.25, 0.75, 0.25, 0.75), seed=43)
    
    comparison = create_comparison(before, after, "REGIONAL CONTROL EFFECT")
    comparison.save("artifacts/images/readme/REAL_regional_control.png")
    print("Saved REAL_regional_control.png")
    
    # Test 3: Attention Manipulation
    print("\n=== Test 3: Attention Manipulation Comparison ===")
    prompt = "a colorful abstract painting with geometric shapes"
    
    before = generate_base_image(sd, prompt, seed=44)
    after = test_attention_manipulation(sd, prompt, seed=44)
    
    comparison = create_comparison(before, after, "ATTENTION MANIPULATION EFFECT")
    comparison.save("artifacts/images/readme/REAL_attention_manipulation.png")
    print("Saved REAL_attention_manipulation.png")
    
    # Test 4: Combined Effects
    print("\n=== Test 4: Combined Effects ===")
    prompt = "a futuristic city skyline at sunset"
    
    # Apply multiple effects
    def apply_combined():
        # Token masking
        def kv_hook(q, k, v, meta):
            seq_len = k.shape[2]
            if 3 < seq_len:
                k = mx.array(k)
                k[:, :, 3, :] = k[:, :, 3, :] * 0.2
            return q, k, v
        
        attn_scores.KV_REGISTRY.set("mid", kv_hook)
        attn_scores.KV_HOOKS_ENABLED = True
        
        # Attention manipulation
        def scores_hook(scores, meta):
            B, H, Tq, Tk = scores.shape
            eye = mx.eye(min(Tq, Tk))
            if Tq != Tk:
                eye_full = mx.zeros((Tq, Tk))
                min_dim = min(Tq, Tk)
                eye_full[:min_dim, :min_dim] = eye
                eye = eye_full
            eye = mx.broadcast_to(eye[None, None, :, :], (B, H, Tq, Tk))
            return scores + eye * 1.5
        
        attn_scores.SCORES_REGISTRY.set("down_2", scores_hook)
        attn_scores.SCORES_HOOKS_ENABLED = True
        
        # Regional control
        def apply_regional(x, meta):
            B, H, W, C = x.shape
            mask = mx.ones_like(x)
            # Amplify top half
            mask[:, :H//2, :, :] = 1.3
            # Reduce bottom half
            mask[:, H//2:, :, :] = 0.8
            return x * mask
        
        regional.REGIONAL_POLICY.set_apply(apply_regional)
        regional.REGIONAL_POLICY_ENABLED = True
        
        # Generate
        latents_gen = sd.generate_latents(
            prompt,
            n_images=1,
            num_steps=2,
            cfg_weight=0.0,
            seed=45
        )
        
        for latents in latents_gen:
            pass
        
        x = sd.decode(latents)
        x = (x * 255).astype(mx.uint8)
        
        # Clear all
        attn_scores.KV_REGISTRY.clear()
        attn_scores.SCORES_REGISTRY.clear()
        attn_scores.KV_HOOKS_ENABLED = False
        attn_scores.SCORES_HOOKS_ENABLED = False
        regional.REGIONAL_POLICY_ENABLED = False
        
        return Image.fromarray(np.array(x[0]))
    
    before = generate_base_image(sd, prompt, seed=45)
    after = apply_combined()
    
    comparison = create_comparison(before, after, "COMBINED EFFECTS")
    comparison.save("artifacts/images/readme/REAL_combined_effects.png")
    print("Saved REAL_combined_effects.png")
    
    print("\n✅ All REAL CorePulse comparison images generated!")
    print("Images saved in artifacts/images/readme/")

if __name__ == "__main__":
    main()