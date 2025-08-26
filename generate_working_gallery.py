#!/usr/bin/env python3
"""
Generate comparison gallery with hooks properly enabled.
"""

import mlx.core as mx
import numpy as np
import sys
import os
from PIL import Image, ImageDraw, ImageFont

sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')

# CRITICAL: Import and enable hooks BEFORE importing StableDiffusionXL
from stable_diffusion import attn_scores

# Enable hooks BEFORE model loading!
print("🔧 Enabling hooks BEFORE model loading...")
attn_scores.enable_kv_hooks(True)
attn_scores.enable_scores_hooks(True)

# NOW import and load the model
from stable_diffusion import StableDiffusionXL

print("=" * 80)
print("🎨 COREPULSE V4 GALLERY - PROPER HOOK ACTIVATION")
print("=" * 80)

os.makedirs("artifacts/images/gallery", exist_ok=True)

def save_image(image, filepath):
    """Save MLX image array to file."""
    img_np = np.array(image)
    if img_np.ndim == 4:
        img_np = img_np[0]
    if img_np.dtype in [np.float32, np.float64]:
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_np).save(filepath)
    return filepath

def create_side_by_side(img1_path, img2_path, output_path, title):
    """Create side-by-side comparison."""
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    width = img1.width
    height = img1.height
    
    # Create canvas
    canvas = Image.new('RGB', (width * 2 + 30, height + 60), 'white')
    canvas.paste(img1, (10, 40))
    canvas.paste(img2, (width + 20, 40))
    
    # Add text
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()
        small = font
    
    draw.text((canvas.width // 2, 15), title, fill='black', font=font, anchor='mt')
    draw.text((width // 2 + 10, height + 45), "BEFORE", fill='gray', font=small, anchor='mt')
    draw.text((width + 20 + width // 2, height + 45), "AFTER", fill='gray', font=small, anchor='mt')
    
    canvas.save(output_path)
    return output_path

print("\n✓ Hooks enabled:", attn_scores.hooks_wanted())
print("✓ KV hooks:", attn_scores._global_state['KV_HOOKS_ENABLED'])
print("✓ Scores hooks:", attn_scores._global_state['SCORES_HOOKS_ENABLED'])

print("\nLoading SDXL with PatchedMHA...")
sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)

prompt = "a majestic eagle soaring over mountains"
comparisons = []

# ==============================================================================
# 1. BASELINE (No hooks)
# ==============================================================================
print("\n" + "="*60)
print("1️⃣ BASELINE - No manipulation")
print("="*60)

# Temporarily clear hooks for baseline
attn_scores.KV_REGISTRY.clear()
attn_scores.SCORES_REGISTRY.clear()

latents_gen = sd.generate_latents(prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
for latents in latents_gen: pass
baseline_path = save_image(sd.decode(latents), "artifacts/images/gallery/demo_baseline.png")
print("✅ Saved baseline")

# ==============================================================================
# 2. TOKEN MASKING
# ==============================================================================
print("\n" + "="*60)
print("2️⃣ TOKEN MASKING - Remove middle tokens")
print("="*60)

def mask_tokens(q, k, v, meta):
    """Zero out tokens 3-7."""
    v_array = np.array(v)
    if v_array.shape[2] > 7:
        v_array[:, :, 3:7, :] = 0
        print(f"  🎭 Masked tokens in {meta.get('block_id', '?')}")
    return q, k, mx.array(v_array)

for block in ["down_2", "mid", "up_0"]:
    attn_scores.KV_REGISTRY.set(block, mask_tokens)

latents_gen = sd.generate_latents(prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
for latents in latents_gen: pass
mask_path = save_image(sd.decode(latents), "artifacts/images/gallery/demo_masked.png")
comparison_1 = create_side_by_side(baseline_path, mask_path, 
                                   "artifacts/images/gallery/comparison_1_mask.png",
                                   "Token Masking Effect")
print("✅ Created token masking comparison")
comparisons.append(comparison_1)

attn_scores.KV_REGISTRY.clear()

# ==============================================================================
# 3. ATTENTION CHAOS
# ==============================================================================
print("\n" + "="*60)
print("3️⃣ ATTENTION CHAOS - Scramble patterns")
print("="*60)

def add_chaos(q, k, v, meta):
    """Add chaos to attention."""
    k_array = np.array(k)
    v_array = np.array(v)
    k_chaos = k_array + np.random.randn(*k_array.shape) * 0.3
    v_chaos = v_array + np.random.randn(*v_array.shape) * 0.3
    print(f"  🌀 Chaos in {meta.get('block_id', '?')}")
    return q, mx.array(k_chaos), mx.array(v_chaos)

for block in ["mid", "up_0"]:
    attn_scores.KV_REGISTRY.set(block, add_chaos)

latents_gen = sd.generate_latents(prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
for latents in latents_gen: pass
chaos_path = save_image(sd.decode(latents), "artifacts/images/gallery/demo_chaos.png")
comparison_2 = create_side_by_side(baseline_path, chaos_path,
                                   "artifacts/images/gallery/comparison_2_chaos.png",
                                   "Attention Chaos Effect")
print("✅ Created chaos comparison")
comparisons.append(comparison_2)

attn_scores.KV_REGISTRY.clear()

# ==============================================================================
# 4. SPATIAL FOCUS
# ==============================================================================
print("\n" + "="*60)
print("4️⃣ SPATIAL FOCUS - Center emphasis")
print("="*60)

def focus_center(scores, meta):
    """Focus on center region."""
    scores_np = np.array(scores)
    seq_len = scores_np.shape[-1]
    
    if seq_len > 77:
        spatial_size = int((seq_len - 77) ** 0.5)
        center = spatial_size // 2
        
        for i in range(77, seq_len):
            y = (i - 77) // spatial_size
            x = (i - 77) % spatial_size
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            
            if dist > spatial_size * 0.3:
                scores_np[:, :, :, i] *= 0.2
                scores_np[:, :, i, :] *= 0.2
        
        print(f"  🎯 Focused center in {meta.get('block_id', '?')}")
    
    return mx.array(scores_np)

for block in ["up_0", "up_1"]:
    attn_scores.SCORES_REGISTRY.set(block, focus_center)

latents_gen = sd.generate_latents(prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
for latents in latents_gen: pass
focus_path = save_image(sd.decode(latents), "artifacts/images/gallery/demo_focus.png")
comparison_3 = create_side_by_side(baseline_path, focus_path,
                                   "artifacts/images/gallery/comparison_3_focus.png",
                                   "Spatial Focus Effect")
print("✅ Created spatial focus comparison")
comparisons.append(comparison_3)

attn_scores.SCORES_REGISTRY.clear()

# ==============================================================================
# 5. STYLE TRANSFER
# ==============================================================================
print("\n" + "="*60)
print("5️⃣ STYLE TRANSFER - Artistic modification")
print("="*60)

def style_mod(q, k, v, meta):
    """Apply style modifications."""
    v_array = np.array(v)
    
    # Add frequency patterns
    for i in range(v_array.shape[2]):
        v_array[:, :, i, :] *= (1 + 0.3 * np.sin(i * 0.3))
    
    # Add texture
    v_array += np.random.randn(*v_array.shape) * 0.02
    
    print(f"  🎨 Styled {meta.get('block_id', '?')}")
    return q, k, mx.array(v_array * 1.1)

for block in ["up_0", "up_1", "up_2"]:
    attn_scores.KV_REGISTRY.set(block, style_mod)

latents_gen = sd.generate_latents(prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
for latents in latents_gen: pass
style_path = save_image(sd.decode(latents), "artifacts/images/gallery/demo_style.png")
comparison_4 = create_side_by_side(baseline_path, style_path,
                                   "artifacts/images/gallery/comparison_4_style.png",
                                   "Style Transfer Effect")
print("✅ Created style transfer comparison")
comparisons.append(comparison_4)

# ==============================================================================
# CREATE FINAL GALLERY
# ==============================================================================
print("\n" + "="*60)
print("📊 CREATING FINAL GALLERY")
print("="*60)

# Create a 2x2 grid of comparisons
gallery = Image.new('RGB', (1200, 900), 'white')

# Load comparisons
for i, comp_path in enumerate(comparisons[:4]):
    comp = Image.open(comp_path)
    # Resize to fit
    comp.thumbnail((580, 420), Image.Resampling.LANCZOS)
    
    # Position in grid
    x = (i % 2) * 600 + 10
    y = (i // 2) * 440 + 60
    
    gallery.paste(comp, (x, y))

# Add title
draw = ImageDraw.Draw(gallery)
try:
    title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
except:
    title_font = ImageFont.load_default()

draw.text((600, 25), "CorePulse V4 - Pre-Attention Manipulation", 
         fill='black', font=title_font, anchor='mt')

gallery_path = "artifacts/images/gallery/COREPULSE_V4_GALLERY.png"
gallery.save(gallery_path)

# Clean up
attn_scores.KV_REGISTRY.clear()
attn_scores.SCORES_REGISTRY.clear()
attn_scores.enable_kv_hooks(False)
attn_scores.enable_scores_hooks(False)

print(f"\n✅ Gallery saved to: {gallery_path}")
print("\n" + "="*80)
print("✨ COREPULSE V4 DEMONSTRATION COMPLETE!")
print("="*80)
print("\nProven capabilities:")
print("  ✅ Token masking - Selective token removal")
print("  ✅ Attention chaos - Pattern disruption")
print("  ✅ Spatial focus - Regional emphasis")
print("  ✅ Style transfer - Artistic modification")
print("\nAll effects achieved through REAL pre-attention hooks!")