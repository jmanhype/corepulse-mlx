#!/usr/bin/env python3
"""
Generate comprehensive before/after comparison gallery for CorePulse V4.
Shows all manipulation capabilities with visual proof.
"""

import mlx.core as mx
import numpy as np
import sys
import os
from PIL import Image

sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import attn_scores
from stable_diffusion import StableDiffusionXL

print("=" * 80)
print("🎨 COREPULSE V4 GALLERY GENERATION")
print("=" * 80)

# Create output directory
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

def create_comparison(img1_path, img2_path, output_path, title):
    """Create side-by-side comparison image."""
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    # Ensure same size
    width = max(img1.width, img2.width)
    height = max(img1.height, img2.height)
    
    # Create comparison canvas
    comparison = Image.new('RGB', (width * 2 + 20, height + 80), 'white')
    
    # Paste images
    comparison.paste(img1, (0, 40))
    comparison.paste(img2, (width + 20, 40))
    
    # Add labels
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Title
    draw.text((comparison.width // 2, 10), title, fill='black', font=font, anchor='mt')
    # Labels
    draw.text((width // 2, height + 50), "BEFORE", fill='gray', font=small_font, anchor='mt')
    draw.text((width + 20 + width // 2, height + 50), "AFTER", fill='gray', font=small_font, anchor='mt')
    
    comparison.save(output_path)
    return output_path

print("\nLoading SDXL model...")
sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)

# Test prompts - simplified for speed
prompts = [
    "a cute cat playing with yarn"
]

techniques = []

# ==============================================================================
# TECHNIQUE 1: BASELINE (No hooks)
# ==============================================================================
print("\n" + "="*60)
print("1️⃣ GENERATING BASELINE IMAGES")
print("="*60)

attn_scores.enable_kv_hooks(False)
attn_scores.enable_scores_hooks(False)

for i, prompt in enumerate(prompts):
    print(f"\n  Generating baseline: {prompt}")
    latents_gen = sd.generate_latents(prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
    for latents in latents_gen: pass
    img_path = f"artifacts/images/gallery/baseline_{i}.png"
    save_image(sd.decode(latents), img_path)
    techniques.append(("baseline", i, img_path))

# ==============================================================================
# TECHNIQUE 2: TOKEN MASKING
# ==============================================================================
print("\n" + "="*60)
print("2️⃣ TOKEN MASKING - Zero out specific tokens")
print("="*60)

attn_scores.enable_kv_hooks(True)

def token_mask_hook(q, k, v, meta):
    """Mask tokens 5-15 to remove middle words."""
    v_array = np.array(v)
    # Zero out tokens 5-15 (middle of prompt)
    if v_array.shape[2] > 15:
        v_array[:, :, 5:15, :] = 0
        print(f"    🎭 Masked tokens 5-15 in {meta.get('block_id', 'unknown')}")
    return q, k, mx.array(v_array)

for block in ["down_2", "mid", "up_0"]:
    attn_scores.KV_REGISTRY.set(block, token_mask_hook)

for i, prompt in enumerate(prompts):
    print(f"\n  Applying token masking: {prompt}")
    latents_gen = sd.generate_latents(prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
    for latents in latents_gen: pass
    img_path = f"artifacts/images/gallery/token_mask_{i}.png"
    save_image(sd.decode(latents), img_path)
    techniques.append(("token_mask", i, img_path))

attn_scores.KV_REGISTRY.clear()

# ==============================================================================
# TECHNIQUE 3: ATTENTION CHAOS
# ==============================================================================
print("\n" + "="*60)
print("3️⃣ ATTENTION CHAOS - Scramble attention patterns")
print("="*60)

def chaos_hook(q, k, v, meta):
    """Add controlled chaos to attention."""
    k_array = np.array(k)
    v_array = np.array(v)
    
    # Add significant noise
    k_chaos = k_array + np.random.randn(*k_array.shape) * 0.5
    v_chaos = v_array + np.random.randn(*v_array.shape) * 0.5
    
    print(f"    🌀 Applied chaos to {meta.get('block_id', 'unknown')}")
    return q, mx.array(k_chaos), mx.array(v_chaos)

for block in ["down_2", "mid", "up_0"]:
    attn_scores.KV_REGISTRY.set(block, chaos_hook)

for i, prompt in enumerate(prompts):
    print(f"\n  Applying attention chaos: {prompt}")
    latents_gen = sd.generate_latents(prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
    for latents in latents_gen: pass
    img_path = f"artifacts/images/gallery/chaos_{i}.png"
    save_image(sd.decode(latents), img_path)
    techniques.append(("chaos", i, img_path))

attn_scores.KV_REGISTRY.clear()

# ==============================================================================
# TECHNIQUE 4: SPATIAL FOCUS (Center attention)
# ==============================================================================
print("\n" + "="*60)
print("4️⃣ SPATIAL FOCUS - Emphasize center region")
print("="*60)

attn_scores.enable_scores_hooks(True)

def spatial_focus_hook(scores, meta):
    """Focus attention on center of image."""
    scores_np = np.array(scores)
    seq_len = scores_np.shape[-1]
    
    if seq_len > 77:  # Has spatial tokens
        spatial_size = int((seq_len - 77) ** 0.5)
        center = spatial_size // 2
        
        # Create center mask
        for i in range(77, seq_len):
            y = (i - 77) // spatial_size
            x = (i - 77) % spatial_size
            
            # Distance from center
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            max_dist = np.sqrt(2 * center**2)
            
            # Reduce attention based on distance
            if dist > max_dist * 0.3:
                scores_np[:, :, :, i] *= 0.3
                scores_np[:, :, i, :] *= 0.3
        
        print(f"    🎯 Applied spatial focus in {meta.get('block_id', 'unknown')}")
    
    return mx.array(scores_np)

for block in ["up_0", "up_1", "up_2"]:
    attn_scores.SCORES_REGISTRY.set(block, spatial_focus_hook)

for i, prompt in enumerate(prompts):
    print(f"\n  Applying spatial focus: {prompt}")
    latents_gen = sd.generate_latents(prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
    for latents in latents_gen: pass
    img_path = f"artifacts/images/gallery/spatial_{i}.png"
    save_image(sd.decode(latents), img_path)
    techniques.append(("spatial", i, img_path))

attn_scores.SCORES_REGISTRY.clear()

# ==============================================================================
# TECHNIQUE 5: STYLE TRANSFER (Frequency manipulation)
# ==============================================================================
print("\n" + "="*60)
print("5️⃣ STYLE TRANSFER - Frequency-based style modification")
print("="*60)

def style_transfer_hook(q, k, v, meta):
    """Apply frequency-based style transfer."""
    v_array = np.array(v)
    
    # Apply frequency filtering for style
    # Amplify high frequencies for sharp style
    noise = np.random.randn(*v_array.shape) * 0.02
    v_styled = v_array * 1.2 + noise
    
    # Add some sinusoidal patterns for artistic effect
    spatial_dims = v_array.shape[2]
    if spatial_dims > 77:
        for i in range(77, spatial_dims):
            v_styled[:, :, i, :] *= (1 + 0.3 * np.sin(i * 0.5))
    
    print(f"    🎨 Applied style transfer in {meta.get('block_id', 'unknown')}")
    return q, k, mx.array(v_styled)

attn_scores.KV_REGISTRY.clear()
for block in ["up_0", "up_1", "up_2"]:
    attn_scores.KV_REGISTRY.set(block, style_transfer_hook)

for i, prompt in enumerate(prompts):
    print(f"\n  Applying style transfer: {prompt}")
    latents_gen = sd.generate_latents(prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
    for latents in latents_gen: pass
    img_path = f"artifacts/images/gallery/style_{i}.png"
    save_image(sd.decode(latents), img_path)
    techniques.append(("style", i, img_path))

# ==============================================================================
# CREATE COMPARISON IMAGES
# ==============================================================================
print("\n" + "="*60)
print("📊 CREATING COMPARISON GALLERY")
print("="*60)

technique_names = {
    "token_mask": "Token Masking",
    "chaos": "Attention Chaos", 
    "spatial": "Spatial Focus",
    "style": "Style Transfer"
}

# Create individual comparisons
for technique_name in technique_names:
    for i, prompt in enumerate(prompts):
        baseline_img = f"artifacts/images/gallery/baseline_{i}.png"
        technique_img = f"artifacts/images/gallery/{technique_name}_{i}.png"
        output_path = f"artifacts/images/gallery/comparison_{technique_name}_{i}.png"
        
        title = f"{technique_names[technique_name]}: {prompt[:40]}..."
        create_comparison(baseline_img, technique_img, output_path, title)
        print(f"  ✅ Created comparison for {technique_name} - prompt {i}")

# ==============================================================================
# CREATE MASTER GALLERY
# ==============================================================================
print("\n" + "="*60)
print("🖼️ CREATING MASTER GALLERY")
print("="*60)

def create_master_gallery():
    """Create a comprehensive gallery showing all techniques."""
    from PIL import Image, ImageDraw, ImageFont
    
    # Load all comparison images
    comparisons = []
    for technique_name in technique_names:
        technique_comparisons = []
        for i in range(len(prompts)):
            img_path = f"artifacts/images/gallery/comparison_{technique_name}_{i}.png"
            if os.path.exists(img_path):
                technique_comparisons.append(Image.open(img_path))
        comparisons.append(technique_comparisons)
    
    if not comparisons or not comparisons[0]:
        print("  ❌ No comparison images found")
        return
    
    # Calculate dimensions
    img_width = comparisons[0][0].width
    img_height = comparisons[0][0].height
    
    rows = len(technique_names)
    cols = len(prompts)
    
    # Create master canvas
    padding = 20
    canvas_width = cols * img_width + (cols + 1) * padding
    canvas_height = rows * img_height + (rows + 1) * padding + 100
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        title_font = ImageFont.load_default()
        label_font = title_font
    
    # Title
    draw.text((canvas_width // 2, 30), "CorePulse V4 - Manipulation Techniques Gallery", 
             fill='black', font=title_font, anchor='mt')
    
    # Place images
    y_offset = 100
    for row, technique_name in enumerate(technique_names):
        for col in range(len(prompts)):
            x = padding + col * (img_width + padding)
            y = y_offset + row * (img_height + padding)
            
            img_path = f"artifacts/images/gallery/comparison_{technique_name}_{col}.png"
            if os.path.exists(img_path):
                img = Image.open(img_path)
                canvas.paste(img, (x, y))
    
    # Save master gallery
    output_path = "artifacts/images/gallery/MASTER_GALLERY.png"
    canvas.save(output_path)
    print(f"  ✅ Created master gallery: {output_path}")
    return output_path

master_path = create_master_gallery()

# Clean up
attn_scores.KV_REGISTRY.clear()
attn_scores.SCORES_REGISTRY.clear()
attn_scores.enable_kv_hooks(False)
attn_scores.enable_scores_hooks(False)

print("\n" + "="*80)
print("✅ GALLERY GENERATION COMPLETE!")
print("="*80)
print(f"\nGenerated {len(technique_names)} techniques × {len(prompts)} prompts = {len(technique_names) * len(prompts)} comparisons")
print("\nTechniques demonstrated:")
for name, display in technique_names.items():
    print(f"  • {display}")
print(f"\nMaster gallery saved to: {master_path}")
print("\nAll hooks have been cleaned up and disabled.")