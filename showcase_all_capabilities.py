#!/usr/bin/env python3
"""
Complete showcase of ALL CorePulse V4 capabilities.
Shows what we CAN do, what we PARTIALLY do, and what we DON'T do.
"""

import mlx.core as mx
import numpy as np
import sys
import os
from PIL import Image, ImageDraw, ImageFont

sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')

# Enable hooks BEFORE importing model
from stable_diffusion import attn_scores
attn_scores.enable_kv_hooks(True)
attn_scores.enable_scores_hooks(True)

from stable_diffusion import StableDiffusionXL

print("=" * 80)
print("🚀 COREPULSE V4 - COMPLETE CAPABILITY SHOWCASE")
print("=" * 80)

os.makedirs("artifacts/images/showcase", exist_ok=True)

def save_image(image, filepath):
    """Save MLX image array to file."""
    img_np = np.array(image)
    if img_np.ndim == 4:
        img_np = img_np[0]
    if img_np.dtype in [np.float32, np.float64]:
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_np).save(filepath)
    return filepath

print("\n✓ Hooks enabled:", attn_scores.hooks_wanted())
print("Loading SDXL with PatchedMHA...")
sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)

# Test prompt
base_prompt = "a vibrant forest with sunlight streaming through trees"
techniques_demonstrated = []

# ==============================================================================
print("\n" + "="*60)
print("✅ WHAT WE CAN DO (FULLY DEMONSTRATED)")
print("="*60)

# Clear hooks for baseline
attn_scores.KV_REGISTRY.clear()
attn_scores.SCORES_REGISTRY.clear()

# 1. BASELINE
print("\n1️⃣ BASELINE - No manipulation")
latents_gen = sd.generate_latents(base_prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
for latents in latents_gen: pass
baseline = save_image(sd.decode(latents), "artifacts/images/showcase/01_baseline.png")
techniques_demonstrated.append(("Baseline", baseline))

# 2. TOKEN-LEVEL MASKING
print("\n2️⃣ TOKEN-LEVEL MASKING - Zero out specific tokens")
def mask_tokens(q, k, v, meta):
    v_array = np.array(v)
    if v_array.shape[2] > 10:
        v_array[:, :, 3:7, :] = 0  # Mask "forest with sunlight"
        print(f"  🎭 Masked tokens 3-7 in {meta.get('block_id', '?')}")
    return q, k, mx.array(v_array)

for block in ["down_2", "mid", "up_0"]:
    attn_scores.KV_REGISTRY.set(block, mask_tokens)

latents_gen = sd.generate_latents(base_prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
for latents in latents_gen: pass
img = save_image(sd.decode(latents), "artifacts/images/showcase/02_token_mask.png")
techniques_demonstrated.append(("Token Masking", img))
attn_scores.KV_REGISTRY.clear()

# 3. ATTENTION MANIPULATION - BOOST
print("\n3️⃣ ATTENTION BOOST - Amplify specific attention")
def boost_attention(scores, meta):
    scores_np = np.array(scores)
    scores_np *= 2.0  # Double attention strength
    print(f"  ⚡ Boosted attention x2 in {meta.get('block_id', '?')}")
    return mx.array(scores_np)

for block in ["mid", "up_0"]:
    attn_scores.SCORES_REGISTRY.set(block, boost_attention)

latents_gen = sd.generate_latents(base_prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
for latents in latents_gen: pass
img = save_image(sd.decode(latents), "artifacts/images/showcase/03_attention_boost.png")
techniques_demonstrated.append(("Attention Boost", img))
attn_scores.SCORES_REGISTRY.clear()

# 4. SPATIAL/REGIONAL CONTROL - CENTER FOCUS
print("\n4️⃣ SPATIAL CONTROL - Center focus")
def center_focus(scores, meta):
    scores_np = np.array(scores)
    seq_len = scores_np.shape[-1]
    
    if seq_len > 77:
        spatial_size = int((seq_len - 77) ** 0.5)
        center = spatial_size // 2
        
        for i in range(77, seq_len):
            y = (i - 77) // spatial_size
            x = (i - 77) % spatial_size
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            
            if dist > spatial_size * 0.4:
                scores_np[:, :, :, i] *= 0.1
                scores_np[:, :, i, :] *= 0.1
        
        print(f"  🎯 Applied center focus in {meta.get('block_id', '?')}")
    
    return mx.array(scores_np)

for block in ["up_0", "up_1"]:
    attn_scores.SCORES_REGISTRY.set(block, center_focus)

latents_gen = sd.generate_latents(base_prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
for latents in latents_gen: pass
img = save_image(sd.decode(latents), "artifacts/images/showcase/04_center_focus.png")
techniques_demonstrated.append(("Center Focus", img))
attn_scores.SCORES_REGISTRY.clear()

# 5. FREQUENCY-BASED STYLE TRANSFER
print("\n5️⃣ FREQUENCY STYLE - Sinusoidal patterns")
def frequency_style(q, k, v, meta):
    v_array = np.array(v)
    
    # Apply sinusoidal modulation
    for i in range(v_array.shape[2]):
        v_array[:, :, i, :] *= (1 + 0.4 * np.sin(i * 0.2))
    
    # Add high-frequency texture
    noise = np.random.randn(*v_array.shape) * 0.02
    v_array = v_array * 1.1 + noise
    
    print(f"  🎨 Applied frequency style in {meta.get('block_id', '?')}")
    return q, k, mx.array(v_array)

for block in ["up_0", "up_1", "up_2"]:
    attn_scores.KV_REGISTRY.set(block, frequency_style)

latents_gen = sd.generate_latents(base_prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
for latents in latents_gen: pass
img = save_image(sd.decode(latents), "artifacts/images/showcase/05_frequency_style.png")
techniques_demonstrated.append(("Frequency Style", img))
attn_scores.KV_REGISTRY.clear()

# 6. MULTI-BLOCK EFFECTS - Different per layer
print("\n6️⃣ MULTI-BLOCK - Different effects at different depths")
def down_block_hook(q, k, v, meta):
    # Structure emphasis in down blocks
    v_array = np.array(v) * 1.3
    print(f"  📐 Structure emphasis in {meta.get('block_id', '?')}")
    return q, k, mx.array(v_array)

def mid_block_hook(q, k, v, meta):
    # Content modification in middle
    v_array = np.array(v)
    v_array += np.random.randn(*v_array.shape) * 0.1
    print(f"  🎲 Content variation in {meta.get('block_id', '?')}")
    return q, k, mx.array(v_array)

def up_block_hook(q, k, v, meta):
    # Detail enhancement in up blocks
    v_array = np.array(v)
    v_array = v_array * 0.9 + np.sin(v_array * 5) * 0.1
    print(f"  ✨ Detail enhancement in {meta.get('block_id', '?')}")
    return q, k, mx.array(v_array)

attn_scores.KV_REGISTRY.set("down_2", down_block_hook)
attn_scores.KV_REGISTRY.set("mid", mid_block_hook)
attn_scores.KV_REGISTRY.set("up_0", up_block_hook)

latents_gen = sd.generate_latents(base_prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
for latents in latents_gen: pass
img = save_image(sd.decode(latents), "artifacts/images/showcase/06_multiblock.png")
techniques_demonstrated.append(("Multi-Block", img))
attn_scores.KV_REGISTRY.clear()

# 7. WAVE PATTERNS
print("\n7️⃣ WAVE PATTERNS - Radial waves")
def wave_pattern(q, k, v, meta):
    v_array = np.array(v)
    seq_len = v_array.shape[2]
    
    if seq_len > 77:
        spatial_size = int((seq_len - 77) ** 0.5)
        center = spatial_size // 2
        
        for i in range(77, seq_len):
            y = (i - 77) // spatial_size
            x = (i - 77) % spatial_size
            
            # Radial wave from center
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            wave = np.sin(dist * 0.5) * 0.3 + 1.0
            v_array[:, :, i, :] *= wave
        
        print(f"  🌊 Applied wave pattern in {meta.get('block_id', '?')}")
    
    return q, k, mx.array(v_array)

for block in ["up_0", "up_1"]:
    attn_scores.KV_REGISTRY.set(block, wave_pattern)

latents_gen = sd.generate_latents(base_prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
for latents in latents_gen: pass
img = save_image(sd.decode(latents), "artifacts/images/showcase/07_wave_pattern.png")
techniques_demonstrated.append(("Wave Pattern", img))
attn_scores.KV_REGISTRY.clear()

# 8. ATTENTION CHAOS
print("\n8️⃣ ATTENTION CHAOS - Scramble patterns")
def chaos_attention(q, k, v, meta):
    k_array = np.array(k)
    v_array = np.array(v)
    
    # Add significant noise
    k_chaos = k_array + np.random.randn(*k_array.shape) * 0.4
    v_chaos = v_array + np.random.randn(*v_array.shape) * 0.4
    
    print(f"  🌀 Applied chaos in {meta.get('block_id', '?')}")
    return q, mx.array(k_chaos), mx.array(v_chaos)

for block in ["mid", "up_0"]:
    attn_scores.KV_REGISTRY.set(block, chaos_attention)

latents_gen = sd.generate_latents(base_prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
for latents in latents_gen: pass
img = save_image(sd.decode(latents), "artifacts/images/showcase/08_chaos.png")
techniques_demonstrated.append(("Chaos", img))
attn_scores.KV_REGISTRY.clear()

# 9. DIAGONAL ATTENTION
print("\n9️⃣ DIAGONAL ATTENTION - Diagonal emphasis")
def diagonal_attention(scores, meta):
    scores_np = np.array(scores)
    seq_len = scores_np.shape[-1]
    
    if seq_len > 77:
        spatial_size = int((seq_len - 77) ** 0.5)
        
        for i in range(77, seq_len):
            y = (i - 77) // spatial_size
            x = (i - 77) % spatial_size
            
            # Emphasize diagonal
            if abs(x - y) < 2:
                scores_np[:, :, :, i] *= 2.0
            else:
                scores_np[:, :, :, i] *= 0.5
        
        print(f"  ↗️ Applied diagonal attention in {meta.get('block_id', '?')}")
    
    return mx.array(scores_np)

for block in ["up_0", "up_1"]:
    attn_scores.SCORES_REGISTRY.set(block, diagonal_attention)

latents_gen = sd.generate_latents(base_prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
for latents in latents_gen: pass
img = save_image(sd.decode(latents), "artifacts/images/showcase/09_diagonal.png")
techniques_demonstrated.append(("Diagonal Attention", img))
attn_scores.SCORES_REGISTRY.clear()

# 10. KALEIDOSCOPE EFFECT
print("\n🔟 KALEIDOSCOPE - Rotational symmetry")
def kaleidoscope(q, k, v, meta):
    v_array = np.array(v)
    seq_len = v_array.shape[2]
    
    if seq_len > 77:
        spatial_size = int((seq_len - 77) ** 0.5)
        center = spatial_size // 2
        
        for i in range(77, seq_len):
            y = (i - 77) // spatial_size
            x = (i - 77) % spatial_size
            
            # Create rotational symmetry
            angle = np.arctan2(y - center, x - center)
            pattern = np.sin(angle * 6) * 0.3 + 1.0
            v_array[:, :, i, :] *= pattern
        
        print(f"  🎪 Applied kaleidoscope in {meta.get('block_id', '?')}")
    
    return q, k, mx.array(v_array)

for block in ["up_0", "up_1"]:
    attn_scores.KV_REGISTRY.set(block, kaleidoscope)

latents_gen = sd.generate_latents(base_prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
for latents in latents_gen: pass
img = save_image(sd.decode(latents), "artifacts/images/showcase/10_kaleidoscope.png")
techniques_demonstrated.append(("Kaleidoscope", img))
attn_scores.KV_REGISTRY.clear()

# ==============================================================================
print("\n" + "="*60)
print("⚠️ WHAT WE PARTIALLY DO")
print("="*60)

# 11. PSEUDO PROMPT INJECTION
print("\n1️⃣1️⃣ PSEUDO PROMPT INJECTION - Modify attention, not true embedding")
def pseudo_inject(q, k, v, meta):
    """We modify attention patterns but don't inject real text embeddings."""
    v_array = np.array(v)
    
    # Modify text tokens (0-77) to simulate different prompt
    if v_array.shape[2] > 77:
        # This changes attention but doesn't inject actual new text
        v_array[:, :, :77, :] += np.random.normal(0, 0.1, v_array[:, :, :77, :].shape)
        print(f"  💉 Pseudo-injected variation in {meta.get('block_id', '?')}")
    
    return q, k, mx.array(v_array)

attn_scores.KV_REGISTRY.set("mid", pseudo_inject)
latents_gen = sd.generate_latents(base_prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
for latents in latents_gen: pass
img = save_image(sd.decode(latents), "artifacts/images/showcase/11_pseudo_inject.png")
techniques_demonstrated.append(("Pseudo Injection", img))
attn_scores.KV_REGISTRY.clear()

# 12. REGIONAL CONTROL (NOT TRUE REGIONAL PROMPTS)
print("\n1️⃣2️⃣ REGIONAL CONTROL - Spatial, not semantic regions")
def regional_control(q, k, v, meta):
    """We control spatial regions but can't inject different prompts per region."""
    v_array = np.array(v)
    seq_len = v_array.shape[2]
    
    if seq_len > 77:
        spatial_size = int((seq_len - 77) ** 0.5)
        
        # Left half vs right half
        for i in range(77, seq_len):
            x = (i - 77) % spatial_size
            
            if x < spatial_size // 2:
                v_array[:, :, i, :] *= 1.3  # Enhance left
            else:
                v_array[:, :, i, :] *= 0.7  # Suppress right
        
        print(f"  🎨 Applied regional control in {meta.get('block_id', '?')}")
    
    return q, k, mx.array(v_array)

attn_scores.KV_REGISTRY.set("up_0", regional_control)
latents_gen = sd.generate_latents(base_prompt, n_images=1, num_steps=2, cfg_weight=1.0, seed=42)
for latents in latents_gen: pass
img = save_image(sd.decode(latents), "artifacts/images/showcase/12_regional.png")
techniques_demonstrated.append(("Regional Control", img))
attn_scores.KV_REGISTRY.clear()

# ==============================================================================
print("\n" + "="*60)
print("❌ WHAT WE DON'T DO (CorePulse Claims)")
print("="*60)

print("""
These features would require:

1. TRUE MULTI-PROMPT INJECTION
   - Would need: Access to text encoder
   - Would need: Generate different CLIP embeddings
   - We have: Only attention manipulation
   
2. PROPER TEXT TOKEN REPLACEMENT
   - Would need: Replace conditioning embeddings
   - Would need: Proper CLIP token generation
   - We have: Only modify existing tokens
   
3. WORD-SPECIFIC TOKEN IDENTIFICATION
   - Would need: Tokenizer integration
   - Would need: Map words to exact token positions
   - We have: Only position-based masking
""")

# ==============================================================================
# CREATE GALLERY
# ==============================================================================
print("\n" + "="*60)
print("📊 CREATING COMPLETE CAPABILITY GALLERY")
print("="*60)

def create_gallery(images_with_labels, output_path):
    """Create a gallery of all techniques."""
    if not images_with_labels:
        return
    
    # Load images
    images = []
    for label, path in images_with_labels:
        if os.path.exists(path):
            img = Image.open(path)
            images.append((label, img))
    
    if not images:
        return
    
    # Calculate grid
    cols = 4
    rows = (len(images) + cols - 1) // cols
    
    # Image size
    img_width = 256
    img_height = 256
    padding = 10
    
    # Create canvas
    canvas_width = cols * (img_width + padding) + padding
    canvas_height = rows * (img_height + padding + 30) + padding + 80
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Font
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        title_font = ImageFont.load_default()
        label_font = title_font
    
    # Title
    draw.text((canvas_width // 2, 30), "CorePulse V4 - Complete Capability Showcase", 
             fill='black', font=title_font, anchor='mt')
    
    # Place images
    for i, (label, img) in enumerate(images):
        row = i // cols
        col = i % cols
        
        x = padding + col * (img_width + padding)
        y = 80 + row * (img_height + padding + 30)
        
        # Resize image
        img.thumbnail((img_width, img_height), Image.Resampling.LANCZOS)
        
        # Paste image
        canvas.paste(img, (x, y))
        
        # Add label
        draw.text((x + img_width // 2, y + img_height + 5), label, 
                 fill='black', font=label_font, anchor='mt')
    
    # Save
    canvas.save(output_path)
    return output_path

gallery_path = create_gallery(techniques_demonstrated, 
                              "artifacts/images/showcase/COMPLETE_SHOWCASE.png")

# Clean up
attn_scores.KV_REGISTRY.clear()
attn_scores.SCORES_REGISTRY.clear()
attn_scores.enable_kv_hooks(False)
attn_scores.enable_scores_hooks(False)

# ==============================================================================
print("\n" + "="*80)
print("✅ COMPLETE CAPABILITY SHOWCASE FINISHED!")
print("="*80)

print("\n📊 SUMMARY:")
print("\n✅ WHAT WE CAN DO (10 techniques):")
for i, (name, _) in enumerate(techniques_demonstrated[:10], 1):
    print(f"  {i}. {name}")

print("\n⚠️ WHAT WE PARTIALLY DO (2 techniques):")
for i, (name, _) in enumerate(techniques_demonstrated[10:12], 11):
    print(f"  {i}. {name}")

print("\n❌ WHAT WE DON'T DO:")
print("  - True multi-prompt injection")
print("  - Proper text token replacement")
print("  - Word-specific token identification")

print(f"\nGallery saved to: {gallery_path}")
print("\nAll techniques use REAL pre-attention hooks, not post-processing!")