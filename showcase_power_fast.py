#!/usr/bin/env python3
"""Showcase CorePulse REAL POWER - Fast version with key trajectories."""

import sys
import os
sys.path.append('/Users/speed/Downloads/corpus-mlx/src/adapters/mlx/mlx-examples/stable_diffusion')

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_diffusion import StableDiffusion
from stable_diffusion import attn_hooks
from tqdm import tqdm
import math

def create_showcase_grid(images, titles, main_title):
    """Create a grid showcase of images."""
    if not images:
        return None
    
    # Size each image
    size = (256, 256)
    images = [img.resize(size, Image.LANCZOS) for img in images]
    
    # Create grid layout
    cols = 3
    rows = (len(images) + cols - 1) // cols
    
    width = size[0] * cols + (cols + 1) * 20
    height = size[1] * rows + (rows + 1) * 20 + 80
    
    canvas = Image.new('RGB', (width, height), '#0a0a0a')
    
    # Add gradient background
    draw = ImageDraw.Draw(canvas)
    for i in range(height):
        color = int(10 + (i/height) * 30)
        draw.rectangle([0, i, width, i+1], fill=(color, color, color + 10))
    
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        title_font = None
        label_font = None
    
    # Title
    draw.text((width//2, 40), main_title, fill='#e74c3c', font=title_font, anchor='mm')
    
    # Add images in grid
    x_start = 20
    y_start = 100
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        row = idx // cols
        col = idx % cols
        
        x = x_start + col * (size[0] + 20)
        y = y_start + row * (size[1] + 40)
        
        # Add glow effect for dramatic images
        if 'EXTREME' in title or '3x' in title or 'CHAOS' in title:
            for offset in range(3, 0, -1):
                glow_color = (255 - offset * 50, 50, 50)
                draw.rectangle([x-offset, y-offset, x+size[0]+offset, y+size[1]+offset], 
                             outline=glow_color, width=2)
        
        # Add border
        border_color = '#f39c12' if idx == 0 else '#34495e'
        draw.rectangle([x-1, y-1, x+size[0]+1, y+size[1]+1], outline=border_color, width=2)
        
        # Paste image
        canvas.paste(img, (x, y))
        
        # Add label
        draw.text((x + size[0]//2, y + size[1] + 10), title, 
                 fill='#ecf0f1', font=label_font, anchor='mt')
    
    return canvas

def generate_power_demos():
    """Generate key power demonstrations quickly."""
    print("\n=== COREPULSE POWER DEMONSTRATIONS ===")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    demos = []
    titles = []
    
    # Demo 1: Baseline
    print("Demo 1: Generating baseline...")
    attn_hooks.ATTN_HOOKS_ENABLED = False
    prompt = "a cyberpunk city with neon lights"
    latents_gen = sd.generate_latents(prompt, n_images=1, num_steps=15, cfg_weight=7.5, seed=42)
    for x_t in tqdm(latents_gen, total=15):
        latents = x_t
    img = sd.decode(latents)[0]
    demos.append(Image.fromarray((np.array(img) * 255).astype(np.uint8)))
    titles.append("BASELINE")
    
    # Demo 2: Extreme Style Boost
    print("Demo 2: Extreme style amplification...")
    attn_hooks.ATTN_HOOKS_ENABLED = True
    attn_hooks.attention_registry.clear()
    
    class ExtremeStyleBoost:
        def __call__(self, *, out=None, meta=None):
            if out is not None:
                block = str(meta.get('block_id', ''))
                if 'up' in block:
                    return out * 3.0  # 3x amplification!
            return out
    
    processor = ExtremeStyleBoost()
    for block in ['up_0', 'up_1', 'up_2']:
        attn_hooks.register_processor(block, processor)
    
    latents_gen = sd.generate_latents(prompt, n_images=1, num_steps=15, cfg_weight=7.5, seed=42)
    for x_t in tqdm(latents_gen, total=15):
        latents = x_t
    img = sd.decode(latents)[0]
    demos.append(Image.fromarray((np.array(img) * 255).astype(np.uint8)))
    titles.append("3x STYLE BOOST")
    
    # Demo 3: Structure Decay
    print("Demo 3: Structure decay...")
    attn_hooks.attention_registry.clear()
    
    class StructureDecay:
        def __call__(self, *, out=None, meta=None):
            if out is not None:
                block = str(meta.get('block_id', ''))
                if 'down' in block:
                    return out * 0.1  # 90% decay!
                elif 'mid' in block:
                    return out * 0.05  # 95% decay!
            return out
    
    processor = StructureDecay()
    for block in ['down_0', 'down_1', 'down_2', 'mid']:
        attn_hooks.register_processor(block, processor)
    
    latents_gen = sd.generate_latents(prompt, n_images=1, num_steps=15, cfg_weight=7.5, seed=42)
    for x_t in tqdm(latents_gen, total=15):
        latents = x_t
    img = sd.decode(latents)[0]
    demos.append(Image.fromarray((np.array(img) * 255).astype(np.uint8)))
    titles.append("90% DECAY")
    
    # Demo 4: Attention Inversion
    print("Demo 4: Attention inversion...")
    attn_hooks.attention_registry.clear()
    
    class AttentionInverter:
        def __call__(self, *, out=None, meta=None):
            if out is not None:
                return -out * 0.8 + 1.0  # Invert attention!
            return out
    
    processor = AttentionInverter()
    for block in ['down_0', 'down_1', 'up_0', 'up_1']:
        attn_hooks.register_processor(block, processor)
    
    latents_gen = sd.generate_latents(prompt, n_images=1, num_steps=15, cfg_weight=7.5, seed=42)
    for x_t in tqdm(latents_gen, total=15):
        latents = x_t
    img = sd.decode(latents)[0]
    demos.append(Image.fromarray((np.array(img) * 255).astype(np.uint8)))
    titles.append("INVERTED")
    
    # Demo 5: Chaotic Oscillation
    print("Demo 5: Chaotic oscillation...")
    attn_hooks.attention_registry.clear()
    
    class ChaoticOscillator:
        def __call__(self, *, out=None, meta=None):
            if out is not None:
                step = meta.get('step_idx', 0)
                # Chaotic function
                multiplier = 0.5 + 1.5 * abs(math.sin(step * 0.7)) * (1 + 0.5 * math.cos(step * 1.3))
                return out * multiplier
            return out
    
    processor = ChaoticOscillator()
    for block in ['down_0', 'down_1', 'down_2', 'mid', 'up_0', 'up_1', 'up_2']:
        attn_hooks.register_processor(block, processor)
    
    latents_gen = sd.generate_latents(prompt, n_images=1, num_steps=15, cfg_weight=7.5, seed=42)
    for x_t in tqdm(latents_gen, total=15):
        latents = x_t
    img = sd.decode(latents)[0]
    demos.append(Image.fromarray((np.array(img) * 255).astype(np.uint8)))
    titles.append("CHAOS MODE")
    
    # Demo 6: Block Isolation (only edges)
    print("Demo 6: Edge block isolation...")
    attn_hooks.attention_registry.clear()
    
    class EdgeIsolator:
        def __call__(self, *, out=None, meta=None):
            if out is not None:
                block = str(meta.get('block_id', ''))
                if block in ['down_0', 'up_2']:
                    return out * 2.5  # Amplify edges
                else:
                    return out * 0.2  # Suppress middle
            return out
    
    processor = EdgeIsolator()
    for block in ['down_0', 'down_1', 'down_2', 'mid', 'up_0', 'up_1', 'up_2']:
        attn_hooks.register_processor(block, processor)
    
    latents_gen = sd.generate_latents(prompt, n_images=1, num_steps=15, cfg_weight=7.5, seed=42)
    for x_t in tqdm(latents_gen, total=15):
        latents = x_t
    img = sd.decode(latents)[0]
    demos.append(Image.fromarray((np.array(img) * 255).astype(np.uint8)))
    titles.append("EDGE ISOLATION")
    
    # Demo 7: Timestep-based Evolution
    print("Demo 7: Timestep evolution...")
    attn_hooks.attention_registry.clear()
    
    class TimestepEvolution:
        def __call__(self, *, out=None, meta=None):
            if out is not None:
                step = meta.get('step_idx', 0)
                progress = step / 15.0
                # Start normal, end extreme
                multiplier = 1.0 + progress * 2.0
                return out * multiplier
            return out
    
    processor = TimestepEvolution()
    for block in ['up_0', 'up_1', 'up_2']:
        attn_hooks.register_processor(block, processor)
    
    latents_gen = sd.generate_latents(prompt, n_images=1, num_steps=15, cfg_weight=7.5, seed=42)
    for x_t in tqdm(latents_gen, total=15):
        latents = x_t
    img = sd.decode(latents)[0]
    demos.append(Image.fromarray((np.array(img) * 255).astype(np.uint8)))
    titles.append("TIME EVOLUTION")
    
    # Demo 8: Extreme Combination
    print("Demo 8: Extreme combination...")
    attn_hooks.attention_registry.clear()
    
    class ExtremeCombination:
        def __call__(self, *, out=None, meta=None):
            if out is not None:
                block = str(meta.get('block_id', ''))
                step = meta.get('step_idx', 0)
                
                # Different behavior per block
                if 'down' in block:
                    return out * (0.3 + 0.7 * abs(math.sin(step * 0.5)))
                elif 'mid' in block:
                    return -out * 0.5 + 0.5  # Partial inversion
                elif 'up' in block:
                    return out * (2.0 + math.cos(step * 0.3))
            return out
    
    processor = ExtremeCombination()
    for block in ['down_0', 'down_1', 'down_2', 'mid', 'up_0', 'up_1', 'up_2']:
        attn_hooks.register_processor(block, processor)
    
    latents_gen = sd.generate_latents(prompt, n_images=1, num_steps=15, cfg_weight=7.5, seed=42)
    for x_t in tqdm(latents_gen, total=15):
        latents = x_t
    img = sd.decode(latents)[0]
    demos.append(Image.fromarray((np.array(img) * 255).astype(np.uint8)))
    titles.append("EXTREME MIX")
    
    # Demo 9: Complete suppression test
    print("Demo 9: Complete suppression...")
    attn_hooks.attention_registry.clear()
    
    class CompleteSuppression:
        def __call__(self, *, out=None, meta=None):
            if out is not None:
                block = str(meta.get('block_id', ''))
                # Suppress everything except final upsampling
                if 'up_2' not in block:
                    return out * 0.01  # 99% suppression
            return out
    
    processor = CompleteSuppression()
    for block in ['down_0', 'down_1', 'down_2', 'mid', 'up_0', 'up_1']:
        attn_hooks.register_processor(block, processor)
    
    latents_gen = sd.generate_latents(prompt, n_images=1, num_steps=15, cfg_weight=7.5, seed=42)
    for x_t in tqdm(latents_gen, total=15):
        latents = x_t
    img = sd.decode(latents)[0]
    demos.append(Image.fromarray((np.array(img) * 255).astype(np.uint8)))
    titles.append("99% SUPPRESSED")
    
    # Clean up
    attn_hooks.attention_registry.clear()
    attn_hooks.ATTN_HOOKS_ENABLED = False
    
    return demos, titles

def main():
    print("=" * 70)
    print("COREPULSE REAL POWER - EXPERIMENTAL TRAJECTORIES")
    print("=" * 70)
    
    # Generate demonstrations
    demos, titles = generate_power_demos()
    
    if demos:
        # Create showcase
        showcase = create_showcase_grid(
            demos, titles,
            "COREPULSE EXPERIMENTAL POWER"
        )
        
        if showcase:
            # Save showcase
            showcase.save("/Users/speed/Downloads/corpus-mlx/artifacts/images/readme/POWER_SHOWCASE.png")
            print("\n✓ Saved POWER_SHOWCASE.png")
            
            # Also save individual demos
            for i, (demo, title) in enumerate(zip(demos, titles)):
                filename = f"/Users/speed/Downloads/corpus-mlx/artifacts/images/readme/DEMO_{i+1}_{title.replace(' ', '_')}.png"
                demo.save(filename)
                print(f"✓ Saved {filename}")
    
    print("\n" + "=" * 70)
    print("POWER SHOWCASE COMPLETE!")
    print("Demonstrated control dimensions:")
    print("• Extreme style amplification (3x)")
    print("• Structure decay (90%)")
    print("• Attention inversion")
    print("• Chaotic oscillation")
    print("• Edge block isolation") 
    print("• Timestep evolution")
    print("• Extreme combination effects")
    print("• Complete suppression (99%)")
    print("=" * 70)

if __name__ == "__main__":
    main()