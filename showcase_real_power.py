#!/usr/bin/env python3
"""Showcase the REAL POWER of CorePulse with experimental trajectories."""

import sys
import os
sys.path.append('/Users/speed/Downloads/corpus-mlx/src/adapters/mlx/mlx-examples/stable_diffusion')

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from stable_diffusion import StableDiffusion
from stable_diffusion import attn_hooks
from tqdm import tqdm
import math

def create_trajectory_visualization(images, titles, trajectory_name, description):
    """Create a trajectory visualization showing evolution."""
    if not images:
        return None
    
    # Size each image
    size = (256, 256)
    images = [img.resize(size, Image.LANCZOS) for img in images]
    
    # Create canvas
    num_images = len(images)
    width = size[0] * num_images + (num_images - 1) * 20 + 40
    height = size[1] + 140
    
    canvas = Image.new('RGB', (width, height), '#0a0a0a')
    
    # Add gradient background
    draw = ImageDraw.Draw(canvas)
    for i in range(height):
        color = int(15 + (i/height) * 25)
        draw.rectangle([0, i, width, i+1], fill=(color, color, color + 5))
    
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        desc_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        title_font = None
        desc_font = None
        label_font = None
    
    # Title and description
    draw.text((width//2, 30), trajectory_name, fill='white', font=title_font, anchor='mm')
    draw.text((width//2, 60), description, fill='#7f8c8d', font=desc_font, anchor='mm')
    
    # Add images with progression
    x_pos = 20
    for i, (img, title) in enumerate(zip(images, titles)):
        # Add border
        bordered = Image.new('RGB', (size[0]+2, size[1]+2), '#e74c3c' if i == len(images)-1 else '#34495e')
        bordered.paste(img, (1, 1))
        canvas.paste(bordered, (x_pos, 90))
        
        # Add label
        draw.text((x_pos + size[0]//2, 90 + size[1] + 10), title, fill='#ecf0f1', font=label_font, anchor='mt')
        
        # Add arrow between images
        if i < len(images) - 1:
            arrow_x = x_pos + size[0] + 10
            arrow_y = 90 + size[1]//2
            draw.text((arrow_x, arrow_y), "→", fill='#f39c12', font=title_font, anchor='mm')
        
        x_pos += size[0] + 20
    
    return canvas

def trajectory_style_evolution():
    """Show style evolution trajectory."""
    print("\n=== TRAJECTORY: Style Evolution ===")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    base_prompt = "a majestic lion portrait"
    
    trajectories = []
    titles = []
    
    # Generate base
    print("Generating base...")
    attn_hooks.ATTN_HOOKS_ENABLED = False
    latents_gen = sd.generate_latents(base_prompt, n_images=1, num_steps=20, cfg_weight=7.5, seed=42)
    for x_t in tqdm(latents_gen, total=20):
        latents = x_t
    img = sd.decode(latents)[0]
    trajectories.append(Image.fromarray((np.array(img) * 255).astype(np.uint8)))
    titles.append("BASE")
    
    # Progressive style amplification
    style_levels = [1.2, 1.5, 2.0, 3.0]
    
    for level in style_levels:
        print(f"Generating with {level}x style boost...")
        attn_hooks.ATTN_HOOKS_ENABLED = True
        attn_hooks.attention_registry.clear()
        
        class StyleAmplifier:
            def __init__(self, strength):
                self.strength = strength
            
            def __call__(self, *, out=None, meta=None):
                if out is not None:
                    block = str(meta.get('block_id', ''))
                    # Amplify output blocks for style
                    if 'up' in block:
                        return out * self.strength
                return out
        
        processor = StyleAmplifier(level)
        for block in ['up_0', 'up_1', 'up_2']:
            attn_hooks.register_processor(block, processor)
        
        latents_gen = sd.generate_latents(base_prompt, n_images=1, num_steps=20, cfg_weight=7.5, seed=42)
        for x_t in tqdm(latents_gen, total=20):
            latents = x_t
        img = sd.decode(latents)[0]
        trajectories.append(Image.fromarray((np.array(img) * 255).astype(np.uint8)))
        titles.append(f"{level}x")
    
    visualization = create_trajectory_visualization(
        trajectories, titles,
        "STYLE AMPLIFICATION TRAJECTORY",
        "Progressive output block amplification from 1x → 3x"
    )
    
    attn_hooks.attention_registry.clear()
    attn_hooks.ATTN_HOOKS_ENABLED = False
    
    return visualization

def trajectory_structure_decay():
    """Show structure decay trajectory."""
    print("\n=== TRAJECTORY: Structure Decay ===")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    base_prompt = "a geometric crystal structure"
    
    trajectories = []
    titles = []
    
    # Generate with progressive structure decay
    decay_levels = [1.0, 0.7, 0.4, 0.2, 0.05]
    
    for level in decay_levels:
        print(f"Generating with {level} structure retention...")
        attn_hooks.ATTN_HOOKS_ENABLED = True if level != 1.0 else False
        attn_hooks.attention_registry.clear()
        
        if level != 1.0:
            class StructureDecayer:
                def __init__(self, retention):
                    self.retention = retention
                
                def __call__(self, *, out=None, meta=None):
                    if out is not None:
                        block = str(meta.get('block_id', ''))
                        # Decay input blocks for structure
                        if 'down' in block:
                            return out * self.retention
                        elif 'mid' in block:
                            return out * (self.retention * 0.7)
                    return out
            
            processor = StructureDecayer(level)
            for block in ['down_0', 'down_1', 'down_2', 'mid']:
                attn_hooks.register_processor(block, processor)
        
        latents_gen = sd.generate_latents(base_prompt, n_images=1, num_steps=20, cfg_weight=7.5, seed=100)
        for x_t in tqdm(latents_gen, total=20):
            latents = x_t
        img = sd.decode(latents)[0]
        trajectories.append(Image.fromarray((np.array(img) * 255).astype(np.uint8)))
        titles.append(f"{int(level*100)}%")
    
    visualization = create_trajectory_visualization(
        trajectories, titles,
        "STRUCTURE DECAY TRAJECTORY",
        "Progressive input block suppression: 100% → 5%"
    )
    
    attn_hooks.attention_registry.clear()
    attn_hooks.ATTN_HOOKS_ENABLED = False
    
    return visualization

def trajectory_oscillating_control():
    """Show oscillating control trajectory."""
    print("\n=== TRAJECTORY: Oscillating Control ===")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    base_prompt = "abstract colorful energy waves"
    
    trajectories = []
    titles = []
    
    # Different oscillation patterns
    oscillation_patterns = [
        ("Linear", lambda step: 1.0),
        ("Sine", lambda step: 1.0 + math.sin(step * 0.5) * 0.8),
        ("Square", lambda step: 2.0 if step % 4 < 2 else 0.5),
        ("Sawtooth", lambda step: 0.5 + (step % 5) * 0.3),
        ("Random", lambda step: np.random.uniform(0.3, 2.0))
    ]
    
    for name, pattern_func in oscillation_patterns:
        print(f"Generating with {name} oscillation...")
        attn_hooks.ATTN_HOOKS_ENABLED = True if name != "Linear" else False
        attn_hooks.attention_registry.clear()
        
        if name != "Linear":
            class OscillatingProcessor:
                def __init__(self, pattern):
                    self.pattern = pattern
                
                def __call__(self, *, out=None, meta=None):
                    if out is not None:
                        step = meta.get('step_idx', 0)
                        multiplier = self.pattern(step)
                        return out * multiplier
                    return out
            
            processor = OscillatingProcessor(pattern_func)
            for block in ['down_0', 'down_1', 'down_2', 'mid', 'up_0', 'up_1', 'up_2']:
                attn_hooks.register_processor(block, processor)
        
        latents_gen = sd.generate_latents(base_prompt, n_images=1, num_steps=20, cfg_weight=7.5, seed=200)
        for x_t in tqdm(latents_gen, total=20):
            latents = x_t
        img = sd.decode(latents)[0]
        trajectories.append(Image.fromarray((np.array(img) * 255).astype(np.uint8)))
        titles.append(name)
    
    visualization = create_trajectory_visualization(
        trajectories, titles,
        "OSCILLATION PATTERN TRAJECTORY",
        "Different temporal attention modulation patterns"
    )
    
    attn_hooks.attention_registry.clear()
    attn_hooks.ATTN_HOOKS_ENABLED = False
    
    return visualization

def trajectory_block_isolation():
    """Show block isolation trajectory."""
    print("\n=== TRAJECTORY: Block Isolation ===")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    base_prompt = "detailed mechanical clockwork"
    
    trajectories = []
    titles = []
    
    # Isolate different blocks
    block_configs = [
        ("ALL", None),
        ("DOWN", ['down_0', 'down_1', 'down_2']),
        ("MID", ['mid']),
        ("UP", ['up_0', 'up_1', 'up_2']),
        ("EDGES", ['down_0', 'up_2'])
    ]
    
    for name, blocks_to_boost in block_configs:
        print(f"Generating with {name} blocks boosted...")
        attn_hooks.ATTN_HOOKS_ENABLED = True if blocks_to_boost else False
        attn_hooks.attention_registry.clear()
        
        if blocks_to_boost:
            class BlockIsolator:
                def __init__(self, target_blocks):
                    self.targets = target_blocks
                
                def __call__(self, *, out=None, meta=None):
                    if out is not None:
                        block = str(meta.get('block_id', ''))
                        if block in self.targets:
                            return out * 2.0  # Boost target blocks
                        else:
                            return out * 0.3  # Suppress others
                    return out
            
            processor = BlockIsolator(blocks_to_boost)
            for block in ['down_0', 'down_1', 'down_2', 'mid', 'up_0', 'up_1', 'up_2']:
                attn_hooks.register_processor(block, processor)
        
        latents_gen = sd.generate_latents(base_prompt, n_images=1, num_steps=20, cfg_weight=7.5, seed=300)
        for x_t in tqdm(latents_gen, total=20):
            latents = x_t
        img = sd.decode(latents)[0]
        trajectories.append(Image.fromarray((np.array(img) * 255).astype(np.uint8)))
        titles.append(name)
    
    visualization = create_trajectory_visualization(
        trajectories, titles,
        "BLOCK ISOLATION TRAJECTORY",
        "Selective amplification of different UNet blocks"
    )
    
    attn_hooks.attention_registry.clear()
    attn_hooks.ATTN_HOOKS_ENABLED = False
    
    return visualization

def create_power_showcase():
    """Create the ultimate power showcase."""
    print("\n=== Creating POWER Showcase ===")
    
    # Generate all trajectories
    trajectories = []
    
    print("Generating all experimental trajectories...")
    
    viz1 = trajectory_style_evolution()
    if viz1:
        trajectories.append(viz1)
    
    viz2 = trajectory_structure_decay()
    if viz2:
        trajectories.append(viz2)
    
    viz3 = trajectory_oscillating_control()
    if viz3:
        trajectories.append(viz3)
    
    viz4 = trajectory_block_isolation()
    if viz4:
        trajectories.append(viz4)
    
    if not trajectories:
        print("No trajectories generated!")
        return
    
    # Create mega showcase
    max_width = max(img.width for img in trajectories)
    total_height = sum(img.height for img in trajectories) + 150
    
    showcase = Image.new('RGB', (max_width, total_height), '#000000')
    
    # Add gradient background
    draw = ImageDraw.Draw(showcase)
    for i in range(total_height):
        r = int(5 + (i/total_height) * 15)
        g = int(5 + (i/total_height) * 10)
        b = int(10 + (i/total_height) * 20)
        draw.rectangle([0, i, max_width, i+1], fill=(r, g, b))
    
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 64)
        subtitle_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except:
        title_font = None
        subtitle_font = None
    
    # Title
    draw.text((max_width//2, 50), "COREPULSE REAL POWER", 
              fill='#e74c3c', font=title_font, anchor='mm')
    draw.text((max_width//2, 95), "Experimental Trajectories & Control Dimensions", 
              fill='#95a5a6', font=subtitle_font, anchor='mm')
    
    # Add all trajectories
    y_pos = 130
    for traj in trajectories:
        # Center if narrower
        x_pos = (max_width - traj.width) // 2
        showcase.paste(traj, (x_pos, y_pos))
        y_pos += traj.height + 10
    
    # Save
    showcase.save("/Users/speed/Downloads/corpus-mlx/artifacts/images/readme/REAL_POWER_SHOWCASE.png")
    print("✓ Saved REAL_POWER_SHOWCASE.png")
    
    # Also save individual trajectories
    for i, traj in enumerate(trajectories):
        traj.save(f"/Users/speed/Downloads/corpus-mlx/artifacts/images/readme/TRAJECTORY_{i+1}.png")
        print(f"✓ Saved TRAJECTORY_{i+1}.png")

def main():
    print("=" * 70)
    print("SHOWCASING COREPULSE REAL POWER")
    print("Experimental Trajectories & Control Dimensions")
    print("=" * 70)
    
    create_power_showcase()
    
    print("\n" + "=" * 70)
    print("POWER SHOWCASE COMPLETE!")
    print("These trajectories show the REAL control dimensions:")
    print("• Style Evolution - Progressive amplification")
    print("• Structure Decay - Controlled decomposition")
    print("• Oscillating Control - Temporal modulation patterns")
    print("• Block Isolation - Selective component amplification")
    print("=" * 70)

if __name__ == "__main__":
    main()