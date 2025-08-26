#!/usr/bin/env python3
"""Create proper demonstration images for CorePulse features."""

import os
import sys
sys.path.append('/Users/speed/Downloads/corpus-mlx/src/adapters/mlx/mlx-examples/stable_diffusion')

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

def create_side_by_side(img1_path, img2_path, title1, title2, output_path):
    """Create a side-by-side comparison with labels."""
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    # Resize to same size
    size = (512, 512)
    img1 = img1.resize(size, Image.LANCZOS)
    img2 = img2.resize(size, Image.LANCZOS)
    
    # Create comparison
    width = size[0] * 2 + 20
    height = size[1] + 80
    comparison = Image.new('RGB', (width, height), 'white')
    
    # Add images
    comparison.paste(img1, (0, 60))
    comparison.paste(img2, (size[0] + 20, 60))
    
    # Add labels
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except:
        font = None
        small_font = None
    
    # Title backgrounds
    draw.rectangle([0, 0, size[0], 50], fill='#2c3e50')
    draw.rectangle([size[0] + 20, 0, width, 50], fill='#2c3e50')
    
    # Titles
    draw.text((size[0]//2, 25), title1, fill='white', font=font, anchor='mm')
    draw.text((size[0] + 20 + size[0]//2, 25), title2, fill='white', font=font, anchor='mm')
    
    # Add arrow
    arrow_y = 60 + size[1]//2
    draw.text((size[0] + 10, arrow_y), "→", fill='#e74c3c', font=font, anchor='mm')
    
    comparison.save(output_path)
    print(f"Created: {output_path}")

def create_prompt_injection_demo():
    """Show how prompt injection replaces content while keeping context."""
    base_path = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme/"
    
    # We'll use the cat and dog images we already generated
    create_side_by_side(
        base_path + "cat_baseline.png",
        base_path + "dog_injection.png",
        "Base: 'cat in garden'",
        "Injected: 'dog' → content",
        base_path + "PROPER_prompt_injection.png"
    )

def create_attention_manipulation_demo():
    """Show attention weight boosting effects."""
    base_path = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme/"
    
    # Use castle images with different attention weights
    create_side_by_side(
        base_path + "castle_normal.png",
        base_path + "castle_boosted.png",
        "Normal Attention",
        "5x 'photorealistic'",
        base_path + "PROPER_attention_manipulation.png"
    )

def create_regional_control_demo():
    """Demonstrate regional masking effects."""
    base_path = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme/"
    
    # Load images
    forest = Image.open(base_path + "forest_baseline.png")
    waterfall = Image.open(base_path + "waterfall_regional.png")
    
    # Create a version showing regional replacement
    size = (512, 512)
    forest = forest.resize(size, Image.LANCZOS)
    waterfall = waterfall.resize(size, Image.LANCZOS)
    
    # Create mask visualization
    mask_visual = Image.new('RGB', (size[0] * 3 + 40, size[1] + 80), 'white')
    
    # Original
    mask_visual.paste(forest, (0, 60))
    
    # Mask representation
    mask_img = Image.new('RGB', size, 'black')
    draw = ImageDraw.Draw(mask_img)
    center = size[0] // 2
    radius = size[0] // 3
    draw.ellipse([center-radius, center-radius, center+radius, center+radius], fill='white')
    mask_visual.paste(mask_img, (size[0] + 20, 60))
    
    # Result (composite)
    result = forest.copy()
    # Create circular crop of waterfall
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([center-radius, center-radius, center+radius, center+radius], fill=255)
    
    # Composite
    result.paste(waterfall, (0, 0), mask)
    mask_visual.paste(result, ((size[0] + 20) * 2, 60))
    
    # Add labels
    draw = ImageDraw.Draw(mask_visual)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except:
        font = None
    
    # Title backgrounds
    for i in range(3):
        x_start = i * (size[0] + 20)
        x_end = x_start + size[0]
        draw.rectangle([x_start, 0, x_end, 50], fill='#2c3e50')
    
    # Titles
    draw.text((size[0]//2, 25), "Original Forest", fill='white', font=font, anchor='mm')
    draw.text((size[0] + 20 + size[0]//2, 25), "Regional Mask", fill='white', font=font, anchor='mm')
    draw.text(((size[0] + 20) * 2 + size[0]//2, 25), "Waterfall Injected", fill='white', font=font, anchor='mm')
    
    # Arrows
    arrow_y = 60 + size[1]//2
    draw.text((size[0] + 10, arrow_y), "→", fill='#e74c3c', font=font, anchor='mm')
    draw.text((size[0] * 2 + 30, arrow_y), "→", fill='#e74c3c', font=font, anchor='mm')
    
    mask_visual.save(base_path + "PROPER_regional_control.png")
    print(f"Created: {base_path}PROPER_regional_control.png")

def create_multiscale_demo():
    """Show multi-scale control at different resolutions."""
    base_path = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme/"
    
    create_side_by_side(
        base_path + "cathedral_structure.png",
        base_path + "cathedral_detailed.png",
        "Structure Level",
        "Detail Level",
        base_path + "PROPER_multiscale_control.png"
    )

def create_showcase_grid():
    """Create the main showcase grid."""
    base_path = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme/"
    
    # Load all proper demos
    demos = {
        "Prompt Injection": Image.open(base_path + "PROPER_prompt_injection.png"),
        "Attention Control": Image.open(base_path + "PROPER_attention_manipulation.png"),
        "Regional Control": Image.open(base_path + "PROPER_regional_control.png"),
        "Multi-Scale": Image.open(base_path + "PROPER_multiscale_control.png")
    }
    
    # Calculate grid size
    demo_height = 512 + 80  # Image height + labels
    demo_width = max(img.width for img in demos.values())
    
    # Create grid (2x2)
    grid_width = demo_width * 2 + 40
    grid_height = demo_height * 2 + 120
    
    grid = Image.new('RGB', (grid_width, grid_height), '#f8f9fa')
    
    # Add title
    draw = ImageDraw.Draw(grid)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        title_font = None
        label_font = None
    
    # Main title
    draw.rectangle([0, 0, grid_width, 80], fill='#1a1a1a')
    draw.text((grid_width//2, 40), "CorePulse MLX - Advanced Diffusion Control", 
              fill='white', font=title_font, anchor='mm')
    
    # Add demos
    positions = [
        (20, 100),  # Top left
        (demo_width + 20, 100),  # Top right
        (20, 100 + demo_height),  # Bottom left
        (demo_width + 20, 100 + demo_height)  # Bottom right
    ]
    
    for (name, img), pos in zip(demos.items(), positions):
        # Resize to fit if needed
        if img.width > demo_width - 40:
            ratio = (demo_width - 40) / img.width
            new_height = int(img.height * ratio)
            img = img.resize((demo_width - 40, new_height), Image.LANCZOS)
        
        # Add technique label
        label_y = pos[1] - 20
        draw.rectangle([pos[0], label_y, pos[0] + img.width, label_y + 25], fill='#3498db')
        draw.text((pos[0] + img.width//2, label_y + 12), name, 
                  fill='white', font=label_font, anchor='mm')
        
        # Paste demo
        grid.paste(img, pos)
    
    grid.save(base_path + "PROPER_COREPULSE_SHOWCASE.png")
    print(f"Created: {base_path}PROPER_COREPULSE_SHOWCASE.png")

def main():
    print("Creating Proper CorePulse Demonstration Images")
    print("=" * 50)
    
    create_prompt_injection_demo()
    create_attention_manipulation_demo()
    create_regional_control_demo()
    create_multiscale_demo()
    create_showcase_grid()
    
    print("=" * 50)
    print("All demonstrations created successfully!")

if __name__ == "__main__":
    main()