#!/usr/bin/env python3
"""Create a showcase grid image from the generated images."""

from PIL import Image, ImageDraw, ImageFont
import os

def create_showcase_grid():
    """Create a grid showcase of CorePulse features."""
    
    # Load the generated images
    base_path = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme/"
    
    images = {
        "Prompt Injection": (
            Image.open(base_path + "cat_baseline.png"),
            Image.open(base_path + "dog_injection.png")
        ),
        "Attention Control": (
            Image.open(base_path + "castle_normal.png"),
            Image.open(base_path + "castle_boosted.png")
        ),
        "Regional Control": (
            Image.open(base_path + "forest_baseline.png"),
            Image.open(base_path + "waterfall_regional.png")
        ),
        "Multi-Scale": (
            Image.open(base_path + "cathedral_structure.png"),
            Image.open(base_path + "cathedral_detailed.png")
        )
    }
    
    # Create a 2x4 grid (2 rows, 4 columns)
    img_size = 512
    grid_width = img_size * 4
    grid_height = img_size * 2
    
    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    
    # Place images
    positions = [
        (0, 0), (img_size, 0), (img_size*2, 0), (img_size*3, 0),
        (0, img_size), (img_size, img_size), (img_size*2, img_size), (img_size*3, img_size)
    ]
    
    idx = 0
    for technique, (before, after) in images.items():
        # Resize images to ensure they fit
        before = before.resize((img_size, img_size), Image.LANCZOS)
        after = after.resize((img_size, img_size), Image.LANCZOS)
        
        # Place before and after
        grid.paste(before, positions[idx])
        grid.paste(after, positions[idx + 4])
        idx += 1
    
    # Save the grid
    grid.save(base_path + "REAL_COREPULSE_SHOWCASE.png")
    print(f"Created showcase grid at {base_path}REAL_COREPULSE_SHOWCASE.png")
    
    # Also save individual comparison images
    for technique, (before, after) in images.items():
        comparison = Image.new('RGB', (img_size * 2, img_size), 'white')
        comparison.paste(before.resize((img_size, img_size), Image.LANCZOS), (0, 0))
        comparison.paste(after.resize((img_size, img_size), Image.LANCZOS), (img_size, 0))
        
        filename = technique.lower().replace(" ", "_").replace("-", "_")
        comparison.save(base_path + f"REAL_{filename}.png")
        print(f"Created comparison image: REAL_{filename}.png")

if __name__ == "__main__":
    create_showcase_grid()