#!/usr/bin/env python3
"""Generate ONE working embedding injection showcase."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from corpus_mlx import create_true_semantic_wrapper
from PIL import Image, ImageDraw, ImageFont
import mlx.core as mx
import numpy as np
import os

def create_comparison_image(before_path, after_path, title, description, output_path):
    """Create a side-by-side comparison image with labels."""
    
    # Load images
    before = Image.open(before_path)
    after = Image.open(after_path)
    
    # Resize to consistent size
    size = (256, 256)
    before = before.resize(size)
    after = after.resize(size)
    
    # Create comparison canvas
    width = size[0] * 2 + 60  # Space for labels
    height = size[1] + 120    # Space for title and labels
    canvas = Image.new('RGB', (width, height), 'white')
    
    # Paste images
    canvas.paste(before, (10, 60))
    canvas.paste(after, (size[0] + 40, 60))
    
    # Add labels
    draw = ImageDraw.Draw(canvas)
    
    try:
        font_title = ImageFont.truetype("Arial.ttf", 16)
        font_label = ImageFont.truetype("Arial.ttf", 12)
    except:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
    
    # Title
    draw.text((width//2 - len(title)*4, 10), title, fill='black', font=font_title)
    
    # Description
    draw.text((10, 30), description, fill='gray', font=font_label)
    
    # Before/After labels
    draw.text((10 + size[0]//2 - 20, height-40), "BEFORE", fill='black', font=font_label)
    draw.text((size[0] + 40 + size[0]//2 - 20, height-40), "AFTER", fill='black', font=font_label)
    
    # Save
    canvas.save(output_path)
    print(f"✅ Created comparison: {output_path}")

def generate_images_for_prompt(sd_model, prompt):
    """Generate image for a given prompt."""
    latents = None
    for step in sd_model.generate_latents(
        prompt,
        negative_text="blurry, ugly",
        num_steps=15,
        cfg_weight=7.5,
        seed=42
    ):
        latents = step
    
    # Decode
    images = sd_model.autoencoder.decode(latents)
    img = images[0]
    img = mx.clip(img, -1, 1)
    img = ((img + 1) * 127.5).astype(mx.uint8)
    return np.array(img)

def main():
    """Generate ONE working embedding injection example."""
    
    print("🧠 TESTING TRUE EMBEDDING INJECTION SHOWCASE")
    print("=" * 60)
    
    # Create wrapper
    wrapper = create_true_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
    
    prompt = "a orange cat playing in a garden"
    original = "cat"
    replacement = "golden retriever dog"
    
    print(f"Prompt: '{prompt}'")
    print(f"Replacement: {original} → {replacement}")
    
    # Generate before
    print("\\n1. Generating BEFORE (cat)...")
    before_img = generate_images_for_prompt(wrapper.sd, prompt)
    before_path = "final_embedding_before.png"
    Image.fromarray(before_img).save(before_path)
    
    # Add replacement and generate after
    print("\\n2. Adding replacement and generating AFTER...")
    wrapper.add_replacement(original, replacement, weight=1.0)
    wrapper.injector.enable_for_prompt(prompt)
    
    after_img = generate_images_for_prompt(wrapper.sd, prompt)
    after_path = "final_embedding_after.png"
    Image.fromarray(after_img).save(after_path)
    
    # Create comparison
    comparison_path = "final_embedding_comparison.png"
    create_comparison_image(
        before_path, after_path,
        "TRUE Embedding Injection: Cat → Dog",
        "Manipulates text conditioning directly - Complete replacement",
        comparison_path
    )
    
    # Check if it worked
    diff = np.mean(np.abs(before_img.astype(float) - after_img.astype(float)))
    print(f"\\n📊 Pixel difference: {diff}")
    
    if diff > 5.0:
        print("✅ SUCCESS: TRUE embedding injection worked!")
    else:
        print("❌ FAILURE: Images are too similar")
        
    wrapper.injector.clear()

if __name__ == "__main__":
    main()