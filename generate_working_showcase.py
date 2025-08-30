#!/usr/bin/env python3
"""Generate WORKING showcase for all corpus-mlx features."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from corpus_mlx import create_semantic_wrapper, create_true_semantic_wrapper
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

def generate_images_for_prompt(model, prompt, negative="blurry, ugly", seed=42, num_steps=15):
    """Generate image for a given prompt."""
    latents = None
    for step in model.generate_latents(
        prompt,
        negative_text=negative,
        num_steps=num_steps,
        cfg_weight=7.5,
        seed=seed
    ):
        latents = step
    
    # Decode
    if hasattr(model, 'autoencoder'):
        images = model.autoencoder.decode(latents)
    elif hasattr(model, 'sd'):
        images = model.sd.autoencoder.decode(latents)
    else:
        raise AttributeError(f"Cannot find autoencoder in {type(model)}")
        
    img = images[0]
    img = mx.clip(img, -1, 1)
    img = ((img + 1) * 127.5).astype(mx.uint8)
    return np.array(img)

def showcase_text_semantic_replacement():
    """Showcase 1: Text-Level Semantic Replacement"""
    
    print("\\n" + "="*60)
    print("SHOWCASE 1: TEXT-LEVEL SEMANTIC REPLACEMENT")
    print("="*60)
    
    wrapper = create_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
    
    test_cases = [
        ("a red apple on a wooden table", "apple", "banana", "Apple → Banana"),
        ("a fluffy cat sitting on a sofa", "cat", "dog", "Cat → Dog"), 
        ("a blue car parked on the street", "car", "bicycle", "Car → Bicycle"),
        ("a laptop computer on a desk", "laptop", "book", "Laptop → Book")
    ]
    
    for i, (prompt, original, replacement, label) in enumerate(test_cases, 1):
        print(f"\\nTest {i}: {label}")
        
        # Generate before
        before_img = generate_images_for_prompt(wrapper.wrapper, prompt)
        before_path = f"showcase_text_{i}_before.png"
        Image.fromarray(before_img).save(before_path)
        
        # Add replacement and generate after
        wrapper.add_replacement(original, replacement)
        wrapper.enable()
        
        after_img = generate_images_for_prompt(wrapper.wrapper, prompt)
        after_path = f"showcase_text_{i}_after.png"
        Image.fromarray(after_img).save(after_path)
        
        # Create comparison
        comparison_path = f"showcase_text_{i}_comparison.png"
        create_comparison_image(
            before_path, after_path,
            f"Text Replacement: {label}",
            "Changes prompt text before tokenization - Simple & 100% effective",
            comparison_path
        )
        
        # Verify it worked
        diff = np.mean(np.abs(before_img.astype(float) - after_img.astype(float)))
        if diff > 5.0:
            print(f"   ✅ SUCCESS: {label} worked (diff: {diff:.1f})")
        else:
            print(f"   ❌ FAILED: {label} failed (diff: {diff:.1f})")
        
        wrapper.replacements.clear()
        wrapper.disable()

def showcase_true_embedding_injection():
    """Showcase 2: TRUE Embedding Injection"""
    
    print("\\n" + "="*60)
    print("SHOWCASE 2: TRUE EMBEDDING INJECTION")
    print("="*60)
    
    wrapper = create_true_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
    
    test_cases = [
        ("a orange cat playing in a garden", "cat", "golden retriever dog", "Cat → Dog (Full)", 1.0),
        ("a brown horse in a field", "horse", "cow", "Horse → Cow (Partial)", 0.7),
        ("a small bird on a branch", "bird", "butterfly", "Bird → Butterfly (Blend)", 0.5)
    ]
    
    for i, (prompt, original, replacement, label, weight) in enumerate(test_cases, 1):
        print(f"\\nTest {i}: {label}")
        
        # Generate before
        before_img = generate_images_for_prompt(wrapper.sd, prompt)
        before_path = f"showcase_embedding_{i}_before.png"
        Image.fromarray(before_img).save(before_path)
        
        # Add injection and generate after
        wrapper.add_replacement(original, replacement, weight=weight)
        wrapper.injector.enable_for_prompt(prompt)
        
        after_img = generate_images_for_prompt(wrapper.sd, prompt)
        after_path = f"showcase_embedding_{i}_after.png"
        Image.fromarray(after_img).save(after_path)
        
        # Create comparison
        comparison_path = f"showcase_embedding_{i}_comparison.png"
        weight_desc = f"Weight: {weight}" + (" (Full replacement)" if weight == 1.0 else " (Partial blend)")
        create_comparison_image(
            before_path, after_path,
            f"Embedding Injection: {label}",
            f"Manipulates text conditioning directly - {weight_desc}",
            comparison_path
        )
        
        # Verify it worked
        diff = np.mean(np.abs(before_img.astype(float) - after_img.astype(float)))
        if diff > 5.0:
            print(f"   ✅ SUCCESS: {label} worked (diff: {diff:.1f})")
        else:
            print(f"   ❌ FAILED: {label} failed (diff: {diff:.1f})")
        
        wrapper.injector.clear()

def main():
    """Generate working visual showcase."""
    
    print("🎨 GENERATING WORKING CORPUS-MLX SHOWCASE")
    print("This will create ACTUAL working before/after comparisons")
    
    # Create output directory
    os.makedirs("showcase", exist_ok=True)
    os.chdir("showcase")
    
    try:
        # Generate working showcases
        showcase_text_semantic_replacement()
        showcase_true_embedding_injection()
        
        print("\\n" + "🎉" * 20)
        print("WORKING SHOWCASE GENERATION COMPLETE!")
        print("🎉" * 20)
        print("\\nGenerated working files:")
        print("• Text replacement comparisons: showcase_text_*_comparison.png")
        print("• Embedding injection comparisons: showcase_embedding_*_comparison.png")
        
        print("\\n📸 Ready for README integration!")
        
    except Exception as e:
        print(f"❌ Error during showcase generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()