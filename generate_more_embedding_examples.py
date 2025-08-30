#!/usr/bin/env python3
"""Generate MORE TRUE embedding injection examples to showcase variety."""

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
    width = size[0] * 2 + 60
    height = size[1] + 120
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

def generate_images_for_prompt(sd_model, prompt, seed=42):
    """Generate image for a given prompt."""
    latents = None
    for step in sd_model.generate_latents(
        prompt,
        negative_text="blurry, ugly",
        num_steps=15,
        cfg_weight=7.5,
        seed=seed
    ):
        latents = step
    
    # Decode
    images = sd_model.autoencoder.decode(latents)
    img = images[0]
    img = mx.clip(img, -1, 1)
    img = ((img + 1) * 127.5).astype(mx.uint8)
    return np.array(img)

def create_embedding_example(wrapper, prompt, original, replacement, label, seed=42):
    """Create one embedding injection example."""
    print(f"\n🧠 Creating: {label}")
    print(f"   Prompt: '{prompt}'")
    print(f"   Replacement: {original} → {replacement}")
    
    # Generate before
    before_img = generate_images_for_prompt(wrapper.sd, prompt, seed=seed)
    before_path = f"temp_before.png"
    Image.fromarray(before_img).save(before_path)
    
    # Add replacement and generate after
    wrapper.add_replacement(original, replacement, weight=1.0)
    wrapper.injector.enable_for_prompt(prompt)
    
    after_img = generate_images_for_prompt(wrapper.sd, prompt, seed=seed)
    after_path = f"temp_after.png"
    Image.fromarray(after_img).save(after_path)
    
    # Create comparison
    safe_label = label.replace(" ", "_").replace("→", "to").replace("(", "").replace(")", "").lower()
    comparison_path = f"embedding_{safe_label}_comparison.png"
    create_comparison_image(
        before_path, after_path,
        f"TRUE Embedding Injection: {label}",
        "Manipulates text conditioning directly",
        comparison_path
    )
    
    # Check if it worked
    diff = np.mean(np.abs(before_img.astype(float) - after_img.astype(float)))
    if diff > 5.0:
        print(f"   ✅ SUCCESS: {label} worked (diff: {diff:.1f})")
        success = True
    else:
        print(f"   ❌ FAILED: {label} failed (diff: {diff:.1f})")
        success = False
    
    # Cleanup
    wrapper.injector.clear()
    os.remove(before_path)
    os.remove(after_path)
    
    return success, comparison_path

def main():
    """Generate multiple TRUE embedding injection examples."""
    
    print("🧠 GENERATING MORE TRUE EMBEDDING INJECTION EXAMPLES")
    print("=" * 70)
    
    # Create wrapper once
    wrapper = create_true_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
    
    # Create output directory
    os.makedirs("showcase", exist_ok=True)
    os.chdir("showcase")
    
    # Define test cases - variety of transformations
    test_cases = [
        # Animals
        ("a brown horse galloping in a field", "horse", "zebra", "Horse → Zebra", 42),
        ("a small bird sitting on a branch", "bird", "colorful butterfly", "Bird → Butterfly", 43),
        ("a white sheep in green grass", "sheep", "goat", "Sheep → Goat", 44),
        ("a black bear in the forest", "bear", "panda", "Bear → Panda", 45),
        
        # Objects  
        ("a red sports car on the road", "car", "motorcycle", "Car → Motorcycle", 46),
        ("a wooden chair in a room", "chair", "sofa", "Chair → Sofa", 47),
        ("a guitar on a stage", "guitar", "piano", "Guitar → Piano", 48),
        ("a bicycle in the park", "bicycle", "scooter", "Bicycle → Scooter", 49),
        
        # Nature/Food
        ("a large oak tree in summer", "oak tree", "cherry blossom tree", "Oak → Cherry Blossom", 50),
        ("a red apple on the table", "apple", "orange", "Apple → Orange", 51),
        ("a slice of pizza on a plate", "pizza", "burger", "Pizza → Burger", 52),
        ("a sunflower in the garden", "sunflower", "rose", "Sunflower → Rose", 53),
    ]
    
    successful_examples = []
    failed_examples = []
    
    for i, (prompt, original, replacement, label, seed) in enumerate(test_cases):
        try:
            success, comparison_path = create_embedding_example(
                wrapper, prompt, original, replacement, label, seed
            )
            
            if success:
                successful_examples.append((label, comparison_path))
            else:
                failed_examples.append(label)
                
        except Exception as e:
            print(f"   ❌ ERROR: {label} - {e}")
            failed_examples.append(label)
    
    # Summary
    print("\n" + "🎉" * 50)
    print("TRUE EMBEDDING INJECTION EXAMPLES COMPLETE!")
    print("🎉" * 50)
    
    print(f"\n✅ SUCCESSFUL EXAMPLES ({len(successful_examples)}):")
    for label, path in successful_examples:
        print(f"   • {label} -> {path}")
    
    if failed_examples:
        print(f"\n❌ FAILED EXAMPLES ({len(failed_examples)}):")
        for label in failed_examples:
            print(f"   • {label}")
    
    print(f"\n📊 SUCCESS RATE: {len(successful_examples)}/{len(test_cases)} ({100*len(successful_examples)/len(test_cases):.1f}%)")
    print(f"\n📁 All successful examples saved in showcase/ directory")
    print("🚀 Ready to update README with more examples!")

if __name__ == "__main__":
    main()