#!/usr/bin/env python3
"""
Generate comprehensive visual showcase for all corpus-mlx features.
Creates before/after comparison images for README display.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from corpus_mlx import (
    create_semantic_wrapper, 
    create_true_semantic_wrapper,
    CorePulseStableDiffusion
)
from adapters.stable_diffusion import StableDiffusion
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
        # Try to use a system font
        font_title = ImageFont.truetype("Arial.ttf", 16)
        font_label = ImageFont.truetype("Arial.ttf", 12)
    except:
        # Fall back to default font
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
    
    print("\n" + "="*60)
    print("SHOWCASE 1: TEXT-LEVEL SEMANTIC REPLACEMENT")
    print("="*60)
    
    # Create wrapper
    wrapper = create_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
    
    test_cases = [
        ("a red apple on a wooden table", "apple", "banana", "Apple → Banana"),
        ("a fluffy cat sitting on a sofa", "cat", "dog", "Cat → Dog"), 
        ("a blue car parked on the street", "car", "bicycle", "Car → Bicycle"),
        ("a laptop computer on a desk", "laptop", "book", "Laptop → Book")
    ]
    
    for i, (prompt, original, replacement, label) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {label}")
        print(f"Prompt: '{prompt}'")
        
        # Generate before (original)
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
        
        wrapper.replacements.clear()
        wrapper.disable()


def showcase_true_embedding_injection():
    """Showcase 2: TRUE Embedding Injection"""
    
    print("\n" + "="*60)
    print("SHOWCASE 2: TRUE EMBEDDING INJECTION")
    print("="*60)
    
    # Create wrapper
    wrapper = create_true_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
    
    test_cases = [
        ("a orange cat playing in a garden", "cat", "golden retriever dog", "Cat → Dog (Full)", 1.0),
        ("a brown horse in a field", "horse", "cow", "Horse → Cow (Partial)", 0.7),
        ("a small bird on a branch", "bird", "butterfly", "Bird → Butterfly (Blend)", 0.5)
    ]
    
    for i, (prompt, original, replacement, label, weight) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {label}")
        print(f"Prompt: '{prompt}' (weight: {weight})")
        
        # Generate before (original)
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
            f"Manipulates K,V tensors in UNet cross-attention - {weight_desc}",
            comparison_path
        )
        
        wrapper.injector.clear()


def showcase_advanced_prompt_injection():
    """Showcase 3: Advanced Prompt Injection"""
    
    print("\n" + "="*60)
    print("SHOWCASE 3: ADVANCED PROMPT INJECTION")
    print("="*60)
    
    # Create wrapper
    base_sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base")
    wrapper = CorePulseStableDiffusion(base_sd)
    
    test_cases = [
        ("a simple wooden chair", "golden metallic shiny", "Time-Windowed", {"start_step": 3, "end_step": 12}),
        ("a plain white house", "cyberpunk neon glowing", "Multi-Block", {"blocks": ["mid", "up_0", "up_1"]}),
        ("a regular coffee cup", "crystal glass transparent", "High Strength", {"strength": 2.0})
    ]
    
    for i, (prompt, injection, label, params) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {label}")
        print(f"Base: '{prompt}' + Injection: '{injection}'")
        
        # Generate before (no injection)
        before_img = generate_images_for_prompt(wrapper.sd, prompt)
        before_path = f"showcase_injection_{i}_before.png"
        Image.fromarray(before_img).save(before_path)
        
        # Add injection and generate after
        wrapper.add_injection(prompt=injection, **params)
        
        after_img = generate_images_for_prompt(wrapper.sd, prompt)
        after_path = f"showcase_injection_{i}_after.png"
        Image.fromarray(after_img).save(after_path)
        
        # Create comparison
        comparison_path = f"showcase_injection_{i}_comparison.png"
        create_comparison_image(
            before_path, after_path,
            f"Prompt Injection: {label}",
            f"Injects '{injection}' during generation - Advanced control",
            comparison_path
        )
        
        wrapper.injections.clear()


def showcase_corepulse_attention():
    """Showcase 4: CorePulse Attention Manipulation"""
    
    print("\n" + "="*60)
    print("SHOWCASE 4: COREPULSE ATTENTION MANIPULATION")
    print("="*60)
    
    # Import CorePulse
    from corpus_mlx.corepulse import CorePulse
    
    # Create base model
    base_sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base")
    corepulse = CorePulse(base_sd)
    
    test_cases = [
        ("a peaceful garden with flowers", "chaos", lambda: corepulse.chaos(intensity=2.0)),
        ("a modern city skyline", "suppression", lambda: corepulse.suppress(factor=0.05)),
        ("a simple mountain landscape", "amplification", lambda: corepulse.amplify(strength=8.0)),
        ("a bright sunny day", "inversion", lambda: corepulse.invert())
    ]
    
    for i, (prompt, effect, apply_effect) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {effect.title()}")
        print(f"Prompt: '{prompt}'")
        
        # Generate before (no effect)
        corepulse.reset()  # Clear any previous effects
        before_img = generate_images_for_prompt(base_sd, prompt)
        before_path = f"showcase_corepulse_{i}_before.png"
        Image.fromarray(before_img).save(before_path)
        
        # Apply effect and generate after
        apply_effect()
        after_img = generate_images_for_prompt(base_sd, prompt)
        after_path = f"showcase_corepulse_{i}_after.png"
        Image.fromarray(after_img).save(after_path)
        
        # Create comparison
        comparison_path = f"showcase_corepulse_{i}_comparison.png"
        create_comparison_image(
            before_path, after_path,
            f"CorePulse: {effect.title()}",
            f"Attention manipulation - {effect} effect applied",
            comparison_path
        )


def create_feature_summary():
    """Create a summary image showing all features."""
    
    print("\n" + "="*60)
    print("CREATING FEATURE SUMMARY")
    print("="*60)
    
    # Collect all comparison images
    comparisons = []
    
    # Find all comparison files
    for pattern in ["showcase_text_*_comparison.png", "showcase_embedding_*_comparison.png", 
                   "showcase_injection_*_comparison.png", "showcase_corepulse_*_comparison.png"]:
        import glob
        comparisons.extend(glob.glob(pattern))
    
    comparisons.sort()
    
    if not comparisons:
        print("No comparison images found!")
        return
    
    print(f"Found {len(comparisons)} comparison images")
    
    # Create grid layout
    cols = 2
    rows = (len(comparisons) + cols - 1) // cols
    
    # Load first image to get dimensions
    first_img = Image.open(comparisons[0])
    img_width, img_height = first_img.size
    
    # Create summary canvas
    canvas_width = img_width * cols + 20 * (cols + 1)
    canvas_height = img_height * rows + 20 * (rows + 1) + 60  # Extra space for title
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    
    # Add title
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    title = "corpus-mlx: Complete Feature Showcase"
    draw.text((canvas_width//2 - len(title)*6, 20), title, fill='black', font=font)
    
    # Place images
    for i, comp_path in enumerate(comparisons):
        row = i // cols
        col = i % cols
        
        x = 20 + col * (img_width + 20)
        y = 80 + row * (img_height + 20)
        
        img = Image.open(comp_path)
        canvas.paste(img, (x, y))
    
    canvas.save("corpus_mlx_complete_showcase.png")
    print("✅ Created complete showcase: corpus_mlx_complete_showcase.png")


def main():
    """Generate complete visual showcase."""
    
    print("🎨 GENERATING CORPUS-MLX VISUAL SHOWCASE")
    print("This will create before/after comparisons for all features")
    
    # Create output directory
    os.makedirs("showcase", exist_ok=True)
    os.chdir("showcase")
    
    try:
        # Generate all showcases
        showcase_text_semantic_replacement()
        showcase_true_embedding_injection()
        showcase_advanced_prompt_injection()
        showcase_corepulse_attention()
        
        # Create summary
        create_feature_summary()
        
        print("\n" + "🎉" * 20)
        print("SHOWCASE GENERATION COMPLETE!")
        print("🎉" * 20)
        print("\nGenerated files:")
        print("• Text replacement comparisons: showcase_text_*_comparison.png")
        print("• Embedding injection comparisons: showcase_embedding_*_comparison.png")
        print("• Prompt injection comparisons: showcase_injection_*_comparison.png")
        print("• CorePulse attention comparisons: showcase_corepulse_*_comparison.png")
        print("• Complete summary: corpus_mlx_complete_showcase.png")
        
        print("\n📸 Ready for README integration!")
        
    except Exception as e:
        print(f"❌ Error during showcase generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()