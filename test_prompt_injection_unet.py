#!/usr/bin/env python3
"""Test the prompt injection UNet - TRUE DataVoid-style prompt injection."""

import sys
import os
sys.path.append('/Users/speed/Downloads/corpus-mlx/src/adapters/mlx/mlx-examples/stable_diffusion')

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from prompt_injection_unet import PromptInjectionSD
from tqdm import tqdm

def create_comparison(baseline, modified, title, description, labels=("BASELINE", "INJECTED")):
    """Create a clean comparison image."""
    
    # Ensure images are PIL
    if not isinstance(baseline, Image.Image):
        baseline = Image.fromarray((np.array(baseline) * 255).astype(np.uint8))
    if not isinstance(modified, Image.Image):
        modified = Image.fromarray((np.array(modified) * 255).astype(np.uint8))
    
    # Resize
    size = (512, 512)
    baseline = baseline.resize(size, Image.LANCZOS)
    modified = modified.resize(size, Image.LANCZOS)
    
    # Canvas
    width = size[0] * 2 + 60
    height = size[1] + 140
    canvas = Image.new('RGB', (width, height), '#1a1a1a')
    draw = ImageDraw.Draw(canvas)
    
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        desc_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        title_font = desc_font = label_font = None
    
    # Title & description
    draw.text((width//2, 30), title, fill='white', font=title_font, anchor='mm')
    draw.text((width//2, 60), description, fill='#888', font=desc_font, anchor='mm')
    
    # Images
    x1, x2 = 20, size[0] + 40
    y = 90
    
    canvas.paste(baseline, (x1, y))
    canvas.paste(modified, (x2, y))
    
    # Labels
    draw.text((x1 + size[0]//2, y + size[1] + 10), labels[0], fill='#666', font=label_font, anchor='mt')
    draw.text((x2 + size[0]//2, y + size[1] + 10), labels[1], fill='#f39c12', font=label_font, anchor='mt')
    
    return canvas

def test_content_swap():
    """Test swapping content (cat/dog) while keeping composition."""
    print("\n=== TEST 1: CONTENT SWAP ===")
    print("Base: 'a blue dog in a garden'")
    print("Injecting 'white cat' into MID blocks (content)")
    print("Keeping garden composition in other blocks\n")
    
    # Initialize our modified SD
    sd = PromptInjectionSD("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    base_prompt = "a blue dog playing in a garden"
    
    # Generate baseline
    print("Generating baseline...")
    latents = sd.generate_latents(base_prompt, n_images=1, num_steps=20, cfg_weight=7.5, seed=42)
    for x_t in tqdm(latents, total=20):
        baseline_latents = x_t
    baseline_img = sd.decode(baseline_latents)[0]
    
    # Now inject different prompt into MID blocks
    sd.inject_prompts({
        'mid': 'a white cat',  # Content blocks
        'down_2': 'a white cat'  # Also some mid-level features
    })
    
    print("\nGenerating with injected prompts...")
    latents = sd.generate_latents(base_prompt, n_images=1, num_steps=20, cfg_weight=7.5, seed=42)
    for x_t in tqdm(latents, total=20):
        injected_latents = x_t
    injected_img = sd.decode(injected_latents)[0]
    
    # Clear injection for next test
    sd.clear_injection()
    
    return create_comparison(
        baseline_img, injected_img,
        "TRUE CONTENT SWAP",
        "MID: 'white cat' | OTHER: 'blue dog in garden'"
    )

def test_style_injection():
    """Test injecting style while keeping content."""
    print("\n=== TEST 2: STYLE INJECTION ===")
    print("Base: 'a robot in a city'")
    print("Injecting 'cyberpunk neon glowing' into UP blocks (style)")
    print("Keeping robot content in MID blocks\n")
    
    sd = PromptInjectionSD("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    base_prompt = "a robot in a city"
    
    # Baseline
    print("Generating baseline...")
    latents = sd.generate_latents(base_prompt, n_images=1, num_steps=20, cfg_weight=7.5, seed=100)
    for x_t in tqdm(latents, total=20):
        baseline_latents = x_t
    baseline_img = sd.decode(baseline_latents)[0]
    
    # Inject style into output blocks
    sd.inject_prompts({
        'up_1': 'cyberpunk neon glowing vibrant',
        'up_2': 'cyberpunk neon glowing vibrant'
    })
    
    print("\nGenerating with style injection...")
    latents = sd.generate_latents(base_prompt, n_images=1, num_steps=20, cfg_weight=7.5, seed=100)
    for x_t in tqdm(latents, total=20):
        styled_latents = x_t
    styled_img = sd.decode(styled_latents)[0]
    
    sd.clear_injection()
    
    return create_comparison(
        baseline_img, styled_img,
        "TRUE STYLE INJECTION",
        "UP: 'cyberpunk neon' | MID: 'robot' (preserved)"
    )

def test_complete_swap():
    """Test complete subject and style swap."""
    print("\n=== TEST 3: COMPLETE SWAP ===")
    print("Base: 'a medieval castle'")
    print("Injecting completely different prompts:")
    print("  DOWN: 'abstract painting'")
    print("  MID: 'futuristic spaceship'")
    print("  UP: 'neon cyberpunk'\n")
    
    sd = PromptInjectionSD("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    base_prompt = "a medieval castle on a hill"
    
    # Baseline
    print("Generating baseline...")
    latents = sd.generate_latents(base_prompt, n_images=1, num_steps=20, cfg_weight=7.5, seed=200)
    for x_t in tqdm(latents, total=20):
        baseline_latents = x_t
    baseline_img = sd.decode(baseline_latents)[0]
    
    # Inject completely different prompts at different levels
    sd.inject_prompts({
        'down_0': 'abstract painting colorful',
        'down_1': 'abstract painting colorful',
        'mid': 'futuristic spaceship sleek metallic',
        'up_0': 'neon cyberpunk glowing',
        'up_1': 'neon cyberpunk glowing',
        'up_2': 'neon cyberpunk glowing'
    })
    
    print("\nGenerating with complete swap...")
    latents = sd.generate_latents(base_prompt, n_images=1, num_steps=20, cfg_weight=7.5, seed=200)
    for x_t in tqdm(latents, total=20):
        swapped_latents = x_t
    swapped_img = sd.decode(swapped_latents)[0]
    
    sd.clear_injection()
    
    return create_comparison(
        baseline_img, swapped_img,
        "COMPLETE PROMPT SWAP",
        "Castle → Abstract+Spaceship+Cyberpunk"
    )

def main():
    print("\n" + "="*70)
    print("TRUE PROMPT INJECTION - DATAVOID STYLE")
    print("Using Modified UNet with Block-Specific Text Embeddings")
    print("="*70)
    
    save_dir = "/Users/speed/Downloads/corpus-mlx/artifacts/images/readme"
    os.makedirs(save_dir, exist_ok=True)
    
    examples = []
    
    # Test 1: Content swap
    ex1 = test_content_swap()
    ex1.save(f"{save_dir}/UNET_content_swap.png")
    examples.append(ex1)
    print("\n✓ Saved content swap example")
    
    # Test 2: Style injection
    ex2 = test_style_injection()
    ex2.save(f"{save_dir}/UNET_style_injection.png")
    examples.append(ex2)
    print("\n✓ Saved style injection example")
    
    # Test 3: Complete swap
    ex3 = test_complete_swap()
    ex3.save(f"{save_dir}/UNET_complete_swap.png")
    examples.append(ex3)
    print("\n✓ Saved complete swap example")
    
    # Create showcase
    if examples:
        height_total = sum(ex.height for ex in examples) + 100
        width_max = max(ex.width for ex in examples)
        
        showcase = Image.new('RGB', (width_max, height_total), '#0a0a0a')
        draw = ImageDraw.Draw(showcase)
        
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
            subtitle_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            title_font = subtitle_font = None
        
        draw.text((width_max//2, 30), "TRUE PROMPT INJECTION", 
                 fill='#e74c3c', font=title_font, anchor='mm')
        draw.text((width_max//2, 65), "Modified UNet with Block-Specific Embeddings",
                 fill='#95a5a6', font=subtitle_font, anchor='mm')
        
        y = 100
        for ex in examples:
            x = (width_max - ex.width) // 2
            showcase.paste(ex, (x, y))
            y += ex.height + 10
        
        showcase.save(f"{save_dir}/UNET_INJECTION_SHOWCASE.png")
        print("\n✓ Saved complete showcase")
    
    print("\n" + "="*70)
    print("TRUE PROMPT INJECTION COMPLETE!")
    print("Successfully demonstrated DataVoid-style techniques:")
    print("  1. Content swap (cat/dog) with preserved composition")
    print("  2. Style injection with preserved content")
    print("  3. Complete multi-level prompt swapping")
    print("="*70)

if __name__ == "__main__":
    main()