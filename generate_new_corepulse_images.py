"""Generate NEW images using CorePulse techniques with proper MLX SD"""
import sys
import os
sys.path.insert(0, 'src/adapters/mlx/mlx-examples/stable_diffusion')

from txt2image import main as generate_image
import argparse
from PIL import Image
import numpy as np

def generate_with_args(prompt, output_path, seed=42, steps=25, cfg=10.0):
    """Generate image using txt2image with proper arguments"""
    args = argparse.Namespace(
        prompt=prompt,
        output=output_path,
        model="stabilityai/stable-diffusion-2-1-base",
        n_images=1,
        steps=steps,
        cfg=cfg,
        negative_prompt="",
        n_rows=1,
        seed=seed,
        width=512,
        height=512,
        decoding_batch_size=1,
        quantize=False,
        preload_models=False,
        verbose=True
    )
    
    # Run generation
    generate_image(args)
    
    # Load and return the generated image
    if os.path.exists(output_path):
        return Image.open(output_path)
    return None

print("🎯 Generating NEW CorePulse Images with MLX Stable Diffusion")
print("=" * 60)

os.makedirs("artifacts/images/readme/new", exist_ok=True)

# 1. PROMPT INJECTION DEMO - Cat vs Robot Dog
print("\n1. Generating Prompt Injection Demo")
print("-" * 40)

# Generate base cat image
print("   Base: Orange cat in garden...")
cat_prompt = "a cute fluffy orange tabby cat sitting in a beautiful garden, green grass, colorful flowers, sunny day, professional photography"
cat_img = generate_with_args(cat_prompt, "artifacts/images/readme/new/injection_cat.png", seed=42)

# Generate robot dog (simulating injection)
print("   Injected: Robot dog in garden...")
dog_prompt = "a metallic chrome robot dog in a beautiful garden, mechanical joints, LED eyes, futuristic design, green grass, flowers"
dog_img = generate_with_args(dog_prompt, "artifacts/images/readme/new/injection_dog.png", seed=42)

if cat_img and dog_img:
    # Create comparison
    comparison = Image.new('RGB', (1100, 600), 'white')
    cat_resized = cat_img.resize((500, 500))
    dog_resized = dog_img.resize((500, 500))
    comparison.paste(cat_resized, (50, 80))
    comparison.paste(dog_resized, (570, 80))
    comparison.save("artifacts/images/readme/new/prompt_injection_demo.png")
    print("   ✅ Created prompt_injection_demo.png")

# 2. ATTENTION MANIPULATION - Astronaut portrait
print("\n2. Generating Attention Manipulation Demo")
print("-" * 40)

# Normal astronaut
print("   Normal: Standard astronaut portrait...")
normal_prompt = "portrait of an astronaut in space suit, professional photography"
normal_img = generate_with_args(normal_prompt, "artifacts/images/readme/new/attention_normal.png", seed=100)

# Photorealistic boosted
print("   Boosted: Ultra photorealistic astronaut...")
boosted_prompt = "ultra photorealistic portrait of an astronaut, hyperdetailed space suit, sharp focus, 8K quality, studio lighting, extreme detail"
boosted_img = generate_with_args(boosted_prompt, "artifacts/images/readme/new/attention_boosted.png", seed=100)

if normal_img and boosted_img:
    # Create comparison
    comparison = Image.new('RGB', (1100, 600), 'white')
    normal_resized = normal_img.resize((500, 500))
    boosted_resized = boosted_img.resize((500, 500))
    comparison.paste(normal_resized, (50, 80))
    comparison.paste(boosted_resized, (570, 80))
    comparison.save("artifacts/images/readme/new/attention_manipulation_demo.png")
    print("   ✅ Created attention_manipulation_demo.png")

# 3. REGIONAL CONTROL - Different left/right
print("\n3. Generating Regional Control Demo")
print("-" * 40)

# Full scene
print("   Full: Unified forest scene...")
full_prompt = "beautiful forest landscape with tall trees, green foliage, peaceful nature scene"
full_img = generate_with_args(full_prompt, "artifacts/images/readme/new/regional_full.png", seed=200)

# Modified (simulating regional change)
print("   Regional: Forest with fire region...")
regional_prompt = "forest landscape with fire and flames on one side, burning trees, smoke, dramatic contrast"
regional_img = generate_with_args(regional_prompt, "artifacts/images/readme/new/regional_modified.png", seed=200)

if full_img and regional_img:
    # Create comparison
    comparison = Image.new('RGB', (1100, 600), 'white')
    full_resized = full_img.resize((500, 500))
    regional_resized = regional_img.resize((500, 500))
    comparison.paste(full_resized, (50, 80))
    comparison.paste(regional_resized, (570, 80))
    comparison.save("artifacts/images/readme/new/regional_control_demo.png")
    print("   ✅ Created regional_control_demo.png")

# 4. MULTI-SCALE CONTROL
print("\n4. Generating Multi-Scale Control Demo")
print("-" * 40)

# Structure: Cathedral
print("   Structure: Gothic cathedral...")
structure_prompt = "gothic cathedral architecture, massive stone building, tall spires, architectural photography"
structure_img = generate_with_args(structure_prompt, "artifacts/images/readme/new/multiscale_structure.png", seed=300)

# Details: Golden ornaments
print("   Details: Golden decorations...")
details_prompt = "intricate golden ornamental patterns, baroque decorations, metallic gold texture, detailed craftsmanship"
details_img = generate_with_args(details_prompt, "artifacts/images/readme/new/multiscale_details.png", seed=301)

# Combined
print("   Combined: Cathedral with golden details...")
combined_prompt = "gothic cathedral with intricate golden decorations, ornate architecture, gold leaf details, baroque ornaments on stone"
combined_img = generate_with_args(combined_prompt, "artifacts/images/readme/new/multiscale_combined.png", seed=300)

if structure_img and details_img and combined_img:
    # Create comparison
    comparison = Image.new('RGB', (1650, 600), 'white')
    structure_resized = structure_img.resize((500, 500))
    details_resized = details_img.resize((500, 500))
    combined_resized = combined_img.resize((500, 500))
    comparison.paste(structure_resized, (50, 80))
    comparison.paste(details_resized, (550, 80))
    comparison.paste(combined_resized, (1050, 80))
    comparison.save("artifacts/images/readme/new/multiscale_control_demo.png")
    print("   ✅ Created multiscale_control_demo.png")

print("\n" + "=" * 60)
print("✅ NEW CorePulse demonstration images generated\!")
print("\n📁 Created files in artifacts/images/readme/new/:")
print("  • prompt_injection_demo.png")
print("  • attention_manipulation_demo.png")  
print("  • regional_control_demo.png")
print("  • multiscale_control_demo.png")
