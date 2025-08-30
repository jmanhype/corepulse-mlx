#!/usr/bin/env python3
"""
Example 2: Generate Comparison Images
Shows baseline vs semantic replacement side by side.
"""

from corpus_mlx import create_semantic_wrapper
from PIL import Image

# Create wrapper
wrapper = create_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")

# Test prompt
prompt = "a photo of a cat sitting on a sofa"

# Generate baseline and replacement in one call
print("Generating comparison: cat → dog")
baseline, replaced = wrapper.generate_comparison(
    prompt=prompt,
    original_obj="cat",
    replacement_obj="dog",
    negative_text="blurry, ugly",
    num_steps=20,
    cfg_weight=7.5,
    seed=42,
    height=256,
    width=256
)

# Save both images
Image.fromarray(baseline).save("example_cat_baseline.png")
Image.fromarray(replaced).save("example_cat_to_dog.png")

print("✅ Saved:")
print("  - example_cat_baseline.png (shows cat)")
print("  - example_cat_to_dog.png (shows dog)")

# Create side-by-side comparison
comparison = Image.new('RGB', (512, 256))
comparison.paste(Image.fromarray(baseline), (0, 0))
comparison.paste(Image.fromarray(replaced), (256, 0))
comparison.save("example_comparison.png")

print("  - example_comparison.png (side-by-side)")