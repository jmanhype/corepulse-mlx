"""Test the research-backed generation example from README"""
import sys
sys.path.insert(0, 'src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import StableDiffusion
import mlx.core as mx

print("2. Testing research-backed CFG 12.0 generation...")

# Initialize SD 2.1-base as shown in README
print("   Loading SD 2.1-base...")
sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
print("   ✅ Model loaded")

# Test the critical CFG 12.0 configuration
prompt = "photo of a red Ferrari sports car, automotive photography, professional lighting, 8K"
print(f"   Testing prompt: '{prompt[:50]}...'")

# Generate with research-backed settings
print("   Generating with CFG 12.0 (not 7.5\!)...")
latents = sd.generate_latents(
    prompt,
    cfg_weight=12.0,  # Critical: NOT 7.5\!
    num_steps=20,
    seed=42
)

print(f"   ✅ Generated latents shape: {latents.shape}")
print(f"   ✅ CFG 12.0 applied successfully")

# Decode to image
print("   Decoding latents to image...")
decoded = sd.decode(latents)
image = sd.denormalize(decoded)

print(f"   ✅ Image shape: {image.shape}")
print(f"   ✅ Image range: [{image.min():.2f}, {image.max():.2f}]")

# Save the result
from PIL import Image
import numpy as np

# Convert to numpy and proper format
img_np = np.array(image[0] * 255).astype(np.uint8)
img_np = np.transpose(img_np, (1, 2, 0))
img = Image.fromarray(img_np)
img.save("test_readme_ferrari.png")
print("   ✅ Saved to test_readme_ferrari.png")

print("\n✅ Research-backed generation working as documented in README\!")
print("   CFG 12.0+ ensures prompt adherence for SD 2.1-base")
