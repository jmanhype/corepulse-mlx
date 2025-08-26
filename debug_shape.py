#!/usr/bin/env python3
"""Debug shape issues with image generation."""

import mlx.core as mx
import numpy as np
import sys
sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')
from stable_diffusion import StableDiffusionXL

# Load model
model = "stabilityai/sdxl-turbo"
sd = StableDiffusionXL(model)

# Generate test image
for latents in sd.generate_latents(
    "a cute dog",
    n_images=1,
    num_steps=2,
    seed=42,
):
    pass

print(f"Latents shape: {latents.shape}")

decoded = sd.decode(latents)
print(f"Decoded shape: {decoded.shape}")

# Convert to numpy
image = np.array(decoded[0])
print(f"Image numpy shape: {image.shape}")

# Check if we need to transpose
if image.shape[0] < image.shape[-1]:  # Channels first
    image = image.transpose(1, 2, 0)
    print(f"Transposed shape: {image.shape}")

# Ensure it's 3-channel RGB
if image.shape[-1] == 1:
    image = np.repeat(image, 3, axis=-1)
elif image.shape[-1] == 4:
    image = image[:, :, :3]
    
print(f"Final shape: {image.shape}")

# Convert to uint8
image = (image * 255).astype(np.uint8)
print(f"Final dtype: {image.dtype}")

# Save 
from PIL import Image
img = Image.fromarray(image)
img.save("test_debug.png")
print("Saved test_debug.png")