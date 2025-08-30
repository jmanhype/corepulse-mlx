#!/usr/bin/env python3
"""Test semantic replacement with CorePulse-like high strength (2.0)"""

import sys
sys.path.append('src')
sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import attn_scores
attn_scores.enable_kv_hooks(True)

from stable_diffusion import StableDiffusionXL
from corpus_mlx import CorePulse

print("="*60)
print("TESTING SEMANTIC REPLACEMENT WITH HIGH STRENGTH")
print("="*60)

# Load model
print("\nLoading SDXL-Turbo...")
model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
corepulse = CorePulse(model)

# Test with very high strength like CorePulse
test_cases = [
    {
        "name": "apple_to_banana",
        "base": "a red apple on a wooden table",
        "inject": "yellow banana fruit",
        "strength": 2.0,  # Like CorePulse!
        "blocks": ["mid"]  # Content block
    },
    {
        "name": "cat_to_dog",
        "base": "a fluffy cat sitting on a couch",
        "inject": "golden retriever dog",
        "strength": 1.8,
        "blocks": ["mid", "up_0"]
    },
    {
        "name": "car_to_bike",
        "base": "a red car on the street",
        "inject": "blue bicycle",
        "strength": 1.5,
        "blocks": ["mid"]
    }
]

for test in test_cases:
    print(f"\n{test['name']}:")
    print(f"  Base: {test['base']}")
    print(f"  Inject: {test['inject']} @ strength {test['strength']}")
    
    corepulse.clear()
    corepulse.add_injection(
        prompt=test['inject'],
        strength=test['strength'],
        blocks=test['blocks'],
        start_step=0,
        end_step=2  # Early injection for SDXL-Turbo
    )
    
    # Generate with 4 steps (SDXL-Turbo optimal)
    latents = model.generate_latents(
        test['base'],
        num_steps=4,
        cfg_weight=0.0,
        seed=42
    )
    
    # Get final and save
    import numpy as np
    from PIL import Image
    import mlx.core as mx
    
    for i, x in enumerate(latents):
        if i == 3:
            decoded = model.decode(x)
            # Convert to numpy and save
            img_array = mx.clip(decoded / 2 + 0.5, 0, 1)
            img_array = (img_array * 255).astype(mx.uint8)
            img_np = np.array(img_array[0])
            if img_np.shape[0] in [3, 4]:
                img_np = np.transpose(img_np, (1, 2, 0))
            img = Image.fromarray(img_np)
            filename = f"semantic_{test['name']}.png"
            img.save(filename)
            print(f"  ✓ Saved: {filename}")

print("\n" + "="*60)
print("CHECK THE IMAGES!")
print("With strength 1.5-2.0 (like original CorePulse),")
print("we should see actual semantic replacement!")
print("="*60)