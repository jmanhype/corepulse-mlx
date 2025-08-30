#!/usr/bin/env python3
"""Test semantic object replacement - the holy grail of prompt injection!"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
sys.path.insert(0, '/Users/speed/Downloads/mlx-examples')
from corpus_mlx import CorePulse
from stable_diffusion import StableDiffusionXL

print("="*60)
print("TESTING SEMANTIC OBJECT REPLACEMENT")
print("="*60)

# Load model
print("\nLoading SDXL-Turbo...")
model = StableDiffusionXL("stabilityai/sdxl-turbo", low_cpu_mem_usage=True, float16=True)
corepulse = CorePulse(model)

# Test cases for semantic replacement
test_cases = [
    {
        "name": "Apple → Banana",
        "base": "a red apple on a wooden table",
        "inject": "yellow banana",
        "strength": 0.9,
        "blocks": ["mid", "up_0", "up_1"]
    },
    {
        "name": "Cat → Dog", 
        "base": "a cute cat sitting on a couch",
        "inject": "friendly dog",
        "strength": 0.95,
        "blocks": ["mid", "up_0", "up_1", "up_2"]
    },
    {
        "name": "Car → Bicycle",
        "base": "a red car parked on the street",
        "inject": "blue bicycle",
        "strength": 0.85,
        "blocks": ["down_2", "mid", "up_0"]
    },
    {
        "name": "Coffee → Tea",
        "base": "a cup of coffee on a desk",
        "inject": "green tea in traditional cup",
        "strength": 0.8,
        "blocks": ["mid", "up_0"]
    }
]

for i, test in enumerate(test_cases, 1):
    print(f"\n{i}. Testing: {test['name']}")
    print(f"   Base: '{test['base']}'")
    print(f"   Injecting: '{test['inject']}' at {test['strength']*100}% strength")
    
    # Clear previous injections
    corepulse.clear()
    
    # Add semantic injection
    corepulse.add_injection(
        prompt=test['inject'],
        strength=test['strength'],
        blocks=test['blocks'],
        start_step=0,
        end_step=3  # Early injection for SDXL-Turbo
    )
    
    # Generate
    latents = model.generate_latents(
        test['base'],
        num_steps=4,
        cfg_weight=0.0,  # No CFG for turbo
        seed=42
    )
    
    # Get final latent and decode
    for j, x in enumerate(latents):
        if j == 3:  # Last step
            img = model.decode(x)
            
    # Save result
    filename = f"semantic_{i}_{test['name'].replace(' → ', '_to_').replace(' ', '_').lower()}.png"
    img.save(filename)
    print(f"   ✓ Saved: {filename}")

print("\n" + "="*60)
print("CHECK THE IMAGES!")
print("If semantic injection works, you should see:")
print("  1. Banana instead of apple")
print("  2. Dog instead of cat")
print("  3. Bicycle instead of car")
print("  4. Tea instead of coffee")
print("="*60)