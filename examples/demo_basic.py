#!/usr/bin/env python3
"""
Basic CorePulse demo showing simple prompt injection.
"""

import sys
sys.path.append('..')

from corpus_mlx import CorePulseStableDiffusion, InjectionConfig
import sys
import os
# Add parent directory to path to import stable_diffusion
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'adapters'))
from stable_diffusion import StableDiffusion


def main():
    """Run basic injection demo."""
    
    # Load base model (using default model)
    print("Loading model...")
    base_model = StableDiffusion(
        model="stabilityai/sdxl-turbo",
        float16=True
    )
    
    # Create CorePulse wrapper
    model = CorePulseStableDiffusion(base_model)
    
    # Set up basic injection
    model.add_injection(
        prompt="ethereal magical aurora",
        strength=0.3,
        blocks=["mid", "up_0", "up_1"]
    )
    
    # Generate with injection
    print("Generating with injection...")
    image = model.generate(
        prompt="majestic lion in african savanna, sunset",
        negative_prompt="",
        num_inference_steps=4,
        guidance_scale=0.0,  # SDXL-Turbo doesn't use CFG
        seed=42
    )
    
    # Save result
    if image:
        image.save("demo_basic_output.png")
        print("Image saved to demo_basic_output.png")
    
    # Clear and generate baseline for comparison
    model.clear_injections()
    
    print("Generating baseline...")
    baseline = model.generate(
        prompt="majestic lion in african savanna, sunset",
        negative_prompt="",
        num_inference_steps=4,
        guidance_scale=0.0,
        seed=42
    )
    
    if baseline:
        baseline.save("demo_basic_baseline.png")
        print("Baseline saved to demo_basic_baseline.png")
    
    print("Demo complete!")


if __name__ == "__main__":
    main()