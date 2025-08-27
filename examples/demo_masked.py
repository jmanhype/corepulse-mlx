#!/usr/bin/env python3
"""
Masked/Regional control demo for CorePulse.
"""

import sys
sys.path.append('..')

from corpus_mlx.corepulse import CorePulse
from corpus_mlx.types import RegionSpec
from stable_diffusion import StableDiffusion


def main():
    """Run regional control demo."""
    
    # Load model
    print("Loading model...")
    base_model = StableDiffusion.from_pretrained(
        "stabilityai/sdxl-turbo",
        low_memory=True
    )
    
    # Create CorePulse instance
    corepulse = CorePulse(base_model)
    
    # Define regions with different prompts
    # Left half - sunset colors
    corepulse.add_regional_prompt(
        prompt="warm golden sunset, orange and purple sky",
        region=(0, 0, 512, 1024),  # x1, y1, x2, y2
        strength=0.4
    )
    
    # Right half - night sky
    corepulse.add_regional_prompt(
        prompt="starry night sky, deep blue cosmos, twinkling stars",
        region=(512, 0, 1024, 1024),
        strength=0.4
    )
    
    # Top third - clouds
    corepulse.add_regional_prompt(
        prompt="dramatic storm clouds, lightning",
        region=(0, 0, 1024, 340),
        strength=0.3
    )
    
    # Generate with regional control
    print("Generating with regional control...")
    image = corepulse.generate(
        prompt="epic landscape with mountains and lake",
        negative_prompt="",
        num_inference_steps=4,
        guidance_scale=0.0,
        seed=42,
        width=1024,
        height=1024
    )
    
    if image:
        image.save("demo_masked_output.png")
        print("Regional control image saved to demo_masked_output.png")
    
    # Generate baseline without regions
    corepulse.clear()
    
    print("Generating baseline...")
    baseline = corepulse.generate(
        prompt="epic landscape with mountains and lake",
        negative_prompt="",
        num_inference_steps=4,
        guidance_scale=0.0,
        seed=42,
        width=1024,
        height=1024
    )
    
    if baseline:
        baseline.save("demo_masked_baseline.png")
        print("Baseline saved to demo_masked_baseline.png")
    
    print("Regional control demo complete!")


if __name__ == "__main__":
    main()