#!/usr/bin/env python3
"""
Prompt weighting demo (Automatic1111-style) for CorePulse.
"""

import sys
sys.path.append('..')

from corpus_mlx.corepulse import CorePulse
from corpus_mlx.prompt_weights import PromptWeightParser
from stable_diffusion import StableDiffusion


def main():
    """Run prompt weighting demo."""
    
    # Load model
    print("Loading model...")
    base_model = StableDiffusion.from_pretrained(
        "stabilityai/sdxl-turbo",
        low_memory=True
    )
    
    # Create CorePulse instance
    corepulse = CorePulse(base_model)
    
    # Example weighted prompts (A1111 syntax)
    weighted_prompts = [
        "a (beautiful:1.5) sunset over the (ocean:1.2)",
        "a [dark:bright:0.5] forest with (glowing:1.8) mushrooms",
        "((masterpiece)), best quality, (detailed:1.3) artwork",
        "(red:1.5) roses and (blue:0.5) violets in a garden"
    ]
    
    for i, prompt in enumerate(weighted_prompts):
        print(f"\nTesting prompt {i+1}: {prompt}")
        
        # Parse weighted prompt
        parser = PromptWeightParser()
        tokens, weights = parser.parse(prompt)
        
        print(f"Parsed tokens: {tokens}")
        print(f"Weights: {weights}")
        
        # Apply weighted injections
        for token, weight in zip(tokens, weights):
            if weight != 1.0:  # Only inject if weight differs from default
                strength = abs(weight - 1.0) * 0.2  # Convert weight to strength
                corepulse.add_injection(
                    prompt=token,
                    strength=min(strength, 0.5),  # Cap at 0.5 for safety
                    blocks=["mid", "up_0"] if weight > 1.0 else ["down_1", "down_2"]
                )
        
        # Generate with weighted prompt
        print(f"Generating with weighted prompt...")
        image = corepulse.generate(
            prompt=" ".join(tokens),  # Use unweighted version as base
            negative_prompt="",
            num_inference_steps=4,
            guidance_scale=0.0,
            seed=42 + i,
            width=1024,
            height=1024
        )
        
        if image:
            image.save(f"demo_weighted_output_{i+1}.png")
            print(f"Saved to demo_weighted_output_{i+1}.png")
        
        # Clear for next prompt
        corepulse.clear()
    
    print("\nPrompt weighting demo complete!")


if __name__ == "__main__":
    main()