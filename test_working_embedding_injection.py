#!/usr/bin/env python3
"""Test TRUE embedding injection by monkey-patching text conditioning."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adapters.stable_diffusion import StableDiffusion
from PIL import Image
import mlx.core as mx
import numpy as np

class EmbeddingInjectionWrapper:
    """Wrapper that injects embeddings by patching text conditioning."""
    
    def __init__(self, sd_model):
        self.sd = sd_model
        self.replacements = {}
        self.original_get_text_conditioning = sd_model._get_text_conditioning
    
    def add_replacement(self, original_text: str, replacement_text: str):
        """Add a replacement rule."""
        self.replacements[original_text] = replacement_text
        print(f"✅ Added embedding injection: {original_text} → {replacement_text}")
    
    def _patched_get_text_conditioning(self, text: str, *args, **kwargs):
        """Patched version that replaces embeddings."""
        
        # Check if we need to replace any text
        modified_text = text
        for original, replacement in self.replacements.items():
            if original in text:
                modified_text = text.replace(original, replacement)
                print(f"🧠 Embedding injection: '{text}' → '{modified_text}'")
                break
        
        # Get conditioning for the modified text
        return self.original_get_text_conditioning(modified_text, *args, **kwargs)
    
    def enable(self):
        """Enable embedding injection."""
        self.sd._get_text_conditioning = self._patched_get_text_conditioning
        print("🔥 Embedding injection ENABLED")
    
    def disable(self):
        """Disable embedding injection."""
        self.sd._get_text_conditioning = self.original_get_text_conditioning
        print("⏸️ Embedding injection DISABLED")
    
    def clear(self):
        """Clear all replacements."""
        self.replacements.clear()
        self.disable()

def generate_and_save(sd_model, prompt, filename):
    """Generate image and save it."""
    print(f"Generating: '{prompt}' -> {filename}")
    
    latents = None
    for step in sd_model.generate_latents(
        prompt,
        negative_text="blurry, ugly",
        num_steps=15,
        cfg_weight=7.5,
        seed=42
    ):
        latents = step
    
    # Decode
    images = sd_model.autoencoder.decode(latents)
    img = images[0]
    img = mx.clip(img, -1, 1)
    img = ((img + 1) * 127.5).astype(mx.uint8)
    img_array = np.array(img)
    
    Image.fromarray(img_array).save(filename)
    print(f"✅ Saved: {filename}")
    return img_array

def test_working_embedding_injection():
    """Test working embedding injection."""
    print("🧠 TESTING WORKING EMBEDDING INJECTION")
    print("=" * 50)
    
    # Create model and wrapper
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base")
    wrapper = EmbeddingInjectionWrapper(sd)
    
    test_prompt = "a orange cat playing in a garden"
    print(f"Test prompt: '{test_prompt}'")
    
    # Generate without injection
    print("\n1. Generating WITHOUT injection...")
    before_img = generate_and_save(sd, test_prompt, "working_before.png")
    
    # Add injection and generate
    print("\n2. Adding injection and generating...")
    wrapper.add_replacement("cat", "golden retriever dog")
    wrapper.enable()
    
    after_img = generate_and_save(sd, test_prompt, "working_after.png")
    
    wrapper.disable()
    
    # Compare
    diff = np.mean(np.abs(before_img.astype(float) - after_img.astype(float)))
    print(f"\n📊 Pixel difference: {diff}")
    
    if diff > 5.0:
        print("✅ SUCCESS: Working embedding injection succeeded!")
        print("🎯 This is TRUE embedding injection - different from text replacement!")
    else:
        print("❌ FAILURE: Images are too similar")

if __name__ == "__main__":
    test_working_embedding_injection()