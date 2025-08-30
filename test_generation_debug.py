#!/usr/bin/env python3
"""
Debug why generated images are blank.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
import numpy as np
from PIL import Image


def test_generation_debug():
    """Debug the generation pipeline."""
    print("Generation Pipeline Debug")
    print("=" * 50)
    
    # Test 1: Basic MLX SD generation
    print("\n1. Testing basic MLX StableDiffusion generation:")
    from adapters.stable_diffusion import StableDiffusion
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base")
    
    # Generate simple latents
    print("   Generating latents...")
    sampler = sd.sampler
    latents = sampler.sample_prior((1, 32, 32, 4), dtype=mx.float32)
    print(f"   Latents shape: {latents.shape}, dtype: {latents.dtype}")
    print(f"   Latents range: [{float(latents.min()):.3f}, {float(latents.max()):.3f}]")
    
    # Get text conditioning
    print("\n   Getting text conditioning...")
    tokens = sd._tokenize(sd.tokenizer, "a red apple", None)
    print(f"   Tokens shape: {tokens.shape}")
    
    conditioning = sd.text_encoder(tokens).last_hidden_state
    print(f"   Conditioning shape: {conditioning.shape}")
    print(f"   Conditioning range: [{float(conditioning.min()):.3f}, {float(conditioning.max()):.3f}]")
    
    # Try a single denoising step
    print("\n   Testing single denoising step...")
    t = mx.array([999.0])
    t_prev = mx.array([900.0])
    
    x_t = latents
    x_t_unet = x_t
    t_unet = mx.broadcast_to(t, [1])
    
    print(f"   Running UNet...")
    eps_pred = sd.unet(x_t_unet, t_unet, encoder_x=conditioning)
    print(f"   UNet output shape: {eps_pred.shape}")
    print(f"   UNet output range: [{float(eps_pred.min()):.3f}, {float(eps_pred.max()):.3f}]")
    
    # Try decoding random latents
    print("\n2. Testing VAE decoder:")
    test_latents = mx.random.normal((1, 32, 32, 4)) * 0.5
    print(f"   Test latents shape: {test_latents.shape}")
    print(f"   Test latents range: [{float(test_latents.min()):.3f}, {float(test_latents.max()):.3f}]")
    
    decoded = sd.autoencoder.decode(test_latents)
    print(f"   Decoded shape: {decoded.shape}")
    print(f"   Decoded range: [{float(decoded.min()):.3f}, {float(decoded.max()):.3f}]")
    
    # Convert to image
    img = decoded[0]  # Get first image
    img = (img * 127.5 + 127.5).astype(mx.uint8)
    img = np.array(img)
    print(f"   Image shape: {img.shape}")
    print(f"   Image range: [{img.min()}, {img.max()}]")
    
    # Check if image is all white
    if np.all(img > 250):
        print("   ❌ Image is all white!")
    elif np.all(img < 5):
        print("   ❌ Image is all black!")
    else:
        print("   ✅ Image has varied pixels")
    
    # Save test image
    Image.fromarray(img).save("debug_test_decode.png")
    print("   Saved: debug_test_decode.png")
    
    # Test 3: Full generation with basic loop
    print("\n3. Testing minimal generation loop:")
    
    # Simple generation
    latents = sampler.sample_prior((1, 32, 32, 4), dtype=mx.float32)
    
    # Just do 2 steps
    for i in range(2):
        print(f"   Step {i+1}")
        t = mx.array([999.0 - i*100])
        t_prev = mx.array([999.0 - (i+1)*100])
        
        # Simple denoising
        noise_pred = sd.unet(latents, mx.broadcast_to(t, [1]), encoder_x=conditioning)
        
        # Update latents (simplified)
        latents = latents - 0.1 * noise_pred
        
        print(f"      Latents range: [{float(latents.min()):.3f}, {float(latents.max()):.3f}]")
    
    # Decode final
    final = sd.autoencoder.decode(latents)
    img = final[0]
    img = (img * 127.5 + 127.5).astype(mx.uint8)
    img = np.array(img)
    
    Image.fromarray(img).save("debug_minimal_generation.png")
    print(f"\n   Saved: debug_minimal_generation.png")
    print(f"   Final image range: [{img.min()}, {img.max()}]")
    
    print("\n" + "=" * 50)
    print("Debug Complete")
    print("\nCheck the generated images to see if they contain anything")


if __name__ == "__main__":
    test_generation_debug()