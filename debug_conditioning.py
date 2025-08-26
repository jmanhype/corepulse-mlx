#!/usr/bin/env python3
"""Debug text conditioning to understand why prompts aren't working."""

import sys
import os
sys.path.append('/Users/speed/Downloads/corpus-mlx/src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import StableDiffusion
import mlx.core as mx


def debug_text_conditioning():
    """Debug the text conditioning pipeline."""
    print("\n=== DEBUGGING TEXT CONDITIONING ===")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    # Test different prompts and their embeddings
    prompts = [
        "a cute dog",
        "a white fluffy cat", 
        "a red car",
        "an empty room"
    ]
    
    for prompt in prompts:
        print(f"\nTesting prompt: '{prompt}'")
        
        # Get text conditioning
        try:
            conditioning = sd._get_text_conditioning(prompt, n_images=1, cfg_weight=7.5)
            print(f"  Conditioning shape: {conditioning.shape}")
            print(f"  Conditioning dtype: {conditioning.dtype}")
            print(f"  Conditioning mean: {mx.mean(conditioning).item():.6f}")
            print(f"  Conditioning std: {mx.std(conditioning).item():.6f}")
            print(f"  Conditioning range: [{mx.min(conditioning).item():.6f}, {mx.max(conditioning).item():.6f}]")
        except Exception as e:
            print(f"  ERROR getting conditioning: {e}")
    
    # Test tokenization
    print(f"\n=== TOKENIZATION TEST ===")
    test_prompt = "a cute dog"
    try:
        tokens = sd._tokenize(sd.tokenizer, test_prompt)
        print(f"Tokens for '{test_prompt}': {tokens.shape}")
        print(f"Token values: {tokens[0][:20]}...")  # First 20 tokens
    except Exception as e:
        print(f"ERROR tokenizing: {e}")
    
    # Test text encoder directly
    print(f"\n=== TEXT ENCODER TEST ===")
    try:
        tokens = sd._tokenize(sd.tokenizer, test_prompt)
        encoded = sd.text_encoder(tokens)
        print(f"Text encoder output type: {type(encoded)}")
        if hasattr(encoded, 'last_hidden_state'):
            print(f"Hidden state shape: {encoded.last_hidden_state.shape}")
        else:
            print(f"Encoded shape: {encoded.shape}")
    except Exception as e:
        print(f"ERROR with text encoder: {e}")


def debug_model_components():
    """Debug model components."""
    print("\n=== DEBUGGING MODEL COMPONENTS ===")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    print(f"UNet type: {type(sd.unet)}")
    print(f"Text encoder type: {type(sd.text_encoder)}")
    print(f"Tokenizer type: {type(sd.tokenizer)}")
    print(f"Autoencoder type: {type(sd.autoencoder)}")
    print(f"Sampler type: {type(sd.sampler)}")
    print(f"Dtype: {sd.dtype}")
    
    # Check if models are loaded
    try:
        print(f"UNet parameters: {len(list(sd.unet.parameters()))} tensors")
    except:
        print("UNet parameters: ERROR accessing")
    
    try:
        print(f"Text encoder parameters: {len(list(sd.text_encoder.parameters()))} tensors")
    except:
        print("Text encoder parameters: ERROR accessing")


def main():
    print("\n" + "="*60)
    print("DEBUGGING TEXT CONDITIONING PIPELINE")
    print("="*60)
    
    # Debug model components
    debug_model_components()
    
    # Debug text conditioning
    debug_text_conditioning()
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()