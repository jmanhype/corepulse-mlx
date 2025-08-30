#!/usr/bin/env python3
"""Debug what type of attention layers are being used."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

sys.path.insert(0, str(Path(__file__).parent / "src" / "adapters" / "mlx" / "mlx-examples" / "stable_diffusion"))
from stable_diffusion import attn_scores

def debug_attention_layers():
    """Check what attention layers are used."""
    print("🔍 DEBUGGING ATTENTION LAYER TYPES")
    print("=" * 50)
    
    # Enable hooks first
    print("Enabling KV hooks...")
    attn_scores.enable_kv_hooks(True)
    print(f"KV_HOOKS_ENABLED: {attn_scores.KV_HOOKS_ENABLED}")
    print(f"hooks_wanted(): {attn_scores.hooks_wanted()}")
    
    # Import and create model
    print("\nCreating Stable Diffusion model...")
    from adapters.stable_diffusion import StableDiffusion
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base")
    
    print("\nChecking UNet attention layers...")
    
    # Check mid block
    if hasattr(sd.unet, 'mid_blocks') and len(sd.unet.mid_blocks) > 1:
        mid_block = sd.unet.mid_blocks[1]
        if hasattr(mid_block, 'transformer_blocks') and len(mid_block.transformer_blocks) > 0:
            first_transformer = mid_block.transformer_blocks[0]
            if hasattr(first_transformer, 'attn1'):
                print(f"mid attn1 type: {type(first_transformer.attn1)}")
            if hasattr(first_transformer, 'attn2'):
                print(f"mid attn2 type: {type(first_transformer.attn2)}")
    
    # Check up blocks
    if hasattr(sd.unet, 'up_blocks'):
        for i, up_block in enumerate(sd.unet.up_blocks):
            if hasattr(up_block, 'attentions') and len(up_block.attentions) > 0:
                first_attention = up_block.attentions[0]
                if hasattr(first_attention, 'transformer_blocks') and len(first_attention.transformer_blocks) > 0:
                    first_transformer = first_attention.transformer_blocks[0]
                    if hasattr(first_transformer, 'attn1'):
                        print(f"up_{i} attn1 type: {type(first_transformer.attn1)}")
                    if hasattr(first_transformer, 'attn2'):
                        print(f"up_{i} attn2 type: {type(first_transformer.attn2)}")

if __name__ == "__main__":
    debug_attention_layers()