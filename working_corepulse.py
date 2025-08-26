#!/usr/bin/env python3
"""
Working CorePulse implementation with attention manipulation.
"""

import mlx.core as mx
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, Optional

# Add stable diffusion to path
sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import StableDiffusionXL
from stable_diffusion import attn_hooks


class WorkingCoreProcessor:
    """
    Working attention processor that modifies attention outputs.
    """
    
    def __init__(self):
        self.step_count = 0
        self.total_steps = 30
    
    def __call__(self, *, out=None, meta: Dict[str, Any] = None) -> Optional[mx.array]:
        """
        Modify attention output based on step progress.
        """
        if out is None or meta is None:
            return None
        
        block_id = meta.get('block_id', '')
        step_idx = meta.get('step_idx', 0)
        sigma = meta.get('sigma', 1.0)
        
        # Update step count
        if step_idx != self.step_count:
            self.step_count = step_idx
            
        # Calculate progress
        progress = step_idx / self.total_steps if self.total_steps > 0 else 0
        
        # Apply different modifications based on phase
        if progress < 0.3:  # Early phase - amplify structure
            if "down" in block_id:
                # Amplify attention in down blocks for structure
                return out * 1.2
                
        elif progress < 0.7:  # Middle phase - enhance content
            if "mid" in block_id:
                # Enhance middle block attention
                return out * 1.3
                
        else:  # Late phase - stylize
            if "up" in block_id:
                # Modify up blocks for style
                # Add slight noise for artistic effect
                noise = mx.random.normal(out.shape, dtype=out.dtype) * 0.05
                return out + noise
        
        return None  # Keep original


def run_working_corepulse():
    """
    Run working CorePulse demonstration.
    """
    print("\n" + "="*80)
    print("🎯 WORKING COREPULSE IMPLEMENTATION")
    print("="*80)
    
    # Load model
    print("\n📦 Loading SDXL...")
    sd = StableDiffusionXL("stabilityai/sdxl-turbo")
    print("✅ Model loaded")
    
    # Test parameters
    prompt = "a majestic castle on a mountain"
    seed = 42
    steps = 10  # More steps to see effect
    
    # 1. Generate baseline
    print("\n" + "-"*60)
    print("1️⃣ BASELINE (no hooks)")
    print("-"*60)
    print(f"Prompt: {prompt}")
    print(f"Steps: {steps}")
    
    for latents_base in sd.generate_latents(
        prompt, 
        n_images=1,
        num_steps=steps,
        seed=seed
    ):
        pass
    
    decoded_base = sd.decode(latents_base)
    img_base = np.array(decoded_base[0])
    img_base = (img_base * 255).astype(np.uint8)
    
    from PIL import Image
    base_path = Path("artifacts/images/readme/WORKING_baseline.png")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_base).save(base_path)
    print(f"💾 Saved: {base_path}")
    
    # 2. Generate with CorePulse
    print("\n" + "-"*60)
    print("2️⃣ WITH COREPULSE")
    print("-"*60)
    
    # Enable hooks
    attn_hooks.ATTN_HOOKS_ENABLED = True
    print("✅ Hooks enabled")
    
    # Create and register processor
    processor = WorkingCoreProcessor()
    processor.total_steps = steps
    
    # Register for all blocks
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_hooks.register_processor(block, processor)
    
    print("✅ Processor registered")
    print("\n🚀 Generating with CorePulse effects...")
    print("  Phase 1 (0-30%): Structure amplification in down blocks")
    print("  Phase 2 (30-70%): Content enhancement in mid blocks")
    print("  Phase 3 (70-100%): Style modification in up blocks")
    
    for latents_core in sd.generate_latents(
        prompt,
        n_images=1, 
        num_steps=steps,
        seed=seed
    ):
        pass
    
    decoded_core = sd.decode(latents_core)
    img_core = np.array(decoded_core[0])
    img_core = (img_core * 255).astype(np.uint8)
    
    core_path = Path("artifacts/images/readme/WORKING_corepulse.png")
    Image.fromarray(img_core).save(core_path)
    print(f"\n💾 Saved: {core_path}")
    
    # Calculate difference
    diff = np.abs(img_base.astype(float) - img_core.astype(float)).mean()
    print(f"\n📊 Average pixel difference: {diff:.2f}")
    
    if diff > 1.0:
        print("✅ CorePulse effects applied successfully!")
    else:
        print("⚠️ Minimal difference detected")
    
    # Clean up
    attn_hooks.ATTN_HOOKS_ENABLED = False
    attn_hooks.attention_registry.clear()
    
    print("\n" + "="*80)
    print("🎉 COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    run_working_corepulse()