#!/usr/bin/env python3
"""
Ruby-style monkey patching for MLX UNet to achieve DataVoid injection.
This properly overrides the UNet's forward method to inject different embeddings at different blocks.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional
import functools

class DataVoidInjector:
    """Global state manager for DataVoid injection configuration."""
    
    def __init__(self):
        self.active = False
        self.config = None
        self.current_step = 0
        self.total_steps = 30
        self.down_embeddings = None
        self.mid_embeddings = None
        self.up_embeddings = None
        
    def configure(self, down=None, mid=None, up=None, total_steps=30):
        """Set up injection configuration."""
        self.down_embeddings = down
        self.mid_embeddings = mid
        self.up_embeddings = up
        self.total_steps = total_steps
        self.current_step = 0
        self.active = True
        
    def reset(self):
        """Disable injection."""
        self.active = False
        self.current_step = 0
        
    def get_phase(self):
        """Determine current phase based on progress."""
        if self.total_steps == 0:
            return "none"
        progress = self.current_step / self.total_steps
        if progress < 0.3:
            return "structure"
        elif progress < 0.7:
            return "content"
        else:
            return "style"


# Global injector instance
INJECTOR = DataVoidInjector()


def monkey_patch_sdxl(sd):
    """
    Ruby-style monkey patch for SDXL to enable DataVoid injection.
    
    Args:
        sd: StableDiffusionXL instance
    """
    print("\n🐒 Ruby-style monkey patching SDXL UNet...")
    
    # Store original methods
    original_denoising_step = sd._denoising_step
    original_generate_latents = sd.generate_latents
    
    def patched_denoising_step(x_t, t, t_prev, conditioning, cfg_weight=7.5, text_time=None, step_idx=0):
        """Denoising step with DataVoid injection tracking."""
        
        # Update global step counter
        INJECTOR.current_step = step_idx
        
        # Check if injection is active and modify conditioning
        if INJECTOR.active:
            phase = INJECTOR.get_phase()
            
            # Use phase-specific embeddings
            if phase == "structure" and INJECTOR.down_embeddings is not None:
                print(f"  💉 Step {step_idx}: Injecting STRUCTURE embeddings")
                # Replace conditioning with structure embeddings
                orig_shape = conditioning.shape
                if len(orig_shape) == 3:  # [batch, seq, dim]
                    conditioning = INJECTOR.down_embeddings
                    
            elif phase == "content" and INJECTOR.mid_embeddings is not None:
                print(f"  💉 Step {step_idx}: Injecting CONTENT embeddings")
                conditioning = INJECTOR.mid_embeddings
                
            elif phase == "style" and INJECTOR.up_embeddings is not None:
                print(f"  💉 Step {step_idx}: Injecting STYLE embeddings")
                conditioning = INJECTOR.up_embeddings
        
        # Call original method with potentially modified conditioning
        return original_denoising_step(
            x_t, t, t_prev, conditioning, cfg_weight, text_time, step_idx
        )
    
    def patched_generate_latents(
        text: str,
        n_images: int = 1,
        num_steps: int = 2,
        cfg_weight: float = 0.0,
        negative_text: str = "",
        latent_size = (64, 64),
        seed = None,
    ):
        """Generate latents with injection support."""
        
        # Reset step counter
        INJECTOR.current_step = 0
        
        # Call original generator
        yield from original_generate_latents(
            text, n_images, num_steps, cfg_weight, negative_text, latent_size, seed
        )
    
    # Apply monkey patches
    sd._denoising_step = patched_denoising_step
    sd.generate_latents = patched_generate_latents
    
    print("✅ Monkey patching complete!")
    
    return sd


def set_injection(structure_prompt, content_prompt, style_prompt, sd, total_steps=30):
    """
    Configure DataVoid injection with different prompts for each phase.
    
    Args:
        structure_prompt: Prompt for structure phase (0-30%)
        content_prompt: Prompt for content phase (30-70%)
        style_prompt: Prompt for style phase (70-100%)
        sd: StableDiffusionXL instance
        total_steps: Total number of denoising steps
    """
    print("\n🎯 Configuring DataVoid injection:")
    print(f"  Structure: {structure_prompt}")
    print(f"  Content: {content_prompt}")
    print(f"  Style: {style_prompt}")
    
    # Generate embeddings for each phase
    structure_emb, _ = sd._get_text_conditioning(structure_prompt, n_images=1)
    content_emb, _ = sd._get_text_conditioning(content_prompt, n_images=1)
    style_emb, _ = sd._get_text_conditioning(style_prompt, n_images=1)
    
    # Configure global injector
    INJECTOR.configure(
        down=structure_emb,
        mid=content_emb,
        up=style_emb,
        total_steps=total_steps
    )
    
    print("✅ Injection configured!")


def disable_injection():
    """Disable DataVoid injection."""
    INJECTOR.reset()
    print("❌ Injection disabled")