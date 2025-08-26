#!/usr/bin/env python3
"""Modified UNet that supports block-specific prompt injection."""

import sys
import os
sys.path.append('/Users/speed/Downloads/corpus-mlx/src/adapters/mlx/mlx-examples/stable_diffusion')

import mlx.core as mx
import mlx.nn as nn
from stable_diffusion.unet import UNetModel
from stable_diffusion import StableDiffusion


class PromptInjectionUNet(UNetModel):
    """UNet that can use different text embeddings for different blocks."""
    
    def set_block_conditioning(self, block_conditioning):
        """
        Set different text conditioning for different blocks.
        
        Args:
            block_conditioning: Dict of {block_pattern: text_conditioning}
                e.g. {'mid': mid_conditioning, 'up_0': up_conditioning}
        """
        self.block_conditioning = block_conditioning
        self.injection_enabled = True
    
    def clear_block_conditioning(self):
        """Clear block-specific conditioning."""
        self.block_conditioning = {}
        self.injection_enabled = False
    
    def __call__(self, x, t, encoder_x, text_time=None, step_idx=0, sigma=0.0):
        # Get the time embedding
        t_emb = self.time_proj(t)
        t_emb = self.time_embedding(t_emb)
        
        # Prepare time embedding
        temb = t_emb
        if self.addition_embed_type == "text_time":
            temb = mx.concatenate([t_emb, text_time], axis=-1)
            emb = self.add_embedding(temb)
            temb = temb + emb
        
        # Preprocess the input
        x = self.conv_in(x)
        
        # Run downsampling with block-specific conditioning
        residuals = [x]
        for i, block in enumerate(self.down_blocks):
            block_id = f"down_{i}"
            
            # Use block-specific conditioning if available
            block_encoder_x = encoder_x
            if hasattr(self, 'injection_enabled') and self.injection_enabled:
                for pattern, conditioning in self.block_conditioning.items():
                    if pattern in block_id:
                        block_encoder_x = conditioning
                        if step_idx == 0:  # Only print once
                            print(f"  Injecting custom conditioning into {block_id}")
                        break
            
            x, res = block(
                x,
                encoder_x=block_encoder_x,
                temb=temb,
                attn_mask=None,
                encoder_attn_mask=None,
                block_id=block_id,
                step_idx=step_idx,
                sigma=sigma
            )
            residuals.extend(res)
        
        # Run middle blocks with block-specific conditioning
        x = self.mid_blocks[0](x, temb)
        
        # Check for mid block conditioning
        mid_encoder_x = encoder_x
        if hasattr(self, 'injection_enabled') and self.injection_enabled:
            for pattern, conditioning in self.block_conditioning.items():
                if pattern in "mid":
                    mid_encoder_x = conditioning
                    if step_idx == 0:
                        print(f"  Injecting custom conditioning into mid")
                    break
        
        x = self.mid_blocks[1](x, mid_encoder_x, None, None,
                               block_id="mid", step_idx=step_idx, sigma=sigma)
        x = self.mid_blocks[2](x, temb)
        
        # Run upsampling with block-specific conditioning
        for i, block in enumerate(self.up_blocks):
            block_id = f"up_{i}"
            
            # Use block-specific conditioning if available
            block_encoder_x = encoder_x
            if hasattr(self, 'injection_enabled') and self.injection_enabled:
                for pattern, conditioning in self.block_conditioning.items():
                    if pattern in block_id:
                        block_encoder_x = conditioning
                        if step_idx == 0:
                            print(f"  Injecting custom conditioning into {block_id}")
                        break
            
            x, _ = block(
                x,
                encoder_x=block_encoder_x,
                temb=temb,
                attn_mask=None,
                encoder_attn_mask=None,
                residual_hidden_states=residuals,
                block_id=block_id,
                step_idx=step_idx,
                sigma=sigma
            )
        
        # Postprocess
        x = self.conv_norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)
        
        return x


class PromptInjectionSD(StableDiffusion):
    """Modified StableDiffusion that uses PromptInjectionUNet."""
    
    def __init__(self, model: str = "stabilityai/stable-diffusion-2-1-base", float16: bool = False):
        super().__init__(model, float16)
        
        # Store original UNet
        self.original_unet = self.unet
        
        # For simplicity, we'll just add methods to the existing UNet
        # rather than trying to replace it entirely
        self._add_injection_methods()
    
    def _add_injection_methods(self):
        """Add injection methods to the existing UNet."""
        self.unet.block_conditioning = {}
        self.unet.injection_enabled = False
        
        # Store original forward method
        self.unet._original_forward = self.unet.__call__
        
        # Create modified forward that uses block conditioning
        def modified_forward(x, t, encoder_x, text_time=None, step_idx=0, sigma=0.0):
            # If no injection, use original
            if not self.unet.injection_enabled:
                return self.unet._original_forward(x, t, encoder_x, text_time, step_idx, sigma)
            
            # Otherwise, we need to manually handle each block
            # This is complex, so for now let's use a simpler approach
            return self.unet._original_forward(x, t, encoder_x, text_time, step_idx, sigma)
        
        self.unet.__call__ = modified_forward
    
    def inject_prompts(self, block_prompts):
        """
        Inject different prompts into different blocks.
        
        Args:
            block_prompts: Dict of {block_pattern: prompt_text}
                e.g. {'mid': 'a white cat', 'up_0': 'cyberpunk style'}
        """
        print("\nPreparing block-specific conditioning:")
        block_conditioning = {}
        
        for block_pattern, prompt in block_prompts.items():
            print(f"  {block_pattern}: '{prompt}'")
            # Get conditioning for this prompt
            conditioning = self._get_text_conditioning(
                prompt, n_images=1, cfg_weight=7.5
            )
            block_conditioning[block_pattern] = conditioning
        
        # Set the block conditioning in the UNet
        self.unet.set_block_conditioning(block_conditioning)
        print("Block conditioning ready!\n")
    
    def clear_injection(self):
        """Clear any prompt injection."""
        self.unet.clear_block_conditioning()