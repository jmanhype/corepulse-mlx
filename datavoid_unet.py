#!/usr/bin/env python3
"""
DataVoid-style UNet with block-specific conditioning injection.
Modifies MLX UNet to swap encoder_x at different blocks for phased injection.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class InjectionConfig:
    """Configuration for block-specific injection."""
    down_blocks: Optional[mx.array] = None  # Structure phase
    mid_blocks: Optional[mx.array] = None   # Content phase  
    up_blocks: Optional[mx.array] = None    # Style phase
    
class DataVoidUNet:
    """
    Wraps MLX UNet to inject different text embeddings at different blocks.
    Implements DataVoid's phased injection strategy.
    """
    
    def __init__(self, original_unet):
        self.unet = original_unet
        self.injection_config = None
        self.current_step = 0
        self.total_steps = 30
        
        # Store original forward method
        self._original_forward = original_unet.__class__.__call__
        
        # Replace the __call__ method directly on the instance
        original_unet.__class__.__call__ = self._injected_forward
        
    def set_injection(self, config: InjectionConfig, total_steps: int = 30):
        """Configure injection for different UNet blocks."""
        self.injection_config = config
        self.total_steps = total_steps
        self.current_step = 0
        
    def _injected_forward(
        self,
        x,
        timestep,
        encoder_x,
        attn_mask=None,
        encoder_attn_mask=None,
        text_time=None,
        step_idx=0,
        sigma=0.0,
    ):
        """Forward with block-specific encoder_x injection."""
        
        # Track current step
        self.current_step = step_idx
        
        # Store original encoder_x
        original_encoder = encoder_x
        
        # Calculate progress for phased injection
        progress = self.current_step / self.total_steps if self.total_steps > 0 else 0
        
        # Compute time embeddings (unchanged)
        temb = self.unet.timesteps(timestep).astype(x.dtype)
        temb = self.unet.time_embedding(temb)
        
        # Add extra text_time conditioning if provided
        if text_time is not None:
            text_emb, time_ids = text_time
            emb = self.unet.add_time_proj(time_ids).flatten(1).astype(x.dtype)
            emb = mx.concatenate([text_emb, emb], axis=-1)
            emb = self.unet.add_embedding(emb)
            temb = temb + emb
            
        # Preprocess input
        x = self.unet.conv_in(x)
        
        # DOWN BLOCKS - Structure phase (0-30%)
        residuals = [x]
        for i, block in enumerate(self.unet.down_blocks):
            # Use structure injection in early steps
            if self.injection_config and progress < 0.3 and self.injection_config.down_blocks is not None:
                block_encoder = self.injection_config.down_blocks
                print(f"  💉 DOWN_BLOCK_{i}: Injecting STRUCTURE embedding at step {self.current_step}")
            else:
                block_encoder = original_encoder
                
            x, res = block(
                x,
                encoder_x=block_encoder,
                temb=temb,
                attn_mask=attn_mask,
                encoder_attn_mask=encoder_attn_mask,
                block_id=f"down_{i}",
                step_idx=step_idx,
                sigma=sigma
            )
            residuals.extend(res)
            
        # MIDDLE BLOCKS - Content phase (30-70%)
        x = self.unet.mid_blocks[0](x, temb)
        
        # Use content injection in middle steps
        if self.injection_config and 0.3 <= progress < 0.7 and self.injection_config.mid_blocks is not None:
            mid_encoder = self.injection_config.mid_blocks
            print(f"  💉 MID_BLOCK: Injecting CONTENT embedding at step {self.current_step}")
        else:
            mid_encoder = original_encoder
            
        x = self.unet.mid_blocks[1](x, mid_encoder, attn_mask, encoder_attn_mask,
                                    block_id="mid", step_idx=step_idx, sigma=sigma)
        x = self.unet.mid_blocks[2](x, temb)
        
        # UP BLOCKS - Style phase (70-100%)  
        for i, block in enumerate(self.unet.up_blocks):
            # Use style injection in late steps
            if self.injection_config and progress >= 0.7 and self.injection_config.up_blocks is not None:
                block_encoder = self.injection_config.up_blocks
                print(f"  💉 UP_BLOCK_{i}: Injecting STYLE embedding at step {self.current_step}")
            else:
                block_encoder = original_encoder
                
            x, _ = block(
                x,
                encoder_x=block_encoder,
                temb=temb,
                attn_mask=attn_mask,
                encoder_attn_mask=encoder_attn_mask,
                residual_hidden_states=residuals,
                block_id=f"up_{i}",
                step_idx=step_idx,
                sigma=sigma
            )
            
        # Postprocess output
        x = self.unet.conv_norm_out(x)
        x = nn.silu(x)
        x = self.unet.conv_out(x)
        
        return x


def inject_datavoid_unet(sd):
    """
    Replace standard UNet with DataVoid injection-capable version.
    
    Args:
        sd: StableDiffusion instance
        
    Returns:
        DataVoidUNet wrapper instance
    """
    print("\n🧬 Injecting DataVoid UNet architecture...")
    
    # Create wrapper around existing UNet
    datavoid_unet = DataVoidUNet(sd.unet)
    
    # Store reference to wrapper
    sd._datavoid_unet = datavoid_unet
    
    print("✅ DataVoid UNet injection complete!")
    return datavoid_unet