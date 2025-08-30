"""
CorePulse injection implementation for MLX.
Handles prompt injection with pre-attention KV manipulation.
"""

import mlx.core as mx
from typing import Optional, Dict, Any, Callable


class InjectionConfig:
    """Configuration for prompt injection."""
    
    def __init__(
        self,
        inject_prompt: str,
        strength: float = 0.3,
        blocks: Optional[list] = None,
        start_step: int = 0,
        end_step: Optional[int] = None
    ):
        self.inject_prompt = inject_prompt
        self.strength = min(max(strength, 0.1), 2.0)  # Allow up to 2.0 for semantic replacement
        self.blocks = blocks or ['mid', 'up_0', 'up_1']
        self.start_step = start_step
        self.end_step = end_step


class PromptInjector:
    """Handles prompt injection into diffusion process."""
    
    def __init__(self, model):
        self.model = model
        self.injections = []
        self.current_step = 0
    
    def add_injection(self, config: InjectionConfig):
        """Add an injection configuration."""
        self.injections.append(config)
    
    def create_injection_hook(self, config: InjectionConfig) -> Callable:
        """Create a hook function for injection."""
        # Get conditioning for injection prompt
        # StableDiffusionXL returns (conditioning, pooled_conditioning)
        inject_result = self.model._get_text_conditioning(config.inject_prompt)
        
        # Handle both SD and SDXL formats
        if isinstance(inject_result, tuple):
            inject_cond = inject_result[0]  # Main conditioning
        else:
            inject_cond = inject_result
        
        def hook(q, k, v, meta=None):
            # Only modify cross-attention layers (text conditioning)
            if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
                block_id = meta.get('block_id', 'unknown')
                
                # Check if this block should be modified
                if block_id not in config.blocks:
                    return q, k, v
                
                # Check if we're in the right step range
                if self.current_step < config.start_step:
                    return q, k, v
                if config.end_step and self.current_step > config.end_step:
                    return q, k, v
                
                batch, heads, seq_len, dim = v.shape
                
                # CRITICAL FIX: The conditioning needs to be properly projected
                # inject_cond shape: (batch, seq_len, 2048)
                # v shape: (batch, heads, seq_len, dim=64)
                
                # Method from working test: direct replacement of text tokens
                if seq_len >= 77:  # Standard CLIP token length
                    v_modified = mx.array(v)  # Copy first
                    
                    # Replace text tokens (first 77) with injection
                    text_tokens = min(77, inject_cond.shape[1])
                    
                    # Project conditioning to match V dimensions
                    # Take first 'dim' dimensions from the 2048-dim conditioning
                    inject_vals = inject_cond[0, :text_tokens, :dim]
                    
                    # Replace (not blend) for stronger effect at high strength
                    if config.strength > 0.9:
                        # Direct replacement like in the working test
                        for b in range(batch):
                            for h in range(heads):
                                v_modified[b, h, :text_tokens, :] = inject_vals
                    else:
                        # Blend for lower strengths
                        for b in range(batch):
                            for h in range(heads):
                                v_modified[b, h, :text_tokens, :] = \
                                    v[b, h, :text_tokens, :] * (1 - config.strength) + \
                                    inject_vals * config.strength
                    
                    return q, k, v_modified
            
            return q, k, v
        
        return hook
    
    def apply_hooks(self, registry):
        """Apply all injection hooks to the registry."""
        for config in self.injections:
            hook = self.create_injection_hook(config)
            for block in config.blocks:
                registry.set(block, hook)
    
    def step(self):
        """Increment the current step."""
        self.current_step += 1
    
    def reset(self):
        """Reset for a new generation."""
        self.current_step = 0
