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
        self.strength = min(max(strength, 0.1), 0.5)  # Clamp to safe range
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
        inject_cond, _ = self.model._get_text_conditioning(config.inject_prompt)
        
        def hook(q, k, v, meta=None):
            # Only modify cross-attention layers
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
                
                # Apply injection with safe scaling
                if seq_len >= inject_cond.shape[1]:
                    embed_len = min(seq_len, inject_cond.shape[1])
                    embed_dim = min(dim, inject_cond.shape[2])
                    inject_vals = inject_cond[0, :embed_len, :embed_dim]
                    
                    # Linear interpolation for safe blending
                    v_modified = mx.array(v)
                    for b in range(batch):
                        for h in range(heads):
                            v_modified[b, h, :embed_len, :embed_dim] = \
                                v[b, h, :embed_len, :embed_dim] * (1 - config.strength) + \
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
