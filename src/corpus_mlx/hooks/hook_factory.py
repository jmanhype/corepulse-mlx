"""
Factory for creating attention hooks.
Following DRY principle - Don't Repeat Yourself.
"""

import mlx.core as mx
from typing import Callable, Optional, List


class HookFactory:
    """Factory for creating various types of attention hooks."""
    
    @staticmethod
    def create_scaling_hook(scale: float, cross_attention_only: bool = True) -> Callable:
        """Create a hook that scales K and V tensors.
        
        Args:
            scale: Scaling factor
            cross_attention_only: Whether to apply only to cross-attention
            
        Returns:
            Hook function
        """
        def hook(q, k, v, meta=None):
            if cross_attention_only and k.shape[2] >= 100:
                return q, k, v
            return q, k * scale, v * scale
        return hook
    
    @staticmethod
    def create_noise_hook(intensity: float, cross_attention_only: bool = True) -> Callable:
        """Create a hook that adds noise to K and V tensors.
        
        Args:
            intensity: Noise intensity
            cross_attention_only: Whether to apply only to cross-attention
            
        Returns:
            Hook function
        """
        def hook(q, k, v, meta=None):
            if cross_attention_only and k.shape[2] >= 100:
                return q, k, v
            noise_k = mx.random.normal(k.shape) * intensity
            noise_v = mx.random.normal(v.shape) * intensity
            return q, k + noise_k, v + noise_v
        return hook
    
    @staticmethod
    def create_masking_hook(mask_indices: List[int], cross_attention_only: bool = True) -> Callable:
        """Create a hook that masks specific indices.
        
        Args:
            mask_indices: Indices to mask
            cross_attention_only: Whether to apply only to cross-attention
            
        Returns:
            Hook function
        """
        def hook(q, k, v, meta=None):
            if cross_attention_only and k.shape[2] >= 100:
                return q, k, v
            k_new = mx.array(k)
            v_new = mx.array(v)
            for idx in mask_indices:
                if idx < k.shape[2]:
                    k_new[:, :, idx, :] = 0
                    v_new[:, :, idx, :] = 0
            return q, k_new, v_new
        return hook
    
    @staticmethod
    def create_conditional_hook(
        condition_fn: Callable,
        true_hook: Callable,
        false_hook: Optional[Callable] = None
    ) -> Callable:
        """Create a conditional hook that applies different hooks based on condition.
        
        Args:
            condition_fn: Function that returns True/False based on q, k, v
            true_hook: Hook to apply when condition is True
            false_hook: Optional hook to apply when condition is False
            
        Returns:
            Hook function
        """
        def hook(q, k, v, meta=None):
            if condition_fn(q, k, v, meta):
                return true_hook(q, k, v, meta)
            elif false_hook:
                return false_hook(q, k, v, meta)
            return q, k, v
        return hook
    
    @staticmethod
    def apply_to_blocks(hook: Callable, blocks: List[str], registry) -> None:
        """Apply a hook to multiple blocks.
        
        Args:
            hook: Hook function to apply
            blocks: List of block names
            registry: KV registry to store hooks
        """
        for block in blocks:
            registry.set(block, hook)