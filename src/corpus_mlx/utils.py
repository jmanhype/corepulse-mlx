"""
Utility classes and functions for CorePulse MLX implementation.
"""

import mlx.core as mx
from typing import Dict, Callable, Optional, Any


class KVRegistry:
    """Registry for key-value manipulation hooks."""
    
    def __init__(self):
        self.hooks: Dict[str, Callable] = {}
        self.active = True
    
    def set(self, block_id: str, hook: Callable):
        """Register a hook for a specific block.
        
        Args:
            block_id: Identifier for the UNet block
            hook: Function to modify (q, k, v) tensors
        """
        self.hooks[block_id] = hook
    
    def get(self, block_id: str) -> Optional[Callable]:
        """Get hook for a specific block.
        
        Args:
            block_id: Identifier for the UNet block
            
        Returns:
            Hook function if registered, None otherwise
        """
        if not self.active:
            return None
        return self.hooks.get(block_id)
    
    def apply(self, block_id: str, q, k, v, meta: Optional[Dict[str, Any]] = None):
        """Apply registered hook to tensors.
        
        Args:
            block_id: Identifier for the UNet block
            q: Query tensor
            k: Key tensor
            v: Value tensor
            meta: Optional metadata dictionary
            
        Returns:
            Modified (q, k, v) tensors
        """
        hook = self.get(block_id)
        if hook:
            if meta is None:
                meta = {'block_id': block_id}
            else:
                meta['block_id'] = block_id
            return hook(q, k, v, meta)
        return q, k, v
    
    def clear(self):
        """Clear all registered hooks."""
        self.hooks.clear()
    
    def disable(self):
        """Temporarily disable all hooks."""
        self.active = False
    
    def enable(self):
        """Re-enable hooks."""
        self.active = True


def normalize_prompt_embeddings(embeddings, target_norm: float = 1.0):
    """Normalize prompt embeddings to prevent overflow.
    
    Args:
        embeddings: Input embeddings tensor
        target_norm: Target L2 norm value
        
    Returns:
        Normalized embeddings
    """
    norms = mx.linalg.norm(embeddings, axis=-1, keepdims=True)
    # Avoid division by zero
    norms = mx.maximum(norms, 1e-12)
    return embeddings * (target_norm / norms)


def blend_tensors(tensor_a, tensor_b, alpha: float = 0.5):
    """Blend two tensors with linear interpolation.
    
    Args:
        tensor_a: First tensor
        tensor_b: Second tensor
        alpha: Blending factor (0 = all A, 1 = all B)
        
    Returns:
        Blended tensor
    """
    alpha = min(max(alpha, 0.0), 1.0)
    return tensor_a * (1 - alpha) + tensor_b * alpha


def create_attention_mask(shape, mask_type: str = "causal"):
    """Create attention mask for self-attention.
    
    Args:
        shape: Shape of the attention matrix (seq_len, seq_len)
        mask_type: Type of mask ("causal", "full", "none")
        
    Returns:
        Attention mask tensor
    """
    seq_len = shape[0]
    
    if mask_type == "causal":
        # Lower triangular mask for causal attention
        mask = mx.tril(mx.ones((seq_len, seq_len)))
    elif mask_type == "full":
        # Full attention (all ones)
        mask = mx.ones((seq_len, seq_len))
    else:
        # No mask
        mask = None
    
    return mask
