"""
Attention manipulation techniques extracted from CorePulse.
Following Single Responsibility Principle.
"""

import mlx.core as mx
from typing import Optional, List, Dict


class AttentionManipulator:
    """Handles all attention manipulation techniques."""
    
    def __init__(self, kv_registry):
        """Initialize with KV registry for hook management.
        
        Args:
            kv_registry: Registry for managing KV hooks
        """
        self.kv_registry = kv_registry
        self._active_technique = None
    
    def amplify(self, strength: float = 5.0, blocks: Optional[List[str]] = None):
        """Amplify attention strength.
        
        Args:
            strength: Amplification factor (1.0 = normal, 5.0 = 5x stronger)
            blocks: Target blocks (default: all)
        """
        if blocks is None:
            blocks = ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]
        
        def amplification_hook(q, k, v, meta=None):
            if k.shape[2] < 100:  # Cross-attention only
                return q, k * strength, v * strength
            return q, k, v
        
        for block in blocks:
            self.kv_registry.set(block, amplification_hook)
        self._active_technique = "amplification"
        return self
    
    def suppress(self, factor: float = 0.05, blocks: Optional[List[str]] = None):
        """Suppress attention strength.
        
        Args:
            factor: Suppression factor (0.05 = 95% reduction)
            blocks: Target blocks
        """
        if blocks is None:
            blocks = ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]
        
        def suppression_hook(q, k, v, meta=None):
            if k.shape[2] < 100:
                return q, k * factor, v * factor
            return q, k, v
        
        for block in blocks:
            self.kv_registry.set(block, suppression_hook)
        self._active_technique = "suppression"
        return self
    
    def chaos(self, intensity: float = 2.0, blocks: Optional[List[str]] = None):
        """Add chaos/noise to attention.
        
        Args:
            intensity: Noise intensity
            blocks: Target blocks
        """
        if blocks is None:
            blocks = ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]
        
        def chaos_hook(q, k, v, meta=None):
            if k.shape[2] < 100:
                noise_k = mx.random.normal(k.shape) * intensity
                noise_v = mx.random.normal(v.shape) * intensity
                return q, k + noise_k, v + noise_v
            return q, k, v
        
        for block in blocks:
            self.kv_registry.set(block, chaos_hook)
        self._active_technique = "chaos"
        return self
    
    def invert(self, blocks: Optional[List[str]] = None):
        """Invert attention (anti-prompt).
        
        Args:
            blocks: Target blocks
        """
        if blocks is None:
            blocks = ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]
        
        def inversion_hook(q, k, v, meta=None):
            if k.shape[2] < 100:
                return q, -k, -v
            return q, k, v
        
        for block in blocks:
            self.kv_registry.set(block, inversion_hook)
        self._active_technique = "inversion"
        return self
    
    def remove_tokens(self, token_range: tuple = (2, 5), blocks: Optional[List[str]] = None):
        """Remove specific tokens from attention.
        
        Args:
            token_range: (start, end) tokens to remove
            blocks: Target blocks
        """
        if blocks is None:
            blocks = ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]
        
        def token_removal_hook(q, k, v, meta=None):
            if k.shape[2] < 100 and k.shape[2] > token_range[1]:
                k_new = mx.array(k)
                v_new = mx.array(v)
                k_new[:, :, token_range[0]:token_range[1], :] = 0
                v_new[:, :, token_range[0]:token_range[1], :] = 0
                return q, k_new, v_new
            return q, k, v
        
        for block in blocks:
            self.kv_registry.set(block, token_removal_hook)
        self._active_technique = "token_removal"
        return self
    
    def progressive_strength(self, strengths: Dict[str, float]):
        """Apply progressive manipulation across blocks.
        
        Args:
            strengths: Dict of block_name -> strength multiplier
        """
        for block, strength in strengths.items():
            def make_hook(s):
                def hook(q, k, v, meta=None):
                    if k.shape[2] < 100:
                        return q, k * s, v * s
                    return q, k, v
                return hook
            
            self.kv_registry.set(block, make_hook(strength))
        self._active_technique = "progressive"
        return self
    
    def isolate_attention_heads(self, head_indices: List[int], blocks: Optional[List[str]] = None):
        """Isolate specific attention heads.
        
        Args:
            head_indices: Which attention heads to keep active
            blocks: Target blocks
        """
        if blocks is None:
            blocks = ["mid", "up_0", "up_1"]
        
        def head_isolation_hook(q, k, v, meta=None):
            if k.shape[2] < 100:
                k_new = mx.zeros_like(k)
                v_new = mx.zeros_like(v)
                for head in head_indices:
                    if head < k.shape[1]:
                        k_new[:, head] = k[:, head]
                        v_new[:, head] = v[:, head]
                return q, k_new, v_new
            return q, k, v
        
        for block in blocks:
            self.kv_registry.set(block, head_isolation_hook)
        self._active_technique = "head_isolation"
        return self
    
    def frequency_domain_manipulation(self, freq_boost: float = 2.0, blocks: Optional[List[str]] = None):
        """Manipulate frequency domain of attention.
        
        Args:
            freq_boost: How much to boost high frequencies
            blocks: Target blocks
        """
        if blocks is None:
            blocks = ["mid", "up_0", "up_1"]
        
        def frequency_hook(q, k, v, meta=None):
            if k.shape[2] < 100:
                high_freq = mx.random.normal(k.shape) * 0.1
                k_new = k + high_freq * freq_boost
                v_new = v + high_freq * freq_boost
                return q, k_new, v_new
            return q, k, v
        
        for block in blocks:
            self.kv_registry.set(block, frequency_hook)
        self._active_technique = "frequency_domain"
        return self