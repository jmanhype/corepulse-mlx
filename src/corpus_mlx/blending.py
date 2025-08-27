"""
Blending strategies for CorePulse MLX.
Provides various methods for combining embeddings and attention values.
"""

import mlx.core as mx
from typing import Optional, Callable, Tuple
from enum import Enum


class BlendMode(Enum):
    """Available blending modes."""
    LINEAR = "linear"
    MULTIPLICATIVE = "multiplicative"
    ADDITIVE = "additive"
    OVERLAY = "overlay"
    SOFTMAX = "softmax"


class EmbeddingBlender:
    """Handles blending of embeddings with various strategies."""
    
    @staticmethod
    def linear_blend(
        source: mx.array,
        target: mx.array,
        alpha: float = 0.5
    ) -> mx.array:
        """Linear interpolation between two embeddings.
        
        Args:
            source: Source embedding
            target: Target embedding
            alpha: Blend factor (0=source, 1=target)
            
        Returns:
            Blended embedding
        """
        alpha = min(max(alpha, 0.0), 1.0)
        return source * (1 - alpha) + target * alpha
    
    @staticmethod
    def multiplicative_blend(
        source: mx.array,
        target: mx.array,
        strength: float = 0.5
    ) -> mx.array:
        """Multiplicative blending with strength control.
        
        Args:
            source: Source embedding
            target: Target embedding
            strength: Multiplication strength
            
        Returns:
            Blended embedding
        """
        return source * (1 - strength) + (source * target) * strength
    
    @staticmethod
    def additive_blend(
        source: mx.array,
        target: mx.array,
        strength: float = 0.3
    ) -> mx.array:
        """Additive blending with normalization.
        
        Args:
            source: Source embedding
            target: Target embedding
            strength: Addition strength
            
        Returns:
            Blended embedding
        """
        result = source + target * strength
        # Normalize to prevent overflow
        norm = mx.linalg.norm(result, axis=-1, keepdims=True)
        norm = mx.maximum(norm, 1e-12)
        return result / norm * mx.linalg.norm(source, axis=-1, keepdims=True)
    
    @staticmethod
    def overlay_blend(
        source: mx.array,
        target: mx.array,
        threshold: float = 0.5
    ) -> mx.array:
        """Overlay blending based on value threshold.
        
        Args:
            source: Source embedding
            target: Target embedding
            threshold: Overlay threshold
            
        Returns:
            Blended embedding
        """
        mask = source > threshold
        return mx.where(mask, target, source)
    
    @staticmethod
    def softmax_blend(
        embeddings: list,
        weights: Optional[list] = None,
        temperature: float = 1.0
    ) -> mx.array:
        """Weighted softmax blending of multiple embeddings.
        
        Args:
            embeddings: List of embeddings to blend
            weights: Optional weights for each embedding
            temperature: Softmax temperature
            
        Returns:
            Blended embedding
        """
        if weights is None:
            weights = [1.0] * len(embeddings)
        
        # Convert weights to softmax probabilities
        weights = mx.array(weights) / temperature
        probs = mx.softmax(weights)
        
        # Weighted sum
        result = mx.zeros_like(embeddings[0])
        for emb, prob in zip(embeddings, probs):
            result = result + emb * prob
        
        return result


class AttentionBlender:
    """Specialized blending for attention mechanisms."""
    
    @staticmethod
    def blend_qkv(
        q: mx.array,
        k: mx.array,
        v: mx.array,
        inject_q: Optional[mx.array] = None,
        inject_k: Optional[mx.array] = None,
        inject_v: Optional[mx.array] = None,
        mode: BlendMode = BlendMode.LINEAR,
        strength: float = 0.3
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Blend query, key, value tensors with injections.
        
        Args:
            q, k, v: Original tensors
            inject_q, inject_k, inject_v: Injection tensors
            mode: Blending mode
            strength: Blend strength
            
        Returns:
            Blended (q, k, v) tensors
        """
        blender = EmbeddingBlender()
        
        if mode == BlendMode.LINEAR:
            blend_fn = lambda s, t: blender.linear_blend(s, t, strength)
        elif mode == BlendMode.MULTIPLICATIVE:
            blend_fn = lambda s, t: blender.multiplicative_blend(s, t, strength)
        elif mode == BlendMode.ADDITIVE:
            blend_fn = lambda s, t: blender.additive_blend(s, t, strength)
        else:
            blend_fn = lambda s, t: blender.linear_blend(s, t, strength)
        
        if inject_q is not None:
            q = blend_fn(q, inject_q)
        if inject_k is not None:
            k = blend_fn(k, inject_k)
        if inject_v is not None:
            v = blend_fn(v, inject_v)
        
        return q, k, v
    
    @staticmethod
    def create_blend_schedule(
        num_steps: int,
        start_strength: float = 0.0,
        end_strength: float = 0.5,
        curve: str = "linear"
    ) -> list:
        """Create a blending strength schedule over denoising steps.
        
        Args:
            num_steps: Total number of steps
            start_strength: Initial strength
            end_strength: Final strength
            curve: Interpolation curve type
            
        Returns:
            List of strength values
        """
        if curve == "linear":
            return [start_strength + (end_strength - start_strength) * i / (num_steps - 1)
                    for i in range(num_steps)]
        elif curve == "cosine":
            import math
            return [start_strength + (end_strength - start_strength) * 
                    (1 - math.cos(i * math.pi / (num_steps - 1))) / 2
                    for i in range(num_steps)]
        elif curve == "exponential":
            import math
            return [start_strength + (end_strength - start_strength) *
                    (math.exp(i / (num_steps - 1)) - 1) / (math.e - 1)
                    for i in range(num_steps)]
        else:
            return [start_strength] * num_steps
