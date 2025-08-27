"""
Masking utilities for CorePulse MLX.
Provides spatial and token-based masking for selective injection.
"""

import mlx.core as mx
import numpy as np
from typing import Optional, Tuple, Union, List


class MaskConfig:
    """Configuration for attention masking."""
    
    def __init__(
        self,
        mask_type: str = "uniform",
        spatial_region: Optional[Tuple[int, int, int, int]] = None,
        token_indices: Optional[List[int]] = None,
        strength: float = 1.0
    ):
        self.mask_type = mask_type
        self.spatial_region = spatial_region  # (x1, y1, x2, y2)
        self.token_indices = token_indices
        self.strength = min(max(strength, 0.0), 1.0)


class AttentionMask:
    """Creates and applies attention masks for selective injection."""
    
    @staticmethod
    def create_spatial_mask(
        height: int,
        width: int,
        region: Tuple[int, int, int, int]
    ) -> mx.array:
        """Create a spatial mask for a specific image region.
        
        Args:
            height: Image height
            width: Image width
            region: (x1, y1, x2, y2) coordinates
            
        Returns:
            Mask array
        """
        x1, y1, x2, y2 = region
        mask = mx.zeros((height, width))
        mask[y1:y2, x1:x2] = 1.0
        return mask
    
    @staticmethod
    def create_token_mask(
        seq_len: int,
        token_indices: List[int],
        smooth: bool = False
    ) -> mx.array:
        """Create a mask for specific token positions.
        
        Args:
            seq_len: Sequence length
            token_indices: Indices to mask
            smooth: Apply smoothing to mask edges
            
        Returns:
            Token mask array
        """
        mask = mx.zeros(seq_len)
        for idx in token_indices:
            if 0 <= idx < seq_len:
                mask[idx] = 1.0
        
        if smooth:
            # Apply gaussian-like smoothing
            kernel = mx.array([0.25, 0.5, 0.25])
            mask = mx.conv1d(mask.reshape(1, 1, -1), 
                           kernel.reshape(1, 1, -1), 
                           padding=1).reshape(-1)
        
        return mask
    
    @staticmethod
    def apply_mask(
        tensor: mx.array,
        mask: mx.array,
        strength: float = 1.0
    ) -> mx.array:
        """Apply a mask to a tensor with specified strength.
        
        Args:
            tensor: Input tensor
            mask: Mask to apply
            strength: Mask strength (0-1)
            
        Returns:
            Masked tensor
        """
        strength = min(max(strength, 0.0), 1.0)
        return tensor * (1 - mask * strength)


class RegionalControl:
    """Manages regional control for selective prompt injection."""
    
    def __init__(self):
        self.regions = []
    
    def add_region(
        self,
        prompt: str,
        bbox: Tuple[int, int, int, int],
        strength: float = 0.5
    ):
        """Add a controlled region with specific prompt.
        
        Args:
            prompt: Prompt for this region
            bbox: Bounding box (x1, y1, x2, y2)
            strength: Control strength
        """
        self.regions.append({
            'prompt': prompt,
            'bbox': bbox,
            'strength': strength
        })
    
    def create_composite_mask(
        self,
        height: int,
        width: int
    ) -> mx.array:
        """Create composite mask from all regions.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            Composite mask array
        """
        mask = mx.zeros((height, width))
        
        for region in self.regions:
            x1, y1, x2, y2 = region['bbox']
            mask[y1:y2, x1:x2] = mx.maximum(
                mask[y1:y2, x1:x2],
                region['strength']
            )
        
        return mask
    
    def clear_regions(self):
        """Clear all defined regions."""
        self.regions.clear()
