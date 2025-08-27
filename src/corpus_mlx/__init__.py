"""
CorePulse MLX - Advanced prompt injection and manipulation for MLX Stable Diffusion.

This package provides CorePulse V4 DataVoid techniques with pre-attention KV manipulation
for Apple Silicon GPUs using the MLX framework.
"""

__version__ = "0.2.1"

# Export only canonical wrappers as per user instruction
from .corepulse import CorePulse
from .sd_wrapper import CorePulseStableDiffusion
from .injection import InjectionConfig, PromptInjector
from .utils import KVRegistry
from .types import CorePulseConfig, GenerationConfig, InjectionSpec, RegionSpec

__all__ = [
    # Main wrapper
    "CorePulse",
    
    # Model-specific wrappers
    "CorePulseStableDiffusion",
    
    # Core components
    "InjectionConfig",
    "PromptInjector",
    "KVRegistry",
    
    # Configuration types
    "CorePulseConfig",
    "GenerationConfig",
    "InjectionSpec",
    "RegionSpec",
]
