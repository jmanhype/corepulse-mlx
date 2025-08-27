"""
CorePulse MLX - Advanced prompt injection and manipulation for MLX Stable Diffusion.

This package provides CorePulse V4 DataVoid techniques with pre-attention KV manipulation
for Apple Silicon GPUs using the MLX framework.
"""

__version__ = "0.2.1"

# Export only canonical wrappers as per user instruction
from .sd_wrapper import CorePulseStableDiffusion
from .injection import InjectionConfig, PromptInjector
from .utils import KVRegistry

__all__ = [
    "CorePulseStableDiffusion",
    "InjectionConfig", 
    "PromptInjector",
    "KVRegistry",
]
