"""
CorePulse MLX - Advanced prompt injection and manipulation for MLX Stable Diffusion.

This package provides CorePulse V4 DataVoid techniques with pre-attention KV manipulation
for Apple Silicon GPUs using the MLX framework.
"""

__version__ = "0.2.1"

# Export only canonical wrappers as per user instruction
from .corepulse import CorePulse
from .sd_wrapper import CorePulseStableDiffusion as CorePulseStableDiffusionBasic
from .sd_wrapper_advanced import CorePulseStableDiffusion  # Advanced version with all features
from .injection import InjectionConfig, PromptInjector
from .utils import KVRegistry
from .types import CorePulseConfig, GenerationConfig, InjectionSpec, RegionSpec
from .semantic_proper import ProperSemanticWrapper, create_semantic_wrapper
from .true_semantic import TrueSemanticWrapper, create_true_semantic_wrapper

__all__ = [
    # Main wrapper
    "CorePulse",
    
    # Model-specific wrappers
    "CorePulseStableDiffusion",
    
    # Semantic replacement
    "ProperSemanticWrapper",
    "create_semantic_wrapper",
    
    # TRUE embedding injection
    "TrueSemanticWrapper", 
    "create_true_semantic_wrapper",
    
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
