"""
Type definitions and configuration objects for CorePulse.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple
from enum import Enum


class BlockType(Enum):
    """UNet block types."""
    DOWN_0 = "down_0"
    DOWN_1 = "down_1"
    DOWN_2 = "down_2"
    MID = "mid"
    UP_0 = "up_0"
    UP_1 = "up_1"
    UP_2 = "up_2"


class BlendStrategy(Enum):
    """Blending strategies for embeddings."""
    LINEAR = "linear"
    MULTIPLICATIVE = "multiplicative"
    ADDITIVE = "additive"
    OVERLAY = "overlay"
    SOFTMAX = "softmax"


class ScheduleCurve(Enum):
    """Schedule curve types."""
    LINEAR = "linear"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    STEP = "step"


@dataclass
class CorePulseConfig:
    """Main configuration for CorePulse."""
    
    # Model settings
    model_id: Optional[str] = None
    device: str = "gpu"
    precision: str = "float16"
    
    # Injection defaults
    default_strength: float = 0.3
    default_blocks: List[str] = field(default_factory=lambda: ["mid", "up_0", "up_1"])
    default_blend_mode: BlendStrategy = BlendStrategy.LINEAR
    
    # Safety settings
    max_strength: float = 0.5
    min_strength: float = 0.1
    safe_mode: bool = True
    
    # Performance
    cache_embeddings: bool = True
    batch_injections: bool = True


@dataclass
class InjectionSpec:
    """Detailed injection specification."""
    
    prompt: str
    strength: float = 0.3
    blocks: Optional[List[BlockType]] = None
    start_step: int = 0
    end_step: Optional[int] = None
    blend_mode: BlendStrategy = BlendStrategy.LINEAR
    
    # Regional control
    region: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    mask: Optional[Any] = None  # MLX array
    
    # Advanced options
    normalize: bool = True
    scale_by_attention: bool = False
    preserve_magnitude: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        self.strength = max(0.0, min(1.0, self.strength))
        if self.blocks is None:
            self.blocks = [BlockType.MID, BlockType.UP_0, BlockType.UP_1]


@dataclass
class RegionSpec:
    """Specification for regional control."""
    
    prompt: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    strength: float = 0.5
    feather: int = 0  # Feathering radius in pixels
    blend_mode: BlendStrategy = BlendStrategy.LINEAR
    
    # Shape options
    shape: str = "rectangle"  # rectangle, ellipse, polygon
    points: Optional[List[Tuple[int, int]]] = None  # For polygon shape
    
    def get_mask_params(self) -> Dict[str, Any]:
        """Get parameters for mask creation."""
        return {
            "bbox": self.bbox,
            "shape": self.shape,
            "feather": self.feather,
            "points": self.points
        }


@dataclass
class GenerationConfig:
    """Configuration for image generation."""
    
    # Basic parameters
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 4
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    
    # Injection specifications
    injections: List[InjectionSpec] = field(default_factory=list)
    regions: List[RegionSpec] = field(default_factory=list)
    
    # Advanced options
    scheduler: str = "euler"
    eta: float = 0.0
    clip_skip: int = 0
    
    # Output options
    num_images: int = 1
    output_format: str = "pil"  # pil, numpy, mlx
    return_dict: bool = False


@dataclass
class AttentionHookConfig:
    """Configuration for attention hooks."""
    
    hook_type: str  # "pre", "post", "replace"
    target_blocks: List[BlockType] = field(default_factory=list)
    target_layers: Optional[List[int]] = None
    target_heads: Optional[List[int]] = None
    
    # Hook parameters
    strength: float = 1.0
    operation: str = "multiply"  # multiply, add, replace
    preserve_norm: bool = True


@dataclass
class PluginConfig:
    """Configuration for plugins."""
    
    name: str
    enabled: bool = True
    priority: int = 0  # Higher priority runs first
    params: Dict[str, Any] = field(default_factory=dict)


class ResultDict(Dict[str, Any]):
    """Enhanced dictionary for generation results."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def images(self):
        """Get generated images."""
        return self.get("images", [])
        
    @property
    def latents(self):
        """Get final latents."""
        return self.get("latents", None)
        
    @property
    def attention_maps(self):
        """Get attention maps if available."""
        return self.get("attention_maps", None)
        
    @property
    def metadata(self):
        """Get generation metadata."""
        return self.get("metadata", {})


def validate_config(config: Union[CorePulseConfig, GenerationConfig]) -> bool:
    """Validate a configuration object.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    if isinstance(config, CorePulseConfig):
        if config.max_strength > 1.0 or config.min_strength < 0.0:
            raise ValueError("Strength values must be between 0 and 1")
        if config.max_strength < config.min_strength:
            raise ValueError("max_strength must be >= min_strength")
            
    elif isinstance(config, GenerationConfig):
        if config.width % 8 != 0 or config.height % 8 != 0:
            raise ValueError("Width and height must be divisible by 8")
        if config.num_inference_steps < 1:
            raise ValueError("Must have at least 1 inference step")
            
    return True