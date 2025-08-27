"""
CorePulse main wrapper - the primary interface for all CorePulse functionality.
"""

import mlx.core as mx
from typing import Optional, Union, List, Dict, Any
from .injection import PromptInjector, InjectionConfig
from .utils import KVRegistry
from .masks import RegionalControl, AttentionMask
from .blending import EmbeddingBlender, AttentionBlender, BlendMode


class CorePulse:
    """
    Main CorePulse wrapper that orchestrates all injection and manipulation techniques.
    This is the primary interface users should interact with.
    """
    
    def __init__(self, base_model=None):
        """Initialize CorePulse with optional base model.
        
        Args:
            base_model: Base diffusion model (SD or SDXL)
        """
        self.model = base_model
        self.injector = PromptInjector(base_model) if base_model else None
        self.kv_registry = KVRegistry()
        self.regional_control = RegionalControl()
        self.attention_blender = AttentionBlender()
        self.embedding_blender = EmbeddingBlender()
        self._hooks_installed = False
        
    def set_model(self, model):
        """Set or update the base model.
        
        Args:
            model: Diffusion model instance
        """
        self.model = model
        self.injector = PromptInjector(model)
        self._hooks_installed = False
        
    def add_injection(
        self,
        prompt: str,
        strength: float = 0.3,
        blocks: Optional[List[str]] = None,
        start_step: int = 0,
        end_step: Optional[int] = None,
        region: Optional[tuple] = None,
        blend_mode: str = "linear"
    ):
        """Add a prompt injection with optional regional control.
        
        Args:
            prompt: Text prompt to inject
            strength: Injection strength (0.1-0.5 recommended)
            blocks: Target UNet blocks (default: ['mid', 'up_0', 'up_1'])
            start_step: Starting denoising step
            end_step: Ending denoising step
            region: Optional (x1, y1, x2, y2) for regional control
            blend_mode: How to blend injections ("linear", "multiplicative", etc)
        """
        # Create injection config
        config = InjectionConfig(
            inject_prompt=prompt,
            strength=strength,
            blocks=blocks,
            start_step=start_step,
            end_step=end_step
        )
        
        # Add to injector
        self.injector.add_injection(config)
        
        # Handle regional control if specified
        if region:
            self.regional_control.add_region(
                prompt=prompt,
                bbox=region,
                strength=strength
            )
            
        # Store blend mode
        self._blend_mode = BlendMode[blend_mode.upper()]
        
    def add_regional_prompt(
        self,
        prompt: str,
        region: tuple,
        strength: float = 0.5
    ):
        """Add a regionally-controlled prompt.
        
        Args:
            prompt: Text prompt for the region
            region: Bounding box (x1, y1, x2, y2)
            strength: Control strength
        """
        self.regional_control.add_region(
            prompt=prompt,
            bbox=region,
            strength=strength
        )
        
        # Also add as injection for this region
        self.add_injection(
            prompt=prompt,
            strength=strength,
            region=region
        )
        
    def create_attention_hook(
        self,
        mode: str = "inject",
        strength: float = 0.3,
        target_blocks: Optional[List[str]] = None
    ):
        """Create a custom attention hook.
        
        Args:
            mode: Hook mode ("inject", "suppress", "amplify", "mask")
            strength: Effect strength
            target_blocks: Which blocks to target
            
        Returns:
            Hook function
        """
        def hook(q, k, v, meta=None):
            block_id = meta.get('block_id', 'unknown') if meta else 'unknown'
            
            # Check if this block should be processed
            if target_blocks and block_id not in target_blocks:
                return q, k, v
                
            if mode == "suppress":
                v = v * (1 - strength)
            elif mode == "amplify":
                v = v * (1 + strength)
            elif mode == "mask" and hasattr(self, '_current_mask'):
                mask = self._current_mask
                v = AttentionMask.apply_mask(v, mask, strength)
                
            return q, k, v
            
        return hook
        
    def install_hooks(self):
        """Install all hooks into the model."""
        if not self.model:
            raise ValueError("No model set. Call set_model() first.")
            
        # Apply injection hooks
        self.injector.apply_hooks(self.kv_registry)
        
        # Mark as installed
        self._hooks_installed = True
        
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        **kwargs
    ):
        """Generate an image with CorePulse enhancements.
        
        Args:
            prompt: Main text prompt
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            seed: Random seed
            width: Image width
            height: Image height
            **kwargs: Additional generation parameters
            
        Returns:
            Generated image(s)
        """
        if not self.model:
            raise ValueError("No model set. Call set_model() first.")
            
        # Install hooks if not already done
        if not self._hooks_installed:
            self.install_hooks()
            
        # Reset step counter
        self.injector.reset()
        
        # Generate regional mask if needed
        if self.regional_control.regions:
            self._current_mask = self.regional_control.create_composite_mask(
                height // 8,  # Latent space dimensions
                width // 8
            )
            
        # Generate with base model
        result = self.model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height,
            **kwargs
        )
        
        return result
        
    def clear(self):
        """Clear all injections and settings."""
        self.injector.injections.clear()
        self.kv_registry.clear()
        self.regional_control.clear_regions()
        self._hooks_installed = False
        
    def create_progressive_injection(
        self,
        prompts: List[str],
        strengths: Optional[List[float]] = None,
        schedule: str = "linear"
    ):
        """Create a progressive injection that changes over steps.
        
        Args:
            prompts: List of prompts to cycle through
            strengths: Optional strength values for each prompt
            schedule: How to transition ("linear", "cosine", "exponential")
        """
        if strengths is None:
            strengths = [0.3] * len(prompts)
            
        # Create blend schedule
        num_prompts = len(prompts)
        schedule_values = self.attention_blender.create_blend_schedule(
            num_prompts,
            start_strength=strengths[0],
            end_strength=strengths[-1],
            curve=schedule
        )
        
        # Add injections with step ranges
        for i, (prompt, strength) in enumerate(zip(prompts, schedule_values)):
            start = i * (50 // num_prompts)  # Assuming 50 steps max
            end = (i + 1) * (50 // num_prompts)
            
            self.add_injection(
                prompt=prompt,
                strength=strength,
                start_step=start,
                end_step=end
            )
            
    @classmethod
    def from_model(cls, model_id: str, **kwargs):
        """Create CorePulse instance with a model.
        
        Args:
            model_id: Model identifier or path
            **kwargs: Model loading arguments
            
        Returns:
            CorePulse instance
        """
        # This would load the appropriate model
        from stable_diffusion import StableDiffusion
        model = StableDiffusion.from_pretrained(model_id, **kwargs)
        return cls(model)