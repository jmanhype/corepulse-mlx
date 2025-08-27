"""
CorePulse Stable Diffusion wrapper for MLX.
Provides the main interface for using CorePulse techniques with Stable Diffusion.
"""

import mlx.core as mx
from typing import Optional, Union, List
from .injection import PromptInjector, InjectionConfig
from .utils import KVRegistry


class CorePulseStableDiffusion:
    """Main wrapper for CorePulse-enhanced Stable Diffusion."""
    
    def __init__(self, base_model):
        """Initialize with a base Stable Diffusion model.
        
        Args:
            base_model: The underlying SD/SDXL model instance
        """
        self.model = base_model
        self.injector = PromptInjector(base_model)
        self.kv_registry = KVRegistry()
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Set up hooks for the model's attention layers."""
        # Hook into the model's UNet blocks
        if hasattr(self.model, 'unet'):
            self._patch_unet_attention(self.model.unet)
    
    def _patch_unet_attention(self, unet):
        """Patch UNet attention layers with CorePulse hooks."""
        # This would integrate with the actual UNet structure
        # For now, just register the hook system
        pass
    
    def add_injection(
        self,
        prompt: str,
        strength: float = 0.3,
        blocks: Optional[List[str]] = None,
        start_step: int = 0,
        end_step: Optional[int] = None
    ):
        """Add a prompt injection configuration.
        
        Args:
            prompt: The prompt text to inject
            strength: Injection strength (0.1-0.5 recommended)
            blocks: Which UNet blocks to target
            start_step: Starting denoising step
            end_step: Ending denoising step (None for all)
        """
        config = InjectionConfig(
            inject_prompt=prompt,
            strength=strength,
            blocks=blocks,
            start_step=start_step,
            end_step=end_step
        )
        self.injector.add_injection(config)
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Generate an image with CorePulse enhancements.
        
        Args:
            prompt: Main text prompt
            negative_prompt: Negative prompt for guidance
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to base model
            
        Returns:
            Generated image(s)
        """
        # Apply injection hooks to registry
        self.injector.apply_hooks(self.kv_registry)
        
        # Reset step counter
        self.injector.reset()
        
        # Generate with the base model
        # The hooks will automatically apply during generation
        result = self.model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            **kwargs
        )
        
        # Update step counter if needed
        # This would integrate with the denoising loop
        
        return result
    
    def clear_injections(self):
        """Clear all injection configurations."""
        self.injector.injections.clear()
        self.kv_registry.clear()
    
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        """Load a model from HuggingFace or local path.
        
        Args:
            model_id: Model identifier or path
            **kwargs: Additional loading arguments
            
        Returns:
            CorePulseStableDiffusion instance
        """
        # This would load the appropriate base model
        # For now, assume it's passed in
        from stable_diffusion import StableDiffusion
        base_model = StableDiffusion.from_pretrained(model_id, **kwargs)
        return cls(base_model)
