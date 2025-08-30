"""
Semantic replacement implementation for corpus-mlx.
This replaces text conditioning embeddings to achieve object replacement.
"""

import mlx.core as mx
from typing import Optional, Dict, Any, Tuple, List
import numpy as np


class SemanticReplacementWrapper:
    """
    Wrapper that intercepts and replaces text conditioning for semantic replacement.
    This achieves true object replacement (apple->banana) by replacing text embeddings.
    """
    
    def __init__(self, base_model):
        """Initialize wrapper around base StableDiffusion model."""
        self.base_model = base_model
        self.replacements = []
        self.original_get_conditioning = base_model._get_text_conditioning
        self.original_denoising_step = base_model._denoising_step
        
        # Monkey-patch the model methods
        base_model._get_text_conditioning = self._patched_get_conditioning
        base_model._denoising_step = self._patched_denoising_step
        
        self.current_step = 0
        self.total_steps = 50
        self.active_replacement = None
    
    def add_semantic_replacement(
        self,
        original_object: str,
        replacement_object: str,
        base_prompt: str = None,
        strength: float = 1.0,
        start_frac: float = 0.0,
        end_frac: float = 0.7
    ):
        """
        Add a semantic replacement configuration.
        
        Args:
            original_object: Object to replace (e.g., "apple")
            replacement_object: Replacement object (e.g., "banana")
            base_prompt: Optional base prompt template with {}
            strength: Replacement strength (1.0 = full replacement)
            start_frac: Start fraction of denoising (0.0 = beginning)
            end_frac: End fraction of denoising (0.7 = 70% through)
        """
        self.replacements.append({
            'original': original_object,
            'replacement': replacement_object,
            'base_prompt': base_prompt,
            'strength': strength,
            'start_frac': start_frac,
            'end_frac': end_frac
        })
    
    def _patched_get_conditioning(
        self,
        text: str,
        n_images: int = 1,
        cfg_weight: float = 7.5,
        negative_text: str = "",
    ):
        """
        Patched version that can replace conditioning based on semantic replacements.
        """
        # Store original prompt for checking
        self.original_prompt = text
        
        # Check if we should replace
        replacement_prompt = None
        for config in self.replacements:
            if config['original'] in text:
                # Found a match - create replacement prompt
                replacement_prompt = text.replace(
                    config['original'], 
                    config['replacement']
                )
                self.active_replacement = config
                self.active_replacement['replacement_prompt'] = replacement_prompt
                print(f"🔄 Semantic replacement: '{config['original']}' -> '{config['replacement']}'")
                break
        
        # Get original conditioning
        original_cond = self.original_get_conditioning(
            text, n_images, cfg_weight, negative_text
        )
        
        if replacement_prompt and self.active_replacement:
            # Get replacement conditioning
            replacement_cond = self.original_get_conditioning(
                replacement_prompt, n_images, cfg_weight, negative_text
            )
            
            # Store both for use during denoising
            self.original_conditioning = original_cond
            self.replacement_conditioning = replacement_cond
            
            # Start with original but will swap during denoising
            return original_cond
        else:
            self.original_conditioning = original_cond
            self.replacement_conditioning = None
            return original_cond
    
    def _patched_denoising_step(
        self, x_t, t, t_prev, conditioning, cfg_weight: float = 7.5, text_time=None
    ):
        """
        Patched denoising step that swaps conditioning based on progress.
        """
        # Track progress
        self.current_step += 1
        frac = self.current_step / self.total_steps
        
        # Check if we should use replacement conditioning
        if (self.active_replacement and 
            self.replacement_conditioning is not None):
            
            config = self.active_replacement
            
            # Check if we're in the replacement window
            if config['start_frac'] <= frac <= config['end_frac']:
                # Use replacement conditioning
                strength = config['strength']
                
                if strength >= 1.0:
                    # Full replacement
                    conditioning = self.replacement_conditioning
                    print(f"  Step {self.current_step}/{self.total_steps}: Using replacement conditioning")
                else:
                    # Blend conditioning
                    conditioning = (
                        self.original_conditioning * (1 - strength) +
                        self.replacement_conditioning * strength
                    )
                    print(f"  Step {self.current_step}/{self.total_steps}: Blending {strength:.0%}")
        
        # Call original denoising step with potentially replaced conditioning
        return self.original_denoising_step(
            x_t, t, t_prev, conditioning, cfg_weight, text_time
        )
    
    def reset(self):
        """Reset state for new generation."""
        self.current_step = 0
        self.active_replacement = None
        self.original_conditioning = None
        self.replacement_conditioning = None
    
    def generate(self, prompt: str, **kwargs):
        """
        Generate with semantic replacement.
        
        Args:
            prompt: Text prompt (will check for replacement targets)
            **kwargs: Other generation parameters
        """
        # Reset state
        self.reset()
        
        # Set total steps if provided
        if 'num_steps' in kwargs:
            self.total_steps = kwargs['num_steps']
        
        # Generate using base model (which now has patched methods)
        return self.base_model.generate(prompt, **kwargs)


def create_semantic_sd(model_path: str = "stable-diffusion-xl-base-1.0"):
    """
    Create a StableDiffusion model with semantic replacement capability.
    
    Returns:
        Tuple of (model, wrapper) for generation with semantic replacement
    """
    # Import the base MLX StableDiffusion model
    if "xl" in model_path.lower():
        from adapters.stable_diffusion import StableDiffusionXL
        base_sd = StableDiffusionXL(
            "stabilityai/stable-diffusion-xl-base-1.0"
        )
    else:
        from adapters.stable_diffusion import StableDiffusion
        base_sd = StableDiffusion(
            "runwayml/stable-diffusion-v1-5"
        )
    
    # Wrap with semantic replacement
    wrapper = SemanticReplacementWrapper(base_sd)
    
    # Also create CorePulse wrapper for compatibility
    from corpus_mlx import CorePulseStableDiffusion
    corpus_model = CorePulseStableDiffusion(base_sd)
    
    return corpus_model, wrapper