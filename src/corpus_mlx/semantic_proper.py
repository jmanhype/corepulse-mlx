"""
Proper semantic replacement for corpus-mlx.
This intercepts and replaces prompts BEFORE tokenization.
"""

import mlx.core as mx
from typing import Dict, Optional, Tuple


class ProperSemanticWrapper:
    """
    Wrapper that achieves TRUE semantic object replacement.
    Intercepts prompts before tokenization and replaces objects.
    """
    
    def __init__(self, corpus_wrapper):
        """
        Initialize semantic wrapper around CorePulseStableDiffusion.
        
        Args:
            corpus_wrapper: CorePulseStableDiffusion instance
        """
        self.wrapper = corpus_wrapper
        self.replacements = {}
        self.active = False
        
        # Store original methods
        self._original_generate_latents = corpus_wrapper.generate_latents
        
        # Patch the wrapper
        corpus_wrapper.generate_latents = self._patched_generate_latents
    
    def add_replacement(self, original: str, replacement: str):
        """
        Add a semantic replacement rule.
        
        Args:
            original: Object to replace (e.g., "apple")
            replacement: Replacement object (e.g., "banana")
        """
        self.replacements[original] = replacement
        print(f"✅ Added replacement: {original} -> {replacement}")
    
    def enable(self):
        """Enable semantic replacement."""
        self.active = True
        print("🔄 Semantic replacement ENABLED")
    
    def disable(self):
        """Disable semantic replacement."""
        self.active = False
        print("⏸️  Semantic replacement DISABLED")
    
    def _apply_replacements(self, text: str) -> Tuple[str, bool]:
        """
        Apply all replacement rules to text.
        
        Returns:
            Tuple of (modified_text, was_replaced)
        """
        if not self.active or not self.replacements:
            return text, False
        
        modified = text
        replaced = False
        
        for original, replacement in self.replacements.items():
            if original in modified:
                modified = modified.replace(original, replacement)
                replaced = True
                print(f"   🔄 Replaced '{original}' with '{replacement}' in prompt")
        
        return modified, replaced
    
    def _patched_generate_latents(self, base_prompt, **kwargs):
        """
        Patched generate_latents that replaces objects in prompts.
        """
        # Apply replacements to base prompt
        modified_prompt, was_replaced = self._apply_replacements(base_prompt)
        
        if was_replaced:
            print(f"   Original: {base_prompt}")
            print(f"   Modified: {modified_prompt}")
        
        # Also check if there are injections to modify
        if hasattr(self.wrapper, 'injections'):
            for injection in self.wrapper.injections:
                if hasattr(injection, 'prompt'):
                    orig = injection.prompt
                    injection.prompt, _ = self._apply_replacements(injection.prompt)
                    if injection.prompt != orig:
                        print(f"   Modified injection: {orig} -> {injection.prompt}")
        
        # Call original with modified prompt
        return self._original_generate_latents(modified_prompt, **kwargs)
    
    def generate_comparison(
        self,
        prompt: str,
        original_obj: str,
        replacement_obj: str,
        **gen_kwargs
    ) -> Tuple:
        """
        Generate comparison images: baseline and replaced.
        
        Returns:
            Tuple of (baseline_image, replaced_image)
        """
        import numpy as np
        
        # Clear any existing replacements
        self.replacements.clear()
        
        # Generate baseline (no replacement)
        print(f"\n1. Generating baseline with '{original_obj}'...")
        self.disable()
        
        latents = None
        for step_latents in self.wrapper.generate_latents(prompt, **gen_kwargs):
            latents = step_latents
        
        # Decode baseline
        images = self.wrapper.sd.autoencoder.decode(latents)
        baseline = images[0]  # Get first image
        # Correct conversion: clamp to [-1, 1] then scale
        baseline = mx.clip(baseline, -1, 1)
        baseline = ((baseline + 1) * 127.5).astype(mx.uint8)
        baseline = np.array(baseline)
        
        # Generate with replacement
        print(f"\n2. Generating with replacement '{original_obj}' -> '{replacement_obj}'...")
        self.add_replacement(original_obj, replacement_obj)
        self.enable()
        
        latents = None
        for step_latents in self.wrapper.generate_latents(prompt, **gen_kwargs):
            latents = step_latents
        
        # Decode replaced
        images = self.wrapper.sd.autoencoder.decode(latents)
        replaced = images[0]  # Get first image
        # Correct conversion: clamp to [-1, 1] then scale
        replaced = mx.clip(replaced, -1, 1)
        replaced = ((replaced + 1) * 127.5).astype(mx.uint8)
        replaced = np.array(replaced)
        
        self.disable()
        return baseline, replaced


def create_semantic_wrapper(model_name: str = "stabilityai/stable-diffusion-2-1-base"):
    """
    Create a corpus-mlx wrapper with semantic replacement capability.
    
    Returns:
        ProperSemanticWrapper instance
    """
    from corpus_mlx import CorePulseStableDiffusion
    from adapters.stable_diffusion import StableDiffusion
    
    # Create base model
    base_sd = StableDiffusion(model_name)
    
    # Create corpus wrapper
    corpus_wrapper = CorePulseStableDiffusion(base_sd)
    
    # Create semantic wrapper
    semantic_wrapper = ProperSemanticWrapper(corpus_wrapper)
    
    return semantic_wrapper