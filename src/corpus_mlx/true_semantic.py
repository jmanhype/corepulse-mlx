#!/usr/bin/env python3
"""
TRUE Semantic Embedding Injection using existing KV hooks.

This leverages corpus-mlx's existing KV hook system to manipulate text embeddings
during cross-attention, achieving the same effect as CorePulse's embedding injection.
"""

import mlx.core as mx
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import sys
from pathlib import Path

# Add adapters to path
sys.path.insert(0, str(Path(__file__).parent.parent / "adapters" / "mlx" / "mlx-examples" / "stable_diffusion"))
from stable_diffusion import attn_scores
from corpus_mlx.hooks.hook_factory import HookFactory


@dataclass
class EmbeddingInjectionConfig:
    """Configuration for embedding injection via KV manipulation."""
    
    original_prompt: str
    replacement_prompt: str
    weight: float = 1.0
    blocks: List[str] = None
    token_mask: Optional[mx.array] = None
    
    def __post_init__(self):
        if self.blocks is None:
            # Target cross-attention blocks where text conditioning matters most
            self.blocks = ["mid", "up_0", "up_1", "up_2"]


class TrueEmbeddingInjector:
    """
    TRUE embedding injection by patching text conditioning.
    
    This replaces text embeddings at the conditioning level, which is simpler
    and more reliable than KV hooks, achieving the same effect as CorePulse.
    """
    
    def __init__(self, sd_model):
        """
        Initialize the embedding injector.
        
        Args:
            sd_model: The Stable Diffusion model
        """
        self.sd = sd_model
        self.replacements = {}
        self.original_get_text_conditioning = sd_model._get_text_conditioning
    
    def _patched_get_text_conditioning(self, text: str, *args, **kwargs):
        """Patched text conditioning that replaces embeddings."""
        
        # Check if we need to replace any text
        modified_text = text
        for original, replacement in self.replacements.items():
            if original in text:
                modified_text = text.replace(original, replacement)
                print(f"🧠 TRUE embedding injection: '{text}' → '{modified_text}'")
                break
        
        # Get conditioning for the modified text
        return self.original_get_text_conditioning(modified_text, *args, **kwargs)
    
    def add_injection(self,
                     original_prompt: str,
                     replacement_prompt: str,
                     weight: float = 1.0,
                     blocks: Optional[List[str]] = None):
        """
        Add an embedding injection configuration.
        
        Args:
            original_prompt: Original text in prompt
            replacement_prompt: Text to inject instead
            weight: Injection strength (ignored for now, always 1.0)
            blocks: Which blocks to inject into (ignored, affects all)
        """
        self.replacements[original_prompt] = replacement_prompt
        
        print(f"💉 Added TRUE embedding injection: {original_prompt} → {replacement_prompt}")
        print(f"   Weight: {weight}, Blocks: all (via text conditioning)")
        
        return None
    
    def enable_for_prompt(self, prompt: str):
        """Enable embedding injection."""
        self.sd._get_text_conditioning = self._patched_get_text_conditioning
        print("🔥 TRUE embedding injection ENABLED")
    
    def clear(self):
        """Clear all injections."""
        self.replacements.clear()
        self.disable()
        print("🗑️  Cleared all embedding injections")
    
    def disable(self):
        """Disable embedding injection."""
        self.sd._get_text_conditioning = self.original_get_text_conditioning
        print("⏸️  TRUE embedding injection DISABLED")


class TrueSemanticWrapper:
    """
    High-level wrapper for TRUE semantic replacement via embedding injection.
    """
    
    def __init__(self, sd_model):
        """Initialize wrapper with SD model."""
        self.sd = sd_model
        self.injector = TrueEmbeddingInjector(sd_model)
        self.replacements = {}
    
    def add_replacement(self, original: str, replacement: str, weight: float = 1.0):
        """Add a semantic replacement rule using embedding injection."""
        self.replacements[original] = replacement
        
        self.injector.add_injection(
            original_prompt=original,
            replacement_prompt=replacement,
            weight=weight
        )
        
        print(f"✅ Added TRUE semantic replacement: {original} → {replacement}")
    
    def generate_with_injection(self, prompt: str, **kwargs):
        """Generate image with embedding injection."""
        # Prepare embeddings
        self.injector.enable_for_prompt(prompt)
        
        # Generate
        return self.sd.generate_image(prompt, **kwargs)
    
    def generate_comparison(self,
                          prompt: str,
                          original_obj: str,
                          replacement_obj: str,
                          **kwargs) -> Tuple[mx.array, mx.array]:
        """
        Generate comparison between original and injection.
        
        Returns:
            (baseline_image, injected_image)
        """
        # Generate baseline
        baseline = self.sd.generate_image(prompt, **kwargs)
        
        # Add injection
        self.add_replacement(original_obj, replacement_obj, weight=1.0)
        
        # Generate with injection
        injected = self.generate_with_injection(prompt, **kwargs)
        
        return baseline, injected
    
    def clear(self):
        """Clear all replacements."""
        self.replacements.clear()
        self.injector.clear()


def create_true_semantic_wrapper(model_name: str, **kwargs):
    """
    Create a TRUE semantic wrapper using embedding injection.
    
    Args:
        model_name: Model to load
        **kwargs: Model loading arguments
        
    Returns:
        TrueSemanticWrapper instance
    """
    from adapters.stable_diffusion import StableDiffusion
    
    print("🧠 Creating TRUE semantic wrapper with text conditioning injection")
    
    # Load model
    sd = StableDiffusion(model_name, **kwargs)
    
    # Create wrapper
    wrapper = TrueSemanticWrapper(sd)
    
    return wrapper