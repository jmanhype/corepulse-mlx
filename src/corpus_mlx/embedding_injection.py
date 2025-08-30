#!/usr/bin/env python3
"""
TRUE Embedding Injection for corpus-mlx.

This module provides real embedding replacement in the UNet during the forward pass,
similar to CorePulse's approach. It manipulates text conditioning embeddings at the
attention layer level, not just at the prompt level.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class EmbeddingInjectionConfig:
    """Configuration for embedding injection."""
    
    # Original and replacement prompts
    original_phrase: str  # e.g., "cat"
    replacement_phrase: str  # e.g., "dog"
    
    # Injection parameters
    weight: float = 1.0  # Strength of replacement (0=original, 1=full replacement)
    blocks: List[str] = None  # Which UNet blocks to inject into
    
    # Timing control
    start_time: float = 0.0  # When to start injection (0-1)
    end_time: float = 1.0  # When to end injection (0-1)
    
    # Token-level control
    token_mask: Optional[mx.array] = None  # Binary mask for selective token replacement
    
    def __post_init__(self):
        if self.blocks is None:
            # Default to middle and up blocks for maximum effect
            self.blocks = ["mid", "up_0", "up_1", "up_2"]


class TokenMaskGenerator:
    """Generate token masks for selective embedding replacement."""
    
    def __init__(self, tokenizer):
        """
        Initialize with a tokenizer.
        
        Args:
            tokenizer: The tokenizer to use for finding token positions
        """
        self.tokenizer = tokenizer
    
    def create_phrase_mask(self, 
                          full_prompt: str, 
                          target_phrase: str,
                          max_length: int = 77) -> mx.array:
        """
        Create a binary mask for tokens corresponding to a target phrase.
        
        Args:
            full_prompt: The complete prompt text
            target_phrase: The phrase to mask (e.g., "cat")
            max_length: Maximum token sequence length
            
        Returns:
            Binary mask array where 1 = replace this token, 0 = keep original
        """
        # Tokenize the full prompt
        tokens_full = self.tokenizer.encode(full_prompt)
        
        # Tokenize just the target phrase
        tokens_target = self.tokenizer.encode(target_phrase)
        
        # Remove special tokens from target
        if len(tokens_target) > 0 and tokens_target[0] == self.tokenizer.bos_token_id:
            tokens_target = tokens_target[1:]
        if len(tokens_target) > 0 and tokens_target[-1] == self.tokenizer.eos_token_id:
            tokens_target = tokens_target[:-1]
        
        # Create mask
        mask = mx.zeros(max_length)
        
        # Find target tokens in full token sequence
        target_len = len(tokens_target)
        if target_len > 0:
            for i in range(len(tokens_full) - target_len + 1):
                if tokens_full[i:i+target_len] == tokens_target:
                    # Found the target phrase, mask these positions
                    mask[i:i+target_len] = 1.0
                    break
        
        return mask


class EmbeddingInjector:
    """
    Injects replacement embeddings into the UNet during generation.
    
    This manipulates the text conditioning embeddings at the cross-attention
    layers, allowing for true semantic replacement during the denoising process.
    """
    
    def __init__(self, sd_model, tokenizer=None):
        """
        Initialize the embedding injector.
        
        Args:
            sd_model: The Stable Diffusion model
            tokenizer: Optional tokenizer for token mask generation
        """
        self.sd = sd_model
        self.tokenizer = tokenizer or getattr(sd_model, 'tokenizer', None)
        self.mask_generator = TokenMaskGenerator(self.tokenizer) if self.tokenizer else None
        
        self.injections: List[EmbeddingInjectionConfig] = []
        self.original_embeddings: Optional[mx.array] = None
        self.replacement_embeddings: Dict[str, mx.array] = {}
        
        # Store original forward methods
        self._original_forwards = {}
        self._hooks_installed = False
    
    def add_injection(self,
                     original_phrase: str,
                     replacement_phrase: str,
                     weight: float = 1.0,
                     blocks: Optional[List[str]] = None,
                     start_time: float = 0.0,
                     end_time: float = 1.0) -> EmbeddingInjectionConfig:
        """
        Add an embedding injection configuration.
        
        Args:
            original_phrase: Original text to replace (e.g., "cat")
            replacement_phrase: Replacement text (e.g., "dog")
            weight: Strength of replacement (0-1)
            blocks: UNet blocks to inject into
            start_time: When to start injection (0-1)
            end_time: When to end injection (0-1)
            
        Returns:
            The injection configuration
        """
        config = EmbeddingInjectionConfig(
            original_phrase=original_phrase,
            replacement_phrase=replacement_phrase,
            weight=weight,
            blocks=blocks,
            start_time=start_time,
            end_time=end_time
        )
        
        # Generate token mask if possible
        if self.mask_generator and self.original_embeddings is not None:
            # This will be generated when we have the full prompt
            config.token_mask = None
        
        self.injections.append(config)
        return config
    
    def prepare_embeddings(self, prompt: str):
        """
        Prepare the replacement embeddings for injection.
        
        Args:
            prompt: The original prompt
        """
        # Get original embeddings
        tokens = self.tokenizer.encode(prompt)
        tokens = mx.array(tokens)[None]
        self.original_embeddings = self.sd.text_encoder(tokens)
        
        # Generate replacement embeddings for each injection
        for config in self.injections:
            # Create modified prompt
            modified_prompt = prompt.replace(config.original_phrase, config.replacement_phrase)
            
            # Get embeddings for modified prompt
            tokens_modified = self.tokenizer.encode(modified_prompt)
            tokens_modified = mx.array(tokens_modified)[None]
            replacement_emb = self.sd.text_encoder(tokens_modified)
            
            # Store replacement embeddings
            self.replacement_embeddings[config.original_phrase] = replacement_emb
            
            # Generate token mask
            if self.mask_generator:
                config.token_mask = self.mask_generator.create_phrase_mask(
                    prompt, config.original_phrase
                )
    
    def inject_embeddings(self, 
                         embeddings: mx.array, 
                         block_name: str,
                         timestep: float = 0.5) -> mx.array:
        """
        Inject replacement embeddings based on configurations.
        
        Args:
            embeddings: Current text embeddings
            block_name: Name of the current UNet block
            timestep: Current timestep (0-1)
            
        Returns:
            Modified embeddings
        """
        result = embeddings
        
        for config in self.injections:
            # Check if this block should be injected
            if block_name not in config.blocks:
                continue
            
            # Check timing
            if timestep < config.start_time or timestep > config.end_time:
                continue
            
            # Get replacement embeddings
            replacement = self.replacement_embeddings.get(config.original_phrase)
            if replacement is None:
                continue
            
            # Apply token mask if available
            if config.token_mask is not None:
                # Expand mask to match embedding dimensions
                mask = config.token_mask[None, :, None]  # [1, seq_len, 1]
                mask = mx.broadcast_to(mask, embeddings.shape)
                
                # Blend original and replacement based on mask and weight
                result = (1 - mask * config.weight) * result + (mask * config.weight) * replacement
            else:
                # Full replacement with weight blending
                result = (1 - config.weight) * result + config.weight * replacement
        
        return result
    
    def install_hooks(self):
        """Install hooks into UNet attention layers."""
        if self._hooks_installed:
            return
        
        # Hook into cross-attention layers
        for name, module in self.sd.unet.named_modules():
            if 'CrossAttention' in module.__class__.__name__ or 'cross_attn' in name:
                # Store original forward
                self._original_forwards[name] = module.forward
                
                # Create hooked forward
                def create_hooked_forward(orig_forward, block_name):
                    def hooked_forward(hidden_states, encoder_hidden_states=None, **kwargs):
                        # Inject into encoder_hidden_states (text embeddings)
                        if encoder_hidden_states is not None:
                            # Determine current timestep (simplified - you may need to track this properly)
                            timestep = getattr(self, '_current_timestep', 0.5)
                            encoder_hidden_states = self.inject_embeddings(
                                encoder_hidden_states, block_name, timestep
                            )
                        
                        return orig_forward(hidden_states, encoder_hidden_states, **kwargs)
                    
                    return hooked_forward
                
                # Install hook
                module.forward = create_hooked_forward(module.forward, name)
        
        self._hooks_installed = True
    
    def remove_hooks(self):
        """Remove installed hooks."""
        if not self._hooks_installed:
            return
        
        # Restore original forwards
        for name, module in self.sd.unet.named_modules():
            if name in self._original_forwards:
                module.forward = self._original_forwards[name]
        
        self._original_forwards.clear()
        self._hooks_installed = False
    
    def clear(self):
        """Clear all injections and embeddings."""
        self.injections.clear()
        self.original_embeddings = None
        self.replacement_embeddings.clear()
    
    def __enter__(self):
        """Context manager entry."""
        self.install_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.remove_hooks()
        self.clear()


class SemanticEmbeddingWrapper:
    """
    High-level wrapper for semantic embedding replacement.
    
    This provides an easy-to-use interface for replacing objects
    by manipulating embeddings in the UNet.
    """
    
    def __init__(self, sd_model):
        """
        Initialize the wrapper.
        
        Args:
            sd_model: The Stable Diffusion model to wrap
        """
        self.sd = sd_model
        self.injector = EmbeddingInjector(sd_model)
        self.enabled = False
    
    def add_replacement(self, original: str, replacement: str, weight: float = 1.0):
        """
        Add a semantic replacement rule.
        
        Args:
            original: Original object/phrase (e.g., "cat")
            replacement: Replacement object/phrase (e.g., "dog")
            weight: Strength of replacement (0-1)
        """
        self.injector.add_injection(
            original_phrase=original,
            replacement_phrase=replacement,
            weight=weight
        )
        print(f"✅ Added embedding replacement: {original} → {replacement} (weight={weight})")
    
    def enable(self):
        """Enable embedding injection."""
        self.injector.install_hooks()
        self.enabled = True
        print("🔄 Embedding injection ENABLED")
    
    def disable(self):
        """Disable embedding injection."""
        self.injector.remove_hooks()
        self.enabled = False
        print("⏸️  Embedding injection DISABLED")
    
    def generate(self, prompt: str, **kwargs):
        """
        Generate with embedding injection.
        
        Args:
            prompt: The prompt to generate from
            **kwargs: Additional generation arguments
            
        Returns:
            Generated image
        """
        if self.enabled:
            # Prepare embeddings for injection
            self.injector.prepare_embeddings(prompt)
        
        # Generate using the model
        return self.sd.generate_image(prompt, **kwargs)
    
    def clear(self):
        """Clear all replacement rules."""
        self.injector.clear()
        print("🗑️  Cleared all embedding replacements")


def create_embedding_injector(model_name: str, **kwargs):
    """
    Create a semantic embedding injector for a model.
    
    Args:
        model_name: Name of the model to load
        **kwargs: Additional model loading arguments
        
    Returns:
        SemanticEmbeddingWrapper instance
    """
    from stable_diffusion import StableDiffusion
    
    # Load model
    sd = StableDiffusion(model_name, **kwargs)
    
    # Create wrapper
    wrapper = SemanticEmbeddingWrapper(sd)
    
    return wrapper