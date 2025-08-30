#!/usr/bin/env python3
"""
UNet Hook System for corpus-mlx.

This module provides TRUE embedding injection by hooking into the UNet's
cross-attention layers during the forward pass, similar to CorePulse.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class EmbeddingInjectionConfig:
    """Configuration for embedding injection at specific layers."""
    
    original_text: str
    replacement_text: str
    weight: float = 1.0  # 0 = original, 1 = full replacement
    blocks: List[str] = None
    token_mask: Optional[mx.array] = None
    timestep_range: Tuple[float, float] = (0.0, 1.0)  # When to apply
    
    def __post_init__(self):
        if self.blocks is None:
            self.blocks = ["mid", "up_blocks.0", "up_blocks.1", "up_blocks.2"]


class UNetHookManager:
    """
    Manages hooks for UNet cross-attention layers to enable embedding injection.
    
    This is the MLX equivalent of CorePulse's UNet patcher.
    """
    
    def __init__(self, unet, text_encoder=None, tokenizer=None):
        """
        Initialize the hook manager.
        
        Args:
            unet: The UNet model to hook
            text_encoder: Text encoder for generating replacement embeddings
            tokenizer: Tokenizer for finding token positions
        """
        self.unet = unet
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        
        # Store original forward methods
        self._original_forwards = {}
        self._hooked_modules = {}
        
        # Injection configurations
        self.injections: List[EmbeddingInjectionConfig] = []
        
        # Cache for embeddings
        self._embedding_cache = {}
        
        # Current timestep tracking
        self._current_timestep = 0.0
        
        # Hook installation status
        self._hooks_installed = False
    
    def add_injection(self,
                     original_text: str,
                     replacement_text: str,
                     weight: float = 1.0,
                     blocks: Optional[List[str]] = None,
                     timestep_range: Tuple[float, float] = (0.0, 1.0)):
        """
        Add an embedding injection configuration.
        
        Args:
            original_text: Text to find in original prompt
            replacement_text: Text to inject instead
            weight: Blending weight (0=original, 1=replacement)
            blocks: Which blocks to inject into
            timestep_range: When during generation to apply (0-1)
        """
        config = EmbeddingInjectionConfig(
            original_text=original_text,
            replacement_text=replacement_text,
            weight=weight,
            blocks=blocks,
            timestep_range=timestep_range
        )
        
        self.injections.append(config)
        return config
    
    def _get_embeddings(self, text: str) -> mx.array:
        """Get text embeddings using the text encoder."""
        if text not in self._embedding_cache:
            if self.tokenizer and self.text_encoder:
                tokens = self.tokenizer.encode(text)
                tokens = mx.array(tokens)[None]
                embeddings = self.text_encoder(tokens)
                self._embedding_cache[text] = embeddings
            else:
                raise ValueError("Text encoder and tokenizer required for embedding generation")
        
        return self._embedding_cache[text]
    
    def _create_token_mask(self, full_text: str, target_phrase: str) -> mx.array:
        """
        Create a binary mask for tokens corresponding to target phrase.
        
        Args:
            full_text: Complete prompt
            target_phrase: Phrase to mask
            
        Returns:
            Binary mask where 1 = replace this token
        """
        if not self.tokenizer:
            return None
        
        # Tokenize full text
        full_tokens = self.tokenizer.encode(full_text)
        
        # Tokenize target phrase
        target_tokens = self.tokenizer.encode(target_phrase)
        
        # Remove special tokens
        if len(target_tokens) > 0:
            if target_tokens[0] == self.tokenizer.bos_token_id:
                target_tokens = target_tokens[1:]
            if target_tokens[-1] == self.tokenizer.eos_token_id:
                target_tokens = target_tokens[:-1]
        
        # Create mask
        max_length = 77  # Standard max length
        mask = mx.zeros(max_length)
        
        # Find target in full tokens
        target_len = len(target_tokens)
        for i in range(len(full_tokens) - target_len + 1):
            if full_tokens[i:i+target_len] == target_tokens:
                mask[i:i+target_len] = 1.0
                break
        
        return mask
    
    def _inject_embeddings(self,
                          encoder_hidden_states: mx.array,
                          block_name: str) -> mx.array:
        """
        Apply embedding injections to encoder hidden states.
        
        This is called during cross-attention to modify the text embeddings.
        
        Args:
            encoder_hidden_states: Original text embeddings
            block_name: Name of current block
            
        Returns:
            Modified embeddings
        """
        result = encoder_hidden_states
        
        for config in self.injections:
            # Check if this block should be injected
            should_inject = False
            for target_block in config.blocks:
                if target_block in block_name:
                    should_inject = True
                    break
            
            if not should_inject:
                continue
            
            # Check timestep range
            if not (config.timestep_range[0] <= self._current_timestep <= config.timestep_range[1]):
                continue
            
            # Get replacement embeddings
            try:
                replacement_embeddings = self._get_embeddings(config.replacement_text)
                
                # Ensure shape compatibility
                if replacement_embeddings.shape != encoder_hidden_states.shape:
                    # Resize if needed (simple approach - may need refinement)
                    if len(replacement_embeddings.shape) == 2:
                        replacement_embeddings = replacement_embeddings[None, :, :]
                    
                    # Truncate or pad to match sequence length
                    seq_len = encoder_hidden_states.shape[1]
                    if replacement_embeddings.shape[1] > seq_len:
                        replacement_embeddings = replacement_embeddings[:, :seq_len, :]
                    elif replacement_embeddings.shape[1] < seq_len:
                        padding = mx.zeros((replacement_embeddings.shape[0], 
                                          seq_len - replacement_embeddings.shape[1],
                                          replacement_embeddings.shape[2]))
                        replacement_embeddings = mx.concatenate([replacement_embeddings, padding], axis=1)
                
                # Apply token mask if available
                if config.token_mask is not None:
                    mask = config.token_mask[None, :, None]  # [1, seq_len, 1]
                    mask = mx.broadcast_to(mask, encoder_hidden_states.shape)
                    
                    # Blend based on mask and weight
                    result = (1 - mask * config.weight) * result + (mask * config.weight) * replacement_embeddings
                else:
                    # Full blending without mask
                    result = (1 - config.weight) * result + config.weight * replacement_embeddings
                    
            except Exception as e:
                print(f"Warning: Failed to inject embeddings: {e}")
                continue
        
        return result
    
    def _create_hooked_transformer(self, original_forward, block_name: str):
        """
        Create a hooked version of a transformer block's forward method.
        
        Args:
            original_forward: Original forward method
            block_name: Name of the block
            
        Returns:
            Hooked forward method
        """
        def hooked_forward(x, memory, attn_mask=None, memory_mask=None):
            # Inject into memory (encoder_hidden_states)
            if memory is not None:
                memory = self._inject_embeddings(memory, block_name)
            
            # Call original forward with modified memory
            return original_forward(x, memory, attn_mask, memory_mask)
        
        return hooked_forward
    
    def install_hooks(self):
        """Install hooks into UNet transformer blocks."""
        if self._hooks_installed:
            return
        
        print("🔧 Installing UNet hooks for embedding injection...")
        
        # Find and hook transformer blocks
        hook_count = 0
        
        # Hook the main transformer blocks in UNet
        for name, module in self.unet.named_modules():
            if isinstance(module, nn.Module):
                # Check if this is a transformer block with cross-attention
                if hasattr(module, 'attn2'):  # Cross-attention layer
                    # Store original forward
                    self._original_forwards[name] = module.__call__
                    
                    # Create and install hooked forward
                    module.__call__ = self._create_hooked_transformer(
                        module.__call__, name
                    )
                    
                    self._hooked_modules[name] = module
                    hook_count += 1
                    print(f"  ✓ Hooked {name}")
        
        print(f"✅ Installed {hook_count} hooks")
        self._hooks_installed = True
    
    def remove_hooks(self):
        """Remove all installed hooks."""
        if not self._hooks_installed:
            return
        
        print("🔧 Removing UNet hooks...")
        
        # Restore original forwards
        for name, module in self._hooked_modules.items():
            if name in self._original_forwards:
                module.__call__ = self._original_forwards[name]
        
        self._original_forwards.clear()
        self._hooked_modules.clear()
        self._hooks_installed = False
        
        print("✅ Hooks removed")
    
    def update_timestep(self, timestep: float):
        """Update current timestep for conditional injection."""
        self._current_timestep = timestep
    
    def clear(self):
        """Clear all injections and caches."""
        self.injections.clear()
        self._embedding_cache.clear()
        self._current_timestep = 0.0
    
    def __enter__(self):
        """Context manager entry."""
        self.install_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.remove_hooks()
        self.clear()


class TrueSemanticInjector:
    """
    High-level interface for TRUE semantic object replacement via embedding injection.
    
    This manipulates embeddings in the UNet, not just text.
    """
    
    def __init__(self, sd_model):
        """
        Initialize the semantic injector.
        
        Args:
            sd_model: The Stable Diffusion model
        """
        self.sd = sd_model
        
        # Create hook manager
        self.hook_manager = UNetHookManager(
            sd_model.unet,
            sd_model.text_encoder,
            sd_model.tokenizer
        )
        
        self.enabled = False
    
    def add_replacement(self, 
                       original: str, 
                       replacement: str,
                       weight: float = 1.0,
                       blocks: Optional[List[str]] = None):
        """
        Add a semantic replacement rule.
        
        This will inject replacement embeddings during UNet forward pass.
        
        Args:
            original: Original object/phrase (e.g., "cat")
            replacement: Replacement object/phrase (e.g., "dog") 
            weight: Injection strength (0-1)
            blocks: Which UNet blocks to inject into
        """
        self.hook_manager.add_injection(
            original_text=original,
            replacement_text=replacement,
            weight=weight,
            blocks=blocks
        )
        print(f"💉 Added embedding injection: {original} → {replacement} (weight={weight})")
    
    def enable(self):
        """Enable embedding injection."""
        self.hook_manager.install_hooks()
        self.enabled = True
        print("🔄 TRUE embedding injection ENABLED")
    
    def disable(self):
        """Disable embedding injection."""
        self.hook_manager.remove_hooks()
        self.enabled = False
        print("⏸️  TRUE embedding injection DISABLED")
    
    def clear(self):
        """Clear all replacement rules."""
        self.hook_manager.clear()
        print("🗑️  Cleared all embedding injections")


def create_true_semantic_injector(model_name: str, **kwargs):
    """
    Create a TRUE semantic injector that manipulates embeddings in UNet.
    
    Args:
        model_name: Name of the model to load
        **kwargs: Additional model loading arguments
        
    Returns:
        TrueSemanticInjector instance
    """
    from adapters.stable_diffusion import StableDiffusion
    
    # Load model
    sd = StableDiffusion(model_name, **kwargs)
    
    # Create injector
    injector = TrueSemanticInjector(sd)
    
    return injector