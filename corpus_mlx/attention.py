"""
Advanced attention manipulation for CorePulse.
Provides per-block control and cross-attention redistribution.
"""

from __future__ import annotations
from typing import Dict, Optional, List, Tuple
import mlx.core as mx
import numpy as np
from .utils import as_mx

Array = mx.array

class AttentionController:
    """
    Controls attention weights at different UNet blocks.
    Allows fine-grained manipulation of cross-attention layers.
    """
    
    def __init__(self, sd):
        self.sd = sd
        self.attention_scales: Dict[str, float] = {}
        self.attention_shifts: Dict[str, Array] = {}
        self.cross_attention_maps: Dict[str, Array] = {}
        self.block_names = self._identify_blocks()
        
        # DataVoid parameters
        self.void_threshold = 0.25  # Below this = potential hallucination
        self.amplification_factor = 2.5  # Product attention boost
        self.redistribution_rate = 0.7  # How aggressively to redirect
        
    def _identify_blocks(self) -> List[str]:
        """Identify UNet blocks with attention layers."""
        blocks = []
        
        # Standard UNet structure
        # Down blocks
        for i in range(4):
            blocks.append(f"down_{i}")
            
        # Mid block
        blocks.append("mid")
        
        # Up blocks
        for i in range(4):
            blocks.append(f"up_{i}")
            
        return blocks
    
    def set_attention_scale(self, block_pattern: str, scale: float):
        """
        Set attention scaling for blocks matching pattern.
        
        Args:
            block_pattern: Block name or pattern (e.g., "down_*", "mid", "up_2")
            scale: Multiplication factor for attention weights
        """
        if "*" in block_pattern:
            # Apply to all matching blocks
            prefix = block_pattern.replace("*", "")
            for block in self.block_names:
                if block.startswith(prefix):
                    self.attention_scales[block] = scale
        else:
            self.attention_scales[block_pattern] = scale
    
    def set_attention_shift(self, block: str, shift: Array):
        """
        Add bias to attention weights in specific block.
        
        Args:
            block: Block name
            shift: Additive bias array
        """
        self.attention_shifts[block] = as_mx(shift)
    
    def apply_datavoid(
        self,
        void_positions: List[int],
        product_positions: List[int],
        blocks: Optional[List[str]] = None
    ):
        """
        Apply DataVoid technique: suppress voids, amplify products.
        Core insight: "Attention is zero-sum. Take from hallucination, give to truth."
        
        Args:
            void_positions: Token indices where hallucinations occur
            product_positions: Token indices for product/truth
            blocks: Specific blocks to apply to (None = all)
        """
        target_blocks = blocks if blocks else self.block_names
        
        for block in target_blocks:
            if block not in self.cross_attention_maps:
                self.cross_attention_maps[block] = {}
            
            self.cross_attention_maps[block]["datavoid"] = {
                "void_positions": void_positions,
                "product_positions": product_positions,
                "void_suppression": 0.1,  # Reduce by 90%
                "product_amplification": self.amplification_factor,
                "redistribution_rate": self.redistribution_rate
            }
    
    def redistribute_attention(
        self,
        from_tokens: List[int],
        to_tokens: List[int],
        redistribution_weight: float = 0.5,
        blocks: Optional[List[str]] = None
    ):
        """
        Redistribute attention from one set of tokens to another.
        
        Args:
            from_tokens: Source token indices
            to_tokens: Target token indices
            redistribution_weight: Amount to redistribute (0.0-1.0)
            blocks: Specific blocks to apply to (None = all)
        """
        target_blocks = blocks if blocks else self.block_names
        
        for block in target_blocks:
            if block not in self.cross_attention_maps:
                self.cross_attention_maps[block] = {}
                
            self.cross_attention_maps[block]["redistribution"] = {
                "from": from_tokens,
                "to": to_tokens,
                "weight": redistribution_weight
            }
    
    def apply_attention_control(
        self,
        attention_weights: Array,
        block_name: str,
        token_embeddings: Optional[Array] = None
    ) -> Array:
        """
        Apply attention manipulations to given weights.
        
        Args:
            attention_weights: Original attention weights [batch, heads, seq, seq]
            block_name: Current UNet block
            token_embeddings: Optional token embeddings for advanced control
            
        Returns:
            Modified attention weights
        """
        weights = attention_weights
        
        # Apply DataVoid technique first if configured
        if block_name in self.cross_attention_maps:
            datavoid = self.cross_attention_maps[block_name].get("datavoid")
            if datavoid:
                weights = self._apply_datavoid_technique(
                    weights,
                    datavoid["void_positions"],
                    datavoid["product_positions"],
                    datavoid["void_suppression"],
                    datavoid["product_amplification"],
                    datavoid["redistribution_rate"]
                )
        
        # Apply scaling
        if block_name in self.attention_scales:
            scale = self.attention_scales[block_name]
            weights = weights * scale
            
        # Apply shifting
        if block_name in self.attention_shifts:
            shift = self.attention_shifts[block_name]
            weights = weights + shift
            
        # Apply redistribution
        if block_name in self.cross_attention_maps:
            redis = self.cross_attention_maps[block_name].get("redistribution")
            if redis:
                weights = self._redistribute_weights(
                    weights,
                    redis["from"],
                    redis["to"],
                    redis["weight"]
                )
        
        # Renormalize
        weights = self._renormalize_attention(weights)
        
        return weights
    
    def _apply_datavoid_technique(
        self,
        weights: Array,
        void_positions: List[int],
        product_positions: List[int],
        void_suppression: float,
        product_amplification: float,
        redistribution_rate: float
    ) -> Array:
        """
        Apply DataVoid technique: Create voids where hallucinations occur,
        fill them with product information.
        
        Zero-entropy principle: Attention stolen from voids must go to products.
        """
        # Step 1: Calculate total attention in voids before suppression
        void_attention_sum = mx.sum(weights[..., void_positions])
        
        # Step 2: Create voids (suppress hallucination-prone areas)
        for void_pos in void_positions:
            weights[..., void_pos] = weights[..., void_pos] * void_suppression
        
        # Step 3: Calculate attention to redistribute
        suppressed_attention = void_attention_sum * (1 - void_suppression) * redistribution_rate
        
        # Step 4: Amplify product positions
        for product_pos in product_positions:
            # Apply amplification
            weights[..., product_pos] = weights[..., product_pos] * product_amplification
            # Add redistributed attention from voids
            weights[..., product_pos] = weights[..., product_pos] + (
                suppressed_attention / len(product_positions)
            )
        
        return weights
    
    def _redistribute_weights(
        self,
        weights: Array,
        from_tokens: List[int],
        to_tokens: List[int],
        amount: float
    ) -> Array:
        """Redistribute attention weights between token sets."""
        # Get attention going to source tokens
        from_attention = weights[..., from_tokens].mean(axis=-1, keepdims=True)
        
        # Scale down source attention
        weights[..., from_tokens] = weights[..., from_tokens] * (1 - amount)
        
        # Add to target tokens
        redistribution = from_attention * amount / len(to_tokens)
        for tok_idx in to_tokens:
            weights[..., tok_idx] = weights[..., tok_idx] + redistribution.squeeze(-1)
            
        return weights
    
    def _renormalize_attention(self, weights: Array) -> Array:
        """Renormalize attention weights to sum to 1."""
        return weights / weights.sum(axis=-1, keepdims=True)


class PerBlockInjectionController:
    """
    Controls prompt injection at specific UNet blocks.
    Enables different prompts to affect different resolution stages.
    """
    
    def __init__(self, sd):
        self.sd = sd
        self.block_injections: Dict[str, List[Dict]] = {}
        
    def add_block_injection(
        self,
        prompt: str,
        blocks: List[str],
        weight: float = 1.0,
        token_mask: Optional[str] = None
    ):
        """
        Add injection that only applies at specific blocks.
        
        Args:
            prompt: Text to inject
            blocks: List of block names
            weight: Injection strength
            token_mask: Optional token focusing
        """
        injection = {
            "prompt": prompt,
            "weight": weight,
            "token_mask": token_mask,
            "embedding": None  # Will be prepared later
        }
        
        for block in blocks:
            if block not in self.block_injections:
                self.block_injections[block] = []
            self.block_injections[block].append(injection)
    
    def prepare_embeddings(self):
        """Pre-compute embeddings for all block injections."""
        from .injection import encode_tokens
        
        for block, injections in self.block_injections.items():
            for inj in injections:
                if inj["embedding"] is None:
                    tokens = encode_tokens(self.sd, inj["prompt"], True, True)
                    embedding = self.sd.text_encoder(tokens).last_hidden_state
                    
                    # Apply token mask if specified
                    if inj["token_mask"]:
                        embedding = self._apply_token_mask(
                            embedding,
                            tokens,
                            inj["token_mask"]
                        )
                    
                    inj["embedding"] = embedding
    
    def get_block_embedding(
        self,
        block_name: str,
        base_embedding: Array
    ) -> Array:
        """
        Get modified embedding for specific block.
        
        Args:
            block_name: Current block
            base_embedding: Original embedding
            
        Returns:
            Modified embedding for this block
        """
        if block_name not in self.block_injections:
            return base_embedding
            
        result = base_embedding
        
        for inj in self.block_injections[block_name]:
            if inj["embedding"] is not None:
                # Blend embeddings
                alpha = inj["weight"]
                result = (1 - alpha) * result + alpha * inj["embedding"]
                
        return result
    
    def _apply_token_mask(
        self,
        embedding: Array,
        tokens: Array,
        mask_text: str
    ) -> Array:
        """Apply token masking to embedding."""
        # Simple implementation - zero out non-masked tokens
        # In practice, would match mask_text to token positions
        return embedding  # TODO: Implement proper token matching


class MultiScaleController:
    """
    Controls generation at multiple resolution scales.
    Enables coarse-to-fine prompt control.
    """
    
    def __init__(self, sd):
        self.sd = sd
        self.scale_configs: List[Dict] = []
        
    def add_scale_config(
        self,
        prompt: str,
        resolution_scale: float,
        start_frac: float,
        end_frac: float,
        weight: float = 1.0
    ):
        """
        Add prompt configuration for specific resolution scale.
        
        Args:
            prompt: Text prompt
            resolution_scale: Resolution multiplier (0.25, 0.5, 1.0)
            start_frac: When to start (0.0-1.0)
            end_frac: When to end (0.0-1.0)
            weight: Blend weight
        """
        self.scale_configs.append({
            "prompt": prompt,
            "resolution_scale": resolution_scale,
            "start_frac": start_frac,
            "end_frac": end_frac,
            "weight": weight
        })
    
    def get_scale_embedding(
        self,
        progress: float,
        current_resolution: Tuple[int, int],
        base_resolution: Tuple[int, int]
    ) -> Optional[Array]:
        """
        Get embedding for current resolution and progress.
        
        Args:
            progress: Current diffusion progress (0.0-1.0)
            current_resolution: Current latent resolution
            base_resolution: Target latent resolution
            
        Returns:
            Blended embedding for this scale/time
        """
        from .injection import encode_tokens
        
        # Calculate current scale
        current_scale = current_resolution[0] / base_resolution[0]
        
        # Find matching configs
        active_configs = []
        for config in self.scale_configs:
            # Check if resolution matches
            if abs(config["resolution_scale"] - current_scale) < 0.1:
                # Check if time window is active
                if config["start_frac"] <= progress <= config["end_frac"]:
                    active_configs.append(config)
        
        if not active_configs:
            return None
            
        # Blend active embeddings
        result = None
        total_weight = 0.0
        
        for config in active_configs:
            tokens = encode_tokens(self.sd, config["prompt"], True, True)
            embedding = self.sd.text_encoder(tokens).last_hidden_state
            
            if result is None:
                result = embedding * config["weight"]
            else:
                result = result + embedding * config["weight"]
                
            total_weight += config["weight"]
        
        if total_weight > 0:
            result = result / total_weight
            
        return result


class DataVoidController:
    """
    Specialized controller for DataVoid technique.
    Automatically detects voids and amplifies products.
    """
    
    def __init__(self, sd):
        self.sd = sd
        self.attention_controller = AttentionController(sd)
        
        # DataVoid configuration
        self.config = {
            "amplification_factor": 2.5,
            "void_threshold": 0.25,
            "product_weight": 0.85,
            "redistribution_rate": 0.7,
            "cross_attention_scale": 2.0
        }
        
    def identify_void_positions(self, attention_weights: Array) -> List[int]:
        """
        Identify positions where attention is below void threshold.
        These are potential hallucination zones.
        """
        # Average attention across batch and heads
        avg_attention = mx.mean(attention_weights, axis=(0, 1))
        # Find positions below threshold
        void_mask = avg_attention < self.config["void_threshold"]
        void_positions = mx.where(void_mask)[0].tolist()
        return void_positions
    
    def identify_product_positions(self, prompt: str, product_keywords: List[str]) -> List[int]:
        """
        Identify token positions corresponding to products.
        """
        # This would use tokenizer to find product token positions
        # Placeholder implementation
        product_positions = []
        for keyword in product_keywords:
            # In real implementation, would tokenize and find positions
            # For now, return example positions
            product_positions.extend([5, 6, 7])  # Example positions
        return list(set(product_positions))
    
    def apply_datavoid_to_prompt(
        self,
        prompt: str,
        product_keywords: List[str],
        blocks: Optional[List[str]] = None
    ):
        """
        Configure DataVoid for a prompt with product keywords.
        """
        # Get product positions
        product_positions = self.identify_product_positions(prompt, product_keywords)
        
        # Void positions will be detected dynamically during generation
        # For now, use heuristic: positions far from products
        all_positions = list(range(77))  # Max token length
        void_positions = [p for p in all_positions 
                         if p not in product_positions 
                         and min([abs(p - pp) for pp in product_positions]) > 3]
        
        # Apply DataVoid
        self.attention_controller.apply_datavoid(
            void_positions=void_positions[:20],  # Limit voids
            product_positions=product_positions,
            blocks=blocks
        )
    
    def validate_product_presence(self, generated_content: str, required_products: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that all required products are present in generated content.
        """
        missing_products = []
        for product in required_products:
            if product.lower() not in generated_content.lower():
                missing_products.append(product)
        return len(missing_products) == 0, missing_products


def create_attention_hooks(sd, attention_controller: AttentionController):
    """
    Create hooks to intercept and modify attention during generation.
    
    Args:
        sd: StableDiffusion instance
        attention_controller: Configured AttentionController
        
    Returns:
        List of hook functions
    """
    hooks = []
    
    def attention_hook(module, block_name):
        def hook_fn(x):
            # This would need integration with MLX's module system
            # For now, return identity
            return x
        return hook_fn
    
    # Would register hooks on actual attention modules
    # This is a placeholder for the integration point
    
    return hooks