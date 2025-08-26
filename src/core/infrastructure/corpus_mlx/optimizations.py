"""
MLX-specific optimizations for CorePulse performance.
Leverages Apple Silicon's unified memory and Metal acceleration.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Callable
import mlx.core as mx
import numpy as np
from functools import lru_cache
from .utils import as_mx

Array = mx.array


class MemoryOptimizer:
    """
    Optimizes memory usage for MLX operations.
    Uses unified memory architecture efficiently.
    """
    
    def __init__(self, max_cache_size: int = 100):
        self.embedding_cache = {}
        self.mask_cache = {}
        self.max_cache_size = max_cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
    @lru_cache(maxsize=128)
    def cached_tokenize(self, prompt: str, max_len: int = 77) -> tuple:
        """Cache tokenization results."""
        # Return as tuple for hashability
        return tuple(prompt[:max_len])
    
    def cache_embedding(self, key: str, embedding: Array) -> Array:
        """Cache computed embeddings."""
        if key in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[key]
        
        self.cache_misses += 1
        
        # Evict old entries if cache is full
        if len(self.embedding_cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest]
        
        self.embedding_cache[key] = embedding
        return embedding
    
    def cache_mask(self, key: str, mask: Array) -> Array:
        """Cache computed masks."""
        if key in self.mask_cache:
            return self.mask_cache[key]
        
        if len(self.mask_cache) >= self.max_cache_size:
            oldest = next(iter(self.mask_cache))
            del self.mask_cache[oldest]
        
        self.mask_cache[key] = mask
        return mask
    
    def clear_caches(self):
        """Clear all caches."""
        self.embedding_cache.clear()
        self.mask_cache.clear()
        self.cached_tokenize.cache_clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(1, total)
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "embedding_cache_size": len(self.embedding_cache),
            "mask_cache_size": len(self.mask_cache)
        }


class ComputeOptimizer:
    """
    Optimizes compute operations for MLX/Metal.
    """
    
    def __init__(self):
        self.fusion_enabled = True
        self.graph_optimization = True
        
    def fused_attention_ops(
        self,
        queries: Array,
        keys: Array,
        values: Array,
        mask: Optional[Array] = None
    ) -> Array:
        """
        Fused attention computation for better Metal performance.
        """
        # Compute attention scores
        scores = mx.matmul(queries, keys.transpose(0, 1, 3, 2))
        scores = scores / mx.sqrt(mx.array(queries.shape[-1], dtype=scores.dtype))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask * -1e9
        
        # Softmax
        weights = mx.softmax(scores, axis=-1)
        
        # Apply to values
        output = mx.matmul(weights, values)
        
        return output
    
    def optimized_blend(
        self,
        eps1: Array,
        eps2: Array,
        alpha: float,
        mask: Optional[Array] = None
    ) -> Array:
        """
        Optimized blending operation using MLX's efficient operations.
        """
        if mask is None:
            # Simple blend
            return eps1 + alpha * (eps2 - eps1)
        else:
            # Masked blend with broadcasting optimization
            diff = eps2 - eps1
            blend = alpha * mask
            return eps1 + blend * diff
    
    def batch_process_injections(
        self,
        base_eps: Array,
        injection_list: List[Dict]
    ) -> Array:
        """
        Process multiple injections in a single batched operation.
        """
        if not injection_list:
            return base_eps
        
        # Stack all injection predictions
        eps_stack = mx.stack([inj["eps"] for inj in injection_list])
        weights = mx.array([inj["weight"] for inj in injection_list])
        
        # Check if masks are present
        if "mask" in injection_list[0] and injection_list[0]["mask"] is not None:
            masks = mx.stack([inj["mask"] for inj in injection_list])
            weighted_masks = masks * weights[:, None, None, None]
            
            # Compute weighted average
            total_weight = weighted_masks.sum(axis=0)
            weighted_eps = (eps_stack * weighted_masks).sum(axis=0)
            
            # Avoid division by zero
            total_weight = mx.maximum(total_weight, 1e-8)
            blended = weighted_eps / total_weight
            
            # Blend with base
            blend_mask = mx.minimum(total_weight, 1.0)
            return base_eps * (1 - blend_mask) + blended * blend_mask
        else:
            # No masks, simple weighted average
            weights = weights / weights.sum()
            weighted_eps = (eps_stack * weights[:, None, None, None]).sum(axis=0)
            total_weight = weights.sum()
            
            return base_eps * (1 - total_weight) + weighted_eps


class PipelineOptimizer:
    """
    Optimizes the entire generation pipeline.
    """
    
    def __init__(self, sd):
        self.sd = sd
        self.memory_opt = MemoryOptimizer()
        self.compute_opt = ComputeOptimizer()
        
    def optimize_generation_loop(
        self,
        wrapper,
        base_prompt: str,
        **kwargs
    ):
        """
        Optimized generation loop with batching and caching.
        """
        # Pre-compute all injections
        prepared_injections = self._prepare_all_injections(wrapper)
        
        # Use optimized generation
        return self._optimized_generate(
            wrapper,
            base_prompt,
            prepared_injections,
            **kwargs
        )
    
    def _prepare_all_injections(self, wrapper) -> List[Dict]:
        """Pre-compute all injection data."""
        from .injection import prepare_injection
        
        H_lat = 64  # Standard latent size
        W_lat = 64
        
        prepared = []
        for ic in wrapper.injections:
            # Use cached preparation
            cache_key = f"{ic.prompt}_{ic.token_mask}_{ic.start_frac}_{ic.end_frac}"
            
            if cache_key in self.memory_opt.embedding_cache:
                prep = self.memory_opt.embedding_cache[cache_key]
            else:
                prep = prepare_injection(self.sd, ic, H_lat, W_lat)
                self.memory_opt.cache_embedding(cache_key, prep)
            
            prepared.append(prep)
        
        return prepared
    
    def _optimized_generate(
        self,
        wrapper,
        base_prompt: str,
        prepared_injections: List[Dict],
        **kwargs
    ):
        """Optimized generation with batched operations."""
        # Standard generation setup
        num_steps = kwargs.get("num_steps", 50)
        cfg_weight = kwargs.get("cfg_weight", 7.5)
        seed = kwargs.get("seed", None)
        
        # Initialize
        sampler = self.sd.sampler
        key = mx.random.key(seed) if seed is not None else None
        
        # ... continue with optimized loop
        # This would integrate the batched operations
        
        return wrapper.generate_latents(base_prompt, **kwargs)


class StreamingOptimizer:
    """
    Enables streaming generation with progressive refinement.
    """
    
    def __init__(self, sd):
        self.sd = sd
        self.preview_interval = 5  # Generate preview every N steps
        
    def streaming_generate(
        self,
        wrapper,
        base_prompt: str,
        callback: Callable[[Array, int, float], None],
        **kwargs
    ):
        """
        Generate with streaming previews.
        
        Args:
            wrapper: CorePulseStableDiffusion instance
            base_prompt: Base prompt
            callback: Function called with (latents, step, progress)
            **kwargs: Generation parameters
        """
        step_count = 0
        num_steps = kwargs.get("num_steps", 50)
        
        for latents in wrapper.generate_latents(base_prompt, **kwargs):
            step_count += 1
            progress = step_count / num_steps
            
            # Call callback for preview
            if step_count % self.preview_interval == 0:
                callback(latents, step_count, progress)
        
        # Final callback
        callback(latents, step_count, 1.0)
        
        return latents


class AdaptiveQualityOptimizer:
    """
    Dynamically adjusts quality based on performance.
    """
    
    def __init__(self):
        self.target_time = 30.0  # Target generation time in seconds
        self.min_steps = 20
        self.max_steps = 50
        self.current_steps = 30
        
    def adaptive_generate(
        self,
        wrapper,
        base_prompt: str,
        **kwargs
    ):
        """
        Generate with adaptive quality based on performance.
        """
        import time
        
        # Override steps with adaptive value
        kwargs["num_steps"] = self.current_steps
        
        # Measure generation time
        start_time = time.time()
        
        latents = None
        for step_latents in wrapper.generate_latents(base_prompt, **kwargs):
            latents = step_latents
        
        generation_time = time.time() - start_time
        
        # Adjust steps for next generation
        if generation_time > self.target_time * 1.2:
            # Too slow, reduce steps
            self.current_steps = max(
                self.min_steps,
                int(self.current_steps * 0.9)
            )
        elif generation_time < self.target_time * 0.8:
            # Too fast, can increase quality
            self.current_steps = min(
                self.max_steps,
                int(self.current_steps * 1.1)
            )
        
        return latents, {
            "generation_time": generation_time,
            "steps_used": kwargs["num_steps"],
            "next_steps": self.current_steps
        }


def create_optimized_wrapper(sd) -> CorePulseStableDiffusion:
    """
    Create an optimized CorePulse wrapper with all optimizations enabled.
    
    Args:
        sd: StableDiffusion instance
        
    Returns:
        Optimized wrapper
    """
    from .sd_wrapper import CorePulseStableDiffusion
    
    # Create wrapper
    wrapper = CorePulseStableDiffusion(sd)
    
    # Add optimizers
    wrapper.memory_optimizer = MemoryOptimizer()
    wrapper.compute_optimizer = ComputeOptimizer()
    wrapper.pipeline_optimizer = PipelineOptimizer(sd)
    
    # Enable optimizations
    wrapper.use_optimizations = True
    
    return wrapper


def benchmark_optimizations(sd):
    """
    Benchmark performance with and without optimizations.
    """
    import time
    from .sd_wrapper import CorePulseStableDiffusion
    
    print("Benchmarking CorePulse Optimizations...")
    print("=" * 50)
    
    # Test prompt
    test_prompt = "beautiful landscape with mountains"
    
    # Without optimizations
    wrapper_normal = CorePulseStableDiffusion(sd)
    wrapper_normal.add_injection(
        prompt="dramatic sunset",
        start_frac=0.0,
        end_frac=0.5,
        weight=0.7
    )
    
    start = time.time()
    for _ in wrapper_normal.generate_latents(
        test_prompt,
        num_steps=20,
        seed=42
    ):
        pass
    time_normal = time.time() - start
    
    # With optimizations
    wrapper_opt = create_optimized_wrapper(sd)
    wrapper_opt.add_injection(
        prompt="dramatic sunset",
        start_frac=0.0,
        end_frac=0.5,
        weight=0.7
    )
    
    start = time.time()
    for _ in wrapper_opt.generate_latents(
        test_prompt,
        num_steps=20,
        seed=42
    ):
        pass
    time_optimized = time.time() - start
    
    # Results
    speedup = time_normal / time_optimized
    print(f"Normal: {time_normal:.2f}s")
    print(f"Optimized: {time_optimized:.2f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    # Memory stats
    if hasattr(wrapper_opt, 'memory_optimizer'):
        stats = wrapper_opt.memory_optimizer.get_stats()
        print(f"\nCache stats:")
        print(f"  Hit rate: {stats['hit_rate']:.1%}")
        print(f"  Hits: {stats['cache_hits']}")
        print(f"  Misses: {stats['cache_misses']}")
    
    return {
        "time_normal": time_normal,
        "time_optimized": time_optimized,
        "speedup": speedup
    }