#!/usr/bin/env python3
"""Debug KV hooks to see why embedding injection isn't working."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from corpus_mlx import create_true_semantic_wrapper
import sys
sys.path.insert(0, str(Path(__file__).parent / "src" / "adapters" / "mlx" / "mlx-examples" / "stable_diffusion"))
from stable_diffusion import attn_scores

def debug_kv_hooks():
    """Debug KV hook system."""
    print("🔍 DEBUGGING KV HOOKS")
    print("=" * 50)
    
    # Check hook registry
    print(f"KV_HOOKS_ENABLED: {attn_scores.KV_HOOKS_ENABLED}")
    print(f"KV_REGISTRY hooks: {list(attn_scores.KV_REGISTRY._hooks.keys())}")
    
    # Create wrapper
    print("\nCreating TRUE semantic wrapper...")
    wrapper = create_true_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
    
    print(f"\nAfter wrapper creation:")
    print(f"KV_HOOKS_ENABLED: {attn_scores.KV_HOOKS_ENABLED}")
    print(f"KV_REGISTRY hooks: {list(attn_scores.KV_REGISTRY._hooks.keys())}")
    
    # Add replacement
    print("\nAdding replacement...")
    wrapper.add_replacement("cat", "dog", weight=1.0)
    
    print(f"\nAfter adding replacement:")
    print(f"KV_HOOKS_ENABLED: {attn_scores.KV_HOOKS_ENABLED}")
    print(f"KV_REGISTRY hooks: {list(attn_scores.KV_REGISTRY._hooks.keys())}")
    
    # Create a test hook that prints when called
    def debug_hook(q, k, v, meta):
        print(f"🔥 HOOK CALLED! q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")
        return q, k, v
    
    # Add debug hook
    attn_scores.KV_REGISTRY.set("mid", debug_hook)
    print(f"\nAdded debug hook to 'mid':")
    print(f"KV_REGISTRY hooks: {list(attn_scores.KV_REGISTRY._hooks.keys())}")
    
    # Test generation with debug hook
    print("\nTesting generation with debug hook...")
    wrapper.injector.enable_for_prompt("a cat")
    
    try:
        gen = wrapper.sd.generate_latents("a cat", num_steps=2, seed=42)
        next(gen)  # Just one step to see if hooks are called
        next(gen)
        print("✅ Generation completed")
    except StopIteration:
        print("✅ Generation completed (StopIteration)")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_kv_hooks()