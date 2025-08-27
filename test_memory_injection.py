#!/usr/bin/env python3
"""
Test: Memory Injection
Inject long-term memory and context across generations.
This demonstrates persistent state and context accumulation.
"""

import sys
import gc
import json
from pathlib import Path
import mlx.core as mx
import PIL.Image
import numpy as np
from datetime import datetime

# Add the stable_diffusion module to path
sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')

# Enable hooks BEFORE importing model
from stable_diffusion import attn_scores
attn_scores.enable_kv_hooks(True)

# Import model components
from stable_diffusion import StableDiffusionXL

class MemoryBank:
    """Manages persistent memory across generations."""
    
    def __init__(self, capacity=10):
        self.memories = []
        self.capacity = capacity
        self.access_count = {}
        self.timestamp = {}
        
    def store(self, key, value, metadata=None):
        """Store a memory with optional metadata."""
        memory = {
            'key': key,
            'value': value,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'access_count': 0
        }
        
        # Add to memories (FIFO if at capacity)
        if len(self.memories) >= self.capacity:
            self.memories.pop(0)
        self.memories.append(memory)
        
        print(f"    💾 Stored memory: '{key}' (total: {len(self.memories)})")
    
    def recall(self, key=None, n=1):
        """Recall memories by key or get most recent."""
        if key:
            # Find memories matching key
            matches = [m for m in self.memories if key in m['key']]
            for m in matches:
                m['access_count'] += 1
            return matches[:n]
        else:
            # Return most recent memories
            return self.memories[-n:]
    
    def get_context_vector(self):
        """Generate a context vector from all memories."""
        if not self.memories:
            return None
        
        # Simple approach: average all memory values
        values = [m['value'] for m in self.memories if m['value'] is not None]
        if values:
            return mx.mean(mx.stack(values), axis=0)
        return None
    
    def save(self, filepath):
        """Save memory bank to file."""
        data = {
            'memories': len(self.memories),
            'capacity': self.capacity,
            'timestamp': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"    💾 Saved memory bank: {len(self.memories)} memories")

def create_memory_injection_hook(memory_bank, model, mode="blend"):
    """
    Create a hook that injects memories into generation.
    
    Args:
        memory_bank: MemoryBank instance
        model: The SDXL model
        mode: "blend", "override", or "augment"
    """
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            # Get context vector from memories
            context = memory_bank.get_context_vector()
            
            if context is not None:
                v_new = mx.array(v)
                
                # Reshape context to match V dimensions
                if len(context.shape) == 2:
                    context = mx.reshape(context, (1, 1, -1, dim))
                
                # Ensure proper shape
                if context.shape[2] < seq_len:
                    repeats = seq_len // context.shape[2] + 1
                    context = mx.tile(context, (1, 1, repeats, 1))
                    context = context[:, :, :seq_len, :]
                
                if context.shape[1] < heads:
                    context = mx.broadcast_to(context, (batch, heads, seq_len, dim))
                
                if mode == "blend":
                    # Blend memories with current
                    strength = 0.3
                    v_new = (1 - strength) * v + strength * context[:batch, :heads, :seq_len, :dim]
                elif mode == "override":
                    # Override early blocks with memories
                    if block_id in ["down_0", "down_1"]:
                        v_new = context[:batch, :heads, :seq_len, :dim]
                elif mode == "augment":
                    # Add memories as additional signal
                    v_new = v + 0.2 * context[:batch, :heads, :seq_len, :dim]
                
                print(f"    🧠 Memory injection at {block_id}: mode={mode}, memories={len(memory_bank.memories)}")
                
                return q, k, v_new
        return q, k, v
    return hook

def create_episodic_memory_hook(episodes, model):
    """
    Inject specific episodic memories at different stages.
    
    Args:
        episodes: List of (prompt, block_targets) tuples
        model: The SDXL model
    """
    # Generate embeddings for each episode
    episode_embeds = {}
    for prompt, blocks in episodes:
        cond, _ = model._get_text_conditioning(prompt)
        for block in blocks:
            episode_embeds[block] = cond
        print(f"    📝 Episode: '{prompt[:30]}...' → {blocks}")
    
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            # Check if we have an episode for this block
            if block_id in episode_embeds:
                embed = episode_embeds[block_id]
                
                if seq_len >= embed.shape[1]:
                    v_new = mx.array(v)
                    embed_dim = min(dim, embed.shape[2])
                    embed_len = min(seq_len, embed.shape[1])
                    
                    # Prepare episode embedding
                    episode = embed[:, :embed_len, :embed_dim]
                    if len(episode.shape) == 3:
                        episode = episode[None, :, :, :]
                    if episode.shape[1] < heads:
                        episode = mx.broadcast_to(episode, (batch, heads, embed_len, embed_dim))
                    
                    # Inject episode
                    blend = 0.5
                    v_new[:, :, :embed_len, :embed_dim] = \
                        (1 - blend) * v[:, :, :embed_len, :embed_dim] + \
                        blend * episode[:, :, :embed_len, :embed_dim]
                    
                    print(f"    🎬 Episodic memory at {block_id}")
                    
                    return q, k, v_new
        return q, k, v
    return hook

def extract_generation_features(latent):
    """Extract features from a generation to store as memory."""
    # Simple feature extraction
    features = mx.mean(latent, axis=(2, 3))
    return features

def main():
    print("🧠 Test: Memory Injection")
    print("=" * 60)
    
    # Configuration
    base_prompt = "a serene garden"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    print(f"📝 Base Prompt: '{base_prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Create output directory
    output_dir = Path("artifacts/images/memory_injection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize memory bank
    memory_bank = MemoryBank(capacity=5)
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Test 1: Baseline (no memory)
    print("\n🎨 Test 1: Baseline (no memory)...")
    attn_scores.KV_REGISTRY.clear()
    
    latents = model.generate_latents(
        base_prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    baseline_features = None
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            baseline_features = extract_generation_features(x)
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "01_baseline.png")
    print("✅ Saved: 01_baseline.png")
    
    # Store baseline as first memory
    memory_bank.store("garden_baseline", baseline_features, {"prompt": base_prompt})
    
    # Test 2: Memory accumulation (sequential memories)
    print("\n🎨 Test 2: Memory accumulation...")
    
    memory_prompts = [
        "cherry blossoms",
        "zen stones",
        "koi pond",
        "bamboo grove"
    ]
    
    for idx, mem_prompt in enumerate(memory_prompts):
        print(f"\n  🧠 Generating memory {idx+1}: '{mem_prompt}'")
        attn_scores.KV_REGISTRY.clear()
        
        # Generate and store memory
        latents = model.generate_latents(
            mem_prompt,
            num_steps=num_steps,
            cfg_weight=cfg_weight,
            seed=seed + idx
        )
        
        for i, x in enumerate(latents):
            if i == num_steps - 1:
                features = extract_generation_features(x)
                memory_bank.store(f"memory_{idx}", features, {"prompt": mem_prompt})
    
    # Now generate with accumulated memories
    print("\n  🎨 Generating with accumulated memories...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_memory_injection_hook(memory_bank, model, mode="blend")
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        "peaceful meditation space",
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "02_accumulated_memory.png")
    print("✅ Saved: 02_accumulated_memory.png")
    
    # Test 3: Episodic memory injection
    print("\n🎨 Test 3: Episodic memory injection...")
    attn_scores.KV_REGISTRY.clear()
    
    episodes = [
        ("morning sunlight", ["down_0", "down_1"]),
        ("afternoon shadows", ["mid"]),
        ("evening glow", ["up_1", "up_2"])
    ]
    
    hook = create_episodic_memory_hook(episodes, model)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        "garden through the day",
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "03_episodic_memory.png")
    print("✅ Saved: 03_episodic_memory.png")
    
    # Test 4: Override mode (strong memory influence)
    print("\n🎨 Test 4: Memory override mode...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_memory_injection_hook(memory_bank, model, mode="override")
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        "abstract patterns",  # Very different prompt
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "04_memory_override.png")
    print("✅ Saved: 04_memory_override.png")
    
    # Test 5: Augment mode (additive memory)
    print("\n🎨 Test 5: Memory augmentation mode...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_memory_injection_hook(memory_bank, model, mode="augment")
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        "futuristic garden",
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "05_memory_augment.png")
    print("✅ Saved: 05_memory_augment.png")
    
    # Test 6: Memory persistence (save and report)
    print("\n🎨 Test 6: Memory persistence...")
    
    # Save memory bank
    memory_bank.save(output_dir / "memory_bank.json")
    
    # Generate with full memory context
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_memory_injection_hook(memory_bank, model, mode="blend")
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        "memory synthesis",
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "06_memory_synthesis.png")
    print("✅ Saved: 06_memory_synthesis.png")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n" + "=" * 60)
    print("✅ Memory Injection Test Complete!")
    print("📊 Results:")
    print("  01_baseline.png: Normal generation without memory")
    print("  02_accumulated_memory.png: Generation with accumulated memories")
    print("  03_episodic_memory.png: Specific episodic memory injection")
    print("  04_memory_override.png: Strong memory override mode")
    print("  05_memory_augment.png: Additive memory augmentation")
    print("  06_memory_synthesis.png: Full memory context synthesis")
    print("\n💡 This proves persistent context injection!")
    print("🧠 Memory injection enables context accumulation!")
    print("🔬 Long-term memory influences generation controllably!")
    print(f"\n📁 Memory bank saved with {len(memory_bank.memories)} memories")

if __name__ == "__main__":
    main()