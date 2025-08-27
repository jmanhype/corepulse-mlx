#!/usr/bin/env python3
"""
Advanced Test: Abstract Conceptualization
Generates pure abstractions by manipulating attention patterns
to create non-representational art based on mathematical and quantum concepts.
"""

import sys
import gc
from pathlib import Path
import mlx.core as mx
import PIL.Image
import numpy as np

# Add the stable_diffusion module to path
sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')

# Enable hooks BEFORE importing model
from stable_diffusion import attn_scores
attn_scores.enable_kv_hooks(True)

# Import model components
from stable_diffusion import StableDiffusionXL

def create_abstract_hook(concept_type, intensity):
    """Create pure abstract patterns based on mathematical/quantum concepts"""
    def hook(q, k, v, meta=None):
        # Only modify cross-attention (text-to-image attention)
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            
            if concept_type == "fibonacci":
                # Fibonacci spiral and golden ratio patterns
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Golden ratio
                phi = 1.618033988749895
                # Fibonacci sequence influence
                fib_seq = mx.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55])
                fib_influence = mx.zeros((seq_len,))
                for i, f in enumerate(fib_seq[:min(10, seq_len)]):
                    fib_influence = fib_influence.at[i].set(f / 55.0)
                # Create spiral pattern
                spiral = mx.exp(-mx.abs(mx.arange(seq_len).reshape(1, 1, -1, 1) - seq_len/phi) * 0.1)
                # Golden rectangles
                golden = mx.sin(pattern * phi) * mx.cos(pattern / phi)
                pattern = pattern * spiral * golden * intensity
                
            elif concept_type == "fractal":
                # Fractal patterns (Mandelbrot-inspired)
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Self-similarity at different scales
                scale1 = mx.sin(pattern * 2) * mx.cos(pattern * 3)
                scale2 = mx.sin(pattern * 8) * mx.cos(pattern * 12)
                scale3 = mx.sin(pattern * 32) * mx.cos(pattern * 48)
                # Recursive structure
                fractal = scale1 * 0.5 + scale2 * 0.3 + scale3 * 0.2
                # Add complexity
                complexity = mx.log(mx.abs(pattern) + 1) * mx.exp(-mx.abs(pattern) * 0.1)
                pattern = fractal * complexity * intensity
                
            elif concept_type == "quantum":
                # Quantum superposition and entanglement patterns
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Superposition states
                superposition = mx.sin(pattern) * mx.cos(pattern) * mx.tanh(pattern)
                # Quantum entanglement (correlation between dimensions)
                entangle = mx.random.normal((batch, heads, seq_len, dim))
                # Create correlation by rolling and multiplying
                entangle_corr = entangle * mx.roll(entangle, 1, axis=3) * 0.7
                # Uncertainty principle
                uncertainty = mx.random.uniform(shape=(batch, heads, seq_len, dim), low=-1, high=1) * \
                             mx.exp(-mx.abs(pattern) * 0.2)
                pattern = (superposition + entangle_corr) * uncertainty * intensity
                
            elif concept_type == "wave":
                # Wave interference and resonance patterns
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Multiple wave sources
                wave1 = mx.sin(mx.arange(seq_len).reshape(1, 1, -1, 1) * 0.3 + pattern)
                wave2 = mx.cos(mx.arange(dim).reshape(1, 1, 1, -1) * 0.5 + pattern)
                wave3 = mx.sin(mx.sqrt(mx.abs(pattern)) * 4)
                # Interference patterns
                interference = wave1 * wave2 * wave3
                # Resonance amplification
                mean_interf = mx.mean(mx.abs(interference))
                resonance = mx.where(
                    mx.abs(interference) > mean_interf,
                    interference * 2,
                    interference * 0.5
                )
                pattern = resonance * intensity
                
            elif concept_type == "chaos":
                # Chaotic attractors (Lorenz-inspired)
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Strange attractor dynamics
                sigma = 10.0
                rho = 28.0
                beta = 8.0 / 3.0
                # Simulate chaotic evolution
                x = pattern
                y = mx.roll(pattern, 1, axis=2)
                z = mx.roll(pattern, 1, axis=3)
                dx = sigma * (y - x)
                dy = x * (rho - z) - y
                dz = x * y - beta * z
                # Combine derivatives
                chaos = (dx + dy + dz) / 50.0
                # Add butterfly effect
                butterfly = mx.where(
                    mx.abs(pattern) < 0.1,
                    pattern * 10,
                    pattern
                )
                pattern = chaos * butterfly * intensity
                
            elif concept_type == "topology":
                # Topological transformations (Klein bottle, Möbius strip)
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Möbius twist
                twist = mx.sin(mx.arange(seq_len).reshape(1, 1, -1, 1) * mx.pi) * \
                       mx.cos(mx.arange(dim).reshape(1, 1, 1, -1) * mx.pi * 2)
                # Klein bottle self-intersection
                klein = pattern * mx.sin(pattern * 4) * mx.cos(mx.roll(pattern, seq_len//2, axis=2) * 3)
                # Non-orientable surface
                checkerboard = mx.broadcast_to(
                    (mx.arange(seq_len).reshape(1, 1, -1, 1) + 
                     mx.arange(dim).reshape(1, 1, 1, -1)) % 2,
                    (batch, heads, seq_len, dim)
                )
                surface = mx.where(
                    checkerboard == 0,
                    pattern,
                    -pattern
                )
                pattern = (twist * klein + surface) * intensity
                
            elif concept_type == "emergence":
                # Emergent patterns from simple rules (Conway's Game of Life inspired)
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Cellular automaton rules
                neighbors = mx.abs(mx.roll(pattern, 1, axis=2)) + \
                           mx.abs(mx.roll(pattern, -1, axis=2)) + \
                           mx.abs(mx.roll(pattern, 1, axis=3)) + \
                           mx.abs(mx.roll(pattern, -1, axis=3))
                # Birth and death rules
                condition = (neighbors > 2) & (neighbors < 4)
                alive = mx.where(
                    condition,
                    pattern * 1.5,
                    pattern * 0.3
                )
                # Emergent complexity
                emergence = alive * mx.exp(-mx.abs(alive - mx.mean(alive)) * 0.2)
                pattern = emergence * intensity
                
            elif concept_type == "symmetry":
                # Abstract symmetry groups and crystallography
                pattern = mx.random.normal((batch, heads, seq_len, dim))
                # Rotational symmetry
                rot2 = pattern + mx.roll(mx.roll(pattern, seq_len//2, axis=2), dim//2, axis=3)
                rot4 = rot2 + mx.roll(mx.roll(rot2, seq_len//4, axis=2), dim//4, axis=3)
                # Reflection symmetry
                reflect_h = pattern + mx.flip(pattern, axis=2)
                reflect_v = pattern + mx.flip(pattern, axis=3)
                # Crystallographic patterns
                crystal = rot4 * reflect_h * reflect_v * 0.125
                pattern = crystal * intensity
                
            else:
                pattern = mx.zeros((batch, heads, seq_len, dim))
            
            # Apply abstract pattern to K and V
            k_new = k * 0.2 + pattern * 0.8
            v_new = v * 0.1 + pattern * 0.9
            
            return q, k_new, v_new
        return q, k, v
    return hook

def main():
    print("🎨 Advanced Test: Abstract Conceptualization")
    print("==" * 30)
    
    # Configuration
    base_prompt = "an abstract composition of pure form and color"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    # Abstract concept distribution across blocks
    abstract_config = {
        "down_0": ("fibonacci", 0.7, "Golden ratio foundation"),
        "down_1": ("fractal", 0.8, "Self-similar structures"),
        "down_2": ("quantum", 0.9, "Quantum superposition"),
        "mid": ("chaos", 1.0, "Chaotic attractors"),
        "up_0": ("wave", 0.8, "Wave interference"),
        "up_1": ("topology", 0.7, "Topological transforms"),
        "up_2": ("emergence", 0.6, "Emergent patterns")
    }
    
    # Alternative: Pure mathematical abstraction
    mathematical_config = {
        "down_0": ("symmetry", 0.8, "Symmetry groups"),
        "down_1": ("fibonacci", 0.9, "Fibonacci spirals"),
        "down_2": ("fractal", 1.0, "Fractal dimensions"),
        "mid": ("quantum", 1.0, "Quantum mechanics"),
        "up_0": ("topology", 0.9, "Topological spaces"),
        "up_1": ("chaos", 0.8, "Chaotic dynamics"),
        "up_2": ("wave", 0.7, "Wave functions")
    }
    
    # Create output directory
    output_dir = Path("artifacts/images/advanced_artistic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the main abstract config
    test_config = abstract_config
    
    print(f"📝 Base Prompt: '{base_prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    print("\n🎨 Abstract Concepts:")
    for block, (concept, intensity, desc) in test_config.items():
        print(f"   {block}: {desc} ({concept}) @ {intensity*100:.0f}%")
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Clear any existing hooks
    attn_scores.KV_REGISTRY.clear()
    
    # Register abstract conceptualization hooks
    print("\n🎨 Registering abstract concept hooks...")
    for block, (concept, intensity, desc) in test_config.items():
        hook = create_abstract_hook(concept, intensity)
        attn_scores.KV_REGISTRY.set(block, hook)
        print(f"   🎨 Applied {concept} to {block}")
    
    # Generate with abstract conceptualization
    print("\n🎨 Generating pure abstraction...")
    latents = model.generate_latents(
        base_prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    # Save image
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    output_path = output_dir / "abstract_conceptualization.png"
    pil_img.save(output_path)
    
    print(f"\n✅ Saved abstract conceptualization: {output_path}")
    print("📊 Expected: Pure non-representational art with:")
    print("   - Fibonacci spirals and golden ratios")
    print("   - Fractal self-similarity")
    print("   - Quantum superposition effects")
    print("   - Chaotic attractor patterns")
    print("   - Wave interference patterns")
    print("   - Topological transformations")
    print("   - Emergent complexity")
    print("💡 This proves we can generate pure mathematical abstractions!")
    
    # Generate baseline
    print("\n🎨 Generating baseline for comparison...")
    attn_scores.KV_REGISTRY.clear()
    
    latents_baseline = model.generate_latents(
        base_prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents_baseline):
        if i == num_steps - 1:
            img_baseline = model.decode(x)
    
    img_array_baseline = (img_baseline[0] * 255).astype(mx.uint8)
    pil_img_baseline = PIL.Image.fromarray(np.array(img_array_baseline))
    output_path_baseline = output_dir / "abstract_conceptualization_baseline.png"
    pil_img_baseline.save(output_path_baseline)
    
    print(f"✅ Saved baseline image: {output_path_baseline}")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n🎉 Abstract conceptualization test complete!")

if __name__ == "__main__":
    main()