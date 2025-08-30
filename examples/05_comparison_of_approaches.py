#!/usr/bin/env python3
"""
Example 5: Comparison of Semantic Replacement Approaches

This demonstrates the difference between:
1. Text-level replacement (what corpus-mlx does)
2. Embedding-level injection (what CorePulse does)
"""

from corpus_mlx import create_semantic_wrapper, CorePulseStableDiffusion
from adapters.stable_diffusion import StableDiffusion
from PIL import Image
import mlx.core as mx
import numpy as np


def demonstrate_approaches():
    """Show the difference between text and embedding approaches."""
    
    print("=" * 60)
    print("SEMANTIC REPLACEMENT APPROACHES COMPARISON")
    print("=" * 60)
    
    # Test prompt
    prompt = "a fluffy cat sitting on a red sofa"
    print(f"\nOriginal prompt: '{prompt}'")
    print("\nGoal: Replace 'cat' with 'dog'")
    
    print("\n" + "-" * 60)
    print("APPROACH 1: Text-Level Replacement (corpus-mlx)")
    print("-" * 60)
    
    # Create semantic wrapper
    wrapper = create_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
    
    # Add replacement rule
    wrapper.add_replacement("cat", "dog")
    print("✅ Rule added: cat → dog")
    
    # Enable and generate
    wrapper.enable()
    print("\nWhat happens internally:")
    print("1. Intercept prompt: 'a fluffy cat sitting on a red sofa'")
    print("2. Apply text replacement: 'a fluffy dog sitting on a red sofa'")
    print("3. Send modified text to tokenizer")
    print("4. Generate image of dog")
    
    # Generate with text replacement
    latents = None
    for step_latents in wrapper.wrapper.generate_latents(
        prompt,
        negative_text="blurry, ugly",
        num_steps=15,
        cfg_weight=7.5,
        seed=42,
        height=256,
        width=256
    ):
        latents = step_latents
    
    # Decode and save
    images = wrapper.wrapper.sd.autoencoder.decode(latents)
    img = images[0]
    img = mx.clip(img, -1, 1)
    img = ((img + 1) * 127.5).astype(mx.uint8)
    img_np = np.array(img)
    
    Image.fromarray(img_np).save("approach1_text_replacement.png")
    print("\n✅ Result saved: approach1_text_replacement.png")
    print("   Shows: Complete dog, no cat features")
    
    wrapper.disable()
    
    print("\n" + "-" * 60)
    print("APPROACH 2: Embedding Injection (CorePulse-style)")
    print("-" * 60)
    
    # Create advanced wrapper
    base_sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base")
    sd = CorePulseStableDiffusion(base_sd)
    
    print("\nWhat CorePulse would do:")
    print("1. Tokenize original: 'a fluffy cat sitting on a red sofa'")
    print("2. Get embeddings for 'cat'")
    print("3. During UNet forward pass:")
    print("   - Intercept at cross-attention layers")
    print("   - Inject 'dog' embeddings with weight")
    print("   - Blend: 0.7*cat_embeddings + 0.3*dog_embeddings")
    print("4. Generate image with mixed features")
    
    # Simulate with prompt injection (not true embedding injection)
    sd.add_injection(
        inject_prompt="golden retriever dog fluffy",
        strength=1.0,  # Strong injection
        blocks=["mid", "up_0", "up_1"]
    )
    
    print("\nNote: corpus-mlx currently uses prompt injection,")
    print("not true embedding manipulation like CorePulse.")
    
    # Generate with injection
    latents = None
    for step_latents in sd.generate_latents(
        prompt,  # Original prompt still says "cat"
        negative_text="blurry, ugly",
        num_steps=15,
        cfg_weight=7.5,
        seed=42,
        height=256,
        width=256
    ):
        latents = step_latents
    
    # Decode and save
    images = sd.sd.autoencoder.decode(latents)
    img = images[0]
    img = mx.clip(img, -1, 1)
    img = ((img + 1) * 127.5).astype(mx.uint8)
    img_np = np.array(img)
    
    Image.fromarray(img_np).save("approach2_embedding_injection.png")
    print("\n✅ Result saved: approach2_embedding_injection.png")
    print("   Shows: Mixed features or partial replacement")
    
    print("\n" + "=" * 60)
    print("KEY DIFFERENCES")
    print("=" * 60)
    
    print("\n📝 Text-Level Replacement:")
    print("  ✅ Complete, clean replacement")
    print("  ✅ 100% predictable")
    print("  ✅ Simple to implement")
    print("  ❌ All-or-nothing")
    print("  ❌ Can't preserve original context")
    
    print("\n🧬 Embedding Injection:")
    print("  ✅ Fine-grained control")
    print("  ✅ Can blend features")
    print("  ✅ Preserves context")
    print("  ❌ More complex")
    print("  ❌ Results vary based on weights")
    
    print("\n" + "=" * 60)
    print("WHICH APPROACH IS BETTER?")
    print("=" * 60)
    
    print("\nIt depends on your needs:")
    print("• Want to completely replace an object? → Text replacement")
    print("• Need subtle blending or regional control? → Embedding injection")
    print("• Prototyping quickly? → Text replacement")
    print("• Need production flexibility? → Embedding injection")
    
    print("\nBoth are valid approaches solving different problems!")


def demonstrate_limitations():
    """Show what each approach can and cannot do."""
    
    print("\n" + "=" * 60)
    print("DEMONSTRATING LIMITATIONS")
    print("=" * 60)
    
    print("\n🚫 What Text Replacement CAN'T Do:")
    print("1. Partial replacement (30% cat, 70% dog)")
    print("2. Regional replacement (cat in center only)")
    print("3. Token masking (suppress 'cat', keep 'fluffy')")
    print("4. Gradual transitions over timesteps")
    
    print("\n🚫 What Embedding Injection CAN'T Do (easily):")
    print("1. Guarantee complete replacement")
    print("2. Predictable results every time")
    print("3. Work without UNet modifications")
    print("4. Simple debugging")
    
    print("\n✨ corpus-mlx provides:")
    print("• Text replacement: ✅ Fully working")
    print("• Prompt injection: ✅ Working")
    print("• True embedding injection: 🚧 Framework created, needs UNet hooks")
    
    print("\n" + "=" * 60)
    print("END OF DEMONSTRATION")
    print("=" * 60)


if __name__ == "__main__":
    # Run the comparison
    demonstrate_approaches()
    
    # Show limitations
    demonstrate_limitations()
    
    print("\n📚 For more details, see:")
    print("   • TECHNICAL_COMPARISON.md")
    print("   • SEMANTIC_REPLACEMENT_GUIDE.md")
    print("   • CorePulse repository for embedding injection examples")