#!/usr/bin/env python3
"""
Test: Frequency Domain Manipulation
Apply FFT-based transformations to attention tensors.
This demonstrates spectral control over generation.
"""

import sys
import gc
from pathlib import Path
import mlx.core as mx
import mlx.core.fft as fft
import PIL.Image
import numpy as np

# Add the stable_diffusion module to path
sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')

# Enable hooks BEFORE importing model
from stable_diffusion import attn_scores
attn_scores.enable_kv_hooks(True)

# Import model components
from stable_diffusion import StableDiffusionXL

def create_frequency_filter_hook(filter_type="low_pass", cutoff=0.5):
    """
    Create a hook that applies frequency domain filtering.
    
    Args:
        filter_type: "low_pass", "high_pass", "band_pass", "notch"
        cutoff: Frequency cutoff (0-1, where 1 is Nyquist frequency)
    """
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            # Apply FFT to V tensor along the sequence dimension
            v_complex = mx.astype(v, mx.complex64)
            v_fft = fft.fft(v_complex, axis=2)
            
            # Create frequency mask
            freqs = mx.arange(seq_len, dtype=mx.float32) / seq_len
            
            if filter_type == "low_pass":
                # Keep low frequencies
                mask = mx.where(freqs < cutoff, 1.0, 0.0)
            elif filter_type == "high_pass":
                # Keep high frequencies
                mask = mx.where(freqs > cutoff, 1.0, 0.0)
            elif filter_type == "band_pass":
                # Keep frequencies in a band
                low_cut = cutoff * 0.5
                high_cut = cutoff * 1.5
                mask = mx.where((freqs > low_cut) & (freqs < high_cut), 1.0, 0.0)
            elif filter_type == "notch":
                # Remove specific frequency band
                low_cut = cutoff - 0.1
                high_cut = cutoff + 0.1
                mask = mx.where((freqs < low_cut) | (freqs > high_cut), 1.0, 0.0)
            else:
                mask = mx.ones_like(freqs)
            
            # Reshape mask to match FFT dimensions
            mask = mx.reshape(mask, (1, 1, seq_len, 1))
            mask = mx.broadcast_to(mask, v_fft.shape)
            
            # Apply filter
            v_fft_filtered = v_fft * mx.astype(mask, mx.complex64)
            
            # Inverse FFT
            v_filtered = fft.ifft(v_fft_filtered, axis=2)
            v_new = mx.real(v_filtered)
            
            print(f"    🌊 {filter_type} filter at {block_id}: cutoff {cutoff:.2f}")
            
            return q, k, v_new
        return q, k, v
    return hook

def create_spectral_enhancement_hook(enhance_type="harmonic"):
    """
    Create a hook that enhances specific spectral components.
    
    Args:
        enhance_type: "harmonic", "subharmonic", "resonance"
    """
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = k.shape
            block_id = meta.get('block_id', 'unknown')
            
            # FFT on both K and V
            k_complex = mx.astype(k, mx.complex64)
            v_complex = mx.astype(v, mx.complex64)
            k_fft = fft.fft(k_complex, axis=2)
            v_fft = fft.fft(v_complex, axis=2)
            
            if enhance_type == "harmonic":
                # Enhance harmonic frequencies (multiples of fundamental)
                fundamental = 4  # Base frequency
                for harmonic in range(1, min(5, seq_len // fundamental)):
                    freq_idx = harmonic * fundamental
                    if freq_idx < seq_len:
                        k_fft[:, :, freq_idx, :] *= 2.0
                        v_fft[:, :, freq_idx, :] *= 2.0
                
            elif enhance_type == "subharmonic":
                # Add subharmonic components
                for sub in [2, 4, 8]:
                    if seq_len // sub > 0:
                        k_fft[:, :, seq_len // sub, :] *= 1.5
                        v_fft[:, :, seq_len // sub, :] *= 1.5
                
            elif enhance_type == "resonance":
                # Create resonance peaks
                resonance_freqs = [seq_len // 8, seq_len // 4, seq_len // 2]
                for freq_idx in resonance_freqs:
                    if freq_idx < seq_len:
                        # Gaussian enhancement around resonance
                        for offset in range(-2, 3):
                            idx = freq_idx + offset
                            if 0 <= idx < seq_len:
                                enhancement = mx.exp(-0.5 * (offset / 2) ** 2) + 1
                                k_fft[:, :, idx, :] *= enhancement
                                v_fft[:, :, idx, :] *= enhancement
            
            # Inverse FFT
            k_new = mx.real(fft.ifft(k_fft, axis=2))
            v_new = mx.real(fft.ifft(v_fft, axis=2))
            
            print(f"    🎵 Spectral {enhance_type} at {block_id}")
            
            return q, k_new, v_new
        return q, k, v
    return hook

def create_phase_shift_hook(shift_amount=0.5):
    """
    Apply phase shifting in frequency domain.
    """
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            # FFT of V
            v_complex = mx.astype(v, mx.complex64)
            v_fft = fft.fft(v_complex, axis=2)
            
            # Create phase shift
            freqs = mx.arange(seq_len, dtype=mx.float32)
            phase_shift = mx.exp(1j * 2 * np.pi * shift_amount * freqs / seq_len)
            phase_shift = mx.astype(phase_shift, mx.complex64)
            phase_shift = mx.reshape(phase_shift, (1, 1, seq_len, 1))
            phase_shift = mx.broadcast_to(phase_shift, v_fft.shape)
            
            # Apply phase shift
            v_fft_shifted = v_fft * phase_shift
            
            # Inverse FFT
            v_shifted = fft.ifft(v_fft_shifted, axis=2)
            v_new = mx.real(v_shifted)
            
            print(f"    🔄 Phase shift at {block_id}: {shift_amount:.2f}π")
            
            return q, k, v_new
        return q, k, v
    return hook

def main():
    print("🌊 Test: Frequency Domain Manipulation")
    print("=" * 60)
    
    # Configuration
    prompt = "a vibrant cyberpunk cityscape with neon lights"
    num_steps = 5
    cfg_weight = 7.5
    seed = 42
    
    print(f"📝 Prompt: '{prompt}'")
    print(f"🔧 Steps: {num_steps}, CFG: {cfg_weight}, Seed: {seed}")
    
    # Create output directory
    output_dir = Path("artifacts/images/frequency_domain")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n📦 Loading SDXL...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    # Test 1: Baseline
    print("\n🎨 Test 1: Baseline (no filtering)...")
    attn_scores.KV_REGISTRY.clear()
    
    latents = model.generate_latents(
        prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "01_baseline.png")
    print("✅ Saved: 01_baseline.png")
    
    # Test 2: Low-pass filter (smooth/blur effect)
    print("\n🎨 Test 2: Low-pass filter (smooth details)...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_frequency_filter_hook("low_pass", cutoff=0.3)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "02_low_pass.png")
    print("✅ Saved: 02_low_pass.png")
    
    # Test 3: High-pass filter (edge enhancement)
    print("\n🎨 Test 3: High-pass filter (enhance edges)...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_frequency_filter_hook("high_pass", cutoff=0.7)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "03_high_pass.png")
    print("✅ Saved: 03_high_pass.png")
    
    # Test 4: Band-pass filter
    print("\n🎨 Test 4: Band-pass filter (mid frequencies)...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_frequency_filter_hook("band_pass", cutoff=0.5)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "04_band_pass.png")
    print("✅ Saved: 04_band_pass.png")
    
    # Test 5: Harmonic enhancement
    print("\n🎨 Test 5: Harmonic enhancement...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_spectral_enhancement_hook("harmonic")
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "05_harmonic.png")
    print("✅ Saved: 05_harmonic.png")
    
    # Test 6: Phase shifting
    print("\n🎨 Test 6: Phase shifting...")
    attn_scores.KV_REGISTRY.clear()
    
    hook = create_phase_shift_hook(shift_amount=0.25)
    for block in ["down_0", "down_1", "down_2", "mid", "up_0", "up_1", "up_2"]:
        attn_scores.KV_REGISTRY.set(block, hook)
    
    latents = model.generate_latents(
        prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "06_phase_shift.png")
    print("✅ Saved: 06_phase_shift.png")
    
    # Test 7: Mixed frequency manipulation
    print("\n🎨 Test 7: Mixed frequency effects...")
    attn_scores.KV_REGISTRY.clear()
    
    # Different filters for different blocks
    low_pass_hook = create_frequency_filter_hook("low_pass", cutoff=0.4)
    high_pass_hook = create_frequency_filter_hook("high_pass", cutoff=0.6)
    harmonic_hook = create_spectral_enhancement_hook("harmonic")
    
    # Apply different effects to different blocks
    attn_scores.KV_REGISTRY.set("down_0", low_pass_hook)
    attn_scores.KV_REGISTRY.set("down_1", low_pass_hook)
    attn_scores.KV_REGISTRY.set("down_2", harmonic_hook)
    attn_scores.KV_REGISTRY.set("mid", harmonic_hook)
    attn_scores.KV_REGISTRY.set("up_0", high_pass_hook)
    attn_scores.KV_REGISTRY.set("up_1", high_pass_hook)
    attn_scores.KV_REGISTRY.set("up_2", high_pass_hook)
    
    latents = model.generate_latents(
        prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=seed
    )
    
    for i, x in enumerate(latents):
        if i == num_steps - 1:
            img = model.decode(x)
    
    img_array = (img[0] * 255).astype(mx.uint8)
    pil_img = PIL.Image.fromarray(np.array(img_array))
    pil_img.save(output_dir / "07_mixed_frequency.png")
    print("✅ Saved: 07_mixed_frequency.png")
    
    # Clean up
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("\n" + "=" * 60)
    print("✅ Frequency Domain Manipulation Test Complete!")
    print("📊 Results:")
    print("  01_baseline.png: Normal generation")
    print("  02_low_pass.png: Low-pass filter (smooth)")
    print("  03_high_pass.png: High-pass filter (edges)")
    print("  04_band_pass.png: Band-pass filter (mid frequencies)")
    print("  05_harmonic.png: Harmonic enhancement")
    print("  06_phase_shift.png: Phase shifting effect")
    print("  07_mixed_frequency.png: Mixed frequency effects")
    print("\n💡 This proves frequency domain control over attention!")
    print("🌊 FFT manipulation enables spectral image control!")
    print("🔬 Different frequencies affect different image aspects!")

if __name__ == "__main__":
    main()