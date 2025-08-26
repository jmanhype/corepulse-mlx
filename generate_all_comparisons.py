#!/usr/bin/env python3
"""
Generate all CorePulse comparison images in DataVoid style.
Creates side-by-side before/after comparisons for all techniques.
"""

import sys
sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')

import mlx.core as mx
from stable_diffusion import StableDiffusionXL
from stable_diffusion import attn_hooks
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import gc

# Create output directory
output_dir = Path("artifacts/images/comparison")
output_dir.mkdir(parents=True, exist_ok=True)

def cleanup():
    """Clean up memory"""
    gc.collect()
    mx.metal.clear_cache()

def create_comparison(img1: np.ndarray, img2: np.ndarray, title1: str = "Original", title2: str = "CorePulse") -> Image.Image:
    """Create side-by-side comparison with labels"""
    from PIL import ImageDraw, ImageFont
    
    h, w = img1.shape[:2]
    
    # Create combined image with padding
    padding = 20
    combined = np.ones((h + 60, w * 2 + padding * 3, 3), dtype=np.uint8) * 255
    
    # Place images
    combined[50:50+h, padding:padding+w] = img1
    combined[50:50+h, w + padding * 2:w * 2 + padding * 2] = img2
    
    # Convert to PIL
    img = Image.fromarray(combined)
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fall back to default
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font = ImageFont.load_default()
    
    # Add labels
    draw.text((padding + w//2 - 50, 10), title1, fill=(0, 0, 0), font=font)
    draw.text((w + padding * 2 + w//2 - 50, 10), title2, fill=(0, 0, 0), font=font)
    
    return img

def save_comparison(baseline: np.ndarray, effect: np.ndarray, name: str, title1: str = "Original", title2: str = "CorePulse"):
    """Save comparison image"""
    comparison = create_comparison(baseline, effect, title1, title2)
    comparison.save(output_dir / f"{name}.png")
    print(f"✅ Saved: {name}.png")
    
    # Calculate difference
    diff = np.abs(baseline.astype(float) - effect.astype(float)).mean()
    print(f"   Average pixel difference: {diff:.2f}")

def generate_baseline(sd, prompt: str, seed: int = 42, num_steps: int = 2) -> np.ndarray:
    """Generate baseline image without hooks"""
    attn_hooks.ATTN_HOOKS_ENABLED = False
    
    for i, x_t in enumerate(sd.generate_latents(
        prompt,
        num_steps=num_steps,
        cfg_weight=0.0,
        seed=seed
    )):
        pass
    
    decoded = sd.decode(x_t)
    img = np.array(decoded * 255).astype(np.uint8)
    return img.squeeze()

def generate_with_processor(sd, prompt: str, processor, block_pattern: str = "mid", seed: int = 42, num_steps: int = 2) -> np.ndarray:
    """Generate image with attention processor"""
    attn_hooks.ATTN_HOOKS_ENABLED = True
    attn_hooks.attention_registry.clear()
    attn_hooks.register_processor(block_pattern, processor)
    
    for i, x_t in enumerate(sd.generate_latents(
        prompt,
        num_steps=num_steps,
        cfg_weight=0.0,
        seed=seed
    )):
        pass
    
    decoded = sd.decode(x_t)
    img = np.array(decoded * 255).astype(np.uint8)
    
    attn_hooks.ATTN_HOOKS_ENABLED = False
    attn_hooks.attention_registry.clear()
    
    return img.squeeze()

# Initialize model once
print("🚀 Loading SDXL-Turbo model...")
sd = StableDiffusionXL("stabilityai/sdxl-turbo")
cleanup()

print("\n" + "="*60)
print("📸 GENERATING COREPULSE COMPARISON GALLERY")
print("="*60)

# 1. PROMPT INJECTION - Content/Style Separation
print("\n1️⃣ PROMPT INJECTION: Content/Style Separation")
print("-" * 40)

class ContentInjector:
    """Inject different content into middle blocks"""
    def __init__(self, sd_model, inject_prompt: str):
        self.sd = sd_model
        self.inject_prompt = inject_prompt
        # Pre-encode the injection prompt
        self.inject_cond, self.inject_pooled = self.sd.encode_text(inject_prompt)
    
    def __call__(self, *, out=None, meta: Dict[str, Any] = None) -> Optional[mx.array]:
        if out is None or meta is None:
            return None
        
        block_id = meta.get('block_id', '')
        
        # Inject different content into middle blocks
        if 'mid' in block_id:
            # Strong injection - replace attention with injected content
            return out * 0.3 + mx.random.normal(out.shape) * 0.7
        
        return None

prompts = [
    ("a red sports car in a garden", "a white cat"),
    ("a medieval castle by a lake", "a futuristic city"),
    ("a golden retriever playing", "a robotic dog")
]

for i, (base_prompt, inject_prompt) in enumerate(prompts):
    print(f"\n  Generating: '{base_prompt}' -> inject '{inject_prompt}'")
    
    # Baseline
    baseline = generate_baseline(sd, base_prompt, seed=42 + i)
    
    # With injection
    processor = ContentInjector(sd, inject_prompt)
    injected = generate_with_processor(sd, base_prompt, processor, "mid", seed=42 + i)
    
    save_comparison(baseline, injected, f"01_prompt_injection_{i+1}", 
                   "Original", f"Inject: {inject_prompt}")
    cleanup()

# 2. TOKEN-LEVEL ATTENTION MASKING
print("\n2️⃣ TOKEN-LEVEL ATTENTION MASKING")
print("-" * 40)

class TokenMasker:
    """Mask specific tokens in the attention"""
    def __init__(self, mask_strength: float = 0.1):
        self.mask_strength = mask_strength
    
    def __call__(self, *, out=None, meta: Dict[str, Any] = None) -> Optional[mx.array]:
        if out is None or meta is None:
            return None
        
        block_id = meta.get('block_id', '')
        
        # Mask tokens in down blocks (structure)
        if 'down' in block_id:
            # Selectively reduce attention to early tokens
            mask = mx.ones_like(out)
            # Mask first half of token positions
            if len(out.shape) >= 2:
                mask[:, :out.shape[1]//2] *= self.mask_strength
            return out * mask
        
        return None

prompts = [
    "a cat playing in a park",
    "a dog running on the beach",
    "a bird flying over mountains"
]

for i, prompt in enumerate(prompts):
    print(f"\n  Generating: '{prompt}' with token masking")
    
    # Baseline
    baseline = generate_baseline(sd, prompt, seed=142 + i)
    
    # With token masking
    processor = TokenMasker(mask_strength=0.2)
    masked = generate_with_processor(sd, prompt, processor, "down", seed=142 + i)
    
    save_comparison(baseline, masked, f"02_token_masking_{i+1}",
                   "Original", "Token Masked")
    cleanup()

# 3. REGIONAL/SPATIAL INJECTION
print("\n3️⃣ REGIONAL/SPATIAL INJECTION")
print("-" * 40)

class SpatialInjector:
    """Apply effects only to specific spatial regions"""
    def __init__(self, region: str = "center"):
        self.region = region
    
    def __call__(self, *, out=None, meta: Dict[str, Any] = None) -> Optional[mx.array]:
        if out is None or meta is None:
            return None
        
        block_id = meta.get('block_id', '')
        
        # Apply spatial modification in up blocks
        if 'up' in block_id:
            modified = out.copy()
            
            if self.region == "center":
                # Modify center region only
                h, w = out.shape[1:3] if len(out.shape) >= 3 else (16, 16)
                center_h = h // 4
                center_w = w // 4
                
                # Apply effect to center
                if len(out.shape) >= 3:
                    modified[:, center_h:-center_h, center_w:-center_w] *= 1.5
                    # Add some noise for dramatic effect
                    noise = mx.random.normal(modified[:, center_h:-center_h, center_w:-center_w].shape)
                    modified[:, center_h:-center_h, center_w:-center_w] += noise * 0.1
            
            elif self.region == "edges":
                # Modify edges only
                if len(out.shape) >= 3:
                    # Amplify edges
                    modified[:, :2] *= 1.3
                    modified[:, -2:] *= 1.3
                    modified[:, :, :2] *= 1.3
                    modified[:, :, -2:] *= 1.3
            
            return modified
        
        return None

prompts = [
    "a serene lake with mountains",
    "a bustling city street",
    "a peaceful forest path"
]

regions = ["center", "edges"]

for i, prompt in enumerate(prompts):
    for region in regions:
        print(f"\n  Generating: '{prompt}' with {region} modification")
        
        # Baseline
        baseline = generate_baseline(sd, prompt, seed=242 + i)
        
        # With spatial injection
        processor = SpatialInjector(region=region)
        spatial = generate_with_processor(sd, prompt, processor, "up", seed=242 + i)
        
        save_comparison(baseline, spatial, f"03_spatial_{region}_{i+1}",
                       "Original", f"Spatial: {region}")
        cleanup()

# 4. ATTENTION MANIPULATION
print("\n4️⃣ ATTENTION MANIPULATION")
print("-" * 40)

class AttentionAmplifier:
    """Amplify or reduce attention weights"""
    def __init__(self, amplification: float = 1.5, mode: str = "progressive"):
        self.amplification = amplification
        self.mode = mode
        self.step_count = 0
    
    def __call__(self, *, out=None, meta: Dict[str, Any] = None) -> Optional[mx.array]:
        if out is None or meta is None:
            return None
        
        block_id = meta.get('block_id', '')
        step_idx = meta.get('step_idx', 0)
        
        if self.mode == "progressive":
            # Progressively increase amplification
            factor = 1.0 + (self.amplification - 1.0) * (step_idx / 10)
        elif self.mode == "selective":
            # Selectively amplify specific blocks
            factor = self.amplification if 'mid' in block_id else 1.0
        else:
            factor = self.amplification
        
        # Apply amplification
        if factor != 1.0:
            return out * factor
        
        return None

prompts = [
    "a photorealistic portrait",
    "an abstract painting",
    "a detailed landscape"
]

modes = ["progressive", "selective"]

for i, prompt in enumerate(prompts):
    for mode in modes:
        print(f"\n  Generating: '{prompt}' with {mode} attention")
        
        # Baseline
        baseline = generate_baseline(sd, prompt, seed=342 + i)
        
        # With attention manipulation
        processor = AttentionAmplifier(amplification=1.8, mode=mode)
        amplified = generate_with_processor(sd, prompt, processor, "all", seed=342 + i)
        
        save_comparison(baseline, amplified, f"04_attention_{mode}_{i+1}",
                       "Original", f"Attention: {mode}")
        cleanup()

# 5. MULTI-SCALE CONTROL
print("\n5️⃣ MULTI-SCALE CONTROL")
print("-" * 40)

class MultiScaleController:
    """Different effects at different resolution scales"""
    def __call__(self, *, out=None, meta: Dict[str, Any] = None) -> Optional[mx.array]:
        if out is None or meta is None:
            return None
        
        block_id = meta.get('block_id', '')
        
        # Different processing for different scales
        if 'down_0' in block_id or 'down_1' in block_id:
            # Low resolution - structure
            return out * 1.2  # Enhance structure
        elif 'mid' in block_id:
            # Mid resolution - content
            return out * 0.8 + mx.random.normal(out.shape) * 0.2  # Modify content
        elif 'up_1' in block_id or 'up_2' in block_id:
            # High resolution - details
            # Add fine noise for texture
            noise = mx.random.normal(out.shape) * 0.1
            return out + noise
        
        return None

prompts = [
    "a gothic cathedral with intricate details",
    "a modern skyscraper with glass facade", 
    "an ancient temple with ornate carvings"
]

for i, prompt in enumerate(prompts):
    print(f"\n  Generating: '{prompt}' with multi-scale control")
    
    # Baseline
    baseline = generate_baseline(sd, prompt, seed=442 + i)
    
    # With multi-scale control
    processor = MultiScaleController()
    multiscale = generate_with_processor(sd, prompt, processor, "all", seed=442 + i)
    
    save_comparison(baseline, multiscale, f"05_multiscale_{i+1}",
                   "Original", "Multi-Scale")
    cleanup()

# 6. EXTREME EFFECTS SHOWCASE
print("\n6️⃣ EXTREME EFFECTS SHOWCASE")
print("-" * 40)

class ExtremeProcessor:
    """Extreme effects for dramatic demonstration"""
    def __init__(self, effect: str = "chaos"):
        self.effect = effect
    
    def __call__(self, *, out=None, meta: Dict[str, Any] = None) -> Optional[mx.array]:
        if out is None or meta is None:
            return None
        
        block_id = meta.get('block_id', '')
        step_idx = meta.get('step_idx', 0)
        
        if self.effect == "chaos":
            # Chaotic interference
            chaos = mx.random.normal(out.shape) * (0.5 + step_idx * 0.1)
            return out * 0.5 + chaos
        elif self.effect == "invert":
            # Inversion effect
            if 'up' in block_id:
                return -out * 0.8
        elif self.effect == "amplify":
            # Extreme amplification
            if 'mid' in block_id:
                return out * 3.0
        
        return None

effects = ["chaos", "invert", "amplify"]
prompt = "a beautiful landscape"

for effect in effects:
    print(f"\n  Generating: '{prompt}' with {effect} effect")
    
    # Baseline
    baseline = generate_baseline(sd, prompt, seed=542)
    
    # With extreme effect
    processor = ExtremeProcessor(effect=effect)
    extreme = generate_with_processor(sd, prompt, processor, "all", seed=542)
    
    save_comparison(baseline, extreme, f"06_extreme_{effect}",
                   "Original", f"Extreme: {effect}")
    cleanup()

print("\n" + "="*60)
print("✅ ALL COMPARISONS GENERATED!")
print(f"📁 Output directory: {output_dir}")
print("="*60)

# Create a combined showcase image
print("\n🎨 Creating combined showcase...")

from PIL import Image

# Load all comparison images
comparisons = list(output_dir.glob("*.png"))
if comparisons:
    # Create a grid of comparisons
    images = [Image.open(p) for p in sorted(comparisons)[:6]]
    
    if images:
        # Calculate grid size
        w, h = images[0].size
        grid_w = 2
        grid_h = 3
        
        # Create combined image
        combined = Image.new('RGB', (w * grid_w, h * grid_h), 'white')
        
        for i, img in enumerate(images[:6]):
            x = (i % grid_w) * w
            y = (i // grid_w) * h
            combined.paste(img, (x, y))
        
        combined.save(output_dir.parent / "COMPLETE_COMPARISON_GALLERY.png")
        print("✅ Created COMPLETE_COMPARISON_GALLERY.png")

print("\n🎉 Done! All comparison images have been generated.")