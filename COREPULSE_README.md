# CorePulse V4 for MLX - Clean Implementation

A production-ready, upstream-friendly implementation of CorePulse attention manipulation for MLX Stable Diffusion with zero regression when disabled.

## 🎯 Key Features

### Clean Hook Architecture
- **Zero regression** when disabled (proven with identical output)
- **Opt-in activation** via `ATTN_HOOKS_ENABLED` flag (default: False)
- **Protocol-based processors** for type safety
- **No monkey-patching** - proper seam integration

### Working Effects Gallery

#### 🏰 Castle Enhancement
![Working Castle](artifacts/images/readme/WORKING_baseline.png) ![Working Castle CorePulse](artifacts/images/readme/WORKING_corepulse.png)
*Left: Baseline | Right: CorePulse with phase-based enhancement (25.30 avg pixel difference)*

#### 🌲 Dreamlike Forest
![Dreamlike Baseline](artifacts/images/readme/SHOWCASE_DREAMLIKE_baseline.png) ![Dreamlike Effect](artifacts/images/readme/SHOWCASE_DREAMLIKE_effect.png)
*Soft, dreamlike atmosphere through attention smoothing in up blocks (27.69 avg difference)*

#### 🏗️ Geometric Architecture
![Geometric Baseline](artifacts/images/readme/SHOWCASE_GEOMETRIC_baseline.png) ![Geometric Effect](artifacts/images/readme/SHOWCASE_GEOMETRIC_effect.png)
*Enhanced geometric structure with edge amplification (49.34 avg difference)*

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/corpus-mlx.git
cd corpus-mlx

# Add to Python path
export PYTHONPATH=$PYTHONPATH:src/adapters/mlx/mlx-examples/stable_diffusion
```

## 🚀 Quick Start

### Basic Usage - Identity Processor (Zero Regression)

```python
from stable_diffusion import StableDiffusionXL
from stable_diffusion import attn_hooks

# Load model
sd = StableDiffusionXL("stabilityai/sdxl-turbo")

# Define identity processor (no modification)
def identity_processor(*, out=None, meta=None):
    return None  # Keep original

# Enable hooks
attn_hooks.ATTN_HOOKS_ENABLED = True

# Register processor
attn_hooks.register_processor("mid", identity_processor)

# Generate - output will be identical to baseline
for latents in sd.generate_latents("a castle", num_steps=10):
    pass
```

### Advanced Usage - Phase-Based Processing

```python
class PhaseProcessor:
    def __call__(self, *, out=None, meta: Dict[str, Any] = None):
        if out is None or meta is None:
            return None
        
        block_id = meta.get('block_id', '')
        step_idx = meta.get('step_idx', 0)
        sigma = meta.get('sigma', 1.0)
        
        # Phase-based modification
        progress = step_idx / total_steps
        
        if progress < 0.3:  # Early - Structure
            if "down" in block_id:
                return out * 1.2
        elif progress < 0.7:  # Middle - Content
            if "mid" in block_id:
                return out * 1.3
        else:  # Late - Style
            if "up" in block_id:
                noise = mx.random.normal(out.shape) * 0.05
                return out + noise
        
        return None
```

## 🎨 Available Processors

### DreamlikeProcessor
Softens and blurs attention for ethereal effects:
- Reduces high-frequency components in up blocks
- Creates soft, dreamlike atmosphere
- Average difference: ~27 pixels

### IntensityProcessor
Progressive amplification for dramatic effect:
- Increases attention strength over time
- Targets mid and down blocks
- Average difference: ~9-10 pixels

### AbstractProcessor
Adds controlled chaos for artistic interpretation:
- Applies noise to later blocks
- Warps middle attention with cosine modulation
- Average difference: ~36 pixels

### GeometricProcessor
Enhances edges and geometric structure:
- Amplifies strong signals, suppresses weak ones
- Targets down blocks for structural enhancement
- Average difference: ~49 pixels

### ColorShiftProcessor
Dynamic color modulation through attention:
- Sinusoidal modulation in up blocks
- Phase-based color shifting
- Average difference: ~22 pixels

## 🔬 Technical Details

### Hook Integration Points

```
UNet Architecture:
├── down_blocks (0, 1, 2) - Structure/Composition
│   └── TransformerBlocks → attention hooks
├── mid_block - Content/Subject
│   └── TransformerBlock → attention hooks
└── up_blocks (0, 1, 2) - Style/Details
    └── TransformerBlocks → attention hooks
```

### Metadata Available to Processors

```python
meta = {
    'block_id': str,     # e.g., "down_0", "mid", "up_2"
    'layer_idx': int,    # Layer within block
    'step_idx': int,     # Current denoising step
    'sigma': float       # Current noise level
}
```

### Performance Characteristics

- **Hooks Disabled**: Zero overhead (identical to vanilla)
- **Hooks Enabled**: ~144 hook calls per generation (2 steps)
- **Processing Time**: < 1ms per hook call
- **Memory Impact**: Negligible

## 🧪 Validation

### Zero Regression Test

```bash
python test_hooks.py
```

Output:
```
🧪 TESTING COREPULSE HOOKS
1️⃣ Baseline (hooks disabled)
2️⃣ With identity hooks
3️⃣ Parity check: Images identical: ✅ YES
🎉 SUCCESS: Zero regression confirmed!
```

### Effect Showcase Generation

```bash
python corepulse_showcase.py
```

Generates comparison images for all effect types with measured pixel differences.

## 🏗️ Architecture

### Clean Separation of Concerns

```
stable_diffusion/
├── attn_hooks.py      # Attention processor system
│   ├── ATTN_HOOKS_ENABLED (global flag)
│   ├── AttentionProcessor (protocol)
│   └── attention_registry (registry)
├── sigma_hooks.py     # Sigma observation system
│   └── SigmaObserver (protocol)
└── unet.py           # Modified with hook calls
    └── TransformerBlock.__call__ → process_attention()
```

### Protocol-Based Design

```python
class AttentionProcessor(Protocol):
    def __call__(self, *, out=None, meta: Dict[str, Any]) -> Optional[mx.array]:
        """Process attention output."""
        ...
```

## 📊 Results Summary

| Effect | Avg Pixel Difference | Visual Impact |
|--------|---------------------|---------------|
| Dreamlike | 27.69 | Soft, ethereal atmosphere |
| Intensity | 9.45 | Subtle but noticeable enhancement |
| Abstract | 36.45 | Strong artistic transformation |
| Geometric | 49.34 | Dramatic structural changes |
| ColorShift | 22.11 | Moderate color variations |

## 🤝 Contributing

This implementation follows upstream-friendly principles:
- No monkey-patching
- Opt-in by default
- Zero regression when disabled
- Clear protocol interfaces
- Comprehensive documentation

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- MLX team for the excellent framework
- Stable Diffusion community for inspiration
- CorePulse original concept by DataVoid

---

**Note**: This is a clean-room implementation designed for upstream integration with MLX Stable Diffusion. All hooks are disabled by default for zero regression.