# CorePulse V4 DataVoid Implementation on MLX

## 🚀 Overview

This repository contains a complete implementation of the CorePulse V4 DataVoid technique for SDXL on Apple Silicon using the MLX framework. The implementation features pre-attention KV manipulation for advanced prompt injection and attention control.

## ✨ Key Features

- **Pre-Attention KV Manipulation**: Modify queries, keys, and values BEFORE attention computation
- **Block-Specific Control**: Individual control over 7 UNet blocks
- **40+ Test Demonstrations**: Comprehensive test suite with various techniques
- **Production Ready**: Fixed scaling issues, clean artifact-free generation
- **MLX Optimized**: Full Metal acceleration on Apple Silicon

## 📁 Repository Structure

```
corpus-mlx/
├── src/adapters/mlx/               # Core implementation
│   └── mlx-examples/stable_diffusion/
│       └── stable_diffusion/
│           ├── attn_mha.py         # PatchedMHA implementation
│           ├── attn_scores.py      # Hook registry system
│           └── unet.py             # Modified UNet with block IDs
│
├── test_*.py                       # 40+ test files
├── fix_injection_scaling.py        # Critical scaling fixes
├── artifacts/images/               # Generated results
│   ├── fixed_scaling/              # Clean working examples
│   ├── embedding_injection/        # Multi-prompt techniques
│   └── [28+ categories]            # Various demonstrations
│
└── *.md                            # Documentation files
```

## 🛠️ Installation

```bash
# Clone the repository
git clone <repo-url>
cd corpus-mlx

# Install dependencies
pip install mlx
pip install -r requirements.txt
```

## 🎯 Quick Start

```python
from stable_diffusion import attn_scores
attn_scores.enable_kv_hooks(True)

from stable_diffusion import StableDiffusionXL

# Load model
model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)

# Create injection hook
def injection_hook(q, k, v, meta=None):
    if k.shape[2] < 100:  # Cross-attention only
        strength = 0.3  # Use 0.1-0.5 range!
        # Your manipulation here
        v_modified = v * (1 - strength) + inject_vals * strength
        return q, k, v_modified
    return q, k, v

# Apply to blocks
attn_scores.KV_REGISTRY.set('mid', injection_hook)
attn_scores.KV_REGISTRY.set('up_0', injection_hook)

# Generate
latents = model.generate_latents("your prompt", num_steps=4)
```

## 📊 Techniques Demonstrated

### Core Capabilities
1. **Embedding Injection** - Multi-prompt blending and replacement
2. **Attention Manipulation** - Suppress/amplify attention weights
3. **Token Masking** - Fine-grained token-level control
4. **Spatial Control** - Regional prompt injection (left/center/right)
5. **Multi-Scale Architecture** - Resolution-aware injection

### Advanced Techniques
6. **Style Mixing** - Blend multiple artistic styles
7. **Concept Fusion** - Create hybrid concepts
8. **Progressive Strength** - Gradual effect increase
9. **Harmonic Resonance** - Frequency-based modulation
10. **Block Sculpting** - Layer-specific modifications

## 🔧 Critical Parameters

### Strength Values (IMPORTANT!)
- **Safe Range**: 0.1 - 0.5
- **Recommended**: 0.2 - 0.3
- **Never Exceed**: 0.5 (causes artifacts)

### Block Selection
- **Early Blocks** (down_0, down_1): Composition, structure
- **Middle Block** (mid): Core content
- **Late Blocks** (up_1, up_2): Style, details

## 📈 Performance

- **Generation Time**: 30-60 seconds (4-6 steps)
- **Memory Usage**: 4-6GB GPU RAM
- **Image Quality**: 512x512 or 1024x1024
- **Hardware**: Apple M1/M2/M3 Silicon

## 🧪 Running Tests

```bash
# Run individual tests
python test_embedding_injection.py
python test_attention_manipulation.py
python test_spatial_control.py

# Run fixed scaling demonstration
python fix_injection_scaling.py

# Run comprehensive showcase
python final_showcase.py
```

## 📸 Results

Generated images are saved to `artifacts/images/` with categories:
- `fixed_scaling/` - Properly scaled examples
- `embedding_injection/` - Multi-prompt techniques
- `attention_manipulation/` - Attention control
- `multi_scale_architecture/` - Scale-aware injection
- And 25+ more categories...

## ⚠️ Important Notes

1. **Always use conservative scaling** (0.1-0.5 range)
2. **Apply hooks before generation** starts
3. **Clear registry between different techniques**
4. **Use middle/late blocks** for best results

## 🐛 Troubleshooting

### Common Issues

1. **Corrupted/Noisy Images**
   - Solution: Reduce strength to 0.1-0.3 range

2. **Broadcasting Errors**
   - Solution: Check tensor dimensions match

3. **Memory Issues**
   - Solution: Use float16 precision, reduce batch size

## 📚 Documentation

- `PROJECT_COMPLETE.md` - Full implementation details
- `COMPLETE_RESULTS.md` - Comprehensive test results
- `COMPREHENSIVE_TEST_SUITE.md` - All test descriptions
- `TRUE_CAPABILITIES.md` - Capability analysis

## 🙏 Credits

- Original CorePulse: https://github.com/DataCTE/CorePulse
- MLX Framework: Apple ML framework
- SDXL-Turbo: Stability AI

## 📄 License

See LICENSE file for details.

---

*Successfully implemented on Apple Silicon - August 2024*