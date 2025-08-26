# CorePulse × MLX API Documentation

## Overview
CorePulse is an advanced prompt injection system for MLX Stable Diffusion that provides fine-grained control over the generation process through time-windowed, token-masked, and spatially-controlled prompt injections.

## Core Features

### 1. Time-Windowed Prompt Injection
Control when prompts influence generation during the diffusion process.

```python
wrapper.add_injection(
    prompt="dark mysterious atmosphere",
    start_frac=0.0,    # Start at beginning (0%)
    end_frac=0.3,      # End at 30% through generation
    weight=0.7         # Blend weight (0.0-1.0)
)
```

**Parameters:**
- `prompt`: Text to inject
- `start_frac`: Start time (0.0 = beginning, 1.0 = end)
- `end_frac`: End time
- `weight`: Blending strength (higher = stronger influence)

### 2. Token-Level Attention Masking
Focus generation on specific tokens within prompts.

```python
wrapper.add_injection(
    prompt="glowing magical crystal orb energy",
    token_mask="crystal orb",  # Focus on these tokens
    weight=0.8
)
```

**Parameters:**
- `token_mask`: String or list of token indices to emphasize
- Masks zero out non-selected tokens in embedding space

### 3. Regional/Spatial Injection
Apply prompts to specific image regions.

```python
# Rectangle in pixel coordinates
wrapper.add_injection(
    prompt="fire flames",
    region=("rect_pix", x, y, width, height, feather),
    weight=0.6
)

# Circle in pixel coordinates
wrapper.add_injection(
    prompt="ice crystal",
    region=("circle_pix", cx, cy, radius, feather),
    weight=0.6
)

# Rectangle in fractional coordinates (0.0-1.0)
wrapper.add_injection(
    prompt="forest",
    region=("rect_frac", fx, fy, fw, fh, feather_frac),
    weight=0.5
)
```

**Region Formats:**
- `rect_pix`: Rectangle in pixels (x, y, width, height, feather_px)
- `circle_pix`: Circle in pixels (center_x, center_y, radius, feather_px)
- `rect_frac`: Rectangle in fractions (fx, fy, fw, fh, feather_frac)
- `circle_frac`: Circle in fractions (fcx, fcy, fr, feather_frac)

### 4. Combined Features
All features can be combined for complex control:

```python
wrapper.add_injection(
    prompt="glowing energy",
    token_mask="energy",           # Token focus
    region=("rect_pix", 128, 128, 256, 256, 20),  # Spatial
    start_frac=0.3,                # Time window start
    end_frac=0.7,                  # Time window end
    weight=0.75
)
```

## Advanced Features

### Attention Manipulation
Fine control over cross-attention layers (in development):

```python
wrapper.add_injection(
    prompt="detailed texture",
    attention_scales={
        "down_1": 1.5,  # Boost early layers
        "mid": 2.0,     # Strong mid-block
        "up_2": 0.8     # Reduce late layers
    }
)
```

### Multi-Scale Control
Different prompts at different resolution stages:

```python
# Coarse structure (early, low-res)
wrapper.add_injection(
    prompt="mountain landscape",
    start_frac=0.0,
    end_frac=0.4,
    resolution_scale=0.25  # Low resolution
)

# Fine details (late, high-res)
wrapper.add_injection(
    prompt="intricate rock textures",
    start_frac=0.6,
    end_frac=1.0,
    resolution_scale=1.0  # Full resolution
)
```

## Usage Examples

### Basic Generation
```python
from stable_diffusion import StableDiffusion
from corpus_mlx.sd_wrapper import CorePulseStableDiffusion

# Initialize
sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
wrapper = CorePulseStableDiffusion(sd)

# Generate with base prompt
latents = None
for step_latents in wrapper.generate_latents(
    "beautiful landscape",
    num_steps=50,
    cfg_weight=7.5,
    seed=42
):
    latents = step_latents

# Decode to image
images = sd.autoencoder.decode(latents)
```

### Creative Composition
```python
# Sky region
wrapper.add_injection(
    prompt="dramatic sunset clouds",
    region=("rect_pix", 0, 0, 512, 200, 30),
    weight=0.7
)

# Ground region
wrapper.add_injection(
    prompt="rocky desert terrain",
    region=("rect_pix", 0, 200, 512, 312, 30),
    weight=0.7
)

# Time-based atmosphere
wrapper.add_injection(
    prompt="golden hour lighting",
    start_frac=0.0,
    end_frac=0.5,
    weight=0.5
)
```

### Style Transfer
```python
# Early: Structure
wrapper.add_injection(
    prompt="portrait photograph",
    start_frac=0.0,
    end_frac=0.3,
    weight=0.8
)

# Late: Style
wrapper.add_injection(
    prompt="oil painting brushstrokes",
    start_frac=0.5,
    end_frac=1.0,
    weight=0.6
)
```

## Architecture

### Noise Prediction Blending
CorePulse works by blending noise predictions at each diffusion step:

```
eps_final = (1 - α) * eps_base + α * eps_injection
```

Where:
- `eps_base`: Noise from base prompt
- `eps_injection`: Noise from injected prompt
- `α`: Blend weight (can be spatially varying via masks)

### Latent Space Operations
- All operations occur in 64×64 latent space (8× downscale from 512×512)
- Masks are computed in latent dimensions for efficiency
- Feathering uses distance transforms or box blur fallback

### Token Processing
- Fixed 77-token sequences for consistency
- Automatic padding/truncation
- Token masks applied via embedding manipulation

## Performance Considerations

### MLX Optimizations
- Uses MLX arrays for GPU acceleration on Apple Silicon
- Float16 precision by default
- Lazy evaluation for efficient computation graphs

### Memory Usage
- Base model: ~2.5GB (float16)
- Per injection: ~10MB (embeddings + masks)
- Batch generation supported

### Speed
- 512×512 @ 50 steps: ~15-30 seconds on M1/M2
- Regional masks add minimal overhead (<5%)
- Token masking: no performance impact

## API Reference

### CorePulseStableDiffusion

#### `__init__(sd, model_kind="sd")`
Initialize wrapper around MLX StableDiffusion instance.

#### `add_injection(**kwargs) -> InjectionConfig`
Add a prompt injection configuration.

**Parameters:**
- `prompt` (str): Text to inject
- `token_mask` (str|list): Tokens to emphasize
- `region` (tuple|array): Spatial mask specification
- `start_frac` (float): Start time [0.0, 1.0]
- `end_frac` (float): End time [0.0, 1.0]
- `weight` (float): Blend strength [0.0, 1.0]

#### `generate_latents(base_prompt, **kwargs)`
Generate image latents with injections.

**Parameters:**
- `base_prompt` (str): Primary prompt
- `negative_text` (str): Negative prompt
- `num_steps` (int): Diffusion steps
- `cfg_weight` (float): Classifier-free guidance
- `n_images` (int): Batch size
- `height` (int): Image height
- `width` (int): Image width
- `seed` (int): Random seed

**Returns:**
- Iterator of latent arrays at each step

#### `clear_injections()`
Remove all injection configurations.

## Troubleshooting

### Common Issues

1. **Token length mismatch errors**
   - Ensure all prompts are padded to 77 tokens
   - Check `corpus_mlx/injection.py` for padding logic

2. **Region mask shape errors**
   - Verify mask dimensions match latent size (H/8 × W/8)
   - Use correct coordinate system (pixels vs fractions)

3. **Memory issues**
   - Reduce batch size
   - Use fewer injection layers
   - Ensure float16 mode is enabled

### Debug Mode
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check injection preparations
for ic in wrapper.injections:
    print(f"Injection: {ic.prompt[:30]}...")
    print(f"  Time: {ic.start_frac:.1f} - {ic.end_frac:.1f}")
    print(f"  Weight: {ic.weight}")
```

## Future Enhancements

### Planned Features
- [ ] Per-block attention control
- [ ] Cross-attention redistribution
- [ ] Learned injection schedules
- [ ] SDXL model support
- [ ] ControlNet integration
- [ ] Dynamic weight scheduling

### Research Directions
- Optimal injection timing strategies
- Semantic region detection
- Attention-guided masking
- Multi-prompt coherence

## Citation
```
CorePulse × MLX: Advanced Prompt Injection for Stable Diffusion
https://github.com/yourusername/corpus-mlx
```