# SDXL (Stable Diffusion XL) with MLX

This implementation supports both Stable Diffusion 2.1 and SDXL models using MLX for Apple Silicon acceleration.

## Features

- ✅ **SDXL Turbo Support**: Ultra-fast 1-4 step generation
- ✅ **High Resolution**: Native 1024x1024 image generation
- ✅ **Optimized for Apple Silicon**: Uses MLX for GPU acceleration
- ✅ **Multiple Models**: Support for both SD 2.1 and SDXL variants

## Quick Start

### Basic SDXL Generation

```python
from stable_diffusion import StableDiffusionXL

# Initialize SDXL
sdxl = StableDiffusionXL(
    model="stabilityai/sdxl-turbo",
    float16=True  # Use float16 for better performance
)

# Generate image
for latents in sdxl.generate_latents(
    text="Your prompt here",
    num_steps=4,  # SDXL Turbo only needs 1-4 steps
    cfg_weight=0.0,  # SDXL Turbo doesn't use CFG
    latent_size=(128, 128)  # For 1024x1024 output
):
    pass

# Decode to image
image = sdxl.decode(latents)
```

### Using the Test Scripts

1. **Quick Test**:
```bash
python quick_sdxl_test.py
```

2. **Full Featured Generation**:
```bash
python test_sdxl.py \
  --prompt "Your detailed prompt" \
  --num-steps 4 \
  --output output.png \
  --width 1024 \
  --height 1024
```

3. **Compare SD vs SDXL**:
```bash
python compare_sd_sdxl.py \
  --prompt "Your comparison prompt" \
  --output-dir ./comparison_output
```

## Model Specifications

### SDXL Turbo
- **Model**: `stabilityai/sdxl-turbo`
- **Resolution**: 1024x1024 (default)
- **Steps**: 1-4 (optimized for speed)
- **CFG**: 0.0 (trained without classifier-free guidance)
- **Speed**: ~5-20 seconds per image on M1/M2/M3

### Stable Diffusion 2.1
- **Model**: `stabilityai/stable-diffusion-2-1-base`
- **Resolution**: 512x512 (default)
- **Steps**: 20-50 (typical)
- **CFG**: 7.5 (recommended)
- **Speed**: ~30-60 seconds per image on M1/M2/M3

## Key Differences from SD 2.1

1. **Dual Text Encoders**: SDXL uses two CLIP text encoders for better prompt understanding
2. **Higher Resolution**: Native 1024x1024 vs 512x512
3. **Conditioning**: Additional conditioning including image size and crop parameters
4. **Architecture**: Larger UNet with more parameters for better quality

## Performance Tips

1. **Use Float16**: Significantly faster with minimal quality loss
   ```python
   sdxl = StableDiffusionXL(model="...", float16=True)
   ```

2. **Batch Generation**: Generate multiple images at once
   ```python
   sdxl.generate_latents(text="...", n_images=4)
   ```

3. **Adjust Steps**: SDXL Turbo works well with just 1-4 steps
   - 1 step: Ultra-fast, lower quality
   - 2-4 steps: Good balance of speed and quality
   - More steps: Diminishing returns for Turbo model

## Memory Requirements

- **SDXL Turbo**: ~6-8 GB VRAM
- **Float16 Mode**: Reduces memory by ~50%
- **Recommended**: M1/M2/M3 Mac with 16GB+ unified memory

## Common Issues

1. **Out of Memory**: Use float16 mode or reduce batch size
2. **Slow Generation**: Ensure you're using the Turbo variant for speed
3. **Model Download**: First run will download ~6GB of model files

## Examples

### Artistic Styles
```bash
# Photorealistic
python test_sdxl.py --prompt "Professional portrait photo, shallow depth of field, studio lighting"

# Anime/Manga
python test_sdxl.py --prompt "Anime character, Studio Ghibli style, cel shaded"

# Digital Art
python test_sdxl.py --prompt "Digital painting, concept art, ArtStation trending"
```

### Resolution Options
```bash
# Square (default)
python test_sdxl.py --width 1024 --height 1024

# Portrait
python test_sdxl.py --width 768 --height 1024

# Landscape  
python test_sdxl.py --width 1024 --height 768

# High Resolution (slower)
python test_sdxl.py --width 1536 --height 1536
```

## Technical Details

The implementation uses:
- **MLX**: Apple's machine learning framework for efficient computation
- **Hugging Face Hub**: Automatic model downloading and caching
- **Safetensors**: Fast and safe model weight loading

## License

Models are subject to their respective licenses:
- SDXL Turbo: [Stability AI License](https://huggingface.co/stabilityai/sdxl-turbo)
- SD 2.1: [CreativeML Open RAIL-M](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)