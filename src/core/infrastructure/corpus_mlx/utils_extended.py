"""Extended utilities for corpus-mlx"""

import mlx.core as mx
import numpy as np
from PIL import Image

def latents_to_pil(latents, sd=None):
    """Convert latents to PIL image
    
    Args:
        latents: MLX array of latents (from diffusion process)
        sd: Optional StableDiffusion instance for VAE decoding
    """
    # Convert to numpy if needed
    if isinstance(latents, mx.array):
        latents_np = np.array(latents)
    else:
        latents_np = latents
    
    # If we have a VAE decoder, use it
    if sd is not None and hasattr(sd, 'vae'):
        # Decode latents to image
        if isinstance(latents, np.ndarray):
            latents = mx.array(latents)
        # VAE decode
        images = sd.vae.decode(latents / sd.vae.config.scaling_factor)
        images = mx.clip(images / 2 + 0.5, 0, 1)
        images_np = np.array(images)
    else:
        # Direct conversion (assumes already in image space)
        images_np = latents_np
    
    # Ensure proper shape
    if images_np.ndim == 4:
        images_np = images_np[0]  # Take first batch
    
    # Convert from CHW to HWC if needed
    if images_np.ndim == 3:
        if images_np.shape[0] in [3, 4]:
            images_np = np.transpose(images_np, (1, 2, 0))
    
    # Scale to 0-255
    if images_np.max() <= 1.0:
        images_np = (images_np * 255).astype(np.uint8)
    else:
        images_np = np.clip(images_np, 0, 255).astype(np.uint8)
    
    # Handle different channel counts
    if images_np.shape[-1] == 4:
        images_np = images_np[:, :, :3]  # Drop alpha
    elif images_np.shape[-1] == 1:
        images_np = np.repeat(images_np, 3, axis=-1)  # Grayscale to RGB
    
    return Image.fromarray(images_np)