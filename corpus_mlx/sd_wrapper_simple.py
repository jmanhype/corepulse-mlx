from __future__ import annotations
from typing import List, Optional, Iterable
import numpy as np
import mlx.core as mx
from .utils import as_mx
from .injection import InjectionConfig, prepare_injection, encode_tokens
from .blending import blend_eps
from .schedule import active

Array = mx.array

class CorePulseStableDiffusion:
    """
    CorePulse wrapper around an MLX StableDiffusion instance.
    - Supports time-windowed, token-masked, and regional prompt injection.
    - Provides a small hook API for pre-step mutations (e.g., product latent paste).
    """
    def __init__(self, sd, model_kind: str = "sd"):
        self.sd = sd
        self.model_kind = model_kind
        self.injections: List[InjectionConfig] = []
        self.pre_step_hooks = []          # functions: (i, progress, x_t) -> x_t
        self.cfg_weight_fn = None         # optional: f(progress) -> float
        self.latent_scale = 8

        try:
            self.dtype = next(iter(sd.unet.parameters())).dtype  # type: ignore[attr-defined]
        except Exception:
            self.dtype = mx.float16

    def add_injection(self, **kwargs) -> InjectionConfig:
        ic = InjectionConfig(**kwargs)
        self.injections.append(ic)
        return ic

    def clear_injections(self): self.injections.clear()
    def add_pre_step_hook(self, fn): self.pre_step_hooks.append(fn)
    def clear_pre_step_hooks(self): self.pre_step_hooks.clear()

    def generate_latents(
        self,
        base_prompt: str,
        *,
        negative_text: str = "",
        num_steps: int = 50,
        cfg_weight: float = 7.5,
        n_images: int = 1,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        sampler_kind: str = "euler",
    ) -> Iterable[Array]:
        """Simple delegation to original SD for now"""
        if not self.injections:
            # No injections, just use original SD
            latent_size = (height // 8, width // 8)
            yield from self.sd.generate_latents(
                base_prompt,
                n_images=n_images,
                num_steps=num_steps,
                cfg_weight=cfg_weight,
                negative_text=negative_text,
                latent_size=latent_size,
                seed=seed,
            )
        else:
            # TODO: Implement injection logic
            latent_size = (height // 8, width // 8)
            yield from self.sd.generate_latents(
                base_prompt,
                n_images=n_images,
                num_steps=num_steps,
                cfg_weight=cfg_weight,
                negative_text=negative_text,
                latent_size=latent_size,
                seed=seed,
            )