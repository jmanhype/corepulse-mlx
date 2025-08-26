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
        assert self.model_kind == "sd", "SDXL path not enabled in this version."
        H_lat, W_lat = height // self.latent_scale, width // self.latent_scale
        B = int(n_images)

        sampler = self.sd.sampler
        key = mx.random.key(seed) if seed is not None else None
        x_t = sampler.sample_prior((B, H_lat, W_lat, 4), dtype=self.dtype, key=key)

        base_tokens = encode_tokens(self.sd, base_prompt, True, True)
        uncond_tokens = encode_tokens(self.sd, negative_text or "", True, True)

        base_emb = self.sd.text_encoder(base_tokens).last_hidden_state
        uncond_emb = self.sd.text_encoder(uncond_tokens).last_hidden_state

        prepared = [prepare_injection(self.sd, ic, H_lat, W_lat) for ic in self.injections]

        steps = list(sampler.timesteps(num_steps))
        n = len(steps)

        for i, (t, t_prev) in enumerate(steps):
            progress = i / max(1, n - 1)

            # optional hook mutations (e.g., paste product latent in region at a boundary)
            for hook in list(self.pre_step_hooks):
                x_t = hook(i, progress, x_t)

            # chooser for cfg at this step
            w = self.cfg_weight_fn(progress) if self.cfg_weight_fn else cfg_weight

            eps_base = self._cfg_eps(x_t, t, base_emb, uncond_emb, w)
            eps_total = eps_base

            for p in prepared:
                if p is None: continue
                ic: InjectionConfig = p["ic"]
                if not active(progress, ic.start_frac, ic.end_frac):
                    continue
                eps_inj = self._cfg_eps(x_t, t, p["embedding"], uncond_emb, w)
                eps_total = blend_eps(eps_total, eps_inj, ic.weight, p["region_mask"])

            x_t = sampler.step(eps_total, x_t, t, t_prev)
            yield x_t

    def _cfg_eps(self, x_t: Array, t: Array, cond_emb: Array, uncond_emb: Array, cfg_weight: float) -> Array:
        eps_uncond = self.sd.unet(x_t, t, encoder_x=uncond_emb)
        eps_cond = self.sd.unet(x_t, t, encoder_x=cond_emb)
        if cfg_weight == 0.0:
            return eps_cond
        w = as_mx(cfg_weight, dtype=eps_cond.dtype)
        return eps_uncond + w * (eps_cond - eps_uncond)
