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

        # Set seed if provided
        if seed is not None:
            mx.random.seed(seed)

        sampler = self.sd.sampler
        x_t = sampler.sample_prior((B, H_lat, W_lat, 4), dtype=self.dtype)

        # Use fixed length for consistency across all embeddings
        fixed_len = 77
        
        # Tokenize base prompt
        base_tok_list = self.sd.tokenizer.tokenize(base_prompt)
        if len(base_tok_list) > fixed_len:
            base_tok_list = base_tok_list[:fixed_len]
        elif len(base_tok_list) < fixed_len:
            base_tok_list = base_tok_list + [0] * (fixed_len - len(base_tok_list))
        base_tokens = as_mx([base_tok_list], dtype=mx.int32)
        
        # Tokenize negative prompt
        uncond_tok_list = self.sd.tokenizer.tokenize(negative_text or "")
        if len(uncond_tok_list) > fixed_len:
            uncond_tok_list = uncond_tok_list[:fixed_len]
        elif len(uncond_tok_list) < fixed_len:
            uncond_tok_list = uncond_tok_list + [0] * (fixed_len - len(uncond_tok_list))
        uncond_tokens = as_mx([uncond_tok_list], dtype=mx.int32)

        base_emb = self.sd.text_encoder(base_tokens).last_hidden_state
        uncond_emb = self.sd.text_encoder(uncond_tokens).last_hidden_state

        # Use same fixed_len for injections to ensure consistent dimensions
        prepared = [prepare_injection(self.sd, ic, H_lat, W_lat, max_tokens=fixed_len) for ic in self.injections]

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
        """Fixed CFG epsilon prediction that matches original SD behavior"""
        if cfg_weight > 1:
            # Concatenate latents and embeddings for CFG
            x_t_unet = mx.concatenate([x_t] * 2, axis=0)
            t_unet = mx.broadcast_to(t, [len(x_t_unet)])
            
            # Concatenate embeddings [uncond, cond]
            encoder_x = mx.concatenate([uncond_emb, cond_emb], axis=0)
            
            # Single UNet call with concatenated inputs
            eps_pred = self.sd.unet(x_t_unet, t_unet, encoder_x=encoder_x)
            
            # Split predictions
            eps_uncond, eps_cond = eps_pred.split(2)
            
            # Apply CFG weighting
            return eps_uncond + cfg_weight * (eps_cond - eps_uncond)
        else:
            # No CFG, just use conditional
            t_single = mx.broadcast_to(t, [len(x_t)])
            return self.sd.unet(x_t, t_single, encoder_x=cond_emb)