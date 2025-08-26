from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import mlx.core as mx
from corpus_mlx.masks import region_mask_in_latent
from corpus_mlx.schedule import active, ramp

def build_product_placement(
    cpsd,
    *,
    region: Tuple,
    mode: str = "inpaint",              # "inpaint" or "image_cond"
    reference_rgba_path: Optional[str] = None,
    phase: Tuple[float,float] = (0.55, 1.0),
    alpha: float = 1.0,
    cfg_cap: Optional[float] = 6.0,
    ramp_steps: int = 8,
):
    """
    Registers a product-placement setup on the CorePulseStableDiffusion wrapper.
    - region: mask spec (rect_frac|rect_pix|circle_*, or an explicit array)
    - mode=inpaint: paste VAE-encoded latent of reference into region at phase start; blend epsilon with alpha=1
    - mode=image_cond: run an additional UNet pass using CLIP embedding from reference image (TODO)
    - cfg_cap: optional function to limit CFG in product phase
    """
    H_lat = None; W_lat = None  # will be resolved on first generate call via hook closure
    pasted = {"done": False}
    start, end = phase

    def _cfg_fn(progress):
        w = cpsd._cfg_weight_default  # set by wrapper on entry
        if cfg_cap is not None and active(progress, start, end):
            return min(w, cfg_cap)
        return w

    # install a per-step hook to paste latent once at phase start (inpaint mode)
    def _pre_step_hook(i, progress, x_t):
        nonlocal H_lat, W_lat
        if H_lat is None or W_lat is None:
            H_lat, W_lat = x_t.shape[1], x_t.shape[2]

        if mode == "inpaint" and (not pasted["done"]) and progress >= start:
            # region mask in latent resolution
            M = region_mask_in_latent(region, H_lat, W_lat)  # (1,H,W,1)
            # Encode reference image to latent
            if reference_rgba_path is None:
                raise ValueError("reference_rgba_path is required for inpaint mode")
            ref_img = _load_image(reference_rgba_path, target=(W_lat*8, H_lat*8))
            # Assumes sd.encode exists; if not, raise a clear error
            if not hasattr(cpsd.sd, "encode"):
                raise AttributeError("StableDiffusion object has no .encode(image) — required for inpaint mode")
            ref_lat = cpsd.sd.encode(ref_img)  # expected NHWC latent
            # Paste
            x_new = x_t * (1.0 - M) + ref_lat * M
            pasted["done"] = True
            return x_new
        return x_t

    # attach
    cpsd.add_pre_step_hook(_pre_step_hook)

    # add a basic injection that is active only in the product phase; for inpaint, we keep weight 1 inside mask
    cpsd.add_injection(
        prompt="", weight=alpha, start_frac=start, end_frac=end, token_mask=None, region=region
    )

    # install cfg cap function
    def _cfg_weight_fn(progress):
        return _cfg_fn(progress)
    cpsd.cfg_weight_fn = _cfg_weight_fn

    # stash default (wrapper will set before loop)
    cpsd._cfg_weight_default = 7.5  # default; overwritten by wrapper per call
    return {"hook": _pre_step_hook, "phase": phase, "alpha": alpha, "mode": mode}

def _load_image(path, target=None):
    """Load RGBA or RGB with Pillow into NHWC float32 in [0,1]; resized to target (W,H) if provided."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if target is not None:
        im = im.resize(target, Image.LANCZOS)
    arr = (np.asarray(im).astype("float32") / 255.0)
    return mx.array(arr)
