from __future__ import annotations
from .utils import as_mx
import mlx.core as mx

def blend_eps(eps_base, eps_inj, alpha: float, mask=None):
    alpha = float(max(0.0, min(1.0, alpha)))
    if mask is None:
        return eps_base * (1.0 - alpha) + eps_inj * alpha
    # mask expected shape (1,H,W,1) -> broadcasts to (B,H,W,C)
    return eps_base * (1.0 - alpha * mask) + eps_inj * (alpha * mask)
