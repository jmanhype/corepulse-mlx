from __future__ import annotations
from typing import Tuple
from corpus_mlx.masks import region_mask_in_latent

def add_regional_prompt(cpsd, prompt: str, region: Tuple, *, start: float=0.0, end: float=1.0, weight: float=0.6, token_mask=None):
    """
    Convenience helper to add a prompt that only affects a region between [start,end].
    """
    cpsd.add_injection(
        prompt=prompt, weight=weight, start_frac=start, end_frac=end, token_mask=token_mask, region=region
    )
