from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict
import numpy as np
import mlx.core as mx
from .utils import as_mx
from .masks import region_mask_in_latent

Array = mx.array

@dataclass
class InjectionConfig:
    prompt: str
    weight: float = 0.5
    start_frac: float = 0.0
    end_frac: float = 1.0
    token_mask: Optional[str] = None
    region: Optional[Union[Tuple, np.ndarray, Array]] = None
    # prepared
    embedding: Optional[Array] = field(default=None, repr=False)
    region_mask_latent: Optional[Array] = field(default=None, repr=False)

def encode_tokens(sd, text: str, prepend_bos: bool=True, append_eos: bool=True) -> Array:
    # MLX tokenizer already includes BOS and EOS tokens automatically
    token_ids = sd.tokenizer.tokenize(text)
    return as_mx([token_ids], dtype=mx.int32)

def encode_text_masked(sd, text: str, mask_phrase: Optional[str]) -> Array:
    # MLX tokenizer already includes BOS and EOS tokens automatically
    tokens = sd.tokenizer.tokenize(text)
    toks_arr = as_mx([tokens], dtype=mx.int32)
    emb = sd.text_encoder(toks_arr).last_hidden_state  # [1, N, D]
    if not mask_phrase:
        return emb
    mask_tokens = sd.tokenizer.tokenize(mask_phrase)
    kept = _find_subsequence(tokens, mask_tokens)
    if not kept:
        kept = [i for i,t in enumerate(tokens) if t in set(mask_tokens)]
    N = emb.shape[1]
    keep_mask = np.zeros((N,), dtype=np.float32)
    # keep BOS/EOS to preserve sequence semantics
    keep_mask[0] = 1.0; keep_mask[-1] = 1.0
    for idx in kept:
        if 0 <= idx < N: keep_mask[idx] = 1.0
    keep_mask = keep_mask.reshape(1, N, 1)
    keep_mask_mx = as_mx(keep_mask, dtype=emb.dtype)
    return emb * keep_mask_mx

def _find_subsequence(tokens: list, subseq: list) -> list:
    if not subseq:
        return []
    for i in range(0, len(tokens) - len(subseq) + 1):
        if tokens[i:i+len(subseq)] == subseq:
            return list(range(i, i+len(subseq)))
    return []

def prepare_injection(sd, ic: InjectionConfig, H_lat: int, W_lat: int, max_tokens: int = 77):
    if not (0.0 <= ic.start_frac <= 1.0 and 0.0 <= ic.end_frac <= 1.0 and ic.start_frac <= ic.end_frac):
        return None
    
    # Tokenize and pad to max_tokens
    tokens = sd.tokenizer.tokenize(ic.prompt)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    elif len(tokens) < max_tokens:
        tokens = tokens + [0] * (max_tokens - len(tokens))
    
    toks_arr = as_mx([tokens], dtype=mx.int32)
    emb = sd.text_encoder(toks_arr).last_hidden_state
    
    # Apply token masking if needed
    if ic.token_mask:
        mask_tokens = sd.tokenizer.tokenize(ic.token_mask)
        kept = _find_subsequence(tokens, mask_tokens)
        if not kept:
            kept = [i for i,t in enumerate(tokens) if t in set(mask_tokens)]
        N = emb.shape[1]
        keep_mask = np.zeros((N,), dtype=np.float32)
        keep_mask[0] = 1.0; keep_mask[-1] = 1.0
        for idx in kept:
            if 0 <= idx < N: keep_mask[idx] = 1.0
        keep_mask = keep_mask.reshape(1, N, 1)
        keep_mask_mx = as_mx(keep_mask, dtype=emb.dtype)
        emb = emb * keep_mask_mx
    
    mask_lat = region_mask_in_latent(ic.region, H_lat, W_lat) if ic.region is not None else None
    ic.embedding = emb
    ic.region_mask_latent = mask_lat
    return {"ic": ic, "embedding": emb, "region_mask": mask_lat}