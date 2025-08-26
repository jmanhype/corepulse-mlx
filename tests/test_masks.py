import numpy as np

def test_rect_mask_shape():
    from corpus_mlx.masks import rect_mask_latent_frac
    M = rect_mask_latent_frac(0.1, 0.2, 0.3, 0.4, 0.1, 64, 64)
    assert M.shape == (1,64,64,1)
