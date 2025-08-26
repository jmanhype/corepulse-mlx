import mlx.core as mx

Array = mx.array

def as_mx(x, dtype=None):
    a = mx.array(x)
    return a.astype(dtype) if dtype is not None else a
