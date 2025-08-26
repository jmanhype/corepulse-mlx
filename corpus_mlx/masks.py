from __future__ import annotations
import numpy as np
from .utils import as_mx
import mlx.core as mx

def _clip(a, lo, hi):
    return max(lo, min(hi, a))

def feather_mask(M: np.ndarray, radius: int) -> np.ndarray:
    """Feather edges; tries scipy if available, else simple box-blur fallback."""
    if radius <= 0:
        return M.astype(np.float32)
    try:
        from scipy.ndimage import distance_transform_edt
        inside = M > 0.5
        outside = ~inside
        dist_in = distance_transform_edt(inside)
        dist_out = distance_transform_edt(outside)
        sdf = dist_out - dist_in
        A = 0.5 + 0.5 * np.clip(sdf / float(radius), -1.0, 1.0)
        return A.astype(np.float32)
    except Exception:
        # box blur fallback
        k = int(max(1, radius))
        pad = k // 2
        P = np.pad(M, ((pad,pad),(pad,pad)), mode="edge").astype(np.float32)
        out = np.zeros_like(M, dtype=np.float32)
        for y in range(M.shape[0]):
            for x in range(M.shape[1]):
                out[y,x] = P[y:y+k, x:x+k].mean()
        return out

def rect_mask_latent_frac(fx, fy, fw, fh, fthr, H_lat, W_lat, feather=True):
    lx = int(round(fx * W_lat))
    ly = int(round(fy * H_lat))
    lw = int(round(fw * W_lat))
    lh = int(round(fh * H_lat))
    fr = int(round(fthr * max(H_lat, W_lat)))
    M = np.zeros((H_lat, W_lat), dtype=np.float32)
    x0, y0 = np.clip(lx, 0, W_lat - 1), np.clip(ly, 0, H_lat - 1)
    x1, y1 = np.clip(lx + lw, 0, W_lat), np.clip(ly + lh, 0, H_lat)
    M[y0:y1, x0:x1] = 1.0
    if feather and fr > 0:
        M = feather_mask(M, fr)
    return as_mx(M[..., None])

def rect_mask_latent_pix(x, y, w, h, feather_px, H_lat, W_lat, scale=8, feather=True):
    lx, ly, lw, lh = [int(round(v / scale)) for v in (x, y, w, h)]
    fr = max(1, int(round(feather_px / scale)))
    M = np.zeros((H_lat, W_lat), dtype=np.float32)
    x0, y0 = np.clip(lx, 0, W_lat - 1), np.clip(ly, 0, H_lat - 1)
    x1, y1 = np.clip(lx + lw, 0, W_lat), np.clip(ly + lh, 0, H_lat)
    M[y0:y1, x0:x1] = 1.0
    if feather and fr > 0:
        M = feather_mask(M, fr)
    return as_mx(M[..., None])

def circle_mask_latent_frac(fcx, fcy, fr, fthr, H_lat, W_lat, feather=True):
    lcx = int(round(fcx * W_lat))
    lcy = int(round(fcy * H_lat))
    lr = int(round(fr * max(H_lat, W_lat)))
    fr2 = int(round(fthr * max(H_lat, W_lat)))
    yy, xx = np.mgrid[0:H_lat, 0:W_lat]
    dist = ((xx - lcx)**2 + (yy - lcy)**2)**0.5
    M = (dist <= lr).astype(np.float32)
    if feather and fr2 > 0:
        M = feather_mask(M, fr2)
    return as_mx(M[..., None])

def circle_mask_latent_pix(cx, cy, r, feather_px, H_lat, W_lat, scale=8, feather=True):
    lcx, lcy, lr = [int(round(v / scale)) for v in (cx, cy, r)]
    fr = max(1, int(round(feather_px / scale)))
    yy, xx = np.mgrid[0:H_lat, 0:W_lat]
    dist = ((xx - lcx)**2 + (yy - lcy)**2)**0.5
    M = (dist <= lr).astype(np.float32)
    if feather and fr > 0:
        M = feather_mask(M, fr)
    return as_mx(M[..., None])

def region_mask_in_latent(spec, H_lat, W_lat):
    import numpy as np
    if spec is None:
        return None
    if isinstance(spec, (np.ndarray, mx.array)):
        M = np.array(spec) if not isinstance(spec, np.ndarray) else spec
        if M.ndim == 2:
            M = M[..., None]
        assert M.shape[0] in (H_lat, 1) and M.shape[1] in (W_lat, 1), "Region mask shape mismatch"
        return as_mx(M.astype(np.float32))
    kind = spec[0]
    if kind == "rect_frac":
        fx, fy, fw, fh, fthr = spec[1:]
        return rect_mask_latent_frac(fx, fy, fw, fh, fthr, H_lat, W_lat)
    if kind == "rect_pix":
        x, y, w, h, feather = spec[1:]
        return rect_mask_latent_pix(x, y, w, h, feather, H_lat, W_lat)
    if kind == "circle_frac":
        fcx, fcy, fr, fthr = spec[1:]
        return circle_mask_latent_frac(fcx, fcy, fr, fthr, H_lat, W_lat)
    if kind == "circle_pix":
        cx, cy, r, feather = spec[1:]
        return circle_mask_latent_pix(cx, cy, r, feather, H_lat, W_lat)
    raise ValueError(f"Unknown region spec kind: {kind}")
