"""Image textures on fill layers: UV tiling, or tri-planar projection from the
baked Position / World Normal maps (seamless across UV islands)."""

import numpy as np

from . import generators, pixels


def bilinear_wrap(img, u, v):
    """Sample an (h, w, c) array at UVs (any shape) with bilinear filtering and wrap."""
    h, w = img.shape[:2]
    x = (u % 1.0) * w - 0.5
    y = (v % 1.0) * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[..., None].astype(np.float32)
    fy = (y - y0)[..., None].astype(np.float32)
    x0 %= w
    y0 %= h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def _uv_grid(res):
    g = (np.arange(res, dtype=np.float32) + 0.5) / res
    return np.meshgrid(g, g)


def sample(ts, layer, image, allow_triplanar=True):
    """(res, res, 3) texture for a fill layer channel, cached until inputs change."""
    res = ts.res
    triplanar = (allow_triplanar and layer.projection == "TRIPLANAR"
                 and ts.baked_map("POSITION") is not None and ts.baked_map("WORLD_NORMAL") is not None)
    key = ("fill", image.name, tuple(image.size), res, layer.tiling, triplanar, generators._generation)

    def compute():
        tex = pixels.read(image)[..., :3]
        t = layer.tiling
        if not triplanar:
            u, v = _uv_grid(res)
            return bilinear_wrap(tex, u * t, v * t)
        pos = generators.baked(ts, "POSITION") * t
        n = generators.baked(ts, "WORLD_NORMAL") * 2.0 - 1.0
        w = np.abs(n) ** 4
        w /= np.maximum(w.sum(-1, keepdims=True), 1e-6)
        x_proj = bilinear_wrap(tex, pos[..., 1], pos[..., 2])
        y_proj = bilinear_wrap(tex, pos[..., 0], pos[..., 2])
        z_proj = bilinear_wrap(tex, pos[..., 0], pos[..., 1])
        return x_proj * w[..., :1] + y_proj * w[..., 1:2] + z_proj * w[..., 2:]

    return generators._cached(key, compute)
