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


def uv_grid(ts):
    """UV coordinates of every texel centre of the texture set's images (tiles
    included: u runs over 0..columns, v over 0..rows)."""
    h, w = ts.shape
    cols, rows = ts.grid
    return np.meshgrid((np.arange(w, dtype=np.float32) + 0.5) / w * cols,
                       (np.arange(h, dtype=np.float32) + 0.5) / h * rows)


def can_triplanar(ts):
    return ts.baked_map("POSITION") is not None and ts.baked_map("WORLD_NORMAL") is not None


def project(ts, tex, tiling, triplanar):
    """Map a tileable (h, w, c) texture onto the texture set: UV tiling, or
    tri-planar from the baked Position / World Normal maps."""
    if not (triplanar and can_triplanar(ts)):
        u, v = uv_grid(ts)
        return bilinear_wrap(tex, u * tiling, v * tiling)
    pos = generators.baked(ts, "POSITION") * tiling
    n = generators.baked(ts, "WORLD_NORMAL") * 2.0 - 1.0
    w = np.abs(n) ** 4
    w /= np.maximum(w.sum(-1, keepdims=True), 1e-6)
    x_proj = bilinear_wrap(tex, pos[..., 1], pos[..., 2])
    y_proj = bilinear_wrap(tex, pos[..., 0], pos[..., 2])
    z_proj = bilinear_wrap(tex, pos[..., 0], pos[..., 1])
    return x_proj * w[..., :1] + y_proj * w[..., 1:2] + z_proj * w[..., 2:]


def sample(ts, layer, image, allow_triplanar=True):
    """(res, res, 3) texture for a fill layer channel, cached until inputs change."""
    triplanar = allow_triplanar and layer.projection == "TRIPLANAR" and can_triplanar(ts)
    key = ("fill", image.name, tuple(image.size), ts.size, layer.tiling, triplanar, generators._generation)
    return generators._cached(key, lambda: project(ts, pixels.read(image)[..., :3], layer.tiling, triplanar))
