"""Procedural grunge library. Every texture tiles seamlessly (periodic noise,
wrapped Voronoi and wrapped line segments), so it can be used for UV tiling,
tri-planar projection, stencils or brush tips. Generated in numpy, nothing to
download, and released with the add-on (GPL)."""

import bpy
import numpy as np

from . import pixels

GRUNGE_TYPES = [
    ("GRUNGE", "Grunge", "Blotchy multi-scale grime"),
    ("DUST", "Dust", "Fine speckles and soft dust patches"),
    ("RUST", "Rust Spots", "Irregular corrosion blobs"),
    ("LEAKS", "Leaks", "Vertical streaks"),
    ("SCRATCHES", "Scratches", "Thin random scratches"),
    ("CONCRETE", "Concrete", "Mottled surface with pits"),
    ("CELLS", "Cells", "Voronoi cells"),
    ("CRACKS", "Cracks", "Network of thin cracks"),
]

SIZE = 512
_cache = {}


def periodic_noise(n, periods, seed):
    """Smooth value noise on an n x n grid that wraps: `periods` lattice cells per
    axis (px, py), so the texture tiles exactly."""
    px, py = periods
    lattice = np.random.default_rng(seed).random((py, px)).astype(np.float32)

    def axis(p):
        c = np.arange(n, dtype=np.float32) * p / n
        i0 = np.floor(c).astype(np.int64)
        f = c - i0
        return i0 % p, (i0 + 1) % p, f * f * (3.0 - 2.0 * f)

    x0, x1, fx = axis(px)
    y0, y1, fy = axis(py)
    top = lattice[y0][:, x0] * (1 - fx) + lattice[y0][:, x1] * fx
    bot = lattice[y1][:, x0] * (1 - fx) + lattice[y1][:, x1] * fx
    return top * (1 - fy[:, None]) + bot * fy[:, None]


def tileable_fbm(n, periods, octaves, seed, gain=0.5):
    total = np.zeros((n, n), np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        total += amp * periodic_noise(n, (periods[0] * 2 ** o, periods[1] * 2 ** o), seed + o * 101)
        norm += amp
        amp *= gain
    return total / norm


def _contrast(x, c, mid=0.5):
    return np.clip((x - mid) * c + 0.5, 0.0, 1.0)


def _voronoi(n, cells, seed):
    """Wrapped jittered-grid Voronoi: distances to the nearest (f1) and second
    nearest (f2) feature point, in cell units. One feature point per cell, so only
    the 3 x 3 neighbouring cells need checking."""
    rng = np.random.default_rng(seed)
    pts = rng.random((cells, cells, 2)).astype(np.float32)
    g = (np.arange(n, dtype=np.float32) + 0.5) * cells / n
    x, y = np.meshgrid(g, g)
    cx, cy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - cx, y - cy
    f1 = np.full((n, n), 1e9, np.float32)
    f2 = np.full((n, n), 1e9, np.float32)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            p = pts[(cy + oy) % cells, (cx + ox) % cells]
            d = np.hypot(ox + p[..., 0] - fx, oy + p[..., 1] - fy)
            f2 = np.where(d < f1, f1, np.minimum(f2, d))
            f1 = np.minimum(f1, d)
    return f1, f2


def _scratches(n, seed, count=70):
    rng = np.random.default_rng(seed)
    g = (np.arange(n, dtype=np.float32) + 0.5) / n
    x, y = np.meshgrid(g, g)
    out = np.zeros((n, n), np.float32)
    for _ in range(count):
        a = rng.normal(0.6, 0.35)  # mostly one direction, like real wear
        d = np.array([np.cos(a), np.sin(a)], np.float32)
        c = rng.random(2).astype(np.float32)
        length, width = rng.uniform(0.05, 0.35), rng.uniform(0.0008, 0.003)
        # Wrapped offsets so lines continue across the tile border.
        dx = (x - c[0] + 0.5) % 1.0 - 0.5
        dy = (y - c[1] + 0.5) % 1.0 - 0.5
        along = dx * d[0] + dy * d[1]
        across = -dx * d[1] + dy * d[0]
        taper = np.clip(1.0 - np.abs(along) / length, 0.0, 1.0)
        line = np.clip(1.0 - np.abs(across) / (width * (0.3 + taper)), 0.0, 1.0) * (taper > 0)
        out = np.maximum(out, line * rng.uniform(0.4, 1.0))
    return out


def grunge_array(kind, seed=0, size=SIZE):
    """(size, size) float32 in [0, 1], seamless."""
    key = (kind, seed, size)
    if key in _cache:
        return _cache[key]
    n = size
    if kind == "GRUNGE":
        a = _contrast(tileable_fbm(n, (4, 4), 6, seed), 2.6)
    elif kind == "DUST":
        patches = tileable_fbm(n, (3, 3), 4, seed)
        speck = tileable_fbm(n, (64, 64), 2, seed + 7)
        a = np.clip(np.clip((speck - 0.68) * 6.0, 0.0, 1.0) * 0.8 + _contrast(patches, 2.0, 0.55) * 0.4,
                    0.0, 1.0)
    elif kind == "RUST":
        blobs = tileable_fbm(n, (5, 5), 6, seed)
        a = np.clip((blobs - 0.55) * 9.0, 0.0, 1.0) * (0.7 + 0.3 * tileable_fbm(n, (40, 40), 2, seed + 3))
    elif kind == "LEAKS":
        streak = tileable_fbm(n, (24, 2), 4, seed)
        a = _contrast(streak, 3.0, 0.55) * tileable_fbm(n, (4, 1), 3, seed + 5)
    elif kind == "SCRATCHES":
        a = _scratches(n, seed)
    elif kind == "CONCRETE":
        base = tileable_fbm(n, (8, 8), 6, seed)
        pits = np.clip((0.28 - tileable_fbm(n, (48, 48), 2, seed + 9)) * 8.0, 0.0, 1.0)
        a = np.clip(_contrast(base, 1.6) - pits * 0.6, 0.0, 1.0)
    elif kind == "CELLS":
        f1, _ = _voronoi(n, 8, seed)
        a = np.clip(f1 / 0.75, 0.0, 1.0)
    elif kind == "CRACKS":
        f1, f2 = _voronoi(n, 6, seed)
        wobble = tileable_fbm(n, (12, 12), 3, seed + 1) * 0.08
        a = np.clip(1.0 - (f2 - f1 + wobble - 0.02) / 0.06, 0.0, 1.0)
    else:
        raise ValueError(kind)
    a = np.ascontiguousarray(a, np.float32)
    _cache[key] = a
    return a


def make_image(kind, seed=0, size=SIZE):
    """A packed Blender image of a grunge texture, for fill textures, stencils or tips."""
    name = f"{pixels.PREFIX}Grunge_{kind.title()}_{seed}"
    img = bpy.data.images.get(name)
    if img is not None:
        return img
    a = grunge_array(kind, seed, size)
    img = bpy.data.images.new(name, size, size, alpha=True)
    img.colorspace_settings.name = "Non-Color"
    pixels.write(img, pixels.rgba(np.repeat(a[..., None], 3, -1), np.ones_like(a)))
    img.pack()
    return img
