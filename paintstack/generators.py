"""Procedural mask generators driven by the texture set's baked mesh maps.

Noise is 3D and sampled at the baked object-space positions, so it continues
across UV seams instead of breaking at island borders."""

from collections import OrderedDict

import numpy as np

from . import pixels

REQUIRES = {
    "EDGE_WEAR": ("CURVATURE",),
    "DIRT": ("AO", "CURVATURE"),
    "CURVATURE": ("CURVATURE",),
    "AO": ("AO",),
    "THICKNESS": ("THICKNESS",),
    "POSITION": ("POSITION",),
    "NOISE": (),
    "ID_MASK": ("ID",),
    "GRUNGE": (),
}

_CACHE_SIZE = 32
_cache = OrderedDict()
_generation = 0


def invalidate():
    """Call after baking: drops every cached map and generator result."""
    global _generation
    _generation += 1
    _cache.clear()
    from . import compositor
    compositor.invalidate_all()


def _cached(key, fn):
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    val = fn()
    _cache[key] = val
    while len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)
    return val


def baked(ts, map_type):
    """Baked map resampled to the texture set resolution, (res, res, 3), or None."""
    img = ts.baked_map(map_type)
    if img is None:
        return None
    res = ts.res

    def load():
        px = pixels.read(img)[..., :3]
        if px.shape[0] != res or px.shape[1] != res:
            ys = (np.arange(res) * px.shape[0] // res)
            xs = (np.arange(res) * px.shape[1] // res)
            px = px[ys][:, xs]
        return np.ascontiguousarray(px)
    return _cached(("map", img.name, res, _generation), load)


def missing(effect, ts):
    return [m for m in REQUIRES.get(effect.type, ()) if ts.baked_map(m) is None]


# Noise

_tables = {}


def _table(seed):
    """Permutation (doubled to avoid wrapping) and lattice values for a seed."""
    if seed not in _tables:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(256).astype(np.int32)
        _tables[seed] = (np.concatenate([perm, perm]), rng.random(256, dtype=np.float32))
    return _tables[seed]


def value_noise(p, seed=0):
    """Smooth 3D value noise in [0, 1] for points p of shape (..., 3).
    Lattice hashing is shared between corners (2 + 4 + 8 table lookups), and the
    corners are combined with nested lerps."""
    perm, vals = _table(seed)
    fl = np.floor(p)
    f = (p - fl).astype(np.float32)
    i = fl.astype(np.int32) & 255
    u = f * f * (3.0 - 2.0 * f)
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    ix, iy, iz = i[..., 0], i[..., 1], i[..., 2]

    def lerp(a, b, t):
        return a + (b - a) * t

    hx = (perm[ix], perm[ix + 1])
    layers = []
    for dz in (0, 1):
        rows = []
        for dy in (0, 1):
            c0 = vals[perm[perm[hx[0] + iy + dy] + iz + dz]]
            c1 = vals[perm[perm[hx[1] + iy + dy] + iz + dz]]
            rows.append(lerp(c0, c1, ux))
        layers.append(lerp(rows[0], rows[1], uy))
    return lerp(layers[0], layers[1], uz)


def fbm(p, octaves=4, seed=0):
    total = np.zeros(p.shape[:-1], np.float32)
    amp, norm = 0.5, 0.0
    for o in range(octaves):
        total += amp * value_noise(p * (2.0 ** o), seed)
        norm += amp
        amp *= 0.5
    return total / norm


def _positions(ts):
    pos = baked(ts, "POSITION")
    if pos is not None:
        return pos
    # No position bake: fall back to UV space (noise will show UV seams).
    res = ts.res
    g = (np.arange(res, dtype=np.float32) + 0.5) / res
    u, v = np.meshgrid(g, g)
    return np.stack([u, v, np.zeros_like(u)], -1)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# Generators

def _source(effect, ts):
    t = effect.type
    if t == "GRUNGE":
        from . import fills, grunge
        tex = grunge.grunge_array(effect.grunge, effect.seed)[..., None]
        return fills.project(ts, tex, effect.grunge_tiling, True)[..., 0]
    if t == "POSITION":
        return baked(ts, "POSITION")[..., "XYZ".index(effect.axis)]
    curv = baked(ts, "CURVATURE")
    if t == "CURVATURE":
        return curv[..., 0]
    if t == "EDGE_WEAR":
        return np.clip((curv[..., 0] - 0.5) * 3.0, 0.0, 1.0)
    if t == "DIRT":
        cavity = np.clip((0.5 - curv[..., 0]) * 3.0, 0.0, 1.0)
        return 0.6 * (1.0 - baked(ts, "AO")[..., 0]) + 0.4 * cavity
    if t == "AO":
        return baked(ts, "AO")[..., 0]
    if t == "THICKNESS":
        return 1.0 - baked(ts, "THICKNESS")[..., 0]
    raise ValueError(t)


def fbm_fast(pos, scale, octaves, seed):
    """fBm on the GPU when a GPU context exists, else the numpy implementation."""
    perm, vals = _table(seed)
    try:
        from .engine import noise_gpu
        return noise_gpu.fbm(pos, scale, octaves, perm, vals)
    except Exception:  # no GPU context (background mode) or unsupported backend
        return fbm(pos * scale, octaves, seed)


def _noise(effect, ts):
    key = ("noise", ts.id_data.name, ts.res, _generation, effect.noise_scale,
           effect.noise_octaves, effect.seed)
    return _cached(key, lambda: fbm_fast(_positions(ts), effect.noise_scale,
                                         effect.noise_octaves, effect.seed))


def id_selection(id_map, color, tolerance):
    """1 where the ID colour is within tolerance of `color`, soft at the boundary."""
    dist = np.linalg.norm(id_map - np.asarray(color, np.float32), axis=-1)
    return 1.0 - smoothstep(tolerance * 0.5, tolerance, dist)


def evaluate(effect, ts):
    """Generator output as (res, res) float in [0, 1], or None if a bake is missing.

    Noise and sources are cached separately from the final threshold, so dragging
    Level / Softness / Noise amount only re-runs a cheap smoothstep."""
    if missing(effect, ts):
        return None
    if effect.type == "ID_MASK":
        key = ("id", ts.id_data.name, ts.res, _generation, tuple(effect.id_color), effect.id_tolerance)
        return _cached(key, lambda: id_selection(baked(ts, "ID"), effect.id_color,
                                                 effect.id_tolerance).astype(np.float32))

    key = ("gen", ts.id_data.name, ts.res, _generation, effect.type, effect.level,
           effect.softness, effect.noise_amount, effect.noise_scale, effect.noise_octaves,
           effect.seed, effect.axis, effect.grunge, effect.grunge_tiling)

    def compute():
        if effect.type == "NOISE":
            v = _noise(effect, ts)
        else:
            src_key = ("src", ts.id_data.name, ts.res, _generation, effect.type, effect.axis,
                       effect.grunge, effect.grunge_tiling, effect.seed if effect.type == "GRUNGE" else 0)
            v = _cached(src_key, lambda: _source(effect, ts))
            if effect.noise_amount > 0.0:
                v = v + (_noise(effect, ts) - 0.5) * effect.noise_amount
        t = 1.0 - effect.level
        return smoothstep(t - effect.softness, t + effect.softness, v).astype(np.float32)
    return _cached(key, compute)
