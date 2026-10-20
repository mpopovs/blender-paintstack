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
}

_CACHE_SIZE = 24
_cache = OrderedDict()
_generation = 0


def invalidate():
    """Call after baking: drops every cached map and generator result."""
    global _generation
    _generation += 1
    _cache.clear()


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

def _hash(ix, iy, iz, seed):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761 + 1013904223)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF).astype(np.float32) * (1.0 / 65535.0)


def value_noise(p, seed=0):
    """Smooth 3D value noise in [0, 1] for points p of shape (..., 3)."""
    i = np.floor(p).astype(np.int64)
    f = (p - i).astype(np.float32)
    u = f * f * (3.0 - 2.0 * f)
    out = np.zeros(p.shape[:-1], np.float32)
    for dx in (0, 1):
        wx = u[..., 0] if dx else 1.0 - u[..., 0]
        for dy in (0, 1):
            wy = u[..., 1] if dy else 1.0 - u[..., 1]
            for dz in (0, 1):
                wz = u[..., 2] if dz else 1.0 - u[..., 2]
                out += wx * wy * wz * _hash(i[..., 0] + dx, i[..., 1] + dy, i[..., 2] + dz, seed)
    return out


def fbm(p, octaves=4, seed=0):
    total = np.zeros(p.shape[:-1], np.float32)
    amp, norm = 0.5, 0.0
    for o in range(octaves):
        total += amp * value_noise(p * (2.0 ** o), seed + o * 31)
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
    if t == "NOISE":
        return fbm(_positions(ts) * effect.noise_scale, effect.noise_octaves, effect.seed)
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


def evaluate(effect, ts):
    """Generator output as (res, res) float in [0, 1], or None if a bake is missing."""
    if missing(effect, ts):
        return None
    key = ("gen", ts.id_data.name, ts.res, _generation, effect.type, effect.level,
           effect.softness, effect.noise_amount, effect.noise_scale, effect.noise_octaves,
           effect.seed, effect.axis)

    def compute():
        v = _source(effect, ts)
        if effect.type != "NOISE" and effect.noise_amount > 0.0:
            n = fbm(_positions(ts) * effect.noise_scale, effect.noise_octaves, effect.seed)
            v = v + (n - 0.5) * effect.noise_amount
        t = 1.0 - effect.level
        return smoothstep(t - effect.softness, t + effect.softness, v).astype(np.float32)
    return _cached(key, compute)
