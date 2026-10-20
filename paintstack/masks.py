"""Layer mask stack: a base painted mask followed by effects (paint, fill,
generators, filters), each blended onto the result below it."""

import numpy as np

from . import generators, pixels

FILTERS = {"LEVELS", "BLUR"}


def _read(img, overrides, cache=None, region=None):
    from .compositor import crop, read
    return crop(read(img, overrides, cache), region)


def box_blur(m, radius, passes=3):
    """Approximate Gaussian blur from repeated separable box filters."""
    w = 2 * radius + 1
    for _ in range(passes):
        p = np.pad(m, [(radius + 1, radius), (0, 0)], mode="edge")
        c = np.cumsum(p, axis=0, dtype=np.float64)
        m = (c[w:] - c[:-w]) / w
        p = np.pad(m, [(0, 0), (radius + 1, radius)], mode="edge")
        c = np.cumsum(p, axis=1, dtype=np.float64)
        m = (c[:, w:] - c[:, :-w]) / w
    return m.astype(np.float32)


def apply_filter(m, effect):
    if effect.type == "LEVELS":
        lo, hi = effect.in_min, max(effect.in_max, effect.in_min + 1e-4)
        return np.clip((m - lo) / (hi - lo), 0.0, 1.0) ** (1.0 / effect.gamma)
    if effect.type == "BLUR":
        return box_blur(m, effect.blur_radius)
    return m


def blend(base, v, alpha, mode):
    if mode == "MULTIPLY":
        out = base * v
    elif mode == "ADD":
        out = base + v
    elif mode == "SUBTRACT":
        out = base - v
    elif mode == "MAX":
        out = np.maximum(base, v)
    elif mode == "MIN":
        out = np.minimum(base, v)
    else:
        out = v
    return np.clip(base + (out - base) * alpha, 0.0, 1.0)


def anchor_value(effect, cache):
    """The anchor's channel result (luminance) remapped by Level / Softness."""
    arr = cache.get(("anchor", effect.anchor_uid, effect.anchor_channel)) if cache else None
    if arr is None:
        return None
    v = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722
    t = 1.0 - effect.level
    return generators.smoothstep(t - effect.softness, t + effect.softness, v).astype(np.float32)


def effect_value(effect, ts, overrides=None, cache=None, region=None):
    """(value, alpha) for a non-filter effect, or None if it has nothing to add."""
    from .compositor import crop
    if effect.type == "PAINT":
        if effect.image is None:
            return None
        px = _read(effect.image, overrides, cache, region)
        return px[..., 0], px[..., 3] * effect.opacity
    if effect.type == "FILL":
        return np.float32(effect.value), effect.opacity
    if effect.type == "ANCHOR":
        v = anchor_value(effect, cache)
        return None if v is None else (v, effect.opacity)
    v = generators.evaluate(effect, ts)
    if v is None:
        return None
    return crop(v, region), effect.opacity


def evaluate(layer, overrides=None, cache=None, region=None):
    """The layer's final mask as a float array (the texture, or just `region`), or
    None when it has no active mask."""
    if not (layer.has_mask and layer.mask_enabled):
        return None
    ts = layer.id_data.paintstack
    if region is not None and any(e.enabled and e.type == "BLUR" for e in layer.mask_effects):
        from .compositor import crop
        return crop(evaluate(layer, overrides, cache), region)  # blur needs neighbours
    if layer.mask_image is not None:
        m = _read(layer.mask_image, overrides, cache, region)[..., 0].copy()
    else:
        from .compositor import region_shape
        m = np.ones(region_shape(ts, region), np.float32)

    for e in layer.mask_effects:
        if not e.enabled:
            continue
        if e.type in FILTERS:
            f = apply_filter(m, e)
            if e.invert:
                f = 1.0 - f
            m = blend(m, f, e.opacity, "NORMAL")
            continue
        ev = effect_value(e, ts, overrides, cache, region)
        if ev is None:
            continue
        v, alpha = ev
        if e.invert:
            v = 1.0 - v
        m = blend(m, np.broadcast_to(v, m.shape), alpha, e.blend_mode)

    return 1.0 - m if layer.mask_invert else m
