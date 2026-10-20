"""Filter layers: adjust everything composited below them (like adjustment layers)."""

import numpy as np

from . import generators


def blur(rgb, radius):
    from .masks import box_blur
    return np.stack([box_blur(rgb[..., i], radius) for i in range(rgb.shape[-1])], -1)


def rgb_to_hsv(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    safe = np.where(d > 1e-6, d, 1.0)
    h = np.where(mx == r, ((g - b) / safe) % 6.0,
                 np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0)) / 6.0
    h = np.where(d > 1e-6, h, 0.0)
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    return np.stack([h, s, mx], -1)


def hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0] % 1.0, hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0).astype(np.int64) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    choices = [np.stack(c, -1) for c in ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))]
    out = np.zeros_like(hsv)
    for k, c in enumerate(choices):
        out = np.where((i == k)[..., None], c, out)
    return out


def _lightness(v, amount):
    return v + (1.0 - v) * amount if amount >= 0 else v * (1.0 + amount)


def hsl(rgb, ch, hue, saturation, lightness):
    if ch.id == "NORMAL":
        return rgb
    if not ch.is_color:
        return np.clip(_lightness(rgb, lightness), 0.0, 1.0)
    hsv = rgb_to_hsv(np.clip(rgb, 0.0, 1.0))
    hsv[..., 0] += hue
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + saturation), 0.0, 1.0)
    hsv[..., 2] = np.clip(_lightness(hsv[..., 2], lightness), 0.0, 1.0)
    return hsv_to_rgb(hsv)


def _warp_field(res, scale):
    from . import grunge

    def compute():
        period = max(1, int(round(scale)))
        return np.stack([grunge.tileable_fbm(res, (period, period), 3, 11),
                         grunge.tileable_fbm(res, (period, period), 3, 23)], -1) - 0.5
    return generators._cached(("warp", res, round(scale, 3)), compute)


def warp(rgb, amount, scale):
    from .fills import bilinear_wrap
    res = rgb.shape[0]
    d = _warp_field(res, scale) * (amount * 0.05)
    g = (np.arange(res, dtype=np.float32) + 0.5) / res
    u, v = np.meshgrid(g, g)
    return bilinear_wrap(rgb, u + d[..., 0], v + d[..., 1])


def apply(base, layer, ch):
    t = layer.filter_type
    if t == "BLUR":
        return blur(base, layer.filter_radius)
    if t == "SHARPEN":
        out = base + layer.filter_amount * (base - blur(base, layer.filter_radius))
        return out if ch.id == "NORMAL" else np.clip(out, 0.0, 1.0)
    if t == "HSL":
        return hsl(base, ch, layer.hue, layer.saturation, layer.lightness)
    if t == "WARP":
        return warp(base, layer.filter_amount, layer.warp_scale)
    return base
