"""Brush tips (alpha shapes) and the brush preset library.

Built-in presets ship with the add-on; user presets are JSON files in Blender's
config directory, so they survive add-on updates and are shared between files."""

import json
import os

import bpy
import numpy as np

from . import pixels

TIP_SIZE = 128

TIPS = [
    ("ROUND", "Round", "Round falloff controlled by Hardness"),
    ("SQUARE", "Square", "Square tip, edge softness from Hardness"),
    ("SPLATTER", "Splatter", "Cluster of random droplets"),
    ("NOISE", "Noise", "Cloudy noise inside a soft circle"),
    ("SCRATCHES", "Scratches", "Thin random strokes"),
    ("IMAGE", "Image", "Any image: luminance x alpha"),
]

_tip_cache = {}


def _grid(n=TIP_SIZE):
    g = (np.arange(n, dtype=np.float32) + 0.5) / n * 2.0 - 1.0
    return np.meshgrid(g, g)


def _tip_square(hardness):
    x, y = _grid()
    d = np.maximum(np.abs(x), np.abs(y))
    t = np.clip((d - min(hardness, 0.99)) / (1.0 - min(hardness, 0.99)), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def _tip_splatter(seed):
    rng = np.random.default_rng(seed)
    x, y = _grid()
    out = np.zeros_like(x)
    for _ in range(45):
        r = rng.uniform(0.03, 0.13)
        cx, cy = rng.normal(0.0, 0.35, 2).clip(-0.85, 0.85)
        d = np.hypot(x - cx, y - cy)
        out = np.maximum(out, np.clip((r - d) / (r * 0.3), 0.0, 1.0) * rng.uniform(0.6, 1.0))
    return out


def _tip_noise(seed):
    from . import generators
    x, y = _grid()
    n = generators.fbm(np.stack([x * 3.0, y * 3.0, np.full_like(x, seed * 1.7)], -1), 5, seed)
    circle = np.clip(1.0 - np.hypot(x, y), 0.0, 1.0) ** 0.7
    return np.clip((n - 0.35) * 2.5, 0.0, 1.0) * circle


def _tip_scratches(seed):
    rng = np.random.default_rng(seed)
    x, y = _grid()
    out = np.zeros_like(x)
    for _ in range(14):
        a = rng.uniform(0, np.pi)
        d = np.array([np.cos(a), np.sin(a)])
        c = rng.uniform(-0.5, 0.5, 2)
        length, width = rng.uniform(0.4, 1.0), rng.uniform(0.008, 0.025)
        along = (x - c[0]) * d[0] + (y - c[1]) * d[1]
        across = -(x - c[0]) * d[1] + (y - c[1]) * d[0]
        taper = np.clip(1.0 - np.abs(along) / length, 0.0, 1.0)
        line = np.clip(1.0 - np.abs(across) / (width * taper + 1e-4), 0.0, 1.0) * (taper > 0)
        out = np.maximum(out, line * rng.uniform(0.5, 1.0))
    circle = np.clip((1.0 - np.hypot(x, y)) * 4.0, 0.0, 1.0)
    return out * circle


def image_luminance(img, max_size=1024):
    """(h, w) float luminance x alpha of an image, downsampled to at most max_size."""
    px = pixels.read(img)
    step = max(1, int(np.ceil(max(px.shape[:2]) / max_size)))
    px = px[::step, ::step]
    lum = px[..., 0] * 0.2126 + px[..., 1] * 0.7152 + px[..., 2] * 0.0722
    return np.ascontiguousarray(np.clip(lum * px[..., 3], 0.0, 1.0), np.float32)


def tip_array(brush):
    """Tip shape as a (n, n) float array, or None for the analytic round tip."""
    t = brush.tip
    if t == "ROUND" or (t == "IMAGE" and brush.tip_image is None):
        return None
    if t == "IMAGE":
        return image_luminance(brush.tip_image, 512)
    key = (t, brush.tip_seed, round(brush.hardness, 3) if t == "SQUARE" else 0)
    if key not in _tip_cache:
        if t == "SQUARE":
            arr = _tip_square(brush.hardness)
        elif t == "SPLATTER":
            arr = _tip_splatter(brush.tip_seed)
        elif t == "NOISE":
            arr = _tip_noise(brush.tip_seed)
        else:
            arr = _tip_scratches(brush.tip_seed)
        _tip_cache[key] = np.ascontiguousarray(arr, np.float32)
    return _tip_cache[key]


# Presets

PRESET_KEYS = ("radius", "hardness", "flow", "opacity", "spacing", "use_pressure", "tip", "tip_seed",
               "angle", "follow_stroke", "size_jitter", "angle_jitter", "flow_jitter", "scatter",
               "lazy_radius")

BUILTIN_PRESETS = {
    "Basic Hard": dict(tip="ROUND", hardness=0.9, flow=1.0, spacing=0.1, size_jitter=0.0,
                       angle_jitter=0.0, flow_jitter=0.0, scatter=0.0),
    "Basic Soft": dict(tip="ROUND", hardness=0.0, flow=0.3, spacing=0.1, size_jitter=0.0,
                       angle_jitter=0.0, flow_jitter=0.0, scatter=0.0),
    "Square": dict(tip="SQUARE", hardness=0.8, flow=1.0, spacing=0.1, follow_stroke=True,
                   angle_jitter=0.0, scatter=0.0),
    "Dirt Splatter": dict(tip="SPLATTER", flow=0.8, spacing=0.6, size_jitter=0.4, angle_jitter=1.0,
                          flow_jitter=0.3, scatter=0.5, follow_stroke=False),
    "Cloudy": dict(tip="NOISE", flow=0.25, spacing=0.3, size_jitter=0.3, angle_jitter=1.0,
                   flow_jitter=0.2, scatter=0.2, follow_stroke=False),
    "Scratches": dict(tip="SCRATCHES", flow=0.9, spacing=0.8, size_jitter=0.3, angle_jitter=1.0,
                      flow_jitter=0.2, scatter=0.3, follow_stroke=False),
    "Ink Line": dict(tip="ROUND", hardness=0.95, flow=1.0, spacing=0.05, lazy_radius=25,
                     size_jitter=0.0, angle_jitter=0.0, flow_jitter=0.0, scatter=0.0),
}


def user_preset_dir(create=False):
    return bpy.utils.user_resource("CONFIG", path=os.path.join("paintstack", "brushes"), create=create)


def _safe(name):
    return "".join(c for c in name if c.isalnum() or c in " -_").strip() or "Brush"


def user_presets():
    d = user_preset_dir()
    if not d or not os.path.isdir(d):
        return {}
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            try:
                with open(os.path.join(d, f), encoding="utf-8") as fh:
                    out[f[:-5]] = json.load(fh)
            except (OSError, ValueError):
                continue
    return out


def apply_preset(brush, name):
    values = BUILTIN_PRESETS.get(name) or user_presets().get(name)
    if values is None:
        return False
    for key, val in values.items():
        if key in PRESET_KEYS:
            setattr(brush, key, val)
    brush.preset_name = name
    return True


def save_preset(brush, name):
    name = _safe(name)
    path = os.path.join(user_preset_dir(create=True), name + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({k: getattr(brush, k) for k in PRESET_KEYS}, fh, indent=2)
    brush.preset_name = name
    return name


def delete_preset(name):
    path = os.path.join(user_preset_dir() or "", _safe(name) + ".json")
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False
