"""CPU layer compositor: flattens a texture set's layer stack into one output
image per channel. The material only samples those outputs, so the shader stays
small no matter how many layers there are."""

import bpy
import numpy as np

from . import pixels
from .channels import enabled_channels


def blend(base, src, mode):
    if mode == "MULTIPLY":
        return base * src
    if mode == "SCREEN":
        return 1.0 - (1.0 - base) * (1.0 - src)
    if mode == "OVERLAY":
        return np.where(base < 0.5, 2.0 * base * src, 1.0 - 2.0 * (1.0 - base) * (1.0 - src))
    if mode == "SOFT_LIGHT":
        return (1.0 - 2.0 * src) * base * base + 2.0 * src * base
    if mode == "ADD":
        return np.minimum(base + src, 1.0)
    if mode == "SUBTRACT":
        return np.maximum(base - src, 0.0)
    if mode == "DARKEN":
        return np.minimum(base, src)
    if mode == "LIGHTEN":
        return np.maximum(base, src)
    if mode == "DIFFERENCE":
        return np.abs(base - src)
    return np.broadcast_to(src, base.shape)


def mix(base, entry):
    src, alpha, mode = entry
    out = blend(base, src, mode)
    if np.ndim(alpha):
        alpha = alpha[..., None]
    return base + (out - base) * alpha


def base_array(ch, res):
    return np.broadcast_to(np.asarray(ch.default, np.float32), (res, res, 3)).copy()


def _read(img, overrides):
    if overrides and img.name in overrides:
        return overrides[img.name]
    return pixels.read(img)


def mask_array(layer, overrides=None):
    """Layer mask as (h, w) float, or None when the layer has no active mask."""
    if not (layer.mask_image and layer.mask_enabled):
        return None
    m = _read(layer.mask_image, overrides)[..., 0]
    return 1.0 - m if layer.mask_invert else m


def layer_entry(layer, ch, overrides=None):
    """(src, alpha, blend_mode) for one layer and channel, or None if it contributes nothing."""
    if not layer.visible:
        return None
    lc = layer.channel(ch.id)
    if lc is None or not lc.enabled:
        return None

    alpha = layer.opacity * lc.opacity
    if layer.type == "FILL":
        val = tuple(lc.color) if ch.is_color else (lc.value,) * 3
        src = np.asarray(val, np.float32)
    else:
        if lc.image is None:
            return None
        px = _read(lc.image, overrides)
        src, alpha = px[..., :3], px[..., 3] * alpha

    mask = mask_array(layer, overrides)
    if mask is not None:
        alpha = mask * alpha
    if not np.ndim(alpha) and alpha <= 0.0:
        return None
    return src, alpha, lc.blend_mode


def composite_entries(base, entries):
    for e in entries:
        base = mix(base, e)
    return base


def composite_channel(ts, ch, overrides=None):
    # Stored top-first, so composite from the end of the list upwards.
    entries = [layer_entry(l, ch, overrides) for l in reversed(ts.layers)]
    return composite_entries(base_array(ch, ts.res), [e for e in entries if e])


def write_output(ts, ch, rgb):
    img = ts.output(ch.id)
    if img is None:
        return
    a = np.ones(rgb.shape[:2], np.float32)
    pixels.write(img, pixels.rgba(np.clip(rgb, 0.0, 1.0), a))


def composite(mat, channels=None):
    from . import material
    ts = mat.paintstack
    if not ts.initialized:
        return
    material.ensure_outputs(mat)
    for ch in channels or enabled_channels(ts):
        write_output(ts, ch, composite_channel(ts, ch))
    material.build(mat)


# Property updates (sliders, toggles) are coalesced into one composite per material.
_pending = set()


def _flush():
    names = list(_pending)
    _pending.clear()
    for name in names:
        mat = bpy.data.materials.get(name)
        if mat is not None:
            composite(mat)
    return None


def request(mat):
    if not mat.paintstack.initialized:
        return
    if not _pending:
        bpy.app.timers.register(_flush, first_interval=0.05)
    _pending.add(mat.name)
