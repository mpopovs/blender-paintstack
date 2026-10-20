"""CPU layer compositor: flattens a texture set's layer stack into one output
image per channel. The material only samples those outputs, so the shader stays
small no matter how many layers there are."""

import bpy
import numpy as np

from . import masks, pixels
from .channels import enabled_channels


def decode_normal(n):
    return n * 2.0 - 1.0


def encode_normal(v):
    v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)
    return v * 0.5 + 0.5


def whiteout(base, detail):
    """Combine two encoded tangent-space normals (detail on top of base)."""
    b, d = decode_normal(base), decode_normal(np.broadcast_to(detail, base.shape))
    return encode_normal(np.stack([b[..., 0] + d[..., 0], b[..., 1] + d[..., 1],
                                   b[..., 2] * d[..., 2]], -1))


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
    if mode == "NORMAL_COMBINE":
        return whiteout(base, src)
    return np.broadcast_to(src, base.shape)


def mix(base, entry):
    src, alpha, mode = entry
    out = blend(base, src, mode)
    if np.ndim(alpha):
        alpha = alpha[..., None]
    return base + (out - base) * alpha


def base_array(ts, ch):
    """Starting value of a channel: its default, or the baked normal for Normal."""
    if ch.id == "NORMAL":
        from . import generators
        baked = generators.baked(ts, "NORMAL")
        if baked is not None:
            return baked.copy()
    return np.broadcast_to(np.asarray(ch.default, np.float32), (ts.res, ts.res, 3)).copy()


def _read(img, overrides):
    if overrides and img.name in overrides:
        return overrides[img.name]
    return pixels.read(img)


def layer_mask(layer, overrides=None, cache=None):
    if cache is None:
        return masks.evaluate(layer, overrides)
    key = ("mask", layer.uid)
    if key not in cache:
        cache[key] = masks.evaluate(layer, overrides)
    return cache[key]


def layer_entry(layer, ch, overrides=None, cache=None):
    """(src, alpha, blend_mode) for one layer and channel, or None if it contributes nothing.
    cache: optional dict shared across channels (and stroke updates) so masks and
    unchanged entries are computed once."""
    if cache is not None and ("entry", layer.uid, ch.id) in cache:
        return cache[("entry", layer.uid, ch.id)]
    entry = _layer_entry(layer, ch, overrides, cache)
    if cache is not None:
        cache[("entry", layer.uid, ch.id)] = entry
    return entry


def _layer_entry(layer, ch, overrides, cache):
    if not layer.visible or layer.type == "FOLDER":
        return None
    lc = layer.channel(ch.id)
    if lc is None or not lc.enabled:
        return None

    alpha = layer.opacity * lc.opacity
    if layer.type == "FILL":
        if lc.texture is not None:
            from . import fills
            src = fills.sample(layer.id_data.paintstack, layer, lc.texture,
                               allow_triplanar=ch.id != "NORMAL")
            if not ch.is_color:
                src = np.repeat(src[..., :1] * 0.2126 + src[..., 1:2] * 0.7152
                                + src[..., 2:3] * 0.0722, 3, -1)
        else:
            val = tuple(lc.color) if ch.is_color else (lc.value,) * 3
            src = np.asarray(val, np.float32)
    else:
        if lc.image is None:
            return None
        px = _read(lc.image, overrides)
        src, alpha = px[..., :3], px[..., 3] * alpha

    mask = layer_mask(layer, overrides, cache)
    if mask is not None:
        alpha = mask * alpha
    if not np.ndim(alpha) and alpha <= 0.0:
        return None
    return src, alpha, lc.blend_mode


def composite_entries(base, entries):
    for e in entries:
        base = mix(base, e)
    return base


def composite_nodes(base, nodes, ch, overrides=None, cache=None):
    """Composite a top-first list of (layer, children) nodes onto base.
    Folders are pass-through: their children composite onto what is below, and the
    folder's opacity and mask fade that result in."""
    for layer, children in reversed(nodes):
        if layer.type == "FOLDER":
            if not layer.visible:
                continue
            inner = composite_nodes(base, children, ch, overrides, cache)
            alpha = layer.opacity
            mask = layer_mask(layer, overrides, cache)
            if mask is not None:
                alpha = (mask * alpha)[..., None]
            base = base + (inner - base) * alpha
        else:
            entry = layer_entry(layer, ch, overrides, cache)
            if entry:
                base = mix(base, entry)
    return base


def composite_channel(ts, ch, overrides=None, cache=None):
    from . import stack
    return composite_nodes(base_array(ts, ch), stack.tree(ts), ch, overrides, cache)


def write_output(ts, ch, rgb):
    img = ts.output(ch.id)
    if img is None:
        return
    if ch.id == "NORMAL":
        rgb = encode_normal(decode_normal(rgb))
    a = np.ones(rgb.shape[:2], np.float32)
    pixels.write(img, pixels.rgba(np.clip(rgb, 0.0, 1.0), a))


def write_mask_view(mat, cache=None):
    """Grey image of the active layer's final mask, for the Layer Mask view."""
    from . import material
    ts = mat.paintstack
    img = material.mask_view_image(mat)
    layer = ts.active_layer
    m = layer_mask(layer, cache=cache) if layer is not None else None
    if m is None:
        m = np.ones((ts.res, ts.res), np.float32)
    pixels.write(img, pixels.rgba(np.repeat(m[..., None], 3, -1), np.ones_like(m)))


def composite(mat, channels=None):
    from . import material
    ts = mat.paintstack
    if not ts.initialized:
        return
    material.ensure_outputs(mat)
    cache = {}
    for ch in channels or enabled_channels(ts):
        write_output(ts, ch, composite_channel(ts, ch, cache=cache))
    if ts.view_mode == "MASK":
        write_mask_view(mat, cache)
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
