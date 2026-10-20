"""CPU layer compositor: flattens a texture set's layer stack into one output
image per channel. The material only samples those outputs, so the shader stays
small no matter how many layers there are.

Performance model
-----------------
* Each material keeps a persistent cache (State) of decoded layer images,
  layer entries and masks. Property updates invalidate only the layer they
  belong to (tracked by per-layer revisions), so re-compositing after a tweak
  reuses every other layer's work. The cache is bounded by a memory budget.
* A "below checkpoint" stores the composite of everything under the most
  recently edited root layer, so repeated tweaks only re-blend that layer and
  the ones above it.
* Strokes composite only the texture-space regions they touched (`region`),
  slicing cached full-resolution entries instead of recomputing them.
* Blending is done in place to avoid full-resolution temporaries."""

from collections import OrderedDict

import bpy
import numpy as np

from . import hardware, masks, pixels
from .channels import BY_ID as BY_ID_, enabled_channels

BUDGET_OVERRIDE_MB = None  # set by tests / scripts; otherwise the add-on preference


# Normals

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


# Blending

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
    return src


def _rows(arr, y0, y1):
    """Rows y0:y1 of a per-texel array; constants pass through unchanged."""
    return arr[y0:y1] if np.ndim(arr) >= 2 else arr


def _mix_rows(base, src, alpha, mode):
    if mode == "NORMAL" and not np.ndim(alpha) and alpha >= 1.0:
        base[...] = src
        return
    out = blend(base, src, mode)
    d = np.subtract(out, base, dtype=np.float32)
    if np.ndim(alpha):
        d *= alpha[..., None]
    else:
        d *= np.float32(alpha)
    base += d


def mix(base, entry):
    """Blend entry onto base IN PLACE (base must be an owned float32 array).
    Large arrays are processed in row bands on the performance cores."""
    src, alpha, mode = entry
    hardware.parallel_rows(lambda y0, y1: _mix_rows(base[y0:y1], _rows(src, y0, y1),
                                                    _rows(alpha, y0, y1), mode),
                           base.shape[0], base.size)
    return base


def _fade_rows(base, inner, alpha):
    d = np.subtract(inner, base, dtype=np.float32)
    d *= alpha[..., None] if np.ndim(alpha) else np.float32(alpha)
    base += d


def fade(base, inner, alpha):
    """base = base + (inner - base) * alpha, in place."""
    hardware.parallel_rows(lambda y0, y1: _fade_rows(base[y0:y1], inner[y0:y1], _rows(alpha, y0, y1)),
                           base.shape[0], base.size)
    return base


# Regions: (y0, y1, x0, x1) in texels, or None for the whole texture

def crop(arr, region):
    if region is None or np.ndim(arr) < 2:
        return arr
    y0, y1, x0, x1 = region
    return arr[y0:y1, x0:x1]


def region_shape(ts, region):
    if region is None:
        return ts.res, ts.res
    return region[1] - region[0], region[3] - region[2]


def base_array(ts, ch, region=None):
    """Starting value of a channel (an owned array): its default, or the baked
    normal for Normal."""
    if ch.id == "NORMAL":
        from . import generators
        baked = generators.baked(ts, "NORMAL")
        if baked is not None:
            return np.array(crop(baked, region), np.float32)
    return np.broadcast_to(np.asarray(ch.default, np.float32), (*region_shape(ts, region), 3)).copy()


# Persistent per-material state

class State:
    def __init__(self):
        self.cache = OrderedDict()  # ("px", image) / ("entry", uid, ch) / ("mask", uid) / ("anchor", ...)
        self.revs = {}  # uid -> revision, bumped when the layer changes
        self.checkpoint = None  # (root_uid, signature, {channel: base below root})


_states = {}


def state(mat):
    st = _states.get(mat.name)
    if st is None:
        st = _states[mat.name] = State()
    return st


def invalidate(mat, uid=None, channel=None):
    """Forget cached work for one layer (uid) or, with no uid, the whole material.
    With a channel, only that channel's entry is dropped (its mask stays valid)."""
    if uid is None:
        _states.pop(mat.name, None)
        return
    st = state(mat)
    st.revs[uid] = st.revs.get(uid, 0) + 1
    if channel is not None:
        for key in (("entry", uid, channel), ("bbox", uid, channel)):
            st.cache.pop(key, None)
        return
    layer = next((l for l in mat.paintstack.layers if l.uid == uid), None)
    names = {i.name for i in _layer_images(layer)} if layer is not None else set()
    for key in [k for k in st.cache if (k[0] != "px" and k[1] == uid) or (k[0] == "px" and k[1] in names)]:
        del st.cache[key]


def invalidate_all():
    _states.clear()


def _budget_bytes():
    mb = BUDGET_OVERRIDE_MB
    if mb is None:
        prefs = bpy.context.preferences.addons.get(__package__)
        mb = getattr(prefs.preferences, "cache_budget_mb", 0) if prefs else 0
    if mb <= 0:  # Auto: scale with (unified) memory
        mb = hardware.auto_cache_budget_mb()
    return mb * 1024 * 1024


def _nbytes(v):
    if isinstance(v, np.ndarray):
        return v.nbytes if v.base is None else 0  # views share their owner's memory
    if isinstance(v, tuple):
        return sum(_nbytes(x) for x in v)
    return 0


def _trim(st, ts):
    """Drop entries of removed layers, then evict oldest entries over budget."""
    uids = {l.uid for l in ts.layers}
    for key in [k for k in st.cache if k[0] in ("entry", "mask") and k[1] not in uids]:
        del st.cache[key]
    budget = _budget_bytes()
    total = sum(_nbytes(v) for v in st.cache.values())
    while total > budget and st.cache:
        _, v = st.cache.popitem(last=False)
        total -= _nbytes(v)


def _layer_images(layer):
    if layer is None:
        return []
    imgs = [lc.image for lc in layer.channels if lc.image]
    if layer.mask_image:
        imgs.append(layer.mask_image)
    imgs += [e.image for e in layer.mask_effects if e.image]
    return imgs


def read(img, overrides=None, cache=None):
    """Decoded RGBA pixels: an override (live stroke data), the cache, or the image."""
    if overrides and img.name in overrides:
        return overrides[img.name]
    if cache is None:
        return pixels.read(img)
    key = ("px", img.name)
    if key not in cache:
        cache[key] = pixels.read(img)
    return cache[key]


def _volatile(layer, overrides):
    """Layers whose images are being painted, or whose mask reads anchors, are
    recomputed every time instead of cached."""
    if any(e.type == "ANCHOR" and e.enabled for e in layer.mask_effects):
        return True
    return bool(overrides) and any(i.name in overrides for i in _layer_images(layer))


# Layers

def layer_mask(layer, overrides=None, cache=None, region=None):
    if cache is None or _volatile(layer, overrides):
        return masks.evaluate(layer, overrides, cache, region)
    key = ("mask", layer.uid)
    if key not in cache:
        cache[key] = masks.evaluate(layer, overrides, cache)
    return crop(cache[key], region)


def _alpha_bbox(alpha):
    """Bounding box (y0, y1, x0, x1) of alpha > 0, or None when not worth cropping."""
    if not np.ndim(alpha):
        return None
    rows = np.flatnonzero(alpha.any(axis=1))
    if rows.size == 0:
        return (0, 0, 0, 0)
    cols = np.flatnonzero(alpha[rows[0]:rows[-1] + 1].any(axis=0))
    box = (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)
    area = (box[1] - box[0]) * (box[3] - box[2])
    return box if area < 0.6 * alpha.size else None


def entry_bbox(layer, ch, cache):
    key = ("bbox", layer.uid, ch.id)
    if key not in cache:
        entry = cache.get(("entry", layer.uid, ch.id))
        cache[key] = _alpha_bbox(entry[1]) if entry else None
    return cache[key]


def covers(layer, ch):
    """True if the layer fully replaces this channel of everything below it."""
    if not layer.visible or (layer.has_mask and layer.mask_enabled):
        return False
    if layer.type == "FOLDER":
        return layer.opacity >= 1.0 and any(covers(c, ch) for c in _children(layer))
    if layer.type != "FILL":
        return False
    lc = layer.channel(ch.id)
    return (lc is not None and lc.enabled and lc.blend_mode == "NORMAL"
            and layer.opacity * lc.opacity >= 1.0)


def _children(folder):
    return [l for l in folder.id_data.paintstack.layers if l.parent_uid == folder.uid]


def layer_entry(layer, ch, overrides=None, cache=None, region=None):
    """(src, alpha, blend_mode) for one layer and channel, or None if it contributes
    nothing. Unchanged layers come from the cache (cropped to region)."""
    if cache is None or _volatile(layer, overrides):
        return _layer_entry(layer, ch, overrides, cache, region)
    key = ("entry", layer.uid, ch.id)
    if key not in cache:
        cache[key] = _layer_entry(layer, ch, overrides, cache, None)
    entry = cache[key]
    if entry is None or region is None:
        return entry
    src, alpha, mode = entry
    return crop(src, region), crop(alpha, region), mode


def _layer_entry(layer, ch, overrides, cache, region):
    if not layer.visible or layer.type in ("FOLDER", "FILTER"):
        return None
    lc = layer.channel(ch.id)
    if lc is None or not lc.enabled:
        return None

    alpha = np.float32(layer.opacity * lc.opacity)
    if layer.type == "FILL":
        if lc.texture is not None:
            from . import fills
            src = fills.sample(layer.id_data.paintstack, layer, lc.texture,
                               allow_triplanar=ch.id != "NORMAL")
            if not ch.is_color:
                src = np.repeat(src[..., :1] * 0.2126 + src[..., 1:2] * 0.7152
                                + src[..., 2:3] * 0.0722, 3, -1)
            src = crop(np.asarray(src, np.float32), region)
        else:
            val = tuple(lc.color) if ch.is_color else (lc.value,) * 3
            src = np.asarray(val, np.float32)
    else:
        if lc.image is None:
            return None
        px = crop(read(lc.image, overrides, cache), region)
        src, alpha = px[..., :3], px[..., 3] * alpha

    mask = layer_mask(layer, overrides, cache, region)
    if mask is not None:
        alpha = mask * alpha
    if not np.ndim(alpha) and alpha <= 0.0:
        return None
    return src, alpha, lc.blend_mode


def composite_entries(base, entries):
    for e in entries:
        base = mix(base, e)
    return base


def composite_nodes(base, nodes, ch, overrides=None, cache=None, region=None):
    """Composite a top-first list of (layer, children) nodes onto base (in place).
    Folders are pass-through: their children composite onto what is below, and the
    folder's opacity and mask fade that result in. Filter layers process what is
    below them. Anchor layers record the result at their position in `cache`."""
    if nodes and not any(l.is_anchor for l in nodes[0][0].id_data.paintstack.layers):
        # Everything below the topmost opaque, unmasked layer is hidden: skip it.
        top = next((k for k, (l, _) in enumerate(nodes) if covers(l, ch)), None)
        if top is not None:
            nodes = nodes[:top + 1]
    for layer, children in reversed(nodes):
        if layer.type in ("FOLDER", "FILTER"):
            lc = layer.channel(ch.id)
            if layer.visible and (layer.type == "FOLDER" or (lc is not None and lc.enabled)):
                if layer.type == "FOLDER" and layer.opacity >= 1.0 and not (layer.has_mask and layer.mask_enabled):
                    base = composite_nodes(base, children, ch, overrides, cache, region)  # in place
                elif layer.type == "FOLDER":
                    inner = composite_nodes(base.copy(), children, ch, overrides, cache, region)
                    alpha = np.float32(layer.opacity)
                else:
                    from . import filters
                    inner = filters.apply(base, layer, ch)
                    alpha = np.float32(layer.opacity * lc.opacity)
                if layer.type == "FILTER" or not (layer.opacity >= 1.0 and not (layer.has_mask and layer.mask_enabled)):
                    mask = layer_mask(layer, overrides, cache, region)
                    if mask is not None:
                        alpha = mask * alpha
                    base = fade(base, inner, alpha)
        else:
            entry = layer_entry(layer, ch, overrides, cache, region)
            if entry:
                box = None
                if region is None and cache is not None and not _volatile(layer, overrides):
                    box = entry_bbox(layer, ch, cache)
                if box is None:
                    base = mix(base, entry)
                elif box[1] > box[0]:
                    # Sparse mask: blend only inside its bounding box (in place on a view).
                    src, alpha, mode = entry
                    mix(base[box[0]:box[1], box[2]:box[3]], (crop(src, box), crop(alpha, box), mode))
        if layer.is_anchor and cache is not None:
            cache[("anchor", layer.uid, ch.id)] = base.copy()
    return base


def composite_channel(ts, ch, overrides=None, cache=None, region=None):
    from . import stack
    return composite_nodes(base_array(ts, ch, region), stack.tree(ts), ch, overrides, cache, region)


def filter_reach(ts):
    """Texels a filter layer reads around each output texel (None = whole image)."""
    reach = 0
    for l in ts.layers:
        if l.type == "FILTER" and l.visible:
            if l.filter_type == "WARP":
                return None
            if l.filter_type in ("BLUR", "SHARPEN"):
                reach += 3 * l.filter_radius + 1
    return reach


# Output

def finalize(ch, rgb):
    if ch.id == "NORMAL":
        rgb = encode_normal(decode_normal(rgb))
    return np.clip(rgb, 0.0, 1.0)


_out_buffers = {}


def write_output(ts, ch, rgb):
    img = ts.output(ch.id)
    if img is None:
        return
    key = (img.name, rgb.shape)
    buf = _out_buffers.get(key)
    if buf is None:
        buf = _out_buffers[key] = np.ones((*rgb.shape[:2], 4), np.float32)
    if ch.id == "NORMAL":
        rgb = encode_normal(decode_normal(rgb))
    hardware.parallel_rows(lambda y0, y1: np.clip(rgb[y0:y1], 0.0, 1.0, out=buf[y0:y1, :, :3]),
                           rgb.shape[0], rgb.size)
    pixels.write(img, buf)


def write_mask_view(mat, cache=None, overrides=None):
    """Grey image of the active layer's final mask, for the Layer Mask view."""
    from . import material
    ts = mat.paintstack
    img = material.mask_view_image(mat)
    layer = ts.active_layer
    m = layer_mask(layer, overrides, cache) if layer is not None else None
    if m is None:
        m = np.ones((ts.res, ts.res), np.float32)
    pixels.write(img, pixels.rgba(np.repeat(m[..., None], 3, -1), np.ones_like(m)))


def anchor_channels(ts):
    """Channels read by Anchor mask effects (they must be composited first)."""
    anchors = {l.uid for l in ts.layers if l.is_anchor}
    out = []
    for l in ts.layers:
        for e in l.mask_effects:
            if e.type == "ANCHOR" and e.enabled and e.anchor_uid in anchors:
                ch = BY_ID_[e.anchor_channel]
                if ch not in out:
                    out.append(ch)
    return out


def _forget_anchors(cache):
    for key in [k for k in cache if k[0] == "anchor"]:
        del cache[key]


def composite_all(ts, channels=None, overrides=None, cache=None, region=None):
    """{channel_id: rgb} for the given channels (default: enabled ones), compositing
    anchor-referenced channels first so their anchors exist when masks need them."""
    cache = {} if cache is None else cache
    _forget_anchors(cache)
    wanted = list(channels or enabled_channels(ts))
    first = anchor_channels(ts)
    results = {}
    for ch in first + [c for c in wanted if c not in first]:
        out = composite_channel(ts, ch, overrides, cache, region)
        if ch in wanted:
            results[ch.id] = out
    return results


def _signature(ts, layers):
    from . import generators
    return (ts.res, tuple(c.id for c in enabled_channels(ts)), generators._generation,
            tuple((l.uid, l.parent_uid, l.type) for l in layers))


def _checkpointed(mat, changed_uid, cache, channels=None):
    """Composite reusing the stored result below the changed layer's root node.
    Returns {channel_id: rgb}, or None when a checkpoint does not apply."""
    from . import stack
    ts = mat.paintstack
    st = state(mat)
    if anchor_channels(ts):
        return None
    nodes = stack.tree(ts)
    lookup = stack.by_uid(ts)
    layer = lookup.get(changed_uid)
    if layer is None:
        return None
    root = ([layer] + stack.ancestors(ts, layer, lookup))[-1]
    i = next(k for k, (l, _) in enumerate(nodes) if l.uid == root.uid)
    start = next(k for k, l in enumerate(ts.layers) if l.uid == root.uid)
    below_layers = list(ts.layers[stack.block_end(ts, start):])
    sig = (_signature(ts, below_layers), tuple(st.revs.get(l.uid, 0) for l in below_layers))
    cp = st.checkpoint
    if cp is None or cp[0] != root.uid or cp[1] != sig:
        bases = {ch.id: composite_nodes(base_array(ts, ch), nodes[i + 1:], ch, None, cache)
                 for ch in enabled_channels(ts)}
        st.checkpoint = cp = (root.uid, sig, bases)
    wanted = {c.id for c in (channels or enabled_channels(ts))}
    return {cid: composite_nodes(b.copy(), nodes[:i + 1], BY_ID_[cid], None, cache)
            for cid, b in cp[2].items() if cid in wanted}


def composite(mat, channels=None, changed=None):
    """Recomposite and write the outputs. `changed`: uid of the single layer that
    changed since the last composite, which enables the below-checkpoint."""
    from . import material
    ts = mat.paintstack
    if not ts.initialized:
        return
    material.ensure_outputs(mat)
    st = state(mat)
    cache = st.cache
    _forget_anchors(cache)
    results = _checkpointed(mat, changed, cache, channels) if changed else None
    if results is None:
        results = composite_all(ts, channels, cache=cache)
    for cid, rgb in results.items():
        write_output(ts, BY_ID_[cid], rgb)
    if ts.view_mode == "MASK":
        write_mask_view(mat, cache)
    _trim(st, ts)
    material.build(mat)


# Property updates (sliders, toggles) are coalesced into one composite per material.
_pending = {}


_VIEW = "__view__"


def _flush():
    pending = dict(_pending)
    _pending.clear()
    for name, uids in pending.items():
        mat = bpy.data.materials.get(name)
        if mat is None:
            continue
        layer_uids = {u for u, _ in uids if u != _VIEW}
        single = next(iter(layer_uids)) if len(layer_uids) == 1 else None
        chans = {c for u, c in uids if u != _VIEW}
        # Only per-channel changes of one layer: recomposite just those channels.
        channels = [BY_ID_[c] for c in chans] if single and None not in chans else None
        composite(mat, channels, changed=single)
    return None


def _schedule(mat, key):
    if not _pending:
        bpy.app.timers.register(_flush, first_interval=0.05)
    _pending.setdefault(mat.name, set()).add(key)


def request(mat, uid=None, channel=None):
    """Schedule a composite. uid: the layer that changed (None = texture-set wide,
    which drops the material's cache); channel: the only channel it affects."""
    if not mat.paintstack.initialized:
        return
    invalidate(mat, uid, channel)
    _schedule(mat, (uid, channel))


def request_view(mat):
    """Schedule a composite for a view change: nothing is invalidated."""
    if mat.paintstack.initialized:
        _schedule(mat, (_VIEW, None))
