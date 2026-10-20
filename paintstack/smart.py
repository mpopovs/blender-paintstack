"""Smart materials and smart masks.

A smart material is a stack of layers (usually a folder); a smart mask is a mask
stack. Both re-bind to the target texture set when applied: generators read the
target mesh's baked maps, so the same asset adapts to every mesh.

Built-ins are Python recipes (specs). User assets are saved as .blend files in
the PaintStack asset library: one Material per file, holding a PaintStack
texture set with the layers and their packed images, marked as an asset so it
shows in Blender's Asset Browser."""

import os
import uuid

import bpy

from . import grunge, pixels

KIND_MATERIAL = "SMART_MATERIAL"
KIND_MASK = "SMART_MASK"
LIBRARY_NAME = "PaintStack"

# Specs: a layer is {"type", "name", "channels": {CHANNEL: {prop: value}}, "effects": [...],
# "children": [...], plus any PS_Layer property}. Channel props are those of PS_LayerChannel,
# with "grunge": (kind, seed) to use a library texture as the fill texture.

_EDGE_WEAR = {"type": "EDGE_WEAR", "level": 0.4, "softness": 0.12, "noise_amount": 0.6, "noise_scale": 25.0}
_DIRT = {"type": "DIRT", "level": 0.45, "softness": 0.2, "noise_amount": 0.4}


def _fill(name, channels, effects=(), **props):
    return {"type": "FILL", "name": name, "channels": channels, "effects": list(effects), **props}


def _only(channels, *enabled):
    """Disable every channel not listed (fill layers otherwise cover all channels)."""
    from .channels import CHANNELS
    out = {c.id: {"enabled": False} for c in CHANNELS}
    for cid in enabled:
        out[cid] = dict(channels.get(cid, {}), enabled=True)
    return out


SMART_MATERIALS = {
    "Painted Steel": [
        _fill("Grime", _only({"BASE_COLOR": {"color": (0.09, 0.08, 0.06)},
                              "ROUGHNESS": {"value": 0.85}}, "BASE_COLOR", "ROUGHNESS"), [_DIRT]),
        _fill("Paint", _only({"BASE_COLOR": {"color": (0.12, 0.27, 0.52)}, "METALLIC": {"value": 0.0},
                              "ROUGHNESS": {"value": 0.45}, "HEIGHT": {"value": 0.53}},
                             "BASE_COLOR", "METALLIC", "ROUGHNESS", "HEIGHT"),
              [dict(_EDGE_WEAR, invert=True),
               {"type": "GRUNGE", "grunge": "RUST", "grunge_tiling": 3.0, "level": 0.12, "softness": 0.05,
                "noise_amount": 0.0, "blend_mode": "MULTIPLY", "invert": True}]),
        _fill("Steel", {"BASE_COLOR": {"color": (0.56, 0.57, 0.58)}, "METALLIC": {"value": 1.0},
                        "ROUGHNESS": {"value": 0.3}, "HEIGHT": {"value": 0.5}}),
    ],
    "Rusty Iron": [
        _fill("Rust", _only({"BASE_COLOR": {"color": (0.35, 0.12, 0.04)}, "METALLIC": {"value": 0.0},
                             "ROUGHNESS": {"value": 0.92}, "HEIGHT": {"value": 0.56}},
                            "BASE_COLOR", "METALLIC", "ROUGHNESS", "HEIGHT"),
              [{"type": "GRUNGE", "grunge": "RUST", "grunge_tiling": 2.0, "level": 0.45, "softness": 0.1,
                "noise_amount": 0.0},
               dict(_DIRT, blend_mode="MAX")]),
        _fill("Iron", {"BASE_COLOR": {"color": (0.33, 0.33, 0.34)}, "METALLIC": {"value": 1.0},
                       "ROUGHNESS": {"value": 0.55, "grunge": ("GRUNGE", 3)}, "HEIGHT": {"value": 0.5}},
              tiling=3.0, projection="TRIPLANAR"),
    ],
    "Worn Gold": [
        _fill("Dirt", _only({"BASE_COLOR": {"color": (0.12, 0.09, 0.05)}, "ROUGHNESS": {"value": 0.8}},
                            "BASE_COLOR", "ROUGHNESS"), [_DIRT]),
        _fill("Polished Edges", _only({"BASE_COLOR": {"color": (1.0, 0.86, 0.57)},
                                       "ROUGHNESS": {"value": 0.12}}, "BASE_COLOR", "ROUGHNESS"),
              [_EDGE_WEAR]),
        _fill("Gold", {"BASE_COLOR": {"color": (0.94, 0.75, 0.38)}, "METALLIC": {"value": 1.0},
                       "ROUGHNESS": {"value": 0.32}, "HEIGHT": {"value": 0.5}}),
    ],
    "Dusty Plastic": [
        _fill("Dust", _only({"BASE_COLOR": {"color": (0.62, 0.6, 0.56)}, "ROUGHNESS": {"value": 0.95}},
                            "BASE_COLOR", "ROUGHNESS"),
              [{"type": "POSITION", "axis": "Z", "level": 0.35, "softness": 0.25, "noise_amount": 0.2},
               {"type": "GRUNGE", "grunge": "DUST", "grunge_tiling": 3.0, "level": 0.6, "softness": 0.3,
                "noise_amount": 0.0, "blend_mode": "MULTIPLY"}]),
        _fill("Plastic", {"BASE_COLOR": {"color": (0.7, 0.08, 0.06)}, "METALLIC": {"value": 0.0},
                          "ROUGHNESS": {"value": 0.38}, "HEIGHT": {"value": 0.5}}),
    ],
    "Concrete": [
        _fill("Leaks", _only({"BASE_COLOR": {"color": (0.25, 0.24, 0.22)}}, "BASE_COLOR"),
              [{"type": "GRUNGE", "grunge": "LEAKS", "grunge_tiling": 2.0, "level": 0.35, "softness": 0.25,
                "noise_amount": 0.0}], opacity=0.6),
        _fill("Concrete", {"BASE_COLOR": {"color": (0.55, 0.54, 0.51), "grunge": ("CONCRETE", 0),
                                          "blend_mode": "MULTIPLY"},
                           "METALLIC": {"value": 0.0}, "ROUGHNESS": {"value": 0.9},
                           "HEIGHT": {"grunge": ("CONCRETE", 0)}},
              tiling=2.0, projection="TRIPLANAR"),
        _fill("Base", {"BASE_COLOR": {"color": (0.62, 0.61, 0.58)}, "METALLIC": {"value": 0.0},
                       "ROUGHNESS": {"value": 0.9}, "HEIGHT": {"value": 0.5}}),
    ],
    "Chrome": [
        _fill("Chrome", {"BASE_COLOR": {"color": (0.9, 0.9, 0.9)}, "METALLIC": {"value": 1.0},
                         "ROUGHNESS": {"value": 0.05}, "HEIGHT": {"value": 0.5}}),
    ],
    "Rubber": [
        _fill("Rubber", {"BASE_COLOR": {"color": (0.04, 0.04, 0.04)}, "METALLIC": {"value": 0.0},
                         "ROUGHNESS": {"value": 0.8}, "HEIGHT": {"value": 0.5}}),
    ],
}

SMART_MASKS = {
    "Edge Wear": [_EDGE_WEAR],
    "Dirt": [_DIRT],
    "Top Dust": [{"type": "POSITION", "axis": "Z", "level": 0.35, "softness": 0.25, "noise_amount": 0.2},
                 {"type": "GRUNGE", "grunge": "DUST", "grunge_tiling": 3.0, "level": 0.6, "softness": 0.3,
                  "noise_amount": 0.0, "blend_mode": "MULTIPLY"}],
    "Rust Spots": [{"type": "GRUNGE", "grunge": "RUST", "grunge_tiling": 2.0, "level": 0.4, "softness": 0.1,
                    "noise_amount": 0.0}],
    "Leaks": [{"type": "GRUNGE", "grunge": "LEAKS", "grunge_tiling": 2.0, "level": 0.4, "softness": 0.25,
               "noise_amount": 0.0}],
    "Scratches": [{"type": "GRUNGE", "grunge": "SCRATCHES", "grunge_tiling": 2.0, "level": 0.8,
                   "softness": 0.2, "noise_amount": 0.0}],
    "Cracks": [{"type": "GRUNGE", "grunge": "CRACKS", "grunge_tiling": 1.5, "level": 0.6, "softness": 0.2,
                "noise_amount": 0.0}],
}


# Applying specs

def _apply_effect(mat, layer, spec):
    from . import ops
    e = ops.add_mask_effect(mat, layer, spec["type"])
    for key, val in spec.items():
        if key != "type":
            setattr(e, key, val)
    return e


def _apply_layer(mat, spec, parent_uid, index):
    from . import ops
    layer = ops.new_layer(mat, spec["type"], spec.get("name"), index=index)
    layer.parent_uid = parent_uid
    for key, val in spec.items():
        if key in ("type", "name", "channels", "effects", "children"):
            continue
        setattr(layer, key, val)
    for cid, props in spec.get("channels", {}).items():
        lc = layer.channel(cid)
        for key, val in props.items():
            if key == "grunge":
                lc.texture = grunge.make_image(*val)
            else:
                setattr(lc, key, val)
    for e in spec.get("effects", ()):
        _apply_effect(mat, layer, e)
    return layer


def apply_spec(mat, name, layer_specs):
    """Insert a smart material spec as a folder above the active layer."""
    from . import ops
    ts = mat.paintstack
    folder = ops.new_layer(mat, "FOLDER", name)
    folder_uid = folder.uid
    index = next(i for i, l in enumerate(ts.layers) if l.uid == folder_uid)
    # Insert children bottom-up right below the folder, so the list stays top-first.
    for spec in reversed(layer_specs):
        _apply_layer(mat, spec, folder_uid, index + 1)
    ts.active_layer_index = next(i for i, l in enumerate(ts.layers) if l.uid == folder_uid)
    return ts.active_layer


def apply_mask_spec(mat, layer, effect_specs):
    for spec in effect_specs:
        _apply_effect(mat, layer, spec)


# Copying layers between texture sets (used by save and load)

def _copy_props(src, dst, skip=()):
    for prop in src.bl_rna.properties:
        ident = prop.identifier
        if ident in ("rna_type",) or ident in skip or prop.is_readonly and prop.type != "COLLECTION":
            continue
        if prop.type == "COLLECTION":
            dst_coll = getattr(dst, ident)
            dst_coll.clear()
            for item in getattr(src, ident):
                _copy_props(item, dst_coll.add())
        else:
            setattr(dst, ident, getattr(src, ident))


def layer_images(layer):
    imgs = [lc.image for lc in layer.channels if lc.image] + [lc.texture for lc in layer.channels if lc.texture]
    imgs += [layer.mask_image] if layer.mask_image else []
    imgs += [e.image for e in layer.mask_effects if e.image]
    return imgs


def copy_layers(src_ts, src_indices, dst_mat, index, copy_images):
    """Copy layers (a contiguous top-first block) into dst_mat's stack at index.
    uids are regenerated and parent / anchor references remapped."""
    dst = dst_mat.paintstack
    remap = {src_ts.layers[i].uid: uuid.uuid4().hex[:8] for i in src_indices}
    new_uids = []
    for k, i in enumerate(src_indices):
        src = src_ts.layers[i]
        layer = dst.layers.add()
        _copy_props(src, layer)
        layer.uid = remap[src.uid]
        layer.parent_uid = remap.get(src.parent_uid, "")
        for e in layer.mask_effects:
            if e.anchor_uid in remap:
                e.anchor_uid = remap[e.anchor_uid]
        if copy_images:
            for lc in layer.channels:
                if lc.image:
                    lc.image = pixels.clone_image(lc.image)
            if layer.mask_image:
                layer.mask_image = pixels.clone_image(layer.mask_image)
            for e in layer.mask_effects:
                if e.image:
                    e.image = pixels.clone_image(e.image)
        size = dst.size
        for img in layer_images(layer):
            if img.name.startswith(pixels.PREFIX) and tuple(img.size) != size \
                    and "Grunge_" not in img.name:
                img.scale(*size)
        # Record the uid before moving: after move() the reference points at whatever
        # item now occupies the old slot.
        new_uids.append(layer.uid)
        dst.layers.move(len(dst.layers) - 1, index + k)
    return new_uids


# Library on disk

def library_dir(create=False):
    return bpy.utils.user_resource("DATAFILES", path=os.path.join("paintstack", "library"), create=create)


def kind_dir(kind, create=False):
    base = library_dir(create)
    if not base:
        return None
    d = os.path.join(base, "smart_materials" if kind == KIND_MATERIAL else "smart_masks")
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def ensure_asset_library():
    """Register the library folder as an Asset Browser library (once)."""
    directory = library_dir(create=True)
    libs = bpy.context.preferences.filepaths.asset_libraries
    for lib in libs:
        if os.path.normpath(bpy.path.abspath(lib.path)) == os.path.normpath(directory):
            return lib
    try:
        lib = libs.new(name=LIBRARY_NAME, directory=directory)
    except (AttributeError, TypeError, RuntimeError):
        try:
            bpy.ops.preferences.asset_library_add(directory=directory)
            lib = libs[-1]
            lib.name = LIBRARY_NAME
        except RuntimeError:
            return None
    return lib


def user_assets(kind):
    d = kind_dir(kind)
    if not d or not os.path.isdir(d):
        return {}
    return {f[:-6]: os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".blend")}


def _safe(name):
    return "".join(c for c in name if c.isalnum() or c in " -_").strip() or "Smart"


def save_asset(src_mat, name, kind, layer_index=None, description=""):
    """Save the layer block at layer_index (or the active layer's mask stack, for a
    smart mask) as an asset .blend. Returns the file path."""
    src = src_mat.paintstack
    name = _safe(name)
    tmp = bpy.data.materials.new(name)
    try:
        ts = tmp.paintstack
        ts.initialized = True
        ts.resolution = src.resolution
        from . import stack
        if kind == KIND_MATERIAL:
            start = src.active_layer_index if layer_index is None else layer_index
            end = stack.block_end(src, start)
            copy_layers(src, list(range(start, end)), tmp, 0, copy_images=True)
            ts.layers[0].parent_uid = ""
        else:
            layer = src.active_layer
            holder = ts.layers.add()
            holder.uid, holder.name, holder.type = "mask", "Mask", "FILL"
            _copy_props(layer, holder, skip=("channels", "uid", "parent_uid", "name", "type"))
            for e in holder.mask_effects:
                if e.image:
                    e.image = pixels.clone_image(e.image)
            if holder.mask_image:
                holder.mask_image = pixels.clone_image(holder.mask_image)
        tmp["paintstack_kind"] = kind
        tmp.use_fake_user = True
        images = set()
        for layer in ts.layers:
            images.update(layer_images(layer))
        for img in images:
            if not img.packed_file:
                img.pack()
        try:
            tmp.asset_mark()
            tmp.asset_data.description = description or f"PaintStack {kind.replace('_', ' ').lower()}"
            tmp.asset_data.tags.new("PaintStack")
            tmp.asset_data.tags.new("Smart Mask" if kind == KIND_MASK else "Smart Material")
        except AttributeError:
            pass
        path = os.path.join(kind_dir(kind, create=True), name + ".blend")
        bpy.data.libraries.write(path, {tmp, *images}, fake_user=True, compress=True)
        ensure_asset_library()
        return path
    finally:
        temp_images = set()
        for layer in tmp.paintstack.layers:
            temp_images.update(i for i in layer_images(layer) if i.users <= 1 and "Grunge_" not in i.name)
        bpy.data.materials.remove(tmp)
        for img in temp_images:
            if img.users == 0:
                bpy.data.images.remove(img)


def _load_material(path):
    with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
        data_to.materials = list(data_from.materials[:1])
    return data_to.materials[0] if data_to.materials else None


def apply_asset(mat, path):
    """Apply a saved smart material (as a block above the active layer) or smart mask
    (appended to the active layer's mask stack). Returns the kind."""
    src_mat = _load_material(path)
    if src_mat is None:
        raise RuntimeError(f"No material in {path}")
    try:
        kind = src_mat.get("paintstack_kind", KIND_MATERIAL)
        src = src_mat.paintstack
        ts = mat.paintstack
        if kind == KIND_MASK:
            layer = ts.active_layer
            holder = src.layers[0]
            for se in holder.mask_effects:
                de = layer.mask_effects.add()
                _copy_props(se, de)
            layer.active_mask_effect_index = len(layer.mask_effects) - 1
        else:
            active = ts.active_layer
            index = ts.active_layer_index if active is not None else 0
            parent = active.parent_uid if active is not None else ""
            uids = copy_layers(src, list(range(len(src.layers))), mat, index, copy_images=False)
            top = next(l for l in ts.layers if l.uid == uids[0])
            top.parent_uid = parent
            ts.active_layer_index = next(i for i, l in enumerate(ts.layers) if l.uid == uids[0])
        return kind
    finally:
        bpy.data.materials.remove(src_mat)
