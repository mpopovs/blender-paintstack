import uuid

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty
from bpy.types import Operator

from . import compositor, material, pixels, undo
from .channels import CHANNELS
from .props import EFFECT_TYPES


def active_material(context):
    obj = context.active_object
    if obj is None or obj.type != "MESH":
        return None
    return obj.active_material


def texture_set(context):
    mat = active_material(context)
    if mat is None or not mat.paintstack.initialized:
        return None, None
    return mat, mat.paintstack


def _image_name(mat, layer, suffix):
    return f"{mat.name}_{layer.uid}_{suffix}"


def ensure_layer_image(mat, layer, lc):
    ch = next(c for c in CHANNELS if c.id == lc.channel)
    lc.image = pixels.new_image(_image_name(mat, layer, ch.key), mat.paintstack.res, ch.default,
                                float_buffer=ch.float_buffer, srgb=ch.srgb, alpha=0.0)
    return lc.image


def add_mask(mat, layer, white=False):
    v = 1.0 if white else 0.0
    layer.mask_image = pixels.new_image(_image_name(mat, layer, "mask"), mat.paintstack.res, (v, v, v))
    return layer.mask_image


def ensure_effect_image(mat, layer, effect):
    effect.image = pixels.new_image(_image_name(mat, layer, f"fx{uuid.uuid4().hex[:4]}"),
                                    mat.paintstack.res, (0.0, 0.0, 0.0), alpha=0.0)
    return effect.image


# Defaults that make each generator useful straight away.
EFFECT_DEFAULTS = {
    "EDGE_WEAR": dict(level=0.35, softness=0.1, noise_amount=0.5, noise_scale=20.0),
    "DIRT": dict(level=0.4, softness=0.2, noise_amount=0.4, noise_scale=8.0),
    "CURVATURE": dict(level=0.5, softness=0.3, noise_amount=0.0),
    "AO": dict(level=0.5, softness=0.3, noise_amount=0.0, invert=True),
    "THICKNESS": dict(level=0.5, softness=0.3, noise_amount=0.0),
    "POSITION": dict(level=0.5, softness=0.3, noise_amount=0.1),
    "NOISE": dict(level=0.5, softness=0.2, noise_scale=6.0),
}


def add_mask_effect(mat, layer, kind):
    e = layer.mask_effects.add()
    e.type = kind
    e.name = next(label for ident, label, _ in EFFECT_TYPES if ident == kind)
    for key, val in EFFECT_DEFAULTS.get(kind, {}).items():
        setattr(e, key, val)
    if kind == "PAINT":
        ensure_effect_image(mat, layer, e)
    layer.active_mask_effect_index = len(layer.mask_effects) - 1
    return layer.mask_effects[layer.active_mask_effect_index]


def layer_images(layer):
    imgs = [lc.image for lc in layer.channels if lc.image]
    if layer.mask_image:
        imgs.append(layer.mask_image)
    imgs += [e.image for e in layer.mask_effects if e.image]
    return imgs


def new_layer(mat, kind, name=None, index=None):
    ts = mat.paintstack
    layer = ts.layers.add()
    layer.type = kind
    layer.uid = uuid.uuid4().hex[:8]
    layer.name = name or f"{kind.title()} {len(ts.layers)}"
    for ch in CHANNELS:
        lc = layer.channels.add()
        lc.channel = ch.id
        lc.color = ch.default
        lc.value = ch.default[0]
    # New layers go above the active one.
    target = ts.active_layer_index if index is None else index
    target = max(0, min(target, len(ts.layers) - 1))
    ts.layers.move(len(ts.layers) - 1, target)
    ts.active_layer_index = target
    return ts.layers[target]


def resize_texture_set(mat):
    ts = mat.paintstack
    res = ts.res
    imgs = [o.image for o in ts.outputs if o.image]
    for layer in ts.layers:
        imgs += layer_images(layer)
    for img in imgs:
        if tuple(img.size) != (res, res):
            img.scale(res, res)
    undo.clear()
    compositor.request(mat)


class PAINTSTACK_OT_init(Operator):
    """Turn the active material into a PaintStack texture set"""
    bl_idname = "paintstack.init"
    bl_label = "Create Texture Set"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context):
        obj = context.active_object
        if not obj.data.uv_layers:
            self.report({"ERROR"}, "The mesh needs a UV map first (unwrap it)")
            return {"CANCELLED"}
        mat = obj.active_material
        if mat is None:
            mat = bpy.data.materials.new(obj.name + "_PaintStack")
            obj.data.materials.append(mat)
            obj.active_material_index = len(obj.data.materials) - 1
        ts = mat.paintstack
        if ts.initialized:
            return {"CANCELLED"}
        ts.initialized = True
        new_layer(mat, "FILL", "Base Material")
        new_layer(mat, "PAINT", "Paint 1", index=0)
        compositor.composite(mat)
        return {"FINISHED"}


class PAINTSTACK_OT_layer_add(Operator):
    """Add a layer above the active one"""
    bl_idname = "paintstack.layer_add"
    bl_label = "Add Layer"
    bl_options = {"REGISTER", "UNDO"}

    type: EnumProperty(items=[("PAINT", "Paint", ""), ("FILL", "Fill", "")])

    @classmethod
    def poll(cls, context):
        return texture_set(context)[0] is not None

    def execute(self, context):
        mat, _ = texture_set(context)
        new_layer(mat, self.type)
        compositor.composite(mat)
        return {"FINISHED"}


class PAINTSTACK_OT_layer_remove(Operator):
    """Delete the active layer"""
    bl_idname = "paintstack.layer_remove"
    bl_label = "Remove Layer"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        mat, ts = texture_set(context)
        return mat is not None and ts.active_layer is not None

    def execute(self, context):
        mat, ts = texture_set(context)
        for img in layer_images(ts.active_layer):
            bpy.data.images.remove(img)
        ts.layers.remove(ts.active_layer_index)
        ts.active_layer_index = min(ts.active_layer_index, len(ts.layers) - 1)
        undo.clear()
        compositor.composite(mat)
        return {"FINISHED"}


class PAINTSTACK_OT_layer_move(Operator):
    """Move the active layer up or down the stack"""
    bl_idname = "paintstack.layer_move"
    bl_label = "Move Layer"
    bl_options = {"REGISTER", "UNDO"}

    direction: EnumProperty(items=[("UP", "Up", ""), ("DOWN", "Down", "")])

    @classmethod
    def poll(cls, context):
        return texture_set(context)[0] is not None

    def execute(self, context):
        mat, ts = texture_set(context)
        i = ts.active_layer_index
        j = i - 1 if self.direction == "UP" else i + 1
        if not 0 <= j < len(ts.layers):
            return {"CANCELLED"}
        ts.layers.move(i, j)
        ts.active_layer_index = j
        compositor.composite(mat)
        return {"FINISHED"}


class PAINTSTACK_OT_layer_duplicate(Operator):
    """Duplicate the active layer, including its painted content"""
    bl_idname = "paintstack.layer_duplicate"
    bl_label = "Duplicate Layer"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        mat, ts = texture_set(context)
        return mat is not None and ts.active_layer is not None

    def execute(self, context):
        mat, ts = texture_set(context)
        src = ts.active_layer
        dst = new_layer(mat, src.type, src.name + " copy")
        src = ts.layers[ts.active_layer_index + 1]  # shifted down by the insert
        for key in ("visible", "opacity", "mask_enabled", "mask_invert", "edit_target"):
            setattr(dst, key, getattr(src, key))
        for slc, dlc in zip(src.channels, dst.channels):
            for key in ("enabled", "blend_mode", "opacity", "color", "value"):
                setattr(dlc, key, getattr(slc, key))
            if slc.image:
                dlc.image = slc.image.copy()
                dlc.image.name = _image_name(mat, dst, dlc.channel.lower())
        if src.mask_image:
            dst.mask_image = src.mask_image.copy()
            dst.mask_image.name = _image_name(mat, dst, "mask")
        for se in src.mask_effects:
            de = dst.mask_effects.add()
            for prop in se.bl_rna.properties:
                if prop.identifier not in ("rna_type", "image") and not prop.is_readonly:
                    setattr(de, prop.identifier, getattr(se, prop.identifier))
            if se.image:
                de.image = se.image.copy()
        compositor.composite(mat)
        return {"FINISHED"}


class PAINTSTACK_OT_mask_add(Operator):
    """Add a mask to the active layer"""
    bl_idname = "paintstack.mask_add"
    bl_label = "Add Mask"
    bl_options = {"REGISTER", "UNDO"}

    white: BoolProperty(name="White", default=True)

    @classmethod
    def poll(cls, context):
        mat, ts = texture_set(context)
        return mat is not None and ts.active_layer is not None and ts.active_layer.mask_image is None

    def execute(self, context):
        mat, ts = texture_set(context)
        add_mask(mat, ts.active_layer, self.white)
        ts.active_layer.edit_target = "MASK"
        compositor.composite(mat)
        return {"FINISHED"}


class PAINTSTACK_OT_mask_remove(Operator):
    """Remove the active layer's mask"""
    bl_idname = "paintstack.mask_remove"
    bl_label = "Remove Mask"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        mat, ts = texture_set(context)
        return mat is not None and ts.active_layer is not None and ts.active_layer.mask_image

    def execute(self, context):
        mat, ts = texture_set(context)
        layer = ts.active_layer
        bpy.data.images.remove(layer.mask_image)
        layer.mask_image = None
        layer.edit_target = "CONTENT"
        undo.clear()
        compositor.composite(mat)
        return {"FINISHED"}


def _active_layer(context):
    mat, ts = texture_set(context)
    return (mat, ts.active_layer) if mat is not None else (None, None)


class PAINTSTACK_OT_effect_add(Operator):
    """Add an effect (generator, filter, paint or fill) to the active layer's mask stack"""
    bl_idname = "paintstack.effect_add"
    bl_label = "Add Mask Effect"
    bl_options = {"REGISTER", "UNDO"}

    type: EnumProperty(items=EFFECT_TYPES)

    @classmethod
    def poll(cls, context):
        return _active_layer(context)[1] is not None

    def execute(self, context):
        mat, layer = _active_layer(context)
        add_mask_effect(mat, layer, self.type)
        layer.edit_target = "MASK"
        compositor.composite(mat)
        return {"FINISHED"}


class PAINTSTACK_OT_effect_remove(Operator):
    """Remove the active mask effect"""
    bl_idname = "paintstack.effect_remove"
    bl_label = "Remove Mask Effect"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        layer = _active_layer(context)[1]
        return layer is not None and layer.active_mask_effect is not None

    def execute(self, context):
        mat, layer = _active_layer(context)
        effect = layer.active_mask_effect
        if effect.image:
            bpy.data.images.remove(effect.image)
        layer.mask_effects.remove(layer.active_mask_effect_index)
        layer.active_mask_effect_index = max(0, layer.active_mask_effect_index - 1)
        undo.clear()
        compositor.composite(mat)
        return {"FINISHED"}


class PAINTSTACK_OT_effect_move(Operator):
    """Move the active mask effect up or down the stack"""
    bl_idname = "paintstack.effect_move"
    bl_label = "Move Mask Effect"
    bl_options = {"REGISTER", "UNDO"}

    direction: EnumProperty(items=[("UP", "Up", ""), ("DOWN", "Down", "")])

    @classmethod
    def poll(cls, context):
        layer = _active_layer(context)[1]
        return layer is not None and layer.active_mask_effect is not None

    def execute(self, context):
        mat, layer = _active_layer(context)
        i = layer.active_mask_effect_index
        # The list is drawn top-first, so "up" means later in the stack.
        j = i + 1 if self.direction == "UP" else i - 1
        if not 0 <= j < len(layer.mask_effects):
            return {"CANCELLED"}
        layer.mask_effects.move(i, j)
        layer.active_mask_effect_index = j
        compositor.composite(mat)
        return {"FINISHED"}


class PAINTSTACK_OT_composite(Operator):
    """Re-flatten the layer stack into the material's channel maps"""
    bl_idname = "paintstack.composite"
    bl_label = "Refresh"

    @classmethod
    def poll(cls, context):
        return texture_set(context)[0] is not None

    def execute(self, context):
        compositor.composite(texture_set(context)[0])
        return {"FINISHED"}


class PAINTSTACK_OT_undo_stroke(Operator):
    """Undo or redo the last paint stroke"""
    bl_idname = "paintstack.undo_stroke"
    bl_label = "Undo Stroke"

    redo: BoolProperty(default=False)

    def execute(self, context):
        ok = undo.redo() if self.redo else undo.undo()
        return {"FINISHED"} if ok else {"CANCELLED"}


classes = (
    PAINTSTACK_OT_init,
    PAINTSTACK_OT_layer_add,
    PAINTSTACK_OT_layer_remove,
    PAINTSTACK_OT_layer_move,
    PAINTSTACK_OT_layer_duplicate,
    PAINTSTACK_OT_mask_add,
    PAINTSTACK_OT_mask_remove,
    PAINTSTACK_OT_effect_add,
    PAINTSTACK_OT_effect_remove,
    PAINTSTACK_OT_effect_move,
    PAINTSTACK_OT_composite,
    PAINTSTACK_OT_undo_stroke,
)
