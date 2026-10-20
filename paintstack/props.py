"""Data model: a texture set lives on a Material, holding a stack of layers.
Layer index 0 is the top of the stack, like Substance Painter's layer panel."""

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       FloatVectorProperty, IntProperty, PointerProperty, StringProperty)
from bpy.types import PropertyGroup

from .channels import CHANNEL_ITEMS

BLEND_MODES = [
    ("NORMAL", "Normal", ""),
    ("MULTIPLY", "Multiply", ""),
    ("SCREEN", "Screen", ""),
    ("OVERLAY", "Overlay", ""),
    ("SOFT_LIGHT", "Soft Light", ""),
    ("ADD", "Linear Dodge (Add)", ""),
    ("SUBTRACT", "Subtract", ""),
    ("DARKEN", "Darken", ""),
    ("LIGHTEN", "Lighten", ""),
    ("DIFFERENCE", "Difference", ""),
]

RESOLUTIONS = [(str(r), str(r), "") for r in (512, 1024, 2048, 4096)]


def _recomposite(self, context):
    from . import compositor
    mat = self.id_data
    if isinstance(mat, bpy.types.Material):
        compositor.request(mat)


class PS_LayerChannel(PropertyGroup):
    channel: EnumProperty(items=CHANNEL_ITEMS)
    enabled: BoolProperty(default=True, update=_recomposite)
    blend_mode: EnumProperty(items=BLEND_MODES, default="NORMAL", update=_recomposite)
    opacity: FloatProperty(min=0.0, max=1.0, default=1.0, subtype="FACTOR", update=_recomposite)
    # Fill layer values
    color: FloatVectorProperty(subtype="COLOR_GAMMA", size=3, min=0.0, max=1.0,
                               default=(0.8, 0.8, 0.8), update=_recomposite)
    value: FloatProperty(min=0.0, max=1.0, default=0.5, subtype="FACTOR", update=_recomposite)
    # Paint layer content (RGBA, alpha = coverage)
    image: PointerProperty(type=bpy.types.Image)


class PS_Layer(PropertyGroup):
    name: StringProperty(default="Layer")
    uid: StringProperty()
    type: EnumProperty(items=[("PAINT", "Paint", "", "BRUSH_DATA", 0),
                              ("FILL", "Fill", "", "IMAGE_RGB", 1)])
    visible: BoolProperty(default=True, update=_recomposite)
    opacity: FloatProperty(min=0.0, max=1.0, default=1.0, subtype="FACTOR", update=_recomposite)
    channels: CollectionProperty(type=PS_LayerChannel)

    mask_image: PointerProperty(type=bpy.types.Image)
    mask_enabled: BoolProperty(default=True, update=_recomposite)
    mask_invert: BoolProperty(default=False, update=_recomposite)
    edit_target: EnumProperty(
        name="Paint Target",
        items=[("CONTENT", "Content", "Paint the layer's channels", "BRUSH_DATA", 0),
               ("MASK", "Mask", "Paint the layer mask", "MOD_MASK", 1)])

    def channel(self, channel_id):
        for lc in self.channels:
            if lc.channel == channel_id:
                return lc
        return None


class PS_OutputMap(PropertyGroup):
    channel: EnumProperty(items=CHANNEL_ITEMS)
    image: PointerProperty(type=bpy.types.Image)


def _resolution_changed(self, context):
    from . import ops
    ops.resize_texture_set(self.id_data)


def _rebuild_material(self, context):
    from . import material
    material.build(self.id_data)


class PS_TextureSet(PropertyGroup):
    initialized: BoolProperty(default=False)
    resolution: EnumProperty(items=RESOLUTIONS, default="2048", update=_resolution_changed)
    use_base_color: BoolProperty(name="Base Color", default=True, update=_recomposite)
    use_metallic: BoolProperty(name="Metallic", default=True, update=_recomposite)
    use_roughness: BoolProperty(name="Roughness", default=True, update=_recomposite)
    use_height: BoolProperty(name="Height", default=True, update=_recomposite)
    use_emission: BoolProperty(name="Emission", default=False, update=_recomposite)
    bump_strength: FloatProperty(name="Bump Strength", min=0.0, max=1.0, default=0.3,
                                 update=_rebuild_material)
    layers: CollectionProperty(type=PS_Layer)
    active_layer_index: IntProperty(default=0)
    outputs: CollectionProperty(type=PS_OutputMap)

    @property
    def res(self):
        return int(self.resolution)

    @property
    def active_layer(self):
        if 0 <= self.active_layer_index < len(self.layers):
            return self.layers[self.active_layer_index]
        return None

    def output(self, channel_id):
        for o in self.outputs:
            if o.channel == channel_id:
                return o.image
        return None


class PS_Brush(PropertyGroup):
    radius: IntProperty(name="Radius", min=1, max=1000, default=40, subtype="PIXEL")
    hardness: FloatProperty(name="Hardness", min=0.0, max=1.0, default=0.6, subtype="FACTOR")
    flow: FloatProperty(name="Flow", min=0.0, max=1.0, default=0.5, subtype="FACTOR")
    opacity: FloatProperty(name="Stroke Opacity", min=0.0, max=1.0, default=1.0, subtype="FACTOR")
    spacing: FloatProperty(name="Spacing", min=0.02, max=2.0, default=0.15,
                           description="Distance between dabs, relative to the radius")
    use_pressure: BoolProperty(name="Pressure → Flow", default=True)
    mirror_x: BoolProperty(name="Mirror X", default=False,
                           description="Mirror strokes across the object's local X axis")
    mode: EnumProperty(items=[("PAINT", "Paint", "", "BRUSH_DATA", 0),
                              ("ERASE", "Erase", "", "X", 1)])

    use_base_color: BoolProperty(name="Base Color", default=True)
    use_metallic: BoolProperty(name="Metallic", default=True)
    use_roughness: BoolProperty(name="Roughness", default=True)
    use_height: BoolProperty(name="Height", default=False)
    use_emission: BoolProperty(name="Emission", default=False)

    base_color: FloatVectorProperty(name="Base Color", subtype="COLOR_GAMMA", size=3,
                                    min=0.0, max=1.0, default=(0.8, 0.1, 0.1))
    metallic: FloatProperty(name="Metallic", min=0.0, max=1.0, default=0.0, subtype="FACTOR")
    roughness: FloatProperty(name="Roughness", min=0.0, max=1.0, default=0.4, subtype="FACTOR")
    height: FloatProperty(name="Height", min=0.0, max=1.0, default=0.7, subtype="FACTOR")
    emission: FloatVectorProperty(name="Emission", subtype="COLOR_GAMMA", size=3,
                                  min=0.0, max=1.0, default=(1.0, 0.5, 0.1))

    def value(self, ch):
        v = getattr(self, ch.key)
        return tuple(v) if ch.is_color else (v, v, v)


classes = (PS_LayerChannel, PS_Layer, PS_OutputMap, PS_TextureSet, PS_Brush)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Material.paintstack = PointerProperty(type=PS_TextureSet)
    bpy.types.Scene.paintstack_brush = PointerProperty(type=PS_Brush)


def unregister():
    del bpy.types.Scene.paintstack_brush
    del bpy.types.Material.paintstack
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
