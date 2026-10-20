"""Data model: a texture set lives on a Material, holding a stack of layers.
Layer index 0 is the top of the stack, like Substance Painter's layer panel."""

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       FloatVectorProperty, IntProperty, PointerProperty, StringProperty)
from bpy.types import PropertyGroup

from .brushes import TIPS as TIP_ITEMS
from .channels import CHANNEL_ITEMS
from .grunge import GRUNGE_TYPES

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
    ("NORMAL_COMBINE", "Normal Map Combine", "Add the layer's normal detail on top (whiteout blend)"),
]

RESOLUTIONS = [(str(r), str(r), "") for r in (256, 512, 1024, 2048, 4096)]

MASK_BLEND_MODES = [
    ("NORMAL", "Normal", ""),
    ("MULTIPLY", "Multiply", ""),
    ("ADD", "Add", ""),
    ("SUBTRACT", "Subtract", ""),
    ("MAX", "Lighten (Max)", ""),
    ("MIN", "Darken (Min)", ""),
]

# Mask stack effects. Generators read the texture set's baked mesh maps.
EFFECT_TYPES = [
    ("PAINT", "Paint", "Hand-painted mask touch-ups"),
    ("FILL", "Fill", "Constant value"),
    ("EDGE_WEAR", "Edge Wear", "Convex edges broken up with noise (needs Curvature)"),
    ("DIRT", "Dirt", "Cavities and occluded areas (needs AO, Curvature)"),
    ("CURVATURE", "Curvature", "Convexity from the baked curvature map"),
    ("AO", "Ambient Occlusion", "Occlusion from the baked AO map"),
    ("THICKNESS", "Thickness", "Thin areas from the baked thickness map"),
    ("POSITION", "Position Gradient", "Gradient along an object axis (needs Position)"),
    ("NOISE", "3D Noise", "Seamless noise sampled at the baked positions"),
    ("ID_MASK", "Color Selection", "Select areas by their baked ID colour (needs ID)"),
    ("GRUNGE", "Grunge", "Tileable grunge texture from the built-in library"),
    ("ANCHOR", "Anchor", "Use another layer's result (an anchor point) as the mask"),
    ("LEVELS", "Levels", "Remap the mask below"),
    ("BLUR", "Blur", "Blur the mask below"),
]

BAKE_MAPS = [
    ("NORMAL", "Normal", "Tangent-space normal from the high poly"),
    ("AO", "Ambient Occlusion", ""),
    ("CURVATURE", "Curvature", ""),
    ("THICKNESS", "Thickness", ""),
    ("POSITION", "Position", "Bounding-box normalised object position"),
    ("WORLD_NORMAL", "World Normal", "Surface normals in the mesh's own space (like Substance's "
                                     "world-space normal map)"),
    ("ID", "ID", "Colour per vertex colour, material or mesh part"),
]


_LAYER_PATH = None


def _owner(prop_group):
    """(layer uid, channel id) a property group belongs to. uid is None for
    texture-set props; channel is set only for a layer channel's own props."""
    global _LAYER_PATH
    if _LAYER_PATH is None:
        import re
        _LAYER_PATH = re.compile(r"paintstack\.layers\[(\d+)\](\.channels\[(\d+)\])?")
    try:
        match = _LAYER_PATH.match(prop_group.path_from_id())
    except ValueError:
        return None, None
    if match is None:
        return None, None
    layers = prop_group.id_data.paintstack.layers
    i = int(match.group(1))
    if i >= len(layers):
        return None, None
    channel = None
    if match.group(3) is not None and isinstance(prop_group, PS_LayerChannel):
        channel = prop_group.channel
    return layers[i].uid, channel


def _recomposite(self, context):
    """Property update: invalidate only the owning layer's cached work, then
    schedule a composite."""
    from . import compositor
    mat = self.id_data
    if isinstance(mat, bpy.types.Material):
        compositor.request(mat, *_owner(self))


def _reblend(self, context):
    """Property update for values that only change how a layer is blended
    (opacity, colour, blend mode, visibility): its cached pixels, mask and
    uploaded textures stay valid, so dragging these sliders stays cheap."""
    from . import compositor
    mat = self.id_data
    if isinstance(mat, bpy.types.Material):
        compositor.request(mat, *_owner(self), blend_only=True)


def _remask(self, context):
    """Property update for a layer's mask stack: only its mask is recomputed (on
    the GPU compositor, from cached sources with the new values as constants)."""
    from . import compositor
    mat = self.id_data
    if isinstance(mat, bpy.types.Material):
        compositor.request(mat, _owner(self)[0], mask_only=True)


class PS_LayerChannel(PropertyGroup):
    channel: EnumProperty(items=CHANNEL_ITEMS)
    enabled: BoolProperty(default=True, update=_reblend)
    blend_mode: EnumProperty(items=BLEND_MODES, default="NORMAL", update=_reblend)
    opacity: FloatProperty(min=0.0, max=1.0, default=1.0, subtype="FACTOR", update=_reblend)
    # Fill layer values
    color: FloatVectorProperty(subtype="COLOR_GAMMA", size=3, min=0.0, max=1.0,
                               default=(0.8, 0.8, 0.8), update=_reblend)
    value: FloatProperty(min=0.0, max=1.0, default=0.5, subtype="FACTOR", update=_reblend)
    # Fill layer texture (overrides color / value when set)
    texture: PointerProperty(type=bpy.types.Image, update=_recomposite)
    # Paint layer content (RGBA, alpha = coverage)
    image: PointerProperty(type=bpy.types.Image)


class PS_MaskEffect(PropertyGroup):
    name: StringProperty(default="Effect")
    type: EnumProperty(items=EFFECT_TYPES, update=_remask)
    enabled: BoolProperty(default=True, update=_remask)
    blend_mode: EnumProperty(items=MASK_BLEND_MODES, default="NORMAL", update=_remask)
    opacity: FloatProperty(min=0.0, max=1.0, default=1.0, subtype="FACTOR", update=_remask)
    invert: BoolProperty(default=False, update=_remask)

    # Generators: smoothstep(level - softness, level + softness, source + noise)
    level: FloatProperty(name="Level", min=0.0, max=1.0, default=0.5, subtype="FACTOR",
                         update=_remask, description="Higher = more coverage")
    softness: FloatProperty(name="Softness", min=0.001, max=1.0, default=0.15, subtype="FACTOR",
                            update=_remask)
    noise_amount: FloatProperty(name="Noise", min=0.0, max=2.0, default=0.4, update=_remask)
    noise_scale: FloatProperty(name="Noise Scale", min=0.1, max=200.0, default=12.0, update=_remask)
    noise_octaves: IntProperty(name="Detail", min=1, max=8, default=4, update=_remask)
    seed: IntProperty(name="Seed", min=0, default=0, update=_remask)
    axis: EnumProperty(name="Axis", items=[("X", "X", ""), ("Y", "Y", ""), ("Z", "Z", "")],
                       default="Z", update=_remask)

    value: FloatProperty(name="Value", min=0.0, max=1.0, default=1.0, subtype="FACTOR",
                         update=_remask)
    in_min: FloatProperty(name="Input Min", min=0.0, max=1.0, default=0.0, update=_remask)
    in_max: FloatProperty(name="Input Max", min=0.0, max=1.0, default=1.0, update=_remask)
    gamma: FloatProperty(name="Gamma", min=0.05, max=10.0, default=1.0, update=_remask)
    blur_radius: IntProperty(name="Radius", min=1, max=64, default=4, subtype="PIXEL",
                             update=_remask)
    id_color: FloatVectorProperty(name="ID Color", subtype="COLOR", size=3, min=0.0, max=1.0,
                                  default=(1.0, 1.0, 1.0), update=_remask)
    id_tolerance: FloatProperty(name="Tolerance", min=0.001, max=1.0, default=0.1, update=_remask)
    grunge: EnumProperty(name="Texture", items=GRUNGE_TYPES, update=_remask)
    grunge_tiling: FloatProperty(name="Tiling", min=0.05, max=50.0, default=2.0, update=_remask)
    anchor_uid: StringProperty(name="Anchor", update=_remask,
                               description="uid of the anchor layer this effect reads")
    anchor_channel: EnumProperty(name="Channel", items=CHANNEL_ITEMS, default="HEIGHT",
                                 update=_remask)
    # PAINT effects: RGBA, value in R, coverage in A
    image: PointerProperty(type=bpy.types.Image)


class PS_Layer(PropertyGroup):
    name: StringProperty(default="Layer")
    uid: StringProperty()
    type: EnumProperty(items=[("PAINT", "Paint", "", "BRUSH_DATA", 0),
                              ("FILL", "Fill", "", "IMAGE_RGB", 1),
                              ("FOLDER", "Folder", "", "FILE_FOLDER", 2),
                              ("FILTER", "Filter", "", "FILTER", 3)])
    is_anchor: BoolProperty(name="Anchor Point", default=False, update=_recomposite,
                            description="Make this point of the stack available to Anchor mask effects")
    filter_type: EnumProperty(
        name="Filter",
        items=[("BLUR", "Blur", "Blur everything below"),
               ("SHARPEN", "Sharpen", "Unsharp mask"),
               ("HSL", "Hue / Saturation / Lightness", "Colour adjustment (lightness only on grey channels)"),
               ("WARP", "Warp", "Displace by noise")],
        update=_recomposite)
    filter_radius: IntProperty(name="Radius", min=1, max=64, default=4, subtype="PIXEL", update=_recomposite)
    filter_amount: FloatProperty(name="Amount", min=0.0, max=5.0, default=1.0, update=_recomposite)
    hue: FloatProperty(name="Hue", min=-0.5, max=0.5, default=0.0, update=_recomposite)
    saturation: FloatProperty(name="Saturation", min=-1.0, max=1.0, default=0.0, update=_recomposite)
    lightness: FloatProperty(name="Lightness", min=-1.0, max=1.0, default=0.0, update=_recomposite)
    warp_scale: FloatProperty(name="Warp Scale", min=0.1, max=100.0, default=8.0, update=_recomposite)
    parent_uid: StringProperty(description="uid of the folder containing this layer ('' = root)")
    expanded: BoolProperty(default=True)
    visible: BoolProperty(default=True, update=_reblend)
    projection: EnumProperty(
        name="Projection",
        items=[("UV", "UV", "Tile fill textures in UV space"),
               ("TRIPLANAR", "Tri-planar", "Project fill textures from 3 axes using the baked "
                                           "Position and World Normal maps (no UV seams)")],
        update=_recomposite)
    tiling: FloatProperty(name="Tiling", min=0.01, max=100.0, default=1.0, update=_recomposite)
    opacity: FloatProperty(min=0.0, max=1.0, default=1.0, subtype="FACTOR", update=_reblend)
    channels: CollectionProperty(type=PS_LayerChannel)

    mask_image: PointerProperty(type=bpy.types.Image)
    mask_enabled: BoolProperty(default=True, update=_remask)
    mask_invert: BoolProperty(default=False, update=_remask)
    edit_target: EnumProperty(
        name="Paint Target",
        items=[("CONTENT", "Content", "Paint the layer's channels", "BRUSH_DATA", 0),
               ("MASK", "Mask", "Paint the layer mask", "MOD_MASK", 1)])
    mask_effects: CollectionProperty(type=PS_MaskEffect)
    active_mask_effect_index: IntProperty(default=0)

    @property
    def has_mask(self):
        return self.mask_image is not None or len(self.mask_effects) > 0

    @property
    def active_mask_effect(self):
        if 0 <= self.active_mask_effect_index < len(self.mask_effects):
            return self.mask_effects[self.active_mask_effect_index]
        return None

    def channel(self, channel_id):
        for lc in self.channels:
            if lc.channel == channel_id:
                return lc
        return None


class PS_OutputMap(PropertyGroup):
    channel: EnumProperty(items=CHANNEL_ITEMS)
    image: PointerProperty(type=bpy.types.Image)


class PS_BakedMap(PropertyGroup):
    map_type: EnumProperty(items=BAKE_MAPS)
    image: PointerProperty(type=bpy.types.Image)


def _poll_high_poly(self, obj):
    return obj.type == "MESH"


class PS_BakeSettings(PropertyGroup):
    high_poly: PointerProperty(name="High Poly", type=bpy.types.Object, poll=_poll_high_poly,
                               description="Optional detailed mesh baked onto this one")
    cage_extrusion: FloatProperty(name="Cage Extrusion", min=0.0, default=0.05, unit="LENGTH",
                                  description="How far outside the low poly rays start")
    max_ray_distance: FloatProperty(name="Max Ray Distance", min=0.0, default=0.0, unit="LENGTH")
    ao_distance: FloatProperty(name="AO Distance", min=0.001, default=0.5, unit="LENGTH")
    thickness_distance: FloatProperty(name="Thickness Distance", min=0.001, default=0.5, unit="LENGTH")
    cage_object: PointerProperty(name="Cage", type=bpy.types.Object, poll=_poll_high_poly,
                                 description="Optional cage mesh (same topology as the low poly) "
                                             "that rays start from; overrides Cage Extrusion")
    id_source: EnumProperty(
        name="ID Source",
        items=[("AUTO", "Auto", "Vertex colours if present, else materials if several, else mesh parts"),
               ("VERTEX_COLOR", "Vertex Color", "Active colour attribute"),
               ("MATERIAL", "Material", "One colour per material"),
               ("MESH_PART", "Mesh Part", "One colour per connected mesh part")])
    samples: IntProperty(name="Samples", min=1, max=1024, default=32)
    margin: IntProperty(name="Margin", min=0, max=64, default=8, subtype="PIXEL")
    use_normal: BoolProperty(name="Normal", default=True)
    use_ao: BoolProperty(name="Ambient Occlusion", default=True)
    use_curvature: BoolProperty(name="Curvature", default=True)
    use_thickness: BoolProperty(name="Thickness", default=True)
    use_position: BoolProperty(name="Position", default=True)
    use_world_normal: BoolProperty(name="World Normal", default=True)
    use_id: BoolProperty(name="ID", default=False)


def _resolution_changed(self, context):
    from . import ops
    ops.resize_texture_set(self.id_data)


def _view_changed(self, context):
    from . import compositor, material
    material.build(self.id_data)
    if self.view_mode == "MASK":
        compositor.request_view(self.id_data)


def _active_layer_changed(self, context):
    if self.view_mode == "MASK":
        from . import compositor
        compositor.request_view(self.id_data)


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
    use_normal: BoolProperty(name="Normal", default=False, update=_recomposite)
    use_emission: BoolProperty(name="Emission", default=False, update=_recomposite)
    view_mode: EnumProperty(
        name="View",
        items=[("MATERIAL", "Material", "Full PBR material"),
               ("BASE_COLOR", "Base Color", ""), ("METALLIC", "Metallic", ""),
               ("ROUGHNESS", "Roughness", ""), ("HEIGHT", "Height", ""),
               ("NORMAL", "Normal", ""), ("EMISSION", "Emission", ""),
               ("MASK", "Layer Mask", "The active layer's final mask")],
        update=_view_changed,
        description="Isolate one channel in the viewport")
    bump_strength: FloatProperty(name="Bump Strength", min=0.0, max=1.0, default=0.3,
                                 update=_rebuild_material)
    layers: CollectionProperty(type=PS_Layer)
    active_layer_index: IntProperty(default=0, update=_active_layer_changed)
    outputs: CollectionProperty(type=PS_OutputMap)
    baked: CollectionProperty(type=PS_BakedMap)
    bake: PointerProperty(type=PS_BakeSettings)
    use_baked_normal: BoolProperty(name="Use Baked Normal", default=True, update=_rebuild_material)
    preview_ao: BoolProperty(name="Multiply AO", default=False, update=_rebuild_material,
                             description="Darken the material's base colour by the baked AO "
                                         "(viewport and render; exports are unaffected)")

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

    def baked_map(self, map_type):
        for b in self.baked:
            if b.map_type == map_type:
                return b.image
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

    preset_name: StringProperty(name="Preset", default="")

    # Tip
    tip: EnumProperty(name="Tip", items=TIP_ITEMS, default="ROUND")
    tip_image: PointerProperty(name="Tip Image", type=bpy.types.Image)
    tip_seed: IntProperty(name="Tip Seed", min=0, default=0)
    angle: FloatProperty(name="Angle", subtype="ANGLE", default=0.0)
    follow_stroke: BoolProperty(name="Follow Stroke", default=False,
                                description="Rotate the tip along the stroke direction")

    # Dynamics
    size_jitter: FloatProperty(name="Size Jitter", min=0.0, max=1.0, default=0.0, subtype="FACTOR")
    angle_jitter: FloatProperty(name="Angle Jitter", min=0.0, max=1.0, default=0.0, subtype="FACTOR")
    flow_jitter: FloatProperty(name="Flow Jitter", min=0.0, max=1.0, default=0.0, subtype="FACTOR")
    scatter: FloatProperty(name="Scatter", min=0.0, max=3.0, default=0.0,
                           description="Random offset of each dab, relative to the radius")
    lazy_radius: IntProperty(name="Lazy Mouse", min=0, max=500, default=0, subtype="PIXEL",
                             description="The brush trails the cursor by this distance for smooth lines")

    # Stencil (screen space)
    use_stencil: BoolProperty(name="Stencil", default=False)
    stencil_image: PointerProperty(name="Stencil Image", type=bpy.types.Image)
    stencil_x: FloatProperty(name="X", default=400.0, subtype="PIXEL")
    stencil_y: FloatProperty(name="Y", default=300.0, subtype="PIXEL")
    stencil_size: FloatProperty(name="Size", min=4.0, default=200.0, subtype="PIXEL",
                                description="Half the stencil's height on screen")
    stencil_rotation: FloatProperty(name="Rotation", subtype="ANGLE", default=0.0)

    use_base_color: BoolProperty(name="Base Color", default=True)
    use_metallic: BoolProperty(name="Metallic", default=True)
    use_roughness: BoolProperty(name="Roughness", default=True)
    use_height: BoolProperty(name="Height", default=False)
    use_normal: BoolProperty(name="Normal", default=False)
    use_emission: BoolProperty(name="Emission", default=False)

    base_color: FloatVectorProperty(name="Base Color", subtype="COLOR_GAMMA", size=3,
                                    min=0.0, max=1.0, default=(0.8, 0.1, 0.1))
    metallic: FloatProperty(name="Metallic", min=0.0, max=1.0, default=0.0, subtype="FACTOR")
    roughness: FloatProperty(name="Roughness", min=0.0, max=1.0, default=0.4, subtype="FACTOR")
    height: FloatProperty(name="Height", min=0.0, max=1.0, default=0.7, subtype="FACTOR")
    emission: FloatVectorProperty(name="Emission", subtype="COLOR_GAMMA", size=3,
                                  min=0.0, max=1.0, default=(1.0, 0.5, 0.1))
    normal: FloatVectorProperty(name="Normal", subtype="COLOR_GAMMA", size=3, min=0.0, max=1.0,
                                default=(0.5, 0.5, 1.0),
                                description="Tangent-space normal to paint (0.5, 0.5, 1 = flat)")

    def value(self, ch):
        v = getattr(self, ch.key)
        return tuple(v) if ch.is_color else (v, v, v)


classes = (PS_LayerChannel, PS_MaskEffect, PS_Layer, PS_OutputMap, PS_BakedMap,
           PS_BakeSettings, PS_TextureSet, PS_Brush)


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
