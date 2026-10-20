import bpy
from bpy.types import Panel, UIList

from . import generators, ops
from .channels import enabled_channels
from .props import BAKE_MAPS, EFFECT_TYPES
from .paint_mode import PAINTSTACK_OT_paint_mode


class PaintStackPanel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PaintStack"


class PAINTSTACK_UL_layers(UIList):
    def draw_item(self, context, layout, data, layer, icon, active_data, active_prop, index):
        row = layout.row(align=True)
        row.label(text="", icon="BRUSH_DATA" if layer.type == "PAINT" else "IMAGE_RGB")
        row.prop(layer, "name", text="", emboss=False)
        if layer.mask_image:
            row.label(text="", icon="MOD_MASK" if layer.edit_target == "MASK" else "SELECT_SET")
        row.prop(layer, "visible", text="", emboss=False,
                 icon="HIDE_OFF" if layer.visible else "HIDE_ON")


EFFECT_ICONS = {"PAINT": "BRUSH_DATA", "FILL": "SNAP_FACE", "LEVELS": "FILTER", "BLUR": "FILTER",
                "ID_MASK": "EYEDROPPER"}


class PAINTSTACK_UL_mask_effects(UIList):
    def draw_item(self, context, layout, data, effect, icon, active_data, active_prop, index):
        row = layout.row(align=True)
        row.label(text="", icon=EFFECT_ICONS.get(effect.type, "TEXTURE"))
        row.prop(effect, "name", text="", emboss=False)
        if effect.type not in ("LEVELS", "BLUR"):
            row.label(text=effect.blend_mode.title())
        row.prop(effect, "enabled", text="", emboss=False,
                 icon="HIDE_OFF" if effect.enabled else "HIDE_ON")

    def filter_items(self, context, data, propname):
        # Show the top of the stack first, like the layer list.
        n = len(getattr(data, propname))
        return [], [n - 1 - i for i in range(n)]


class PAINTSTACK_MT_add_effect(bpy.types.Menu):
    bl_label = "Add Mask Effect"

    def draw(self, context):
        layout = self.layout
        for ident, label, _ in EFFECT_TYPES:
            if ident in ("LEVELS", "PAINT", "ID_MASK"):
                layout.separator()
            layout.operator("paintstack.effect_add", text=label,
                            icon=EFFECT_ICONS.get(ident, "TEXTURE")).type = ident


def draw_effect(layout, effect, ts):
    col = layout.column(align=True)
    if effect.type not in ("LEVELS", "BLUR"):
        row = col.row(align=True)
        row.prop(effect, "blend_mode", text="")
        row.prop(effect, "invert", text="Invert", toggle=True)
    else:
        col.prop(effect, "invert")
    col.prop(effect, "opacity")

    t = effect.type
    missing = generators.missing(effect, ts)
    if missing:
        layout.label(text="Bake first: " + ", ".join(m.title() for m in missing), icon="ERROR")
    if t == "FILL":
        col.prop(effect, "value")
    elif t == "LEVELS":
        col.prop(effect, "in_min")
        col.prop(effect, "in_max")
        col.prop(effect, "gamma")
    elif t == "BLUR":
        col.prop(effect, "blur_radius")
    elif t == "ID_MASK":
        col = layout.column(align=True)
        row = col.row(align=True)
        row.prop(effect, "id_color", text="")
        row.operator("paintstack.id_pick", text="Pick", icon="EYEDROPPER")
        col.prop(effect, "id_tolerance")
    elif t in generators.REQUIRES:
        col = layout.column(align=True)
        col.prop(effect, "level")
        col.prop(effect, "softness")
        if t == "POSITION":
            col.row().prop(effect, "axis", expand=True)
        col = layout.column(align=True)
        if t != "NOISE":
            col.prop(effect, "noise_amount")
        col.prop(effect, "noise_scale")
        col.prop(effect, "noise_octaves")
        col.prop(effect, "seed")
        if ts.baked_map("POSITION") is None:
            layout.label(text="Bake Position for seamless noise", icon="INFO")


class PAINTSTACK_PT_main(PaintStackPanel, Panel):
    bl_label = "PaintStack"

    def draw(self, context):
        layout = self.layout
        mat = ops.active_material(context)
        if context.active_object is None or context.active_object.type != "MESH":
            layout.label(text="Select a mesh object", icon="INFO")
            return
        if mat is None or not mat.paintstack.initialized:
            layout.operator("paintstack.init", icon="ADD")
            return
        ts = mat.paintstack
        layout.label(text=f"Texture Set: {mat.name}", icon="MATERIAL")
        layout.prop(ts, "resolution")
        col = layout.column(heading="Channels", align=True)
        for ch in ("base_color", "metallic", "roughness", "height", "emission"):
            col.prop(ts, "use_" + ch)
        if ts.use_height:
            layout.prop(ts, "bump_strength")

        layout.separator()
        if PAINTSTACK_OT_paint_mode.running:
            layout.label(text="Painting… (Esc to exit)", icon="BRUSH_DATA")
        else:
            layout.operator("paintstack.paint_mode", icon="BRUSH_DATA")
        row = layout.row(align=True)
        row.operator("paintstack.undo_stroke", text="Undo Stroke", icon="LOOP_BACK").redo = False
        row.operator("paintstack.undo_stroke", text="Redo", icon="LOOP_FORWARDS").redo = True
        layout.operator("paintstack.composite", icon="FILE_REFRESH")


class PAINTSTACK_PT_layers(PaintStackPanel, Panel):
    bl_label = "Layers"
    bl_parent_id = "PAINTSTACK_PT_main"

    @classmethod
    def poll(cls, context):
        return ops.texture_set(context)[0] is not None

    def draw(self, context):
        layout = self.layout
        _, ts = ops.texture_set(context)
        row = layout.row()
        row.template_list("PAINTSTACK_UL_layers", "", ts, "layers", ts, "active_layer_index", rows=6)
        col = row.column(align=True)
        col.operator("paintstack.layer_add", text="", icon="BRUSH_DATA").type = "PAINT"
        col.operator("paintstack.layer_add", text="", icon="IMAGE_RGB").type = "FILL"
        col.operator("paintstack.layer_duplicate", text="", icon="DUPLICATE")
        col.operator("paintstack.layer_remove", text="", icon="REMOVE")
        col.separator()
        col.operator("paintstack.layer_move", text="", icon="TRIA_UP").direction = "UP"
        col.operator("paintstack.layer_move", text="", icon="TRIA_DOWN").direction = "DOWN"

        layer = ts.active_layer
        if layer is None:
            return
        layout.prop(layer, "opacity")

        box = layout.box()
        if layer.has_mask:
            row = box.row(align=True)
            row.prop(layer, "edit_target", expand=True)
            row = box.row(align=True)
            row.prop(layer, "mask_enabled", text="Mask")
            row.prop(layer, "mask_invert", text="Invert")
            if layer.mask_image:
                row.operator("paintstack.mask_remove", text="", icon="X")
        else:
            row = box.row(align=True)
            row.label(text="Mask", icon="MOD_MASK")
            row.operator("paintstack.mask_add", text="White").white = True
            row.operator("paintstack.mask_add", text="Black").white = False

        row = box.row()
        row.template_list("PAINTSTACK_UL_mask_effects", "", layer, "mask_effects",
                          layer, "active_mask_effect_index", rows=3)
        col = row.column(align=True)
        col.menu("PAINTSTACK_MT_add_effect", text="", icon="ADD")
        col.operator("paintstack.effect_remove", text="", icon="REMOVE")
        col.separator()
        col.operator("paintstack.effect_move", text="", icon="TRIA_UP").direction = "UP"
        col.operator("paintstack.effect_move", text="", icon="TRIA_DOWN").direction = "DOWN"
        effect = layer.active_mask_effect
        if effect is not None:
            draw_effect(box, effect, ts)

        box = layout.box()
        for ch in enabled_channels(ts):
            lc = layer.channel(ch.id)
            if lc is None:
                continue
            row = box.row(align=True)
            row.prop(lc, "enabled", text="")
            sub = row.row(align=True)
            sub.active = lc.enabled
            sub.label(text=ch.label)
            if layer.type == "FILL":
                sub.prop(lc, "color" if ch.is_color else "value", text="")
            sub.prop(lc, "blend_mode", text="")
            sub.prop(lc, "opacity", text="")


class PAINTSTACK_PT_brush(PaintStackPanel, Panel):
    bl_label = "Brush"
    bl_parent_id = "PAINTSTACK_PT_main"

    @classmethod
    def poll(cls, context):
        return ops.texture_set(context)[0] is not None

    def draw(self, context):
        layout = self.layout
        b = context.scene.paintstack_brush
        _, ts = ops.texture_set(context)
        layout.row().prop(b, "mode", expand=True)
        col = layout.column(align=True)
        col.prop(b, "radius")
        col.prop(b, "hardness")
        col.prop(b, "flow")
        col.prop(b, "opacity")
        col.prop(b, "spacing")
        row = layout.row()
        row.prop(b, "use_pressure")
        row.prop(b, "mirror_x")

        layer = ts.active_layer
        if layer and (layer.edit_target == "MASK" or layer.type == "FILL"):
            layout.label(text="Painting the layer mask", icon="MOD_MASK")
            return
        box = layout.box()
        box.label(text="Paint Channels")
        for ch in enabled_channels(ts):
            row = box.row(align=True)
            row.prop(b, "use_" + ch.key, text="")
            sub = row.row(align=True)
            sub.active = getattr(b, "use_" + ch.key)
            sub.prop(b, ch.key)


class PAINTSTACK_PT_bake(PaintStackPanel, Panel):
    bl_label = "Bake Mesh Maps"
    bl_parent_id = "PAINTSTACK_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return ops.texture_set(context)[0] is not None

    def draw(self, context):
        layout = self.layout
        _, ts = ops.texture_set(context)
        s = ts.bake
        layout.prop(s, "high_poly")
        col = layout.column(heading="Maps", align=True)
        for ident, label, _ in BAKE_MAPS:
            row = col.row(align=True)
            row.prop(s, "use_" + ident.lower(), text=label)
            row.label(text="", icon="CHECKMARK" if ts.baked_map(ident) else "BLANK1")
        col = layout.column(align=True)
        if s.high_poly:
            col.prop(s, "cage_object")
            sub = col.column(align=True)
            sub.active = s.cage_object is None
            sub.prop(s, "cage_extrusion")
            col.prop(s, "max_ray_distance")
        if s.use_id:
            col.prop(s, "id_source")
        col.prop(s, "ao_distance")
        col.prop(s, "thickness_distance")
        col.prop(s, "samples")
        col.prop(s, "margin")
        row = layout.row()
        if ts.baked_map("NORMAL"):
            row.prop(ts, "use_baked_normal")
        if ts.baked_map("AO"):
            row.prop(ts, "preview_ao")
        layout.operator("paintstack.bake", icon="RENDER_STILL")


class PAINTSTACK_PT_export(PaintStackPanel, Panel):
    bl_label = "Export"
    bl_parent_id = "PAINTSTACK_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return ops.texture_set(context)[0] is not None

    def draw(self, context):
        self.layout.operator("paintstack.export", icon="EXPORT")


classes = (PAINTSTACK_UL_layers, PAINTSTACK_UL_mask_effects, PAINTSTACK_MT_add_effect,
           PAINTSTACK_PT_main, PAINTSTACK_PT_layers, PAINTSTACK_PT_brush, PAINTSTACK_PT_bake,
           PAINTSTACK_PT_export)
