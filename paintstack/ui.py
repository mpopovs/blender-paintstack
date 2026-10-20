import bpy
from bpy.types import Panel, UIList

from . import ops
from .channels import enabled_channels
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
        if layer.mask_image:
            row = box.row(align=True)
            row.prop(layer, "edit_target", expand=True)
            row = box.row(align=True)
            row.prop(layer, "mask_enabled", text="Mask")
            row.prop(layer, "mask_invert", text="Invert")
            row.operator("paintstack.mask_remove", text="", icon="X")
        else:
            row = box.row(align=True)
            row.label(text="Mask", icon="MOD_MASK")
            row.operator("paintstack.mask_add", text="White").white = True
            row.operator("paintstack.mask_add", text="Black").white = False

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


class PAINTSTACK_PT_export(PaintStackPanel, Panel):
    bl_label = "Export"
    bl_parent_id = "PAINTSTACK_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return ops.texture_set(context)[0] is not None

    def draw(self, context):
        self.layout.operator("paintstack.export", icon="EXPORT")


classes = (PAINTSTACK_UL_layers, PAINTSTACK_PT_main, PAINTSTACK_PT_layers,
           PAINTSTACK_PT_brush, PAINTSTACK_PT_export)
