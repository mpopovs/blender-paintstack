import bpy
from bpy.types import Panel, UIList

from . import brushes, generators, ops, stack
from .channels import enabled_channels
from .props import BAKE_MAPS, EFFECT_TYPES
from .paint_mode import PAINTSTACK_OT_paint_mode


class PaintStackPanel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PaintStack"


LAYER_ICONS = {"PAINT": "BRUSH_DATA", "FILL": "IMAGE_RGB", "FOLDER": "FILE_FOLDER", "FILTER": "FILTER"}


class PAINTSTACK_UL_layers(UIList):
    def draw_item(self, context, layout, data, layer, icon, active_data, active_prop, index):
        row = layout.row(align=True)
        for _ in range(stack.depth(data, layer)):
            row.label(text="", icon="BLANK1")
        if layer.type == "FOLDER":
            row.prop(layer, "expanded", text="", emboss=False,
                     icon="DISCLOSURE_TRI_DOWN" if layer.expanded else "DISCLOSURE_TRI_RIGHT")
        row.label(text="", icon=LAYER_ICONS[layer.type])
        row.prop(layer, "name", text="", emboss=False)
        if layer.is_anchor:
            row.label(text="", icon="PINNED")
        if layer.has_mask:
            row.label(text="", icon="MOD_MASK" if layer.edit_target == "MASK" else "SELECT_SET")
        row.prop(layer, "visible", text="", emboss=False,
                 icon="HIDE_OFF" if layer.visible else "HIDE_ON")

    def filter_items(self, context, data, propname):
        # Hide the contents of collapsed folders.
        lookup = stack.by_uid(data)
        flags = [0 if stack.hidden_by_collapse(data, l, lookup) else self.bitflag_filter_item
                 for l in data.layers]
        return flags, []


EFFECT_ICONS = {"PAINT": "BRUSH_DATA", "FILL": "SNAP_FACE", "LEVELS": "FILTER", "BLUR": "FILTER",
                "ID_MASK": "EYEDROPPER", "GRUNGE": "IMAGE_DATA", "ANCHOR": "PINNED"}


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


class PAINTSTACK_MT_anchors(bpy.types.Menu):
    bl_label = "Anchor Layers"

    def draw(self, context):
        _, ts = ops.texture_set(context)
        anchors = [l for l in ts.layers if l.is_anchor] if ts else []
        if not anchors:
            self.layout.label(text="No anchor layers")
        for l in anchors:
            self.layout.operator("paintstack.effect_set_anchor", text=l.name, icon="PINNED").uid = l.uid


class PAINTSTACK_MT_add_effect(bpy.types.Menu):
    bl_label = "Add Mask Effect"

    def draw(self, context):
        layout = self.layout
        for ident, label, _ in EFFECT_TYPES:
            if ident in ("LEVELS", "PAINT", "ID_MASK", "GRUNGE"):
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
    elif t == "ANCHOR":
        lookup = stack.by_uid(ts)
        anchor = lookup.get(effect.anchor_uid)
        col = layout.column(align=True)
        col.menu("PAINTSTACK_MT_anchors", text=anchor.name if anchor else "Choose Anchor…", icon="PINNED")
        col.prop(effect, "anchor_channel")
        col.prop(effect, "level")
        col.prop(effect, "softness")
        if not any(l.is_anchor for l in ts.layers):
            layout.label(text="Tick 'Anchor Point' on a layer first", icon="INFO")
    elif t == "GRUNGE":
        col = layout.column(align=True)
        col.prop(effect, "grunge")
        col.prop(effect, "grunge_tiling")
        col.prop(effect, "seed")
        col.prop(effect, "level")
        col.prop(effect, "softness")
        if ts.baked_map("POSITION") is None or ts.baked_map("WORLD_NORMAL") is None:
            layout.label(text="Bake Position + World Normal for seamless tri-planar", icon="INFO")
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
        for ch in ("base_color", "metallic", "roughness", "height", "normal", "emission"):
            col.prop(ts, "use_" + ch)
        if ts.use_height:
            layout.prop(ts, "bump_strength")
        layout.prop(ts, "view_mode")

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
        col.operator("paintstack.layer_add", text="", icon="FILE_FOLDER").type = "FOLDER"
        col.operator("paintstack.layer_add", text="", icon="FILTER").type = "FILTER"
        col.operator("paintstack.layer_duplicate", text="", icon="DUPLICATE")
        col.operator("paintstack.layer_remove", text="", icon="REMOVE")
        col.separator()
        col.operator("paintstack.layer_group", text="", icon="NEWFOLDER")
        col.operator("paintstack.layer_ungroup", text="", icon="X")
        col.separator()
        col.operator("paintstack.layer_move", text="", icon="TRIA_UP").direction = "UP"
        col.operator("paintstack.layer_move", text="", icon="TRIA_DOWN").direction = "DOWN"

        layer = ts.active_layer
        if layer is None:
            return
        row = layout.row()
        row.prop(layer, "opacity")
        row.prop(layer, "is_anchor", text="", icon="PINNED")

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

        if layer.type == "FOLDER":
            layout.label(text="Folder: opacity and mask apply to its contents", icon="INFO")
            return
        if layer.type == "FILTER":
            box = layout.box()
            box.prop(layer, "filter_type")
            col = box.column(align=True)
            if layer.filter_type in ("BLUR", "SHARPEN"):
                col.prop(layer, "filter_radius")
            if layer.filter_type in ("SHARPEN", "WARP"):
                col.prop(layer, "filter_amount")
            if layer.filter_type == "WARP":
                col.prop(layer, "warp_scale")
            if layer.filter_type == "HSL":
                col.prop(layer, "hue")
                col.prop(layer, "saturation")
                col.prop(layer, "lightness")
            row = box.row(align=True)
            for ch in enabled_channels(ts):
                lc = layer.channel(ch.id)
                row.prop(lc, "enabled", text=ch.label, toggle=True)
            return
        box = layout.box()
        if layer.type == "FILL":
            row = box.row(align=True)
            row.prop(layer, "projection", expand=True)
            row.prop(layer, "tiling")
            if layer.projection == "TRIPLANAR" and not (ts.baked_map("POSITION")
                                                        and ts.baked_map("WORLD_NORMAL")):
                box.label(text="Bake Position + World Normal for tri-planar", icon="ERROR")
        for ch in enabled_channels(ts):
            lc = layer.channel(ch.id)
            if lc is None:
                continue
            row = box.row(align=True)
            row.prop(lc, "enabled", text="")
            sub = row.row(align=True)
            sub.active = lc.enabled
            sub.label(text=ch.label)
            if layer.type == "FILL" and lc.texture is None:
                sub.prop(lc, "color" if ch.is_color else "value", text="")
            sub.prop(lc, "blend_mode", text="")
            sub.prop(lc, "opacity", text="")
            if layer.type == "FILL":
                sub = box.row(align=True)
                sub.active = lc.enabled
                sub.template_ID(lc, "texture", open="image.open")


class PAINTSTACK_MT_brush_presets(bpy.types.Menu):
    bl_label = "Brush Presets"

    def draw(self, context):
        layout = self.layout
        for name in brushes.BUILTIN_PRESETS:
            layout.operator("paintstack.brush_preset_apply", text=name).name = name
        user = brushes.user_presets()
        if user:
            layout.separator()
            for name in user:
                layout.operator("paintstack.brush_preset_apply", text=name, icon="USER").name = name
        layout.separator()
        layout.operator("paintstack.brush_preset_save", icon="ADD")
        layout.operator("paintstack.brush_preset_delete", icon="TRASH")


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
        layout.menu("PAINTSTACK_MT_brush_presets", text=b.preset_name or "Brush Presets", icon="BRUSH_DATA")
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
        layout.prop(b, "lazy_radius")

        layer = ts.active_layer
        if layer and (layer.edit_target == "MASK" or layer.type in ("FILL", "FOLDER")):
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


class PAINTSTACK_PT_brush_tip(PaintStackPanel, Panel):
    bl_label = "Tip & Dynamics"
    bl_parent_id = "PAINTSTACK_PT_brush"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        b = context.scene.paintstack_brush
        layout.prop(b, "tip")
        if b.tip == "IMAGE":
            layout.template_ID(b, "tip_image", open="image.open")
        elif b.tip not in ("ROUND", "SQUARE"):
            layout.prop(b, "tip_seed")
        row = layout.row(align=True)
        row.prop(b, "angle")
        row.prop(b, "follow_stroke", toggle=True)
        col = layout.column(align=True)
        col.prop(b, "size_jitter")
        col.prop(b, "angle_jitter")
        col.prop(b, "flow_jitter")
        col.prop(b, "scatter")


class PAINTSTACK_PT_brush_stencil(PaintStackPanel, Panel):
    bl_label = "Stencil"
    bl_parent_id = "PAINTSTACK_PT_brush"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        self.layout.prop(context.scene.paintstack_brush, "use_stencil", text="")

    def draw(self, context):
        layout = self.layout
        b = context.scene.paintstack_brush
        layout.active = b.use_stencil
        layout.template_ID(b, "stencil_image", open="image.open")
        col = layout.column(align=True)
        row = col.row(align=True)
        row.prop(b, "stencil_x")
        row.prop(b, "stencil_y")
        col.prop(b, "stencil_size")
        col.prop(b, "stencil_rotation")
        layout.label(text="In paint mode: hold S and move / wheel / Ctrl+wheel", icon="INFO")


class PAINTSTACK_MT_smart(bpy.types.Menu):
    bl_label = "Smart Assets"
    kind = "SMART_MATERIAL"

    def draw(self, context):
        from . import smart
        layout = self.layout
        builtins = smart.SMART_MATERIALS if self.kind == "SMART_MATERIAL" else smart.SMART_MASKS
        for name in builtins:
            op = layout.operator("paintstack.smart_apply", text=name)
            op.kind, op.name, op.path = self.kind, name, ""
        user = smart.user_assets(self.kind)
        if user:
            layout.separator()
            for name, path in user.items():
                op = layout.operator("paintstack.smart_apply", text=name, icon="ASSET_MANAGER")
                op.kind, op.name, op.path = self.kind, name, path


class PAINTSTACK_MT_smart_materials(PAINTSTACK_MT_smart):
    bl_label = "Smart Materials"
    kind = "SMART_MATERIAL"


class PAINTSTACK_MT_smart_masks(PAINTSTACK_MT_smart):
    bl_label = "Smart Masks"
    kind = "SMART_MASK"


class PAINTSTACK_MT_smart_delete(bpy.types.Menu):
    bl_label = "Delete Saved Asset"

    def draw(self, context):
        from . import smart
        layout = self.layout
        found = False
        for kind in (smart.KIND_MATERIAL, smart.KIND_MASK):
            for name, path in smart.user_assets(kind).items():
                layout.operator("paintstack.smart_delete", text=name, icon="TRASH").path = path
                found = True
        if not found:
            layout.label(text="No saved assets")


class PAINTSTACK_MT_grunge(bpy.types.Menu):
    bl_label = "Grunge Library"

    def draw(self, context):
        from .grunge import GRUNGE_TYPES
        layout = self.layout
        for use_as, label in (("STENCIL", "As Stencil"), ("TIP", "As Brush Tip"), ("IMAGE", "As Image")):
            layout.label(text=label)
            for ident, name, _ in GRUNGE_TYPES:
                op = layout.operator("paintstack.grunge_image", text=name)
                op.kind, op.use_as = ident, use_as
            layout.separator()


class PAINTSTACK_PT_smart(PaintStackPanel, Panel):
    bl_label = "Smart Assets"
    bl_parent_id = "PAINTSTACK_PT_main"

    @classmethod
    def poll(cls, context):
        return ops.texture_set(context)[0] is not None

    def draw(self, context):
        layout = self.layout
        row = layout.row(align=True)
        row.menu("PAINTSTACK_MT_smart_materials", text="Smart Material", icon="MATERIAL")
        row.menu("PAINTSTACK_MT_smart_masks", text="Smart Mask", icon="MOD_MASK")
        row = layout.row(align=True)
        row.operator("paintstack.smart_save", text="Save Material", icon="ADD").kind = "SMART_MATERIAL"
        row.operator("paintstack.smart_save", text="Save Mask", icon="ADD").kind = "SMART_MASK"
        row.menu("PAINTSTACK_MT_smart_delete", text="", icon="TRASH")
        layout.menu("PAINTSTACK_MT_grunge", icon="IMAGE_DATA")
        layout.label(text="Saved assets also appear in the Asset Browser ('PaintStack')", icon="INFO")


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


class PAINTSTACK_PT_system(PaintStackPanel, Panel):
    bl_label = "System"
    bl_parent_id = "PAINTSTACK_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        from . import hardware
        layout = self.layout
        info = hardware.summary()
        col = layout.column(align=True)
        col.label(text=f"{info['gpu_backend']} · {info['gpu']}", icon="SYSTEM")
        col.label(text=f"{info['ram_gb']} GB · {info['threads']} threads · scale {info['pixel_size']:g}")
        from . import compositor
        _, ts = ops.texture_set(context)
        if ts is not None:
            col.label(text=f"Compositor: {'GPU' if compositor.use_gpu(ts) else 'CPU'}")
        layout.operator("paintstack.diagnostics", icon="CHECKMARK")


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
           PAINTSTACK_MT_brush_presets, PAINTSTACK_MT_anchors, PAINTSTACK_MT_smart_materials,
           PAINTSTACK_MT_smart_masks, PAINTSTACK_MT_smart_delete, PAINTSTACK_MT_grunge,
           PAINTSTACK_PT_main, PAINTSTACK_PT_layers, PAINTSTACK_PT_smart,
           PAINTSTACK_PT_brush, PAINTSTACK_PT_brush_tip, PAINTSTACK_PT_brush_stencil,
           PAINTSTACK_PT_bake, PAINTSTACK_PT_export, PAINTSTACK_PT_system)
