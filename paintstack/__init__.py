"""PaintStack: Substance-Painter-style layered PBR texture painting for Blender."""

import bpy
from bpy.app.handlers import persistent

from . import baker, diagnostics, export, ops, paint_mode, picking, props, ui, undo
from .pixels import PREFIX

class PaintStackPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    cache_budget_mb: bpy.props.IntProperty(
        name="Compositor Cache (MB)", min=0, max=65536, default=0,
        description="Memory kept for decoded layers, entries and masks per texture set "
                    "(0 = Auto: 15% of system memory, which Apple Silicon shares with the GPU)")
    threads: bpy.props.IntProperty(
        name="Threads", min=0, max=64, default=0,
        description="CPU threads for compositing (0 = Auto: the performance cores)")
    compositor_backend: bpy.props.EnumProperty(
        name="Compositor",
        items=[("AUTO", "Auto", "GPU on a hardware GPU (Metal on Apple Silicon), CPU otherwise"),
               ("GPU", "GPU", "Blend layers on the GPU"),
               ("CPU", "CPU", "Blend layers on the CPU (multithreaded)")])
    gpu_display: bpy.props.BoolProperty(
        name="GPU Display", default=True,
        description="GPU compositor results go straight to the viewport; the CPU images are "
                    "updated when you pause for a second, and before saving or painting")
    gpu_live_preview: bpy.props.BoolProperty(
        name="GPU Live Stroke Preview", default=True,
        description="While painting, write stroke regions straight into the viewport's GPU "
                    "textures (Metal / OpenGL) instead of re-uploading whole images")

    def draw(self, context):
        from . import hardware
        layout = self.layout
        info = hardware.summary()
        box = layout.box()
        box.label(text=f"{info['os']} · {info['gpu_backend']} · {info['gpu']}", icon="SYSTEM")
        box.label(text=f"{info['ram_gb']} GB memory · {hardware.performance_cores()} performance cores "
                       f"· display scale {info['pixel_size']:g}")
        col = layout.column()
        row = col.row()
        row.prop(self, "cache_budget_mb")
        row.label(text=f"Auto = {info['auto_cache_mb']} MB")
        col.prop(self, "threads")
        col.prop(self, "compositor_backend")
        col.prop(self, "gpu_display")
        col.prop(self, "gpu_live_preview")
        layout.operator("paintstack.diagnostics", icon="CHECKMARK")


_classes = ((PaintStackPreferences,) + ops.classes + paint_mode.classes + picking.classes
            + baker.classes + export.classes + diagnostics.classes + ui.classes)


@persistent
def _pack_images(_):
    """Generated images are lost on save unless packed into the .blend. GPU results
    that only reached the viewport are written to the images first."""
    from . import compositor
    compositor.sync_all()
    for img in bpy.data.images:
        if img.name.startswith(PREFIX) and (img.is_dirty or not img.packed_file):
            img.pack()


@persistent
def _on_load(_):
    from . import compositor
    undo.clear()
    compositor.invalidate_all()


def register():
    props.register()
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.app.handlers.save_pre.append(_pack_images)
    bpy.app.handlers.load_post.append(_on_load)


def unregister():
    bpy.app.handlers.load_post.remove(_on_load)
    bpy.app.handlers.save_pre.remove(_pack_images)
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
    props.unregister()
