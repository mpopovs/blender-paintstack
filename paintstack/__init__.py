"""PaintStack: Substance-Painter-style layered PBR texture painting for Blender."""

import bpy
from bpy.app.handlers import persistent

from . import export, ops, paint_mode, props, ui, undo
from .pixels import PREFIX

_classes = ops.classes + paint_mode.classes + export.classes + ui.classes


@persistent
def _pack_images(_):
    """Generated images are lost on save unless packed into the .blend."""
    for img in bpy.data.images:
        if img.name.startswith(PREFIX) and (img.is_dirty or not img.packed_file):
            img.pack()


@persistent
def _on_load(_):
    undo.clear()


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
