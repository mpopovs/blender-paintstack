"""Test environment helpers, so the same suites run in two places:

* `bpy` as a Python module (Linux CI, background, software OpenGL), and
* the real Blender app, e.g. on a Mac with Metal (see run_all.py / run_on_mac.sh),
  where the tests run inside the UI session and must not reset it."""

import bpy
import gpu


def init_gpu():
    """The bpy module needs an explicit GPU context; the Blender app has one."""
    if hasattr(gpu, "init"):
        try:
            gpu.init()
        except Exception:
            pass


def _unregister():
    import paintstack
    try:
        paintstack.unregister()
    except Exception:
        pass


def _clear_session():
    """Empty the current scene without reloading the file (safe inside a UI session).

    The window's scene, view layer and workspaces stay as they are: swapping the
    window to a new scene and deleting the old one from a timer leaves the window
    pointing at freed data until the next redraw, which crashes Blender."""
    scene = bpy.context.window.scene if bpy.context.window else bpy.context.scene
    for sc in [s for s in bpy.data.scenes if s != scene]:
        bpy.data.scenes.remove(sc)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights,
                 bpy.data.textures, bpy.data.node_groups):
        for idb in list(coll):
            coll.remove(idb)
    for img in list(bpy.data.images):
        if img.type not in {"RENDER_RESULT", "COMPOSITING"}:  # owned by the UI
            bpy.data.images.remove(img)
    for w in list(bpy.data.worlds):
        if w != scene.world:
            bpy.data.worlds.remove(w)


# The UI-session reset (forced on to exercise it with the bpy module too).
SOFT_RESET = not bpy.app.background


def fresh_scene():
    """Empty scene with PaintStack (re)registered from this checkout."""
    import paintstack
    from paintstack import compositor, undo
    _unregister()
    if SOFT_RESET:
        _clear_session()
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    compositor.invalidate_all()
    undo.clear()
    paintstack.register()
    if SOFT_RESET:
        # The scene survives the soft reset: put its PaintStack settings (brush,
        # tool options) back to their defaults like a reloaded file would.
        scene = bpy.context.scene
        for name in [p.identifier for p in scene.bl_rna.properties]:
            if name.startswith("paintstack"):
                scene.property_unset(name)


init_gpu()
