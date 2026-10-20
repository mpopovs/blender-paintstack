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
    """Fresh, empty scene without reloading the file (safe inside a UI session)."""
    wm = bpy.context.window_manager
    new = bpy.data.scenes.new("PaintStack Test")
    for win in wm.windows:
        win.scene = new
    for sc in [s for s in bpy.data.scenes if s != new]:
        bpy.data.scenes.remove(sc)
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.images,
                 bpy.data.cameras, bpy.data.lights, bpy.data.worlds, bpy.data.textures,
                 bpy.data.node_groups):
        for idb in list(coll):
            coll.remove(idb)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll)
    bpy.data.orphans_purge(do_recursive=True)


def fresh_scene():
    """Empty scene with PaintStack (re)registered from this checkout."""
    import paintstack
    from paintstack import compositor, undo
    _unregister()
    if bpy.app.background:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    else:
        _clear_session()
    compositor.invalidate_all()
    undo.clear()
    paintstack.register()


init_gpu()
