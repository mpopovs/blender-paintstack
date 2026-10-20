"""Phase 5 headless test: the performance machinery must never change results.

After every kind of edit, the cached / checkpointed / per-channel / region-limited
paths are compared with a from-scratch composite (no cache).

    python tests/test_phase5.py
"""

import math
import os
import sys

import bpy
import gpu
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
gpu.init()

import paintstack  # noqa: E402
from paintstack import compositor, pixels, smart  # noqa: E402
from paintstack.channels import enabled_channels  # noqa: E402
from paintstack.engine.projector import Projector  # noqa: E402
from paintstack.stroke import Stroke  # noqa: E402
from test_headless import W, H, check, front_view  # noqa: E402
from test_phase3 import override  # noqa: E402


def outputs(ts):
    return {c.id: pixels.read(ts.output(c.id))[..., :3].copy() for c in enabled_channels(ts)}


def reference(ts):
    """From-scratch result: no persistent cache, no checkpoint, full regions."""
    res = compositor.composite_all(ts, cache={})
    return {cid: compositor.finalize(compositor.BY_ID_[cid], rgb) for cid, rgb in res.items()}


def same(ts, label, tol=2e-3):
    got, ref = outputs(ts), reference(ts)
    worst = max(float(np.abs(got[c] - ref[c]).max()) for c in ref)
    # 8-bit outputs quantise to 1/255.
    check(worst <= max(tol, 1.0 / 255 + 1e-4), f"{label}: cached == from scratch (max diff {worst:.4f})")


def flush():
    compositor._flush()


def L(ts, uid):
    return next(l for l in ts.layers if l.uid == uid)


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    paintstack.register()
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24)
    obj = bpy.context.active_object
    with override(obj):
        bpy.ops.paintstack.init()
    mat = obj.active_material
    ts = mat.paintstack
    ts.resolution = "256"
    ts.use_normal = True
    s = ts.bake
    s.samples = 2
    s.use_normal = s.use_id = s.use_thickness = False
    with override(obj):
        bpy.ops.paintstack.bake()
        bpy.ops.paintstack.smart_apply(kind="SMART_MATERIAL", name="Painted Steel")
    folder_uid = ts.active_layer.uid
    with override(obj):
        bpy.ops.paintstack.layer_add(type="PAINT")  # inside the folder
    inner_uid = ts.active_layer.uid
    compositor.composite(mat)
    same(ts, "initial")

    # Property tweaks go through the update callbacks -> request() -> _flush()
    paint = next(l for l in ts.layers if l.name == "Paint")
    paint.channel("BASE_COLOR").color = (0.9, 0.1, 0.1)  # per-channel invalidation
    flush()
    same(ts, "channel colour tweak")
    paint.opacity = 0.6  # layer-level
    flush()
    same(ts, "layer opacity tweak")
    paint.mask_effects[0].level = 0.3  # mask effect
    flush()
    same(ts, "mask effect tweak")
    L(ts, folder_uid).opacity = 0.7  # folder fade path (not in place)
    flush()
    same(ts, "folder opacity")
    L(ts, folder_uid).opacity = 1.0
    flush()
    same(ts, "folder opacity back to 1 (in-place path)")
    ts.use_emission = True  # texture-set wide
    flush()
    same(ts, "channel enabled")

    # Checkpoint reuse: repeated tweaks of the same root layer
    L(ts, folder_uid).expanded = False  # so the new layer lands above the folder, at the root
    ts.active_layer_index = 0
    with override(obj):
        bpy.ops.paintstack.layer_add(type="FILL")
    top_uid = ts.active_layer.uid
    top = ts.active_layer
    top.channel("ROUGHNESS").value = 0.2
    for v in (0.3, 0.5, 0.8):
        top.opacity = v
        flush()
    st = compositor.state(mat)
    check(st.checkpoint is not None and st.checkpoint[0] == top_uid, "checkpoint kept for the edited root layer")
    same(ts, "after repeated checkpointed tweaks")
    # Editing a layer below the checkpoint must not reuse it
    L(ts, folder_uid).visible = False
    flush()
    same(ts, "edit below the checkpoint")
    L(ts, folder_uid).visible = True
    flush()

    # Cover culling: an opaque fill on top hides everything for its channels
    cover = ts.active_layer
    for lc in cover.channels:
        lc.enabled = lc.channel == "METALLIC"
    cover.opacity = 1.0
    cover.channel("METALLIC").value = 0.25
    flush()
    check(np.allclose(outputs(ts)["METALLIC"], 0.25, atol=0.01), "opaque cover fill wins")
    same(ts, "cover culling")

    # Structural ops (add / move / group / remove) reuse cached entries
    with override(obj):
        bpy.ops.paintstack.layer_move(direction="DOWN")
        bpy.ops.paintstack.layer_group()
        bpy.ops.paintstack.layer_ungroup()
    same(ts, "move / group / ungroup")
    with override(obj):
        bpy.ops.paintstack.effect_add(type="GRUNGE")
    same(ts, "mask effect added")
    with override(obj):
        bpy.ops.paintstack.layer_add(type="FILTER")
    ts.active_layer.filter_type = "BLUR"
    flush()
    same(ts, "blur filter layer")

    # Region strokes: on a root layer, inside a folder, near a blur filter, on a mask
    proj = Projector(obj, bpy.context.evaluated_depsgraph_get(), mat, ts.res)
    b = bpy.context.scene.paintstack_brush
    b.radius, b.flow, b.hardness, b.opacity, b.spacing = 18, 1.0, 0.7, 1.0, 0.2
    b.use_height = b.use_normal = True

    def stroke(label, uid):
        ts.active_layer_index = next(i for i, l in enumerate(ts.layers) if l.uid == uid)
        st_ = Stroke(proj, front_view(), mat, b, seed=1)
        for i in range(24):
            a = i / 24 * 2 * math.pi
            st_.add_point(W / 2 + 60 * math.cos(a), H / 2 + 40 * math.sin(a))
            if i % 6 == 5:
                st_.update()
        st_.finish()
        same(ts, label)

    stroke("region stroke inside a folder, under a blur filter", inner_uid)
    ts.active_layer_index = 0
    with override(obj):
        bpy.ops.paintstack.layer_add(type="PAINT")
    stroke("region stroke on a root layer", ts.active_layer.uid)
    L(ts, inner_uid).edit_target = "MASK"
    with override(obj):
        ts.active_layer_index = next(i for i, l in enumerate(ts.layers) if l.uid == inner_uid)
        bpy.ops.paintstack.mask_add(white=True)
    b.mode = "ERASE"
    stroke("region stroke painting a mask", inner_uid)
    b.mode = "PAINT"

    # Warp filter forces whole-texture updates
    filt = next(l for l in ts.layers if l.type == "FILTER")
    filt.filter_type = "WARP"
    flush()
    stroke("stroke with a warp filter (whole texture)", inner_uid)

    # Undo restores and the cache follows
    from paintstack import undo
    undo.undo()
    same(ts, "after undo")
    undo.redo()
    same(ts, "after redo")

    # Regions are a small part of the texture for a small stroke
    proj.begin_view(*front_view())
    proj.clear_mask()
    proj.dabs([(W / 2, H / 2, 10, 1.0, 1.0, 0.0)])
    regions = proj.take_regions(4)
    area = sum((r[1] - r[0]) * (r[3] - r[2]) for r in regions) / ts.res ** 2
    m = proj.read_mask() > 0
    inside = np.zeros_like(m)
    for y0, y1, x0, x1 in regions:
        inside[y0:y1, x0:x1] = True
    check(area < 0.15 and not (m & ~inside).any(), f"dab regions cover the dab ({area:.1%} of the texture)")
    r0 = regions[0]
    check(np.allclose(proj.read_mask(r0), proj.read_mask()[r0[0]:r0[1], r0[2]:r0[3]]),
          "partial readback == slice of full readback")

    # The cache respects its memory budget
    compositor.DEFAULT_BUDGET_MB, old = 1, compositor.DEFAULT_BUDGET_MB
    compositor.composite(mat)
    total = sum(compositor._nbytes(v) for v in compositor.state(mat).cache.values())
    check(total <= 1024 * 1024, f"cache trimmed to its budget ({total / 1e6:.2f} MB)")
    compositor.DEFAULT_BUDGET_MB = old
    same(ts, "after cache eviction")

    print("FAILED" if check.failed else "ALL PASSED")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
