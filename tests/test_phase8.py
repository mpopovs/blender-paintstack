"""Phase 8 test: cheap slider changes (blend-only invalidation, GPU uploads
reused), strokes that keep their outputs on the GPU, and the results still
matching the CPU compositor.

    python tests/test_phase8.py
"""

import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _env import fresh_scene  # noqa: E402  (also sets up the GPU)

from paintstack import compositor, pixels  # noqa: E402
from paintstack.engine import gpu_compositor, live, projector as projector_mod  # noqa: E402
from paintstack.engine.projector import Projector  # noqa: E402
from paintstack.stroke import Stroke  # noqa: E402
from test_headless import W, H, check, front_view  # noqa: E402
from test_phase3 import override  # noqa: E402
from test_phase7 import TOL, outputs  # noqa: E402


def scene(res=256):
    fresh_scene()
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24)
    obj = bpy.context.active_object
    with override(obj):
        bpy.ops.paintstack.init()
    mat = obj.active_material
    ts = mat.paintstack
    ts.resolution = str(res)
    s = ts.bake
    s.samples = 2
    s.use_normal = s.use_id = s.use_thickness = False
    with override(obj):
        bpy.ops.paintstack.bake()
        bpy.ops.paintstack.layer_add(type="PAINT")
        bpy.ops.paintstack.layer_add(type="FILL")
        bpy.ops.paintstack.effect_add(type="DIRT")
    ts.active_layer.name = "Tint"
    return obj, mat


def reference(mat):
    """From-scratch CPU composite (no cache), written to the images."""
    compositor.BACKEND_OVERRIDE = "CPU"
    compositor.invalidate(mat)
    compositor.composite(mat)
    return outputs(mat.paintstack)


def same(a, b, tol=TOL):
    return max(float(np.abs(a[c] - b[c]).max()) for c in a) <= tol


class Counter:
    """Counts make_texture calls (texture uploads)."""

    def __enter__(self):
        self.n = 0
        self.orig = projector_mod.make_texture

        def counted(*args, **kw):
            self.n += 1
            return self.orig(*args, **kw)
        projector_mod.make_texture = counted
        gpu_compositor.make_texture = counted
        return self

    def __exit__(self, *exc):
        projector_mod.make_texture = self.orig
        gpu_compositor.make_texture = self.orig


def stroke(obj, mat, live, seed=0):
    b = bpy.context.scene.paintstack_brush
    b.radius, b.flow, b.spacing = 30, 0.8, 0.2
    proj = Projector(obj, bpy.context.evaluated_depsgraph_get(), mat, mat.paintstack.res)
    s = Stroke(proj, front_view(), mat, b, seed=seed, live=live)
    for i in range(24):
        a = i / 24 * 2 * math.pi
        s.add_point(W / 2 + 60 * math.cos(a), H / 2 + 60 * math.sin(a))
        if i % 6 == 5:
            s.update()
    return s


def gpu_mask_checks():
    """Every mask effect, blend mode, invert and filter: GPU mask == CPU mask."""
    from paintstack import masks, ops, pixels as px_mod
    from paintstack.engine import gpu_masks
    fresh_scene()
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16)
    obj = bpy.context.active_object
    with override(obj):
        bpy.ops.paintstack.init()
    mat = obj.active_material
    ts = mat.paintstack
    ts.resolution = "256"
    s = ts.bake
    s.samples = 2
    s.use_normal = False
    s.use_id = s.use_thickness = True
    with override(obj):
        bpy.ops.paintstack.bake()
        bpy.ops.paintstack.layer_add(type="FILL")
    layer = ts.active_layer
    ops.add_mask(mat, layer)
    rng = np.random.default_rng(3)
    px_mod.write(layer.mask_image, rng.random((256, 256, 4)).astype(np.float32))
    gc = gpu_compositor.instance()
    gc.namespace = mat.name
    modes = ["NORMAL", "MULTIPLY", "ADD", "SUBTRACT", "MAX", "MIN"]
    worst = (0.0, "")

    def compare(label, varied=True):
        nonlocal worst
        cpu = masks.evaluate(layer, None, {})
        t = gpu_masks.evaluate(gc, layer, compositor.state(mat).cache, set())
        g = gc.read(t)[..., 0]
        gc.release(t)
        d = float(np.abs(g - cpu).max())
        if d > worst[0]:
            worst = (d, label)
        check(d < 4e-3 and (not varied or 0.005 < float(cpu.std())), f"GPU mask == CPU: {label} (max diff {d:.4f}, "
                                                    f"mean {float(cpu.mean()):.2f}, std {float(cpu.std()):.2f})")

    compare("mask image only")

    def add_effect(i, kind, stacked):
        if not stacked:
            layer.mask_effects.clear()
        e = ops.add_mask_effect(mat, layer, kind)
        if kind == "PAINT":
            ops.ensure_effect_image(mat, layer, e)
            px_mod.write(e.image, rng.random((256, 256, 4)).astype(np.float32))
        if kind not in ("LEVELS", "BLUR"):
            e.blend_mode = modes[i % len(modes)]
        e.opacity = 0.75
        e.invert = i % 3 == 0
        if kind == "FILL":
            e.value = 0.3
        if kind == "LEVELS":
            e.in_min, e.in_max, e.gamma = 0.1, 0.8, 1.7
        if kind == "ID_MASK":
            e.id_color = tuple(generators_id_color(ts))
        if hasattr(e, "level"):
            e.level, e.softness = 0.55, 0.15
        compare(f"{'stack + ' if stacked else ''}{kind} ({e.blend_mode}{', inverted' if e.invert else ''})",
                varied=not stacked)

    kinds = ["PAINT", "FILL", "EDGE_WEAR", "DIRT", "CURVATURE", "AO", "THICKNESS", "POSITION", "NOISE",
             "ID_MASK", "GRUNGE", "LEVELS", "BLUR"]
    for stacked in (False, True):
        layer.mask_effects.clear()
        for i, kind in enumerate(kinds):
            add_effect(i, kind, stacked)
    layer.mask_invert = True
    compare("inverted layer mask")
    print(f"worst GPU mask difference {worst[0]:.4f} ({worst[1]})")


def udim_plane():
    """Two quads facing the camera: the left one in UDIM tile 1001, the right one
    in 1002. World x = 2u - 2 and z = 2v - 1 across both."""
    me = bpy.data.meshes.new("UDIMPlane")
    verts = [(-2, 0, -1), (0, 0, -1), (0, 0, 1), (-2, 0, 1), (2, 0, -1), (2, 0, 1)]
    me.from_pydata(verts, [], [(0, 1, 2, 3), (1, 4, 5, 2)])
    uv = me.uv_layers.new(name="UVMap")
    coords = [(0, 0), (1, 0), (1, 1), (0, 1), (1, 0), (2, 0), (2, 1), (1, 1)]
    for loop, c in zip(uv.data, coords):
        loop.uv = c
    obj = bpy.data.objects.new("UDIMPlane", me)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def udim_checks():
    import tempfile
    from paintstack import export, ops, pixels as px_mod
    fresh_scene()
    obj = udim_plane()
    with override(obj):
        bpy.ops.paintstack.init()
    mat = obj.active_material
    ts = mat.paintstack
    ts.resolution = "256"
    # Paint something before switching to UDIM: it must stay in tile 1001.
    with override(obj):
        bpy.ops.paintstack.layer_add(type="PAINT")
    layer = ts.active_layer
    lc = layer.channel("BASE_COLOR")
    ops.ensure_layer_image(mat, layer, lc)
    before = np.random.default_rng(5).random((256, 256, 4)).astype(np.float32)
    px_mod.write(lc.image, before)

    with override(obj):
        check(bpy.ops.paintstack.udim_detect() == {"FINISHED"}, "detect UDIM tiles")
    check(ts.tiles == [1001, 1002] and ts.grid == (2, 1), f"tiles {ts.tiles}, grid {ts.grid}")
    check(tuple(lc.image.size) == (512, 256) and tuple(ts.output("BASE_COLOR").size) == (512, 256),
          "images re-laid out to 2 tiles side by side")
    after = px_mod.read(lc.image)
    check(np.abs(after[:, :256] - before).max() < 1.0 / 255 + 1e-6, "tile 1001 content kept in place")
    check(np.allclose(after[:, 256:], after[0, 256]), "new tile starts uniform")
    ts.layers.remove(ts.active_layer_index)

    # Projection across the tile border
    proj = Projector(obj, bpy.context.evaluated_depsgraph_get(), mat, ts.res)
    check(proj.size == (512, 256), "stroke mask covers both tiles")
    view = front_view()
    proj.begin_view(*view)
    proj.set_tip(None)
    proj.clear_mask()
    proj.dabs([(W / 2, H / 2, 60, 1.0, 1.0, 0.0)])
    m = proj.read_mask() > 0.5
    h, w = 256, 512
    u = (np.arange(w) + 0.5) / w * 2
    v = (np.arange(h) + 0.5) / h
    uu, vv = np.meshgrid(u, v)
    world = np.stack([2 * uu - 2, np.zeros_like(uu), 2 * vv - 1, np.ones_like(uu)], -1)
    clip = world @ np.array(view[2], np.float32).T
    sx = (clip[..., 0] / clip[..., 3] * 0.5 + 0.5) * W
    sy = (clip[..., 1] / clip[..., 3] * 0.5 + 0.5) * H
    ref = np.hypot(sx - W / 2, sy - H / 2) < 60
    iou = (m & ref).sum() / max((m | ref).sum(), 1)
    check(iou > 0.95 and m[:, :256].any() and m[:, 256:].any(),
          f"dab across the tile border matches the reference (IoU {iou:.3f})")
    regions = proj.regions_for([(W / 2, H / 2, 60)])
    check(all(r[1] <= 256 and r[3] <= 512 for r in regions), "regions stay inside the layout")
    proj.free()

    # Bake through the atlas UV map
    s = ts.bake
    s.samples = 2
    s.use_normal = s.use_id = s.use_thickness = s.use_curvature = False
    with override(obj):
        bpy.ops.paintstack.bake()
    pos = px_mod.read(ts.baked_map("POSITION"))
    check(tuple(ts.baked_map("POSITION").size) == (512, 256), "baked maps cover both tiles")
    left, right = pos[64:192, 16:240, 0].mean(), pos[64:192, 272:496, 0].mean()
    check(right > left + 0.3, f"position bake follows the tiles (x: {left:.2f} | {right:.2f})")
    from paintstack import picking
    l_px = picking.sample(ts.baked_map("POSITION"), (0.5, 0.5), ts.grid)
    r_px = picking.sample(ts.baked_map("POSITION"), (1.5, 0.5), ts.grid)
    check(r_px[0] > l_px[0] + 0.3, f"picking reads the right tile ({l_px[0]:.2f} | {r_px[0]:.2f})")
    check([l.name for l in obj.data.uv_layers] == ["UVMap"] and obj.data.uv_layers.active.name == "UVMap",
          "temporary atlas UV map removed after baking")

    # Layers on the atlas: GPU == CPU (non-square images, filters, generator masks)
    from paintstack import grunge
    with override(obj):
        bpy.ops.paintstack.layer_add(type="PAINT")
    paint = ts.active_layer
    plc = paint.channel("BASE_COLOR")
    ops.ensure_layer_image(mat, paint, plc)
    px_mod.write(plc.image, np.random.default_rng(9).random((256, 512, 4)).astype(np.float32))
    compositor.invalidate(mat, paint.uid)  # what a stroke does after writing a layer
    paint_uid = paint.uid  # (layer references go stale when layers are added)
    with override(obj):
        bpy.ops.paintstack.layer_add(type="FILL")
        bpy.ops.paintstack.effect_add(type="NOISE")
        bpy.ops.paintstack.effect_add(type="BLUR")
    fill = ts.active_layer
    fill.channel("BASE_COLOR").color = (0.9, 0.1, 0.1)
    fill.channel("ROUGHNESS").texture = grunge.make_image("CELLS", 0)
    with override(obj):
        bpy.ops.paintstack.layer_add(type="FILTER")
    filt = ts.active_layer
    for kind in ("BLUR", "WARP"):
        filt.filter_type = kind
        compositor.BACKEND_OVERRIDE = "CPU"
        compositor.composite(mat)
        cpu = outputs(ts)
        compositor.BACKEND_OVERRIDE = "GPU"
        compositor.DEFER_OVERRIDE = False
        compositor.composite(mat)
        gpu_ = outputs(ts)
        d = max(float(np.abs(cpu[c] - gpu_[c]).max()) for c in cpu)
        varied = min(float(cpu[c].std()) for c in ("BASE_COLOR", "ROUGHNESS"))
        check(d <= TOL and varied > 0.01 and compositor.use_gpu(ts),
              f"UDIM atlas with a {kind} filter: GPU == CPU (max diff {d:.4f}, std {varied:.3f})")
    # A live stroke across the tile border, on top of a pending GPU composite
    ts.active_layer_index = next(i for i, l in enumerate(ts.layers) if l.uid == paint_uid)
    plc = ts.active_layer.channel("BASE_COLOR")
    compositor.BACKEND_OVERRIDE = "GPU"
    compositor.DEFER_OVERRIDE = True
    compositor.composite(mat)
    before = px_mod.read(plc.image).copy()
    st_ = stroke(obj, mat, True)
    st_.finish()
    compositor.sync_outputs(mat)
    got = outputs(ts)
    changed = np.abs(px_mod.read(plc.image) - before).max(-1) > 0.01
    compositor.DEFER_OVERRIDE = None
    ref = reference(mat)
    d = max(float(np.abs(got[c] - ref[c]).max()) for c in got)
    check(changed[:, :256].any() and changed[:, 256:].any() and d <= TOL,
          f"UDIM stroke paints both tiles; result == CPU composite (max diff {d:.4f})")
    compositor.BACKEND_OVERRIDE = compositor.DEFER_OVERRIDE = None

    # Material samples the layout; export writes one file per tile
    mapping = mat.node_tree.nodes.get("PS_UDIM Layout")
    check(mapping is not None and tuple(mapping.inputs["Scale"].default_value)[:2] == (0.5, 1.0),
          "material maps UVs to the tile layout")
    d = tempfile.mkdtemp()
    with override(obj):
        bpy.ops.paintstack.export(directory=d, preset="BLENDER", padding=0)
    files = sorted(os.listdir(d))
    check("Material_BaseColor.1001.png" in files or any(f.endswith("_BaseColor.1001.png") for f in files),
          f"UDIM export names ({files[:3]}...)")
    tile2 = [f for f in files if f.endswith("_BaseColor.1002.png")]
    if tile2:
        img = bpy.data.images.load(os.path.join(d, tile2[0]))
        check(tuple(img.size) == (256, 256), "each exported tile has the tile resolution")
        bpy.data.images.remove(img)

    # Too large for the GPU: refused, previous resolution kept
    ts.udim_tiles = "1001-1005"
    ts.resolution = "4096"  # 5 x 4096 = 20480 > 16384
    check(ts.resolution == "256" and tuple(ts.output("BASE_COLOR").size) == (1280, 256),
          f"oversized layout refused ({ts.resolution}, {ops._size_error})")


def generators_id_color(ts):
    from paintstack import generators
    idm = generators.baked(ts, "ID")
    return idm[idm.shape[0] // 2, idm.shape[1] // 2]


def main():
    udim_checks()
    gpu_mask_checks()
    obj, mat = scene()
    ts = mat.paintstack
    tint = ts.layers[0]
    st = compositor.state(mat)

    # Blend-only properties keep the layer's decoded pixels and mask
    compositor.BACKEND_OVERRIDE = "CPU"
    compositor.composite(mat)
    mask = st.cache.get(("mask", tint.uid))
    check(mask is not None, "fill layer mask cached")
    tint.opacity = 0.6
    tint.channel("BASE_COLOR").color = (0.9, 0.2, 0.1)
    tint.channel("ROUGHNESS").blend_mode = "MULTIPLY"
    check(st.cache.get(("mask", tint.uid)) is mask, "opacity / colour / blend mode keep the cached mask")
    compositor._flush()
    got = outputs(ts)
    check(same(got, reference(mat), 1e-6), "blend-only update == from-scratch composite (CPU)")
    tint.channel("BASE_COLOR").texture = None  # a real (non blend-only) channel change still works
    compositor._flush()
    check(same(outputs(ts), reference(mat), 1e-6), "channel texture change recomposites")

    # GPU: a slider change uploads nothing; result matches the CPU
    compositor.BACKEND_OVERRIDE = "GPU"
    compositor.DEFER_OVERRIDE = False
    compositor.composite(mat)
    with Counter() as c:
        tint.opacity = 0.3
        compositor._flush()
        tint.channel("BASE_COLOR").color = (0.1, 0.8, 0.3)
        compositor._flush()
    check(c.n == 0, f"GPU slider changes upload nothing ({c.n} uploads)")
    gpu_out = outputs(ts)
    check(same(gpu_out, reference(mat)), "GPU after slider changes == CPU")
    compositor.BACKEND_OVERRIDE = "GPU"
    compositor.composite(mat)
    with Counter() as c:
        ts.layers[1].opacity = 0.5  # the paint layer
        compositor._flush()
    check(c.n == 0, f"paint layer opacity: no upload ({c.n})")

    # Strokes on top of a pending GPU composite
    with override(obj):
        ts.active_layer_index = 1  # paint layer
    for live_mode in (True, False):
        compositor.BACKEND_OVERRIDE = "GPU"
        compositor.DEFER_OVERRIDE = True
        compositor.composite(mat)
        pending = set(compositor.state(mat).gpu_results)
        check(pending == {c.id for c in compositor.enabled_channels(ts)}, "composite left GPU results pending")
        s = stroke(obj, mat, live_mode)
        check(bool(s.sinks) == live_mode and (live_mode or bool(s.outputs)),
              f"stroke took the pending outputs (live={live_mode}, sinks={len(s.sinks)})")
        if live_mode and "ROUGHNESS" in s.sinks:
            # The viewport texture mid-stroke: level 0 and a mip level built on the GPU.
            gc = gpu_compositor.instance()
            sink = gc.read(s.sinks["ROUGHNESS"])[..., 0]
            img = ts.output("ROUGHNESS")
            lvl0 = live.read_level(img, 0)[..., 0]
            lvl2 = live.read_level(img, 2)[..., 0]
            box = sink.reshape(sink.shape[0] // 4, 4, sink.shape[1] // 4, 4).mean((1, 3))
            check(np.abs(sink - pixels.read(img)[..., 0]).max() > 0.05, "stroke changed the sink")
            check(np.abs(lvl0 - sink).max() <= TOL, f"viewport level 0 == stroke result "
                                                    f"({np.abs(lvl0 - sink).max():.4f})")
            check(np.abs(lvl2 - box).max() <= 2 * TOL, f"viewport mip 2 == 4x4 average "
                                                       f"({np.abs(lvl2 - box).max():.4f})")
        s.finish()
        if live_mode:
            check(set(compositor.state(mat).gpu_results) >= {c.id for c in s.channels},
                  "live stroke leaves its outputs pending on the GPU")
        compositor.sync_outputs(mat)
        got = outputs(ts)
        compositor.DEFER_OVERRIDE = None
        ref = reference(mat)
        check(same(got, ref), f"stroke result == CPU composite (live={live_mode}, max diff "
                              f"{max(float(np.abs(got[c] - ref[c]).max()) for c in got):.4f})")

    compositor.BACKEND_OVERRIDE = compositor.DEFER_OVERRIDE = None
    print("FAILED" if check.failed else "ALL PASSED")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
