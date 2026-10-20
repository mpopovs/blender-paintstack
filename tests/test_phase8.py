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


def main():
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
