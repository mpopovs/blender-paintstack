"""Performance benchmark: composite, single-layer tweak and stroke timings on a
realistic stack (smart-material folder with generator masks, grunge fill,
painted layer).

    python tests/bench.py 1024 2048
"""

import json
import math
import os
import sys
import time

import bpy
import gpu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
gpu.init()

import paintstack  # noqa: E402
from paintstack import compositor, smart  # noqa: E402
from paintstack.engine.projector import Projector  # noqa: E402
from paintstack.stroke import Stroke  # noqa: E402
from test_headless import W, H, front_view  # noqa: E402
from test_phase3 import override  # noqa: E402


def timed(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best * 1000


def scene(res):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    paintstack.register()
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32)
    obj = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    with override(obj):
        bpy.ops.paintstack.init()
    mat = obj.active_material
    ts = mat.paintstack
    ts.resolution = "512"
    s = ts.bake
    s.samples = 2
    s.use_normal = s.use_id = s.use_thickness = False
    with override(obj):
        bpy.ops.paintstack.bake()
    ts.resolution = str(res)
    with override(obj):
        bpy.ops.paintstack.smart_apply(kind="SMART_MATERIAL", name="Painted Steel")
        ts.active_layer_index = 0
        bpy.ops.paintstack.layer_add(type="FILL")
        bpy.ops.paintstack.effect_add(type="GRUNGE")
        bpy.ops.paintstack.layer_add(type="PAINT")
    # A root-level fill above the folder, the layer whose sliders get dragged.
    ts.layers[0].expanded = False
    ts.active_layer_index = 0
    with override(obj):
        bpy.ops.paintstack.layer_add(type="FILL")
        bpy.ops.paintstack.effect_add(type="DIRT")
    ts.active_layer.name = "Tint"
    return obj, mat


def stroke_bench(obj, mat, n_points=40, update_every=4):
    ts = mat.paintstack
    b = bpy.context.scene.paintstack_brush
    b.radius, b.flow, b.spacing = 25, 0.8, 0.2
    b.use_height = True
    proj = Projector(obj, bpy.context.evaluated_depsgraph_get(), mat, ts.res)
    t0 = time.perf_counter()
    s = Stroke(proj, front_view(), mat, b, seed=0)
    start = (time.perf_counter() - t0) * 1000
    updates = []
    for i in range(n_points):
        a = i / n_points * 2 * math.pi
        s.add_point(W / 2 + 70 * math.cos(a), H / 2 + 70 * math.sin(a))
        if i % update_every == update_every - 1:
            t = time.perf_counter()
            s.update()
            updates.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    s.finish()
    finish = (time.perf_counter() - t) * 1000
    return start, sum(updates) / len(updates), finish


def main():
    sizes = [int(a) for a in sys.argv[1:] if a.isdigit()] or [1024, 2048]
    rows = []
    for res in sizes:
        obj, mat = scene(res)
        ts = mat.paintstack
        compositor.composite(mat)  # warm generator caches
        full = timed(lambda: compositor.composite(mat))
        top = ts.layers[0]

        def tweak():
            # A slider drag: the property update invalidates the layer, then the
            # coalesced composite runs (through the same code path as the UI).
            top.opacity = 0.5 if top.opacity == 1.0 else 1.0
            compositor._flush()

        def tweak_color():
            lc = top.channel("BASE_COLOR")
            lc.color = (0.2, 0.3, 0.4) if lc.color[0] > 0.5 else (0.8, 0.3, 0.4)
            compositor._flush()
        tweak()
        tw = timed(tweak)
        tweak_color()
        twc = timed(tweak_color)
        start, upd, fin = stroke_bench(obj, mat)
        row = {"res": res, "composite_ms": round(full), "tweak_top_layer_ms": round(tw),
               "tweak_color_ms": round(twc),
               "stroke_start_ms": round(start), "stroke_update_ms": round(upd), "stroke_finish_ms": round(fin)}
        rows.append(row)
        print(json.dumps(row))
        paintstack.unregister()
    return rows


if __name__ == "__main__":
    main()
