"""Phase 6 headless test: Apple Silicon / platform features — threaded compositing,
GPU live stroke preview (incl. mip levels), Retina brush scaling, Metal-safe
formats and the diagnostics self-test.

    python tests/test_phase6.py
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
from paintstack import compositor, diagnostics, hardware, pixels  # noqa: E402
from paintstack.engine import live  # noqa: E402
from paintstack.engine.projector import Projector  # noqa: E402
from paintstack.stroke import Stroke  # noqa: E402
from test_headless import W, H, check, front_view  # noqa: E402
from test_phase3 import override  # noqa: E402


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    paintstack.register()

    # --- Hardware detection
    check(hardware.performance_cores() >= 1 and hardware.threads() >= 1, "core count detected")
    b = hardware.auto_cache_budget_mb()
    check(512 <= b <= 6144, f"auto cache budget scales with memory ({b} MB)")
    seen = np.zeros(1000, int)

    def mark(y0, y1):
        seen[y0:y1] += 1
    hardware.parallel_rows(mark, 1000, hardware.PARALLEL_MIN * 2)
    check((seen == 1).all(), "parallel row chunks cover every row exactly once")

    # --- Threaded blending is bit-identical to single-threaded
    rng = np.random.default_rng(0)
    base = rng.random((1024, 1024, 3)).astype(np.float32)
    src = rng.random((1024, 1024, 3)).astype(np.float32)
    alpha = rng.random((1024, 1024)).astype(np.float32)
    old = hardware.PARALLEL_MIN
    for mode in ("NORMAL", "MULTIPLY", "OVERLAY", "NORMAL_COMBINE"):
        hardware.PARALLEL_MIN = 1 << 62
        single = compositor.mix(base.copy(), (src, alpha, mode))
        hardware.PARALLEL_MIN = 1
        threaded = compositor.mix(base.copy(), (src, alpha, mode))
        check(np.array_equal(single, threaded), f"threaded {mode} blend == single-threaded")
    hardware.PARALLEL_MIN = old

    # --- Scene
    bpy.ops.mesh.primitive_plane_add(size=2, rotation=(math.pi / 2, 0, 0))
    obj = bpy.context.active_object
    with override(obj):
        bpy.ops.paintstack.init()
    mat = obj.active_material
    ts = mat.paintstack
    ts.resolution = "256"
    compositor.composite(mat)
    proj = Projector(obj, bpy.context.evaluated_depsgraph_get(), mat, ts.res)
    br = bpy.context.scene.paintstack_brush
    br.radius, br.flow, br.hardness, br.opacity = 20, 1.0, 1.0, 1.0

    # --- Retina: brush sizes are UI pixels, doubled in physical pixels
    areas = []
    for scale in (1.0, 2.0):
        s = Stroke(proj, front_view(), mat, br, seed=0, px_scale=scale, live=False)
        s.add_point(W / 2, H / 2)
        areas.append((proj.read_mask() > 0.5).sum())
        s.finish()
        from paintstack import undo
        undo.undo()
    ratio = areas[1] / areas[0]
    check(abs(ratio - 4.0) < 0.3, f"px_scale 2 doubles the brush radius (area x{ratio:.2f})")

    # --- Live preview: GPU textures update during the stroke, CPU images at the end
    img = ts.output("BASE_COLOR")
    cpu_before = pixels.read(img).copy()
    br.base_color = (0.0, 0.0, 1.0)
    s = Stroke(proj, front_view(), mat, br, seed=0, live=True)
    s.add_point(W / 2 - 30, H / 2)
    s.add_point(W / 2 + 30, H / 2)
    s.update()
    check(np.array_equal(pixels.read(img), cpu_before), "live preview leaves the CPU image alone mid-stroke")
    gpu_now = live.read_level(img, 0)
    painted = s.outputs["BASE_COLOR"]
    hit = painted[..., 2] > 0.9
    want = live.srgb_to_linear(np.float32(1.0))  # sRGB texture: sampling returns linear
    check(hit.sum() > 50 and np.allclose(gpu_now[hit][:, 2], want, atol=0.02)
          and np.allclose(gpu_now[hit][:, 0], 0.0, atol=0.02), "GPU texture shows the stroke mid-stroke")
    mip = live.read_level(img, 2)
    check(mip[..., 2].max() > mip[..., 0].max() + 0.2, "mip levels updated too (zoomed-out view)")
    s.finish()
    after_live = pixels.read(img).copy()
    check(not np.array_equal(after_live, cpu_before), "CPU image written when the stroke ends")
    from paintstack import undo
    undo.undo()
    s = Stroke(proj, front_view(), mat, br, seed=0, live=False)
    s.add_point(W / 2 - 30, H / 2)
    s.add_point(W / 2 + 30, H / 2)
    s.finish()
    check(np.array_equal(pixels.read(img), after_live), "live and CPU preview give identical results")

    # --- Metal-safe formats
    check(proj.mask.texture_color.format == "RGBA16F", "stroke mask is half float (blendable on Metal)")

    # --- Diagnostics self-test
    report = diagnostics.run()
    check(report.failures == 0, f"diagnostics: {report.failures} failure(s)")
    with bpy.context.temp_override():
        r = bpy.ops.paintstack.diagnostics()
    check(r == {"FINISHED"} and "PaintStack Diagnostics" in bpy.data.texts, "diagnostics operator writes a report")

    # --- macOS key hints
    from paintstack import paint_mode
    real = hardware.is_mac
    hardware.is_mac = lambda: True
    check("⌘Z" in paint_mode.status_text(), "status bar shows Cmd keys on macOS")
    hardware.is_mac = real

    print("FAILED" if check.failed else "ALL PASSED")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
