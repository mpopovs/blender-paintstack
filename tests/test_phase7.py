"""Phase 7 test: the GPU compositor must match the CPU compositor on every kind
of stack, and its deferred viewport display must reach the CPU images.

    python tests/test_phase7.py
"""

import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _env import fresh_scene  # noqa: E402  (also sets up the GPU)

from paintstack import compositor, grunge, pixels  # noqa: E402
from paintstack.channels import enabled_channels  # noqa: E402
from paintstack.engine import gpu_compositor, live  # noqa: E402
from paintstack.engine.projector import Projector  # noqa: E402
from paintstack.stroke import Stroke  # noqa: E402
from test_headless import W, H, check, front_view  # noqa: E402
from test_phase3 import override  # noqa: E402

TOL = 1.0 / 255 + 2e-3  # 8-bit outputs, half-float GPU accumulation


def outputs(ts):
    return {c.id: pixels.read(ts.output(c.id))[..., :3].copy() for c in enabled_channels(ts)}


def compare(mat, label):
    """Composite on the CPU and on the GPU (written straight to the images)."""
    ts = mat.paintstack
    compositor.BACKEND_OVERRIDE = "CPU"
    compositor.composite(mat)
    cpu = outputs(ts)
    compositor.BACKEND_OVERRIDE = "GPU"
    compositor.DEFER_OVERRIDE = False
    compositor.composite(mat)
    gpu = outputs(ts)
    worst = max(float(np.abs(cpu[c] - gpu[c]).max()) for c in cpu)
    check(worst <= TOL, f"{label}: GPU == CPU (max diff {worst:.4f})")
    return worst


def main():
    fresh_scene()
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24)
    obj = bpy.context.active_object
    with override(obj):
        bpy.ops.paintstack.init()
    mat = obj.active_material
    ts = mat.paintstack
    ts.resolution = "256"
    ts.use_normal = True
    ts.use_emission = True
    s = ts.bake
    s.samples = 2
    s.use_normal = s.use_id = s.use_thickness = False
    with override(obj):
        bpy.ops.paintstack.bake()

    compare(mat, "default stack")

    # Every blend mode, with texture sources and masks
    with override(obj):
        bpy.ops.paintstack.layer_add(type="FILL")
        bpy.ops.paintstack.effect_add(type="GRUNGE")
    fill = ts.active_layer
    fill.channel("BASE_COLOR").texture = grunge.make_image("CELLS", 0)
    fill.opacity = 0.8
    for mode in ("NORMAL", "MULTIPLY", "SCREEN", "OVERLAY", "SOFT_LIGHT", "ADD", "SUBTRACT",
                 "DARKEN", "LIGHTEN", "DIFFERENCE"):
        for lc in fill.channels:
            if lc.channel != "NORMAL":
                lc.blend_mode = mode
        compare(mat, f"blend mode {mode}")
    # A real (unit-length, z-dominant) normal map; a grey image would encode
    # degenerate directions that flip sign at 0.5.
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32) / 64
    n = np.stack([0.3 * np.sin(xx * 2 * np.pi), 0.3 * np.cos(yy * 2 * np.pi), np.ones_like(xx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    nimg = bpy.data.images.new("bumps", 64, 64, alpha=True, float_buffer=True)
    nimg.colorspace_settings.name = "Non-Color"
    nimg.pixels.foreach_set(np.concatenate([n * 0.5 + 0.5, np.ones((64, 64, 1), np.float32)], -1).ravel())
    fill.channel("NORMAL").texture = nimg
    fill.channel("NORMAL").blend_mode = "NORMAL_COMBINE"
    compare(mat, "normal map combine over the baked normal")

    # Smart material folder (generators, nested masks) + opaque and faded folders
    with override(obj):
        bpy.ops.paintstack.smart_apply(kind="SMART_MATERIAL", name="Painted Steel")
    folder_uid = ts.active_layer.uid
    compare(mat, "smart material folder (in place)")
    next(l for l in ts.layers if l.uid == folder_uid).opacity = 0.6
    compare(mat, "faded folder")
    with override(obj):
        bpy.ops.paintstack.mask_add(white=False)
        bpy.ops.paintstack.effect_add(type="EDGE_WEAR")
    compare(mat, "folder with a mask")

    # Painted layer content
    proj = Projector(obj, bpy.context.evaluated_depsgraph_get(), mat, ts.res)
    br = bpy.context.scene.paintstack_brush
    br.radius, br.flow, br.hardness = 30, 0.7, 0.5
    ts.active_layer_index = 0
    with override(obj):
        bpy.ops.paintstack.layer_add(type="PAINT")
    st = Stroke(proj, front_view(), mat, br, seed=0, live=False)
    for i in range(12):
        st.add_point(W / 2 + 70 * math.cos(i / 2), H / 2 + 50 * math.sin(i / 2))
    st.finish()
    compare(mat, "painted layer")

    # Filters
    ts.active_layer_index = 0
    with override(obj):
        bpy.ops.paintstack.layer_add(type="FILTER")
    filt = ts.active_layer
    for kind, setup in (("BLUR", dict(filter_radius=5)),
                        ("SHARPEN", dict(filter_radius=3, filter_amount=1.5)),
                        ("HSL", dict(hue=0.3, saturation=-0.4, lightness=0.2)),
                        ("WARP", dict(filter_amount=1.0, warp_scale=6.0))):
        filt.filter_type = kind
        for k, v in setup.items():
            setattr(filt, k, v)
        compare(mat, f"filter {kind}")
    filt.opacity = 0.5
    compare(mat, "filter at half opacity")
    with override(obj):
        bpy.ops.paintstack.layer_remove()

    # GPU checkpoint: repeated tweaks of one root layer reuse the stored base
    for layer in ts.layers:  # so the probe lands at the root, above everything
        if layer.type == "FOLDER":
            layer.expanded = False
    ts.active_layer_index = 0
    with override(obj):
        bpy.ops.paintstack.layer_add(type="FILL")
    top = ts.active_layer
    top.name = "Probe"
    for lc in top.channels:
        lc.enabled = lc.channel in ("ROUGHNESS", "BASE_COLOR")
    compositor.BACKEND_OVERRIDE = "GPU"
    compositor.DEFER_OVERRIDE = False
    uid = top.uid
    for v in (0.3, 0.6, 0.9):
        top.opacity = v
        compositor._flush()
    stt = compositor.state(mat)
    check(stt.gpu_checkpoint is not None and stt.gpu_checkpoint[0] == uid, "GPU checkpoint kept for the edited layer")
    gpu_out = outputs(ts)
    compositor.BACKEND_OVERRIDE = "CPU"
    compositor.composite(mat)
    cpu_out = outputs(ts)
    worst = max(float(np.abs(cpu_out[c] - gpu_out[c]).max()) for c in cpu_out)
    check(worst <= TOL, f"checkpointed GPU tweaks == CPU (max diff {worst:.4f})")

    # Deferred display: the viewport texture updates now, the CPU image on sync
    compositor.BACKEND_OVERRIDE = "GPU"
    compositor.DEFER_OVERRIDE = True
    img = ts.output("ROUGHNESS")
    before = pixels.read(img).copy()
    top.channel("ROUGHNESS").value = 0.05 if top.channel("ROUGHNESS").value > 0.5 else 0.95
    top.opacity = 1.0
    compositor._flush()
    check(np.array_equal(pixels.read(img), before), "deferred: CPU image untouched right after the composite")
    shown = live.read_level(img, 0)[..., 0]
    check(stt.gpu_results and abs(float(shown.mean()) - float(before[..., 0].mean())) > 0.02,
          "deferred: the viewport texture already shows the change")
    compositor.sync_outputs(mat)
    synced = pixels.read(img)[..., 0]
    check(not stt.gpu_results and np.allclose(synced, shown, atol=TOL), "sync writes the shown result to the CPU image")
    compositor.DEFER_OVERRIDE = False
    compositor.BACKEND_OVERRIDE = "CPU"
    compositor.composite(mat)
    check(np.allclose(pixels.read(img)[..., 0], synced, atol=TOL), "synced image == CPU composite")

    # A stroke after a deferred composite starts from up-to-date CPU images
    compositor.BACKEND_OVERRIDE = "GPU"
    compositor.DEFER_OVERRIDE = True
    top.channel("ROUGHNESS").value = 0.5
    compositor._flush()
    check(bool(stt.gpu_results), "deferred results pending")
    ts.active_layer_index = next(i for i, l in enumerate(ts.layers) if l.type == "PAINT")
    st = Stroke(proj, front_view(), mat, br, seed=0, live=False)
    check(not stt.gpu_results, "stroke start syncs pending GPU results")
    st.finish()

    # Saving syncs too (save_pre handler)
    top.channel("ROUGHNESS").value = 0.7
    compositor._flush()
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "sync.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)
    check(not stt.gpu_results, "saving syncs pending GPU results")
    compositor.DEFER_OVERRIDE = None

    # Failure on the GPU falls back to the CPU compositor
    real = gpu_compositor.GPUCompositor.blend

    def broken(self, *a, **k):
        raise RuntimeError("simulated backend failure")
    gpu_compositor.GPUCompositor.blend = broken
    compositor.BACKEND_OVERRIDE = "GPU"
    compositor.FORCE_SYNC = True
    top.opacity = 0.4
    compositor._flush()
    gpu_compositor.GPUCompositor.blend = real
    check(compositor.BACKEND_OVERRIDE == "CPU", "GPU failure switches to the CPU compositor")
    got = outputs(ts)
    compositor.composite(mat)
    check(all(np.array_equal(got[c], v) for c, v in outputs(ts).items()), "fallback result is the CPU result")
    compositor.FORCE_SYNC = False
    compositor.BACKEND_OVERRIDE = None

    # Anchors keep stacks on the CPU
    top.is_anchor = True
    with override(obj):
        bpy.ops.paintstack.layer_add(type="FILL")
        bpy.ops.paintstack.effect_add(type="ANCHOR")
    ts.active_layer.active_mask_effect.anchor_uid = uid
    compositor.BACKEND_OVERRIDE = "GPU"
    check(not compositor.use_gpu(ts), "stacks with anchors use the CPU compositor")
    ts.layers.remove(ts.active_layer_index)
    top.is_anchor = False

    # A GPU result still waiting for its CPU sync when the resolution changes is
    # stale: syncing must drop it, not write it into the resized image.
    compositor.DEFER_OVERRIDE = True
    compositor.composite(mat)
    old = ts.res
    ts.resolution = str(old * 2)
    try:
        compositor.invalidate_all()
        ok = True
    except Exception as e:  # noqa: BLE001
        ok = False
        print(e)
    check(ok and tuple(ts.output("BASE_COLOR").size) == (old * 2, old * 2),
          "pending GPU result from before a resolution change is dropped")
    compositor.BACKEND_OVERRIDE = compositor.DEFER_OVERRIDE = None

    print("FAILED" if check.failed else "ALL PASSED")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
