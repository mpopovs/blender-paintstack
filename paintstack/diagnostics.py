"""Self-test for the GPU paths on the machine Blender runs on (e.g. Metal on Apple
Silicon). Every check compares the GPU result with a CPU reference, so a backend
difference shows up as a FAIL instead of a silently wrong stroke. Run it from the
add-on preferences or the PaintStack panel; the report is saved to the text
datablock "PaintStack Diagnostics"."""

import math
import time

import bpy
import numpy as np
from bpy.types import Operator
from mathutils import Matrix

from . import hardware

W = H = 400


def _view():
    """Perspective camera at (0, -5, 0) looking at the origin along +Y."""
    cam = Matrix.Translation((0, -5, 0)) @ Matrix.Rotation(math.pi / 2, 4, "X")
    view = cam.inverted()
    f, n, fa = 1 / math.tan(math.radians(25)), 0.1, 100.0
    proj = Matrix(((f, 0, 0, 0), (0, f, 0, 0),
                   (0, 0, (fa + n) / (n - fa), 2 * fa * n / (n - fa)), (0, 0, -1, 0)))
    return W, H, proj @ view, view, True


def _plane(segments=16):
    """A plane facing the camera (y = 0, x and z in [-1, 1]); UV = (x, z) remapped."""
    g = np.linspace(-1, 1, segments + 1, dtype=np.float32)
    pos, uv = [], []
    for i in range(segments):
        for j in range(segments):
            quad = [(g[i], g[j]), (g[i + 1], g[j]), (g[i + 1], g[j + 1]), (g[i], g[j + 1])]
            for a, b, c in ((0, 1, 2), (0, 2, 3)):
                for k in (a, b, c):
                    x, z = quad[k]
                    pos.append((x, 0.0, z))
                    uv.append(((x + 1) / 2, (z + 1) / 2))
    pos = np.array(pos, np.float32)
    # Winding gives normals towards -Y (the camera) for the facing test.
    pos = pos.reshape(-1, 3, 3)[:, ::-1].reshape(-1, 3)
    uv = np.array(uv, np.float32).reshape(-1, 3, 2)[:, ::-1].reshape(-1, 2)
    nor = np.tile(np.array([[0.0, -1.0, 0.0]], np.float32), (len(pos), 1))
    return pos, nor, uv


def _expected_dab(res, view, cx, cy, radius):
    """CPU reference: texels of the plane whose screen projection is inside the dab."""
    w, h, persp = view[0], view[1], np.array(view[2], np.float32)
    g = (np.arange(res, dtype=np.float32) + 0.5) / res
    u, v = np.meshgrid(g, g)
    world = np.stack([u * 2 - 1, np.zeros_like(u), v * 2 - 1, np.ones_like(u)], -1)
    clip = world @ persp.T
    sx = (clip[..., 0] / clip[..., 3] * 0.5 + 0.5) * w
    sy = (clip[..., 1] / clip[..., 3] * 0.5 + 0.5) * h
    return np.hypot(sx - cx, sy - cy) < radius


class Report:
    def __init__(self):
        self.lines, self.failures = [], 0

    def check(self, ok, label, detail=""):
        self.lines.append(f"{'PASS' if ok else 'FAIL'}  {label}{'  (' + detail + ')' if detail else ''}")
        self.failures += 0 if ok else 1

    def info(self, text):
        self.lines.append(f"      {text}")


def run():
    import gpu
    from . import generators
    from .engine import live, noise_gpu, shaders
    from .engine.projector import Projector

    r = Report()
    for k, v in hardware.summary().items():
        r.info(f"{k}: {v}")

    # Shaders compile on this backend
    for name, fn in (("depth", shaders.depth_shader), ("dab", shaders.dab_shader),
                     ("coverage", shaders.coverage_shader), ("noise", noise_gpu._get_shader),
                     ("live copy", live._copy_shader)):
        try:
            fn()
            r.check(True, f"shader '{name}' compiles")
        except Exception as e:
            r.check(False, f"shader '{name}' compiles", str(e)[:200])

    res = 512
    pos, nor, uv = _plane()
    proj = Projector.from_arrays(pos, nor, uv, res)
    view = _view()

    def dab(*dabs, tip=None, stencil=None):
        proj.begin_view(*view)
        proj.set_tip(tip)
        if stencil is not None:
            proj.set_stencil(*stencil)
        proj.clear_mask()
        proj.dabs(list(dabs))
        return proj.read_mask()

    # Projection, depth and facing: GPU dab == CPU reference
    m = dab((W / 2, H / 2, 40, 1.0, 1.0, 0.0)) > 0.5
    ref = _expected_dab(res, view, W / 2, H / 2, 40)
    iou = (m & ref).sum() / max((m | ref).sum(), 1)
    r.check(iou > 0.97, "dab projection matches the CPU reference", f"IoU {iou:.3f}")

    # Blending into the half-float stroke mask accumulates ("over")
    v = dab((W / 2, H / 2, 40, 1.0, 0.5, 0.0), (W / 2, H / 2, 40, 1.0, 0.5, 0.0))
    centre = float(v[ref].mean())
    r.check(abs(centre - 0.75) < 0.01, "stroke mask blending (0.5 over 0.5 = 0.75)", f"{centre:.3f}")

    # Filtered half-float tip texture
    from . import brushes

    class _B:
        tip, tip_seed, hardness, tip_image = "SQUARE", 0, 1.0, None
    sq = dab((W / 2, H / 2, 40, 1.0, 1.0, 0.0), tip=brushes.tip_array(_B())) > 0.5
    ratio = sq.sum() / max(m.sum(), 1)
    r.check(abs(ratio - 4 / math.pi) < 0.08, "brush tip texture sampling", f"square/round {ratio:.2f}")

    # Stencil (left half black)
    sten = np.zeros((32, 32), np.float32)
    sten[:, 16:] = 1.0
    st = dab((W / 2, H / 2, 40, 1.0, 1.0, 0.0), stencil=(sten, W / 2, H / 2, 100.0, 0.0)) > 0.5
    cols = np.nonzero(st.any(0))[0]
    r.check(st.any() and cols.min() >= res // 2 - 4, "stencil masks the brush")

    # Partial readback
    region = (200, 300, 180, 330)
    part = proj.read_mask(region)
    full = proj.read_mask()
    r.check(np.allclose(part, full[200:300, 180:330]), "partial GPU readback")

    # GPU noise == CPU noise
    pts = np.random.default_rng(0).random((256, 256, 3)).astype(np.float32)
    perm, vals = generators._table(1)
    t = time.perf_counter()
    gn = noise_gpu.fbm(pts, 8.0, 4, perm, vals)
    t_gpu = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    cn = generators.fbm(pts * 8.0, 4, 1)
    t_cpu = (time.perf_counter() - t) * 1000
    diff = float(np.abs(gn - cn).max())
    r.check(diff < 2e-3, "GPU noise matches CPU noise", f"max diff {diff:.1e}")
    r.info(f"noise 256x256: GPU {t_gpu:.0f} ms, CPU {t_cpu:.0f} ms")

    # Live preview writes into image textures (level 0 and mips)
    r.info(f"sRGB texture writes encode: {live.srgb_encodes()}")
    for float_buffer, srgb in ((False, True), (False, False), (True, False)):
        img = bpy.data.images.new("PS_diag_live", 64, 64, alpha=True, float_buffer=float_buffer)
        try:
            img.colorspace_settings.name = "sRGB" if srgb else "Non-Color"
            data = np.full((64, 64, 4), 0.2, np.float32)
            img.pixels.foreach_set(data.ravel())
            data[16:48, 16:48, :3] = 0.8
            live.push(img, data, (16, 48, 16, 48))
            got0 = live.read_level(img, 0)[32, 32, 0]
            got1 = live.read_level(img, 1)[16, 16, 0]
            want = live.srgb_to_linear(np.float32(0.8)) if srgb else 0.8
            ok = abs(got0 - want) < 0.02 and abs(got1 - want) < 0.02
            kind = "8-bit sRGB" if srgb else ("float" if float_buffer else "8-bit linear")
            r.check(ok, f"live preview writes ({kind}, level 0 + mip 1)",
                    f"{got0:.3f}, {got1:.3f}, want {want:.3f}")
        except Exception as e:
            r.check(False, "live preview writes", str(e)[:200])
        finally:
            bpy.data.images.remove(img)

    # Timings
    big = np.random.default_rng(0).random((2048, 2048, 3)).astype(np.float32)
    alpha = np.random.default_rng(1).random((2048, 2048)).astype(np.float32)
    from . import compositor
    old = hardware.PARALLEL_MIN
    try:
        hardware.PARALLEL_MIN = 1 << 62
        t = time.perf_counter()
        compositor.mix(big.copy(), (np.float32(0.5), alpha, "NORMAL"))
        single = (time.perf_counter() - t) * 1000
    finally:
        hardware.PARALLEL_MIN = old
    t = time.perf_counter()
    compositor.mix(big.copy(), (np.float32(0.5), alpha, "NORMAL"))
    threaded = (time.perf_counter() - t) * 1000
    r.info(f"2K blend: 1 thread {single:.0f} ms, {hardware.threads()} threads {threaded:.0f} ms")

    _compositor_checks(r)

    proj2 = Projector.from_arrays(pos, nor, uv, 2048)
    proj2.begin_view(*view)
    proj2.clear_mask()
    t = time.perf_counter()
    proj2.dabs([(150 + i * 2, 200, 20, 0.5, 0.5, 0.0) for i in range(50)])
    proj2.read_mask((800, 1300, 600, 1500))
    r.info(f"2K: 50 dabs + region readback {(time.perf_counter() - t) * 1000:.0f} ms")
    proj.free()
    proj2.free()

    r.lines.insert(0, f"PaintStack diagnostics: {r.failures} failure(s)")
    return r


def _compositor_checks(r, res=2048):
    """GPU vs CPU compositor on a temporary texture set (not attached to any object)."""
    from . import compositor, grunge, ops, pixels
    from .channels import enabled_channels
    mat = bpy.data.materials.new("PS_diagnostics")
    images = []
    try:
        ts = mat.paintstack
        ts.initialized = True
        ts.resolution = str(res)
        ops.new_layer(mat, "FILL", "Base")
        layer = ops.new_layer(mat, "FILL", "Rust")
        layer.channel("BASE_COLOR").color = (0.5, 0.2, 0.05)
        layer.channel("BASE_COLOR").texture = grunge.make_image("RUST", 0)
        ops.add_mask_effect(mat, layer, "GRUNGE")
        for mode, name in (("MULTIPLY", "Dirt"), ("OVERLAY", "Tint")):
            l = ops.new_layer(mat, "FILL", name)
            l.channel("BASE_COLOR").blend_mode = mode
            l.opacity = 0.6
            ops.add_mask_effect(mat, l, "NOISE")
        filt = ops.new_layer(mat, "FILTER", "Blur")
        filt.filter_type = "BLUR"
        filt.filter_radius = 2

        def run(backend):
            compositor.BACKEND_OVERRIDE = backend
            compositor.FORCE_SYNC = True
            compositor.invalidate(mat)
            compositor.composite(mat)  # warm caches and uploads
            t = time.perf_counter()
            compositor.composite(mat)
            ms = (time.perf_counter() - t) * 1000
            out = {c.id: pixels.read(ts.output(c.id))[..., :3].copy() for c in enabled_channels(ts)}
            return ms, out

        cpu_ms, cpu = run("CPU")
        try:
            gpu_ms, gpu_out = run("GPU")
        except Exception as e:
            r.check(False, "GPU compositor runs", str(e)[:200])
            return
        fell_back = compositor.BACKEND_OVERRIDE == "CPU"
        worst = max(float(np.abs(cpu[c] - gpu_out[c]).max()) for c in cpu)
        r.check(not fell_back and worst <= 1.0 / 255 + 2e-3, "GPU compositor matches the CPU compositor",
                f"max diff {worst:.4f}")
        r.info(f"{res}px composite, 5 layers + blur: CPU {cpu_ms:.0f} ms, GPU {gpu_ms:.0f} ms")
        r.info(f"Auto compositor here: {'GPU' if hardware.gpu_is_hardware() else 'CPU'}")
    finally:
        compositor.BACKEND_OVERRIDE = None
        compositor.FORCE_SYNC = False
        ts = mat.paintstack
        images = [o.image for o in ts.outputs if o.image]
        for l in ts.layers:
            images += [i for i in (l.mask_image,) if i] + [e.image for e in l.mask_effects if e.image]
        compositor.invalidate(mat)
        bpy.data.materials.remove(mat)
        for img in images:
            bpy.data.images.remove(img)


class PAINTSTACK_OT_diagnostics(Operator):
    """Test PaintStack's GPU paths on this machine (Metal on macOS) and measure speed.
    The report is saved to the text 'PaintStack Diagnostics'"""
    bl_idname = "paintstack.diagnostics"
    bl_label = "Run Diagnostics"

    def execute(self, context):
        report = run()
        text = bpy.data.texts.get("PaintStack Diagnostics") or bpy.data.texts.new("PaintStack Diagnostics")
        text.clear()
        text.write("\n".join(report.lines) + "\n")
        print("\n".join(report.lines))
        level = {"INFO"} if report.failures == 0 else {"WARNING"}
        self.report(level, f"{report.lines[0]} - see the text 'PaintStack Diagnostics'")
        return {"FINISHED"}


classes = (PAINTSTACK_OT_diagnostics,)
