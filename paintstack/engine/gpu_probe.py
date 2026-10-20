"""Low-level GPU probe: checks the individual GPU operations PaintStack relies on
(clears, push constants, uniform buffers, gl_FragCoord and texture orientation,
float textures, depth test, blending, scissor, readback) and reports the raw
values, so a backend difference (e.g. Metal vs OpenGL) shows up as one line."""

import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

N = 8  # probe target size (N x N)
QUAD = ((-1, -1), (1, -1), (1, 1), (-1, 1))


def _shader(frag, constants=(), samplers=(), ubo=None, varying=False):
    info = gpu.types.GPUShaderCreateInfo()
    if ubo:
        info.typedef_source(ubo)
        info.uniform_buf(0, "Probe", "pb")
    for kind, ident in constants:
        info.push_constant(kind, ident)
    for slot, ident in enumerate(samplers):
        info.sampler(slot, "FLOAT_2D", ident)
    info.vertex_in(0, "VEC2", "pos")
    if varying:
        iface = gpu.types.GPUStageInterfaceInfo("ps_probe_iface")
        iface.smooth("VEC2", "uvInterp")
        info.vertex_out(iface)
        info.vertex_source("void main()\n{\n  uvInterp = pos * 0.5 + 0.5;\n"
                           "  gl_Position = vec4(pos, 0.0, 1.0);\n}\n")
    else:
        info.vertex_source("void main()\n{\n  gl_Position = vec4(pos, 0.0, 1.0);\n}\n")
    info.fragment_out(0, "VEC4", "fragColor")
    info.fragment_source(frag)
    return gpu.shader.create_from_info(info)


def _read(tex):
    return np.array(tex.read().to_list(), np.float32).reshape(tex.height, tex.width, -1)


def _draw(off, sh, setup=None, quad=QUAD, clear=(0.0, 0.0, 0.0, 0.0)):
    batch = batch_for_shader(sh, "TRI_FAN", {"pos": quad})
    with off.bind():
        gpu.state.active_framebuffer_get().clear(color=clear)
        gpu.state.blend_set("NONE")
        sh.bind()
        if setup:
            setup(sh)
        batch.draw(sh)
    return _read(off.texture_color)


def _fmt(v):
    return "(" + ", ".join(f"{x:.3g}" for x in np.ravel(v)) + ")"


def _orientation(out, src):
    """How an (N, N) result relates to its source: identity, flipped, or other."""
    if np.allclose(out, src, atol=1e-3):
        return "identity"
    if np.allclose(out, src[::-1], atol=1e-3):
        return "FLIPPED-Y"
    if np.allclose(out, src[:, ::-1], atol=1e-3):
        return "FLIPPED-X"
    return f"OTHER row0 {_fmt(out[0, :4])} rowN {_fmt(out[-1, :4])}"


def run():
    """Returns a list of (name, ok, detail). ok is the result expected on OpenGL."""
    results = []

    def probe(name, fn):
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001  (report anything)
            ok, detail = False, f"ERROR {type(e).__name__}: {str(e)[:160]}"
        results.append((name, ok, detail))

    off = gpu.types.GPUOffScreen(N, N, format="RGBA32F")
    src = (np.arange(N)[:, None] * 10 + np.arange(N)[None, :]).astype(np.float32)  # row*10 + col
    rgba = np.zeros((N, N, 4), np.float32)
    rgba[..., 0] = src
    rgba[..., 3] = 1.0
    try:
        def clear():
            with off.bind():
                fb = gpu.state.active_framebuffer_get()
                fb.clear(color=(0.25, 0.5, 0.75, 1.0))
                rc = np.array(fb.read_color(0, 0, N, N, 4, 0, "FLOAT").to_list(), np.float32).reshape(N, N, 4)
            tr = _read(off.texture_color)
            ok = np.allclose(tr[0, 0], (0.25, 0.5, 0.75, 1)) and np.allclose(rc[0, 0], (0.25, 0.5, 0.75, 1))
            return ok, f"texture.read {_fmt(tr[0, 0])}, read_color {_fmt(rc[0, 0])}"
        probe("offscreen clear + readback", clear)

        def push_const():
            sh = _shader("void main()\n{\n  fragColor = color;\n}\n", [("VEC4", "color")])
            out = _draw(off, sh, lambda s: s.uniform_float("color", (0.1, 0.2, 0.3, 0.4)))
            return np.allclose(out[N // 2, N // 2], (0.1, 0.2, 0.3, 0.4)), _fmt(out[N // 2, N // 2])
        probe("draw with a VEC4 push constant", push_const)

        def push_int():
            sh = _shader("void main()\n{\n  fragColor = vec4(float(count), 0.0, 0.0, 1.0);\n}\n",
                         [("INT", "count")])
            out = _draw(off, sh, lambda s: s.uniform_int("count", 7))
            return abs(out[1, 1, 0] - 7) < 1e-3, _fmt(out[1, 1, 0])
        probe("draw with an INT push constant", push_int)

        def two_consts():
            sh = _shader("void main()\n{\n  fragColor = vec4(a.x, b.y, float(n), 1.0);\n}\n",
                         [("VEC4", "a"), ("VEC4", "b"), ("INT", "n")])

            def setup(s):
                s.uniform_float("a", (0.5, 0, 0, 0))
                s.uniform_float("b", (0, 0.25, 0, 0))
                s.uniform_int("n", 3)
            out = _draw(off, sh, setup)
            return np.allclose(out[1, 1, :3], (0.5, 0.25, 3)), _fmt(out[1, 1, :3])
        probe("draw with several push constants", two_consts)

        def fragcoord():
            sh = _shader("void main()\n{\n  fragColor = vec4(gl_FragCoord.xy, 0.0, 1.0);\n}\n")
            out = _draw(off, sh)
            ok = np.allclose(out[0, 0, :2], (0.5, 0.5)) and np.allclose(out[N - 1, 0, :2], (0.5, N - 0.5))
            return ok, f"read row 0 col 0 = {_fmt(out[0, 0, :2])}, row {N - 1} col 0 = {_fmt(out[N - 1, 0, :2])}"
        probe("gl_FragCoord matches readback rows", fragcoord)

        def ndc():
            sh = _shader("void main()\n{\n  fragColor = vec4(1.0);\n}\n")
            out = _draw(off, sh, quad=((-1, -1), (1, -1), (1, 0), (-1, 0)))  # bottom half (NDC y < 0)
            rows = np.nonzero(out[:, 0, 0] > 0.5)[0]
            return rows.tolist() == list(range(N // 2)), f"rows set: {rows.tolist()}"
        probe("NDC y<0 lands in readback rows 0..N/2", ndc)

        def upload(fmt, value_scale=1.0):
            def fn():
                tex = gpu.types.GPUTexture((N, N), format=fmt,
                                           data=gpu.types.Buffer("FLOAT", rgba.size, (rgba * value_scale).ravel()))
                back = _read(tex)[..., 0] / value_scale
                return np.allclose(back, src, atol=0.05), "texture.read: " + _orientation(back, src)
            return fn
        probe("texture upload -> read (RGBA32F)", upload("RGBA32F"))
        probe("texture upload -> read (RGBA16F)", upload("RGBA16F"))

        tex = gpu.types.GPUTexture((N, N), format="RGBA32F", data=gpu.types.Buffer("FLOAT", rgba.size, rgba.ravel()))

        def numpy_view():
            from . import projector
            raw = tex.read()
            try:
                plain = np.asarray(raw, np.float32).size
            except Exception:  # noqa: BLE001
                plain = "error"
            arr = projector.buffer_array(tex.read(), N * N * 4).reshape(N, N, 4)
            ok = np.allclose(arr, rgba)
            return ok, (f"plain np.asarray sees {plain} of {N * N * 4} values; "
                        f"PaintStack reads via {projector._buffer_path}")
        probe("readback buffer -> numpy", numpy_view)

        def fetch():
            sh = _shader("void main()\n{\n  fragColor = texelFetch(src, ivec2(gl_FragCoord.xy), 0);\n}\n",
                         samplers=["src"])
            out = _draw(off, sh, lambda s: s.uniform_sampler("src", tex))[..., 0]
            return np.allclose(out, src), "texelFetch(gl_FragCoord): " + _orientation(out, src)
        probe("texelFetch at gl_FragCoord copies a texture", fetch)

        def sample_uv():
            sh = _shader("void main()\n{\n  fragColor = texelFetch(src, ivec2(uvInterp * vec2(textureSize(src, 0))), 0);\n}\n",
                         samplers=["src"], varying=True)
            out = _draw(off, sh, lambda s: s.uniform_sampler("src", tex))[..., 0]
            return np.allclose(out, src), "texelFetch(varying uv): " + _orientation(out, src)
        probe("texelFetch at a varying UV copies a texture", sample_uv)

        def r32f():
            vals = np.array([[0.0, 1.0, 200.0, 255.0]], np.float32)
            t = gpu.types.GPUTexture((4, 1), format="R32F", data=gpu.types.Buffer("FLOAT", 4, vals.ravel()))
            sh = _shader("void main()\n{\n  fragColor = vec4(texelFetch(src, ivec2(2, 0), 0).r,"
                         " texelFetch(src, ivec2(3, 0), 0).r, 0.0, 1.0);\n}\n", samplers=["src"])
            out = _draw(off, sh, lambda s: s.uniform_sampler("src", t))
            return np.allclose(out[1, 1, :2], (200, 255)), _fmt(out[1, 1, :2])
        probe("R32F texture holds values > 1", r32f)

        ubo_src = """
struct Probe {
  mat4 m;
  vec4 a;
  vec4 b;
};
"""

        def ubo():
            sh = _shader("void main()\n{\n  fragColor = vec4(pb.m[0][0], pb.m[3][1], pb.a.y, pb.b.w);\n}\n",
                         ubo=ubo_src)
            m = np.arange(16, dtype=np.float32).reshape(4, 4)  # m[col][row] in GLSL = data[col*4 + row]
            data = np.concatenate([m.ravel(), (1, 2, 3, 4), (5, 6, 7, 8)]).astype(np.float32)
            buf = gpu.types.GPUUniformBuf(data.tobytes())
            out = _draw(off, sh, lambda s: s.uniform_block("pb", buf))
            want = (0, 13, 2, 8)
            return np.allclose(out[1, 1], want), f"{_fmt(out[1, 1])} want {_fmt(want)}"
        probe("uniform buffer (bytes) read in the shader", ubo)

        def ubo_buffer():
            sh = _shader("void main()\n{\n  fragColor = vec4(pb.m[0][0], pb.m[3][1], pb.a.y, pb.b.w);\n}\n",
                         ubo=ubo_src)
            m = np.arange(16, dtype=np.float32).reshape(4, 4)
            data = np.concatenate([m.ravel(), (1, 2, 3, 4), (5, 6, 7, 8)]).astype(np.float32)
            buf = gpu.types.GPUUniformBuf(gpu.types.Buffer("FLOAT", data.size, data))
            out = _draw(off, sh, lambda s: s.uniform_block("pb", buf))
            want = (0, 13, 2, 8)
            return np.allclose(out[1, 1], want), f"{_fmt(out[1, 1])} want {_fmt(want)}"
        probe("uniform buffer (gpu.Buffer) read in the shader", ubo_buffer)

        def depth():
            info = gpu.types.GPUShaderCreateInfo()
            info.push_constant("VEC4", "color")
            info.push_constant("FLOAT", "z")
            info.vertex_in(0, "VEC2", "pos")
            info.fragment_out(0, "VEC4", "fragColor")
            info.vertex_source("void main()\n{\n  gl_Position = vec4(pos, z, 1.0);\n}\n")
            info.fragment_source("void main()\n{\n  fragColor = color;\n}\n")
            sh = gpu.shader.create_from_info(info)
            batch = batch_for_shader(sh, "TRI_FAN", {"pos": QUAD})
            with off.bind():
                gpu.state.active_framebuffer_get().clear(color=(0, 0, 0, 0), depth=1.0)
                gpu.state.depth_test_set("LESS_EQUAL")
                gpu.state.depth_mask_set(True)
                sh.bind()
                for color, z in (((1, 0, 0, 1), -0.5), ((0, 1, 0, 1), 0.5)):  # near red, then far green
                    sh.uniform_float("color", color)
                    sh.uniform_float("z", z)
                    batch.draw(sh)
                gpu.state.depth_test_set("NONE")
                gpu.state.depth_mask_set(False)
            out = _read(off.texture_color)[1, 1]
            return np.allclose(out, (1, 0, 0, 1)), f"{_fmt(out)} (want red: near quad wins)"
        probe("depth test in an offscreen", depth)

        def blend():
            mask = gpu.types.GPUOffScreen(N, N, format="RGBA16F")
            try:
                sh = _shader("void main()\n{\n  fragColor = color;\n}\n", [("VEC4", "color")])
                batch = batch_for_shader(sh, "TRI_FAN", {"pos": QUAD})
                with mask.bind():
                    gpu.state.active_framebuffer_get().clear(color=(0, 0, 0, 0))
                    gpu.state.blend_set("ALPHA")
                    sh.bind()
                    sh.uniform_float("color", (1, 1, 1, 0.5))
                    batch.draw(sh)
                    batch.draw(sh)
                    gpu.state.blend_set("NONE")
                out = _read(mask.texture_color)[1, 1]
            finally:
                mask.free()
            return abs(out[3] - 0.75) < 0.01, f"{_fmt(out)} (want alpha 0.75)"
        probe("ALPHA blending into RGBA16F", blend)

        def scissor():
            sh = _shader("void main()\n{\n  fragColor = vec4(1.0);\n}\n")
            batch = batch_for_shader(sh, "TRI_FAN", {"pos": QUAD})
            with off.bind():
                gpu.state.active_framebuffer_get().clear(color=(0, 0, 0, 0))
                gpu.state.scissor_test_set(True)
                gpu.state.scissor_set(0, 0, 2, 3)
                sh.bind()
                batch.draw(sh)
                gpu.state.scissor_test_set(False)
            out = _read(off.texture_color)[..., 0] > 0.5
            ys, xs = np.nonzero(out)
            box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else None
            return box == (0, 0, 2, 3), f"lit box (x0, y0, x1, y1) = {box}"
        probe("scissor rectangle", scissor)

        def fb_texture():
            t = gpu.types.GPUTexture((N, N), format="RGBA16F")
            fb = gpu.types.GPUFrameBuffer(color_slots=t)
            sh = _shader("void main()\n{\n  fragColor = vec4(0.3, 0.6, 0.9, 1.0);\n}\n")
            batch = batch_for_shader(sh, "TRI_FAN", {"pos": QUAD})
            with fb.bind():
                gpu.state.viewport_set(0, 0, N, N)
                sh.bind()
                batch.draw(sh)
            out = _read(t)[1, 1]
            return np.allclose(out, (0.3, 0.6, 0.9, 1), atol=2e-3), _fmt(out)
        probe("draw into a GPUFrameBuffer(texture)", fb_texture)
    finally:
        off.free()
    return results


# What the GPU compositor relies on (its shaders copy and blend with texelFetch at
# gl_FragCoord into GPUFrameBuffer textures, with push constants).
COMPOSITOR_NEEDS = ("offscreen clear + readback", "draw with a VEC4 push constant",
                    "draw with several push constants", "texelFetch at gl_FragCoord copies a texture",
                    "draw into a GPUFrameBuffer(texture)")
_verdicts = {}


def supports(needs):
    """True if every named probe gives the expected result here (probed once per session)."""
    if "all" not in _verdicts:
        try:
            _verdicts["all"] = {name: ok for name, ok, _ in run()}
        except Exception:  # noqa: BLE001  (no usable GPU)
            _verdicts["all"] = {}
    got = _verdicts["all"]
    return all(got.get(n, False) for n in needs)
