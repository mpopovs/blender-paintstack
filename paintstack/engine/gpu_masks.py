"""Layer mask stacks on the GPU, for the GPU compositor.

GLSL port of masks.evaluate and generators.evaluate: the mask image, paint and
fill effects, generators (sources and 3D noise from the baked maps), colour
selection, levels and blur, with the same blend modes and clamping. Generator
sources and the noise field are uploaded / computed once and kept (keyed on the
cached CPU arrays they come from); Level, Softness, Noise amount, opacity and
blend modes are shader constants, so dragging them costs no CPU work.

Anchor effects need CPU composites mid-stack; stacks with anchors use the CPU
compositor, so they never reach this module."""

import gpu
import numpy as np

MASK_BLENDS = {"NORMAL": 0, "MULTIPLY": 1, "ADD": 2, "SUBTRACT": 3, "MAX": 4, "MIN": 5}

# kind: 0 constant (params3.x), 1 texture red (alpha = its alpha if params2.w),
# 2 generator source + noise -> smoothstep, 3 noise -> smoothstep,
# 4 ID colour selection, 5 levels of the mask so far (params3 = lo, hi, gamma).
_EFFECT = """
float sstep(float e0, float e1, float x)
{
  float t = clamp((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

void main()
{
  ivec2 p = ivec2(gl_FragCoord.xy);
  float m = texelFetch(baseTex, p, 0).r;
  int kind = int(params.w + 0.5);
  float v;
  float a = params.x;
  if (kind == 0) {
    v = params3.x;
  }
  else if (kind == 1) {
    vec4 t = texelFetch(valTex, p, 0);
    v = t.r;
    if (params2.w > 0.5) {
      a *= t.a;
    }
  }
  else if (kind == 2 || kind == 3) {
    float n = texelFetch(noiseTex, p, 0).r;
    v = (kind == 3) ? n : texelFetch(valTex, p, 0).r + (n - 0.5) * params2.z;
    v = sstep(params2.x - params2.y, params2.x + params2.y, v);
  }
  else if (kind == 4) {
    float d = length(texelFetch(valTex, p, 0).rgb - idColor.rgb);
    v = 1.0 - sstep(idColor.a * 0.5, idColor.a, d);
  }
  else {
    float hi = max(params3.y, params3.x + 1e-4);
    v = pow(clamp((m - params3.x) / (hi - params3.x), 0.0, 1.0), 1.0 / params3.z);
  }
  if (params.z > 0.5) {
    v = 1.0 - v;
  }
  int mode = int(params.y + 0.5);
  float o = v;
  if (mode == 1) o = m * v;
  else if (mode == 2) o = m + v;
  else if (mode == 3) o = m - v;
  else if (mode == 4) o = max(m, v);
  else if (mode == 5) o = min(m, v);
  fragColor = vec4(vec3(clamp(m + (o - m) * a, 0.0, 1.0)), 1.0);
}
"""

_shader = None


def _effect_shader():
    global _shader
    if _shader is None:
        from gpu_extras.batch import batch_for_shader
        info = gpu.types.GPUShaderCreateInfo()
        for ident in ("params", "params2", "params3", "idColor"):
            info.push_constant("VEC4", ident)
        for slot, ident in enumerate(("baseTex", "valTex", "noiseTex")):
            info.sampler(slot, "FLOAT_2D", ident)
        info.vertex_in(0, "VEC2", "pos")
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source("void main()\n{\n  gl_Position = vec4(pos, 0.0, 1.0);\n}\n")
        info.fragment_source(_EFFECT)
        sh = gpu.shader.create_from_info(info)
        _shader = (sh, batch_for_shader(sh, "TRI_FAN", {"pos": ((-1, -1), (1, -1), (1, 1), (-1, 1))}))
    return _shader


def _pass(gc, base, kind, opacity=1.0, mode="NORMAL", invert=False, val=None, noise=None,
          params2=(0.0, 0.0, 0.0, 0.0), params3=(0.0, 1.0, 1.0, 0.0), id_color=(0.0, 0.0, 0.0, 1.0)):
    """One effect onto the mask so far: returns a new target and releases base."""
    from .projector import _white
    sh, batch = _effect_shader()
    out = gc.acquire(base.w, base.h)
    with out.fb.bind():
        gpu.state.blend_set("NONE")
        gpu.state.viewport_set(0, 0, out.w, out.h)
        sh.bind()
        sh.uniform_float("params", (float(opacity), float(MASK_BLENDS.get(mode, 0)), float(invert), float(kind)))
        sh.uniform_float("params2", tuple(float(x) for x in params2))
        sh.uniform_float("params3", tuple(float(x) for x in params3))
        sh.uniform_float("idColor", tuple(float(x) for x in id_color))
        sh.uniform_sampler("baseTex", base.tex)
        sh.uniform_sampler("valTex", val if val is not None else _white())
        sh.uniform_sampler("noiseTex", noise if noise is not None else _white())
        batch.draw(sh)
        for ident in ("baseTex", "valTex", "noiseTex"):
            sh.uniform_sampler(ident, _white())
    gc.release(base)
    return out


def supported(layer):
    """True if this layer's mask stack can be evaluated on the GPU."""
    return not any(e.enabled and e.type == "ANCHOR" for e in layer.mask_effects)


def _noise(gc, ts, effect, keys):
    """The generator's fBm noise field as a (cached) GPU texture."""
    from .. import generators
    from . import noise_gpu
    pos = generators._positions(ts)
    key = ("noise", ts.size, generators._generation, round(effect.noise_scale, 6),
           effect.noise_octaves, effect.seed)
    keys.add(key)

    def make():
        pos_tex = gc.upload(("positions", ts.size, generators._generation), [pos],
                            lambda: np.concatenate([pos, np.ones(pos.shape[:2] + (1,), np.float32)], -1),
                            "RGBA32F")
        keys.add(("positions", ts.size, generators._generation))
        perm, vals = generators._table(effect.seed)
        return noise_gpu.fbm_target(gc, pos_tex, ts.size, effect.noise_scale, effect.noise_octaves, perm, vals)
    return gc.memo(key, [pos], make)


def _generator(gc, m, effect, ts, keys):
    from .. import generators
    if generators.missing(effect, ts):
        return m
    t = 1.0 - effect.level
    if effect.type == "ID_MASK":
        idm = generators.baked(ts, "ID")
        key = ("idmap", ts.size, generators._generation)
        keys.add(key)
        tex = gc.upload(key, [idm], lambda: np.concatenate(
            [idm, np.ones(idm.shape[:2] + (1,), np.float32)], -1), "RGBA32F")
        return _pass(gc, m, 4, effect.opacity, effect.blend_mode, effect.invert, val=tex,
                     id_color=(*effect.id_color, effect.id_tolerance))
    if effect.type == "NOISE":
        return _pass(gc, m, 3, effect.opacity, effect.blend_mode, effect.invert,
                     noise=_noise(gc, ts, effect, keys).tex, params2=(t, effect.softness, 0.0, 0.0))
    src_key = ("src", ts.id_data.name, ts.size, generators._generation, effect.type, effect.axis,
               effect.grunge, effect.grunge_tiling, effect.seed if effect.type == "GRUNGE" else 0)
    src = generators._cached(src_key, lambda: generators._source(effect, ts))
    key = ("gsrc",) + src_key[1:]
    keys.add(key)
    tex = gc.upload(key, [src], lambda: np.ascontiguousarray(src, np.float32), "R32F")
    noise = _noise(gc, ts, effect, keys).tex if effect.noise_amount > 0.0 else None
    return _pass(gc, m, 2, effect.opacity, effect.blend_mode, effect.invert, val=tex, noise=noise,
                 params2=(t, effect.softness, effect.noise_amount if noise is not None else 0.0, 0.0))


def evaluate(gc, layer, cache, keys):
    """The layer's final mask as a GPU target (red channel), or None when it has no
    active mask. The caller releases the target."""
    from ..compositor import read
    if not (layer.has_mask and layer.mask_enabled):
        return None
    ts = layer.id_data.paintstack
    m = gc.acquire(*ts.size)
    with m.fb.bind():  # pooled targets hold old data: start from a defined value
        gpu.state.active_framebuffer_get().clear(color=(1.0, 1.0, 1.0, 1.0))
    if layer.mask_image is not None:
        px = read(layer.mask_image, None, cache)
        key = ("px", layer.mask_image.name)
        keys.add(key)
        m = _pass(gc, m, 1, val=gc.upload(key, [px], lambda: px), params2=(0, 0, 0, 0.0))
    else:
        m = _pass(gc, m, 0, params3=(1.0, 1.0, 1.0, 0.0))

    for e in layer.mask_effects:
        if not e.enabled:
            continue
        if e.type == "LEVELS":
            m = _pass(gc, m, 5, e.opacity, "NORMAL", e.invert, params3=(e.in_min, e.in_max, e.gamma, 0.0))
        elif e.type == "BLUR":
            blurred = gc.blur(m, e.blur_radius)
            m = _pass(gc, m, 1, e.opacity, "NORMAL", e.invert, val=blurred.tex, params2=(0, 0, 0, 0.0))
            gc.release(blurred)
        elif e.type == "PAINT":
            if e.image is None:
                continue
            px = read(e.image, None, cache)
            key = ("px", e.image.name)
            keys.add(key)
            m = _pass(gc, m, 1, e.opacity, e.blend_mode, e.invert, val=gc.upload(key, [px], lambda: px),
                      params2=(0, 0, 0, 1.0))
        elif e.type == "FILL":
            m = _pass(gc, m, 0, e.opacity, e.blend_mode, e.invert, params3=(e.value, 1.0, 1.0, 0.0))
        else:
            m = _generator(gc, m, e, ts, keys)
    if layer.mask_invert:
        m = _pass(gc, m, 5, 1.0, "NORMAL", True, params3=(0.0, 1.0, 1.0, 0.0))
    return m
