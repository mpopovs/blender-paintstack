"""GPU compositor: blends the layer stack on the GPU (Metal on Apple Silicon).

The CPU still prepares each layer's entry (source colour, alpha with masks) and
keeps it cached; uploads are cached too, keyed on the identity of those arrays,
so a re-composite after a tweak only uploads what changed. Blend modes, folder
fades and filters are GLSL ports of the CPU code (compositor.py / filters.py)
and are tested against it. Accumulation runs in RGBA32F render targets, written
without hardware blending, which every backend supports; layer data is uploaded
as half floats (well below 8-bit output precision).

Anchor masks need CPU results mid-composite, so stacks with anchors use the CPU
compositor."""

import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from .projector import _white, make_texture, read_texture

BLEND_IDS = {"NORMAL": 0, "MULTIPLY": 1, "SCREEN": 2, "OVERLAY": 3, "SOFT_LIGHT": 4, "ADD": 5,
             "SUBTRACT": 6, "DARKEN": 7, "LIGHTEN": 8, "DIFFERENCE": 9, "NORMAL_COMBINE": 10}

_VERT = "void main()\n{\n  gl_Position = vec4(pos, 0.0, 1.0);\n}\n"

_COMMON = """
vec3 encode_normal(vec3 v)
{
  return v / max(length(v), 1e-6) * 0.5 + 0.5;
}
"""

_BLEND = _COMMON + """
vec3 blend(vec3 b, vec3 s, int mode)
{
  if (mode == 1) return b * s;
  if (mode == 2) return 1.0 - (1.0 - b) * (1.0 - s);
  if (mode == 3) {
    return mix(1.0 - 2.0 * (1.0 - b) * (1.0 - s), 2.0 * b * s, vec3(lessThan(b, vec3(0.5))));
  }
  if (mode == 4) return (1.0 - 2.0 * s) * b * b + 2.0 * s * b;
  if (mode == 5) return min(b + s, vec3(1.0));
  if (mode == 6) return max(b - s, vec3(0.0));
  if (mode == 7) return min(b, s);
  if (mode == 8) return max(b, s);
  if (mode == 9) return abs(b - s);
  if (mode == 10) {
    vec3 bb = b * 2.0 - 1.0;
    vec3 d = s * 2.0 - 1.0;
    return encode_normal(vec3(bb.x + d.x, bb.y + d.y, bb.z * d.z));
  }
  return s;
}

void main()
{
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec3 base = texelFetch(baseTex, p, 0).rgb;
  vec4 lay = texelFetch(layerTex, p, 0);
  vec3 src = (params.z > 0.5) ? lay.rgb : constSrc.rgb;
  float a = ((params.w > 0.5) ? lay.a : 1.0) * params.x;
  vec3 o = blend(base, src, int(params.y + 0.5));
  fragColor = vec4(base + (o - base) * a, 1.0);
}
"""

_FADE = """
void main()
{
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec3 base = texelFetch(baseTex, p, 0).rgb;
  vec3 inner = texelFetch(innerTex, p, 0).rgb;
  float a = params.x * ((params.y > 0.5) ? texelFetch(maskTex, p, 0).r : 1.0);
  fragColor = vec4(base + (inner - base) * a, 1.0);
}
"""

_BLUR = """
void main()
{
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 size = textureSize(src, 0);
  ivec2 dir = ivec2(params.xy);
  int r = int(params.z + 0.5);
  vec3 acc = vec3(0.0);
  for (int k = -r; k <= r; k++) {
    acc += texelFetch(src, clamp(p + dir * k, ivec2(0), size - 1), 0).rgb;
  }
  fragColor = vec4(acc / float(2 * r + 1), 1.0);
}
"""

_SHARPEN = """
void main()
{
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec3 b = texelFetch(baseTex, p, 0).rgb;
  vec3 bl = texelFetch(blurTex, p, 0).rgb;
  vec3 o = b + params.x * (b - bl);
  fragColor = vec4((params.y > 0.5) ? clamp(o, 0.0, 1.0) : o, 1.0);
}
"""

_HSL = """
vec3 rgb_to_hsv(vec3 c)
{
  float mx = max(c.r, max(c.g, c.b));
  float mn = min(c.r, min(c.g, c.b));
  float d = mx - mn;
  float safe = (d > 1e-6) ? d : 1.0;
  float h;
  if (mx == c.r) h = mod((c.g - c.b) / safe, 6.0);
  else if (mx == c.g) h = (c.b - c.r) / safe + 2.0;
  else h = (c.r - c.g) / safe + 4.0;
  h = (d > 1e-6) ? h / 6.0 : 0.0;
  float s = (mx > 1e-6) ? d / max(mx, 1e-6) : 0.0;
  return vec3(h, s, mx);
}

vec3 hsv_to_rgb(vec3 c)
{
  float h = mod(c.x, 1.0);
  float s = c.y, v = c.z;
  int i = int(floor(h * 6.0)) % 6;
  float f = h * 6.0 - floor(h * 6.0);
  float p = v * (1.0 - s), q = v * (1.0 - f * s), t = v * (1.0 - (1.0 - f) * s);
  if (i == 0) return vec3(v, t, p);
  if (i == 1) return vec3(q, v, p);
  if (i == 2) return vec3(p, v, t);
  if (i == 3) return vec3(p, q, v);
  if (i == 4) return vec3(t, p, v);
  return vec3(v, p, q);
}

vec3 lightness(vec3 v, float amount)
{
  return (amount >= 0.0) ? v + (1.0 - v) * amount : v * (1.0 + amount);
}

void main()
{
  vec3 c = texelFetch(baseTex, ivec2(gl_FragCoord.xy), 0).rgb;
  int kind = int(params2.x + 0.5);
  if (kind == 2) {
    fragColor = vec4(c, 1.0);
    return;
  }
  if (kind == 1) {
    fragColor = vec4(clamp(lightness(c, params.z), 0.0, 1.0), 1.0);
    return;
  }
  vec3 hsv = rgb_to_hsv(clamp(c, 0.0, 1.0));
  hsv.x += params.x;
  hsv.y = clamp(hsv.y * (1.0 + params.y), 0.0, 1.0);
  hsv.z = clamp(lightness(vec3(hsv.z), params.z).x, 0.0, 1.0);
  fragColor = vec4(hsv_to_rgb(hsv), 1.0);
}
"""

_WARP = """
vec3 fetch_wrap(ivec2 q, ivec2 size)
{
  return texelFetch(baseTex, ((q % size) + size) % size, 0).rgb;
}

void main()
{
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 size = textureSize(baseTex, 0);
  vec2 d = texelFetch(fieldTex, p, 0).rg * params.x;
  vec2 uv = (vec2(p) + 0.5) / vec2(size) + d;
  vec2 x = (uv - floor(uv)) * vec2(size) - 0.5;
  vec2 x0 = floor(x);
  vec2 f = x - x0;
  ivec2 i0 = ivec2(x0);
  vec3 top = fetch_wrap(i0, size) * (1.0 - f.x) + fetch_wrap(i0 + ivec2(1, 0), size) * f.x;
  vec3 bot = fetch_wrap(i0 + ivec2(0, 1), size) * (1.0 - f.x) + fetch_wrap(i0 + ivec2(1, 1), size) * f.x;
  fragColor = vec4(top * (1.0 - f.y) + bot * f.y, 1.0);
}
"""

_COPY = """
void main()
{
  fragColor = texelFetch(src, ivec2(gl_FragCoord.xy), 0);
}
"""

# Output for display: clip / renormalise like compositor.finalize, optional sRGB
# linearisation (for backends that encode writes to sRGB textures).
_FINAL = _COMMON + """
vec3 to_linear(vec3 c)
{
  return mix(pow((c + 0.055) / 1.055, vec3(2.4)), c / 12.92, vec3(lessThanEqual(c, vec3(0.04045))));
}

void main()
{
  vec3 c = texelFetch(src, ivec2(gl_FragCoord.xy), 0).rgb;
  if (params.x > 0.5) c = encode_normal(c * 2.0 - 1.0);
  c = clamp(c, 0.0, 1.0);
  if (params.y > 0.5) c = to_linear(c);
  fragColor = vec4(c, 1.0);
}
"""

_DOWN = """
void main()
{
  ivec2 p = ivec2(gl_FragCoord.xy) * 2;
  fragColor = 0.25 * (texelFetch(src, p, 0) + texelFetch(src, p + ivec2(1, 0), 0)
                      + texelFetch(src, p + ivec2(0, 1), 0) + texelFetch(src, p + ivec2(1, 1), 0));
}
"""

_SPECS = {
    "blend": (_BLEND, ["baseTex", "layerTex"], [("VEC4", "constSrc"), ("VEC4", "params")]),
    "fade": (_FADE, ["baseTex", "innerTex", "maskTex"], [("VEC4", "params")]),
    "blur": (_BLUR, ["src"], [("VEC4", "params")]),
    "sharpen": (_SHARPEN, ["baseTex", "blurTex"], [("VEC4", "params")]),
    "hsl": (_HSL, ["baseTex"], [("VEC4", "params"), ("VEC4", "params2")]),
    "warp": (_WARP, ["baseTex", "fieldTex"], [("VEC4", "params")]),
    "copy": (_COPY, ["src"], []),
    "final": (_FINAL, ["src"], [("VEC4", "params")]),
    "down": (_DOWN, ["src"], []),
}

_shaders = {}


def shader(name):
    if name not in _shaders:
        frag, samplers, constants = _SPECS[name]
        info = gpu.types.GPUShaderCreateInfo()
        for kind, ident in constants:
            info.push_constant(kind, ident)
        for slot, ident in enumerate(samplers):
            info.sampler(slot, "FLOAT_2D", ident)
        info.vertex_in(0, "VEC2", "pos")
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source(_VERT)
        info.fragment_source(frag)
        sh = gpu.shader.create_from_info(info)
        batch = batch_for_shader(sh, "TRI_FAN", {"pos": ((-1, -1), (1, -1), (1, 1), (-1, 1))})
        _shaders[name] = (sh, batch, samplers)
    return _shaders[name]


class Target:
    """A render target texture with its framebuffer. Half float: 8-bit outputs need
    far less precision, and it halves memory and bandwidth (unified memory on
    Apple Silicon is shared with everything else)."""

    def __init__(self, w, h, fmt="RGBA16F"):
        self.tex = gpu.types.GPUTexture((w, h), format=fmt)
        self.fb = gpu.types.GPUFrameBuffer(color_slots=self.tex)
        self.w, self.h = w, h


class GPUCompositor:
    def __init__(self):
        self.pool = {}  # (w, h) -> free Targets
        self.uploads = {}  # (namespace, key...) -> (refs, texture)
        self.namespace = ""  # material being composited

    # Targets

    def acquire(self, w, h):
        free = self.pool.setdefault((w, h), [])
        return free.pop() if free else Target(w, h)

    def release(self, t):
        if t is not None:
            self.pool.setdefault((t.w, t.h), []).append(t)

    def run(self, name, out, textures=(), **constants):
        sh, batch, samplers = shader(name)
        with out.fb.bind():
            gpu.state.blend_set("NONE")
            gpu.state.viewport_set(0, 0, out.w, out.h)
            sh.bind()
            for ident, value in constants.items():
                sh.uniform_float(ident, value)
            for ident, tex in zip(samplers, textures):
                sh.uniform_sampler(ident, tex)
            batch.draw(sh)
            # Release sampler bindings (see engine.live.release_samplers).
            for ident in samplers:
                sh.uniform_sampler(ident, _white())
        return out

    def op(self, name, size, textures=(), **constants):
        return self.run(name, self.acquire(*size), textures, **constants)

    # Uploads (cached on the identity of the CPU arrays they came from)

    def upload(self, key, arrays, build, fmt="RGBA16F"):
        key = (self.namespace,) + tuple(key)
        hit = self.uploads.get(key)
        if hit is not None and len(hit[0]) == len(arrays) and all(a is b for a, b in zip(hit[0], arrays)):
            return hit[1]
        tex = make_texture(build(), fmt)
        self.uploads[key] = (tuple(arrays), tex)
        return tex

    def forget_unused(self, namespace, keys):
        """Drop uploads of one material that the last composite did not use, and
        keep only a few spare render targets per size."""
        keep = {(namespace,) + tuple(k) for k in keys}
        for key in [k for k in self.uploads if k[0] == namespace and k not in keep
                    and k[1] not in ("warp",)]:
            del self.uploads[key]
        for size, free in self.pool.items():
            del free[4:]

    def from_array(self, rgb):
        h, w = rgb.shape[:2]
        t = self.acquire(w, h)
        tmp = make_texture(np.concatenate([rgb, np.ones((h, w, 1), np.float32)], -1), "RGBA32F")
        return self.run("copy", t, (tmp,))

    def copy(self, t):
        return self.op("copy", (t.w, t.h), (t.tex,))

    def read(self, t):
        return read_texture(t.tex, 4)[..., :3].copy()

    # Operations

    def blend(self, base, entry, key):
        src, alpha, mode = entry
        src_tex, alpha_tex = np.ndim(src) >= 2, np.ndim(alpha) >= 2
        layer = _white()
        if src_tex or alpha_tex:
            h, w = base.h, base.w

            def build():
                rgb = src if src_tex else np.zeros((h, w, 3), np.float32)
                a = alpha if alpha_tex else np.ones((h, w), np.float32)
                return np.concatenate([rgb, a[..., None]], -1)
            layer = self.upload(key, [src if src_tex else None, alpha if alpha_tex else None], build)
        const = tuple(np.broadcast_to(src, (3,))) if not src_tex else (0.0, 0.0, 0.0)
        scalar = 1.0 if alpha_tex else float(alpha)
        out = self.op("blend", (base.w, base.h), (base.tex, layer),
                      constSrc=(*const, 0.0),
                      params=(scalar, float(BLEND_IDS.get(mode, 0)), float(src_tex), float(alpha_tex)))
        self.release(base)
        return out

    def fade(self, base, inner, alpha, key):
        alpha_tex = np.ndim(alpha) >= 2
        mask = _white()
        if alpha_tex:
            mask = self.upload(key, [alpha], lambda: np.ascontiguousarray(alpha, np.float32), "R16F")
        out = self.op("fade", (base.w, base.h), (base.tex, inner.tex, mask),
                      params=(1.0 if alpha_tex else float(alpha), float(alpha_tex), 0.0, 0.0))
        self.release(base)
        self.release(inner)
        return out

    def blur(self, t, radius):
        """Three box passes per axis, rows then columns, like masks.box_blur."""
        cur = self.copy(t)
        for _ in range(3):
            for d in ((0.0, 1.0), (1.0, 0.0)):
                nxt = self.op("blur", (cur.w, cur.h), (cur.tex,), params=(d[0], d[1], float(radius), 0.0))
                self.release(cur)
                cur = nxt
        return cur

    def filter(self, base, layer, ch):
        """New target with the filter applied to base (base is kept)."""
        size = (base.w, base.h)
        t = layer.filter_type
        if t == "BLUR":
            return self.blur(base, layer.filter_radius)
        if t == "SHARPEN":
            bl = self.blur(base, layer.filter_radius)
            out = self.op("sharpen", size, (base.tex, bl.tex),
                          params=(float(layer.filter_amount), float(ch.id != "NORMAL"), 0.0, 0.0))
            self.release(bl)
            return out
        if t == "HSL":
            kind = 2.0 if ch.id == "NORMAL" else (0.0 if ch.is_color else 1.0)
            return self.op("hsl", size, (base.tex,), params=(layer.hue, layer.saturation, layer.lightness, 0.0),
                           params2=(kind, 0.0, 0.0, 0.0))
        if t == "WARP":
            from .. import filters
            field = filters._warp_field(base.w, layer.warp_scale)
            ftex = self.upload(("warp", base.w, round(layer.warp_scale, 3)), [field],
                               lambda: np.concatenate([field, np.zeros(field.shape[:2] + (2,), np.float32)], -1),
                               "RGBA32F")
            return self.op("warp", size, (base.tex, ftex), params=(layer.filter_amount * 0.05, 0.0, 0.0, 0.0))
        return self.copy(base)

    # Output for display

    def to_image(self, t, image, is_normal):
        """Write a composited result into an image's GPU texture (and its mips)."""
        from . import live
        tex = gpu.texture.from_image(image)
        to_linear = tex.format.startswith("SRGB") and live.srgb_encodes()
        level = self.op("final", (t.w, t.h), (t.tex,), params=(float(is_normal), 0.0, 0.0, 0.0))
        mip = 0
        try:
            while True:
                src = level
                if to_linear:  # what the backend will re-encode on write
                    src = self.op("final", (level.w, level.h), (level.tex,), params=(0.0, 1.0, 0.0, 0.0))
                try:
                    fb = gpu.types.GPUFrameBuffer(color_slots={"texture": tex, "mip": mip})
                except Exception:
                    if src is not level:
                        self.release(src)
                    break
                sh, batch, samplers = shader("copy")
                with fb.bind():
                    gpu.state.blend_set("NONE")
                    gpu.state.viewport_set(0, 0, src.w, src.h)
                    sh.bind()
                    sh.uniform_sampler("src", src.tex)
                    batch.draw(sh)
                    sh.uniform_sampler("src", _white())
                if src is not level:
                    self.release(src)
                if level.w <= 1 or level.h <= 1 or mip >= 12:
                    break
                nxt = self.op("down", (max(1, level.w // 2), max(1, level.h // 2)), (level.tex,))
                self.release(level)
                level = nxt
                mip += 1
        finally:
            self.release(level)


_instance = None


def instance():
    global _instance
    if _instance is None:
        _instance = GPUCompositor()
    return _instance


def reset():
    global _instance
    _instance = None
