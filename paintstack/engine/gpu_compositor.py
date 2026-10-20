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

/* params = (opacity, blend mode, source is a texture, source alpha is coverage),
   params2.x = has a mask texture. */
void main()
{
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec3 base = texelFetch(baseTex, p, 0).rgb;
  vec4 lay = texelFetch(layerTex, p, 0);
  vec3 src = (params.z > 0.5) ? lay.rgb : constSrc.rgb;
  float a = ((params.w > 0.5) ? lay.a : 1.0) * params.x;
  if (params2.x > 0.5) {
    a *= texelFetch(maskTex, p, 0).r;
  }
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
  vec2 d = texelFetch(fieldTex, p, 0).rg * params.x * params.yz;  /* tile units -> image */
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
    "blend": (_BLEND, ["baseTex", "layerTex", "maskTex"],
              [("VEC4", "constSrc"), ("VEC4", "params"), ("VEC4", "params2")]),
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

    def run(self, name, out, textures=(), rect=None, **constants):
        """Draw a full-screen pass into target `out` (or only into rect = (x, y, w, h))."""
        return self.draw(name, out.fb, rect or (0, 0, out.w, out.h), textures, **constants) and out

    def draw(self, name, fb, rect, textures=(), **constants):
        sh, batch, samplers = shader(name)
        with fb.bind():
            gpu.state.blend_set("NONE")
            gpu.state.viewport_set(*rect)
            sh.bind()
            for ident, value in constants.items():
                sh.uniform_float(ident, value)
            for ident, tex in zip(samplers, textures):
                sh.uniform_sampler(ident, tex)
            batch.draw(sh)
            # Release sampler bindings (see engine.live.release_samplers).
            for ident in samplers:
                sh.uniform_sampler(ident, _white())
        return True

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

    def memo(self, key, arrays, make):
        """Like upload, for GPU objects made by make() (e.g. a computed noise field)."""
        key = (self.namespace,) + tuple(key)
        hit = self.uploads.get(key)
        if hit is not None and len(hit[0]) == len(arrays) and all(a is b for a, b in zip(hit[0], arrays)):
            return hit[1]
        obj = make()
        self.uploads[key] = (tuple(arrays), obj)
        return obj

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

    def write_region(self, t, y0, x0, rgba):
        """Paste an (h, w, 4) array into a target at (y0, x0)."""
        from .live import _blit
        _blit(t.tex, 0, x0, y0, np.ascontiguousarray(rgba, np.float32))

    def read_region(self, t, y0, y1, x0, x1):
        """(y1 - y0, x1 - x0, 4) block of a target."""
        from .projector import buffer_array
        with t.fb.bind():
            buf = gpu.state.active_framebuffer_get().read_color(x0, y0, x1 - x0, y1 - y0, 4, 0, "FLOAT")
        return buffer_array(buf, (y1 - y0) * (x1 - x0) * 4).reshape(y1 - y0, x1 - x0, 4)

    # Operations

    def mask_texture(self, mask, key):
        """One-channel upload of a layer mask, shared by every channel's blend."""
        if np.ndim(mask) < 2:
            return None
        return self.upload(key, [mask], lambda: mask, "R16F")

    def blend(self, base, parts):
        """Blend one layer channel (compositor.Parts) onto base."""
        src = parts.src
        src_tex = np.ndim(src) >= 2
        layer = _white()
        if src_tex:
            if src.shape[-1] == 4:  # painted pixels: upload as they are
                layer = self.upload(parts.src_key, [src], lambda: src)
            else:
                layer = self.upload(parts.src_key, [src], lambda: np.concatenate(
                    [src, np.ones(src.shape[:2] + (1,), np.float32)], -1))
        if parts.mask_tex is not None:
            mask = parts.mask_tex
        else:
            mask = self.mask_texture(parts.mask, parts.mask_key) if parts.mask is not None else None
        scale = parts.scale * (1.0 if mask is not None or parts.mask is None else float(parts.mask))
        const = tuple(np.broadcast_to(np.asarray(src, np.float32), (3,))) if not src_tex else (0.0, 0.0, 0.0)
        out = self.op("blend", (base.w, base.h), (base.tex, layer, mask if mask is not None else _white()),
                      constSrc=(*const, 0.0),
                      params=(scale, float(BLEND_IDS.get(parts.mode, 0)), float(src_tex),
                              float(parts.src_alpha)),
                      params2=(float(mask is not None), 0.0, 0.0, 0.0))
        self.release(base)
        return out

    def fade(self, base, inner, scale, mask=None, mask_key=None, mask_tex=None):
        """base + (inner - base) * scale * mask (a CPU mask, or mask_tex on the GPU)."""
        mtex = mask_tex if mask_tex is not None else (
            self.mask_texture(mask, mask_key) if mask is not None else None)
        if mask is not None and mtex is None:
            scale *= float(mask)
        out = self.op("fade", (base.w, base.h), (base.tex, inner.tex, mtex if mtex is not None else _white()),
                      params=(float(scale), float(mtex is not None), 0.0, 0.0))
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
            field = filters._warp_field(base.h, base.w, layer.warp_scale)
            ftex = self.upload(("warp", base.h, base.w, round(layer.warp_scale, 3)), [field],
                               lambda: np.concatenate([field, np.zeros(field.shape[:2] + (2,), np.float32)], -1),
                               "RGBA32F")
            tile = min(base.w, base.h)
            return self.op("warp", size, (base.tex, ftex),
                           params=(layer.filter_amount * 0.05, tile / base.w, tile / base.h, 0.0))
        return self.copy(base)

    # Output for display

    def to_image_region(self, t, image, block):
        """Write an aligned block (y0, y1, x0, x1, levels from live.aligned) of a
        target holding final values into the image's GPU texture and its first
        mip levels; the conversion and the mips are computed on the GPU."""
        from . import live
        y0, y1, x0, x1, levels = block
        tex = gpu.texture.from_image(image)
        to_linear = float(tex.format.startswith("SRGB") and live.srgb_encodes())
        level, owned = t, []
        try:
            for mip in range(levels):
                ly0, ly1, lx0, lx1 = y0 >> mip, y1 >> mip, x0 >> mip, x1 >> mip
                if ly1 <= ly0 or lx1 <= lx0:
                    break
                rect = (lx0, ly0, lx1 - lx0, ly1 - ly0)
                if mip:
                    nxt = self.acquire(max(1, level.w // 2), max(1, level.h // 2))
                    owned.append(nxt)
                    self.run("down", nxt, (level.tex,), rect=rect)
                    level = nxt
                try:
                    fb = gpu.types.GPUFrameBuffer(color_slots={"texture": tex, "mip": mip})
                except Exception:
                    if mip == 0:
                        raise
                    break  # no further mip levels
                self.draw("final", fb, rect, (level.tex,), params=(0.0, to_linear, 0.0, 0.0))
        finally:
            for o in owned:
                self.release(o)

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
