"""GLSL shaders for projection painting.

Per-stroke view data lives in a uniform buffer (ViewParams); only per-dab values
are push constants, which keeps them well inside the 128-byte guarantee."""

import gpu

_cache = {}

VIEW_STRUCT = """
struct ViewParams {
  mat4 viewProj;
  vec4 viewO;    /* view origin, is_perspective */
  vec4 viewF;    /* view forward, depth bias */
  vec4 region;   /* width, height, use_tip_texture, use_stencil */
  vec4 stencil;  /* center x, center y, half height (px), rotation */
  vec4 stencil2; /* aspect (w / h), unused, unused, unused */
};
"""

_DEPTH_VERT = """
void main()
{
  wpos = pos;
  gl_Position = vp.viewProj * vec4(pos, 1.0);
}
"""

_DEPTH_FRAG = """
void main()
{
  fragColor = vec4(dot(wpos - vp.viewO.xyz, vp.viewF.xyz), 0.0, 0.0, 1.0);
}
"""

_UV_VERT = """
void main()
{
  wpos = pos;
  wnor = nor;
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
"""

# One brush dab rendered in UV space: each texel is projected to the screen and
# tested against the brush tip, the stencil, the depth prepass and the surface angle.
_DAB_FRAG = """
vec2 rotate(vec2 v, float a)
{
  float c = cos(a), s = sin(a);
  return vec2(c * v.x + s * v.y, -s * v.x + c * v.y);
}

void main()
{
  vec4 clip = vp.viewProj * vec4(wpos, 1.0);
  if (clip.w <= 0.0) {
    discard;
  }
  vec2 ndc = clip.xy / clip.w;
  vec2 scr = (ndc * 0.5 + 0.5) * vp.region.xy;
  vec2 local = rotate(scr - brush.xy, dab.x) / brush.z;

  float shape;
  if (vp.region.z > 0.5) {
    if (abs(local.x) >= 1.0 || abs(local.y) >= 1.0) {
      discard;
    }
    shape = texture(tipTex, local * 0.5 + 0.5).r;
  }
  else {
    float t = length(local);
    if (t >= 1.0) {
      discard;
    }
    shape = 1.0 - smoothstep(min(brush.w, 0.999), 1.0, t);
  }

  if (vp.region.w > 0.5) {
    vec2 s = rotate(scr - vp.stencil.xy, vp.stencil.w) / vp.stencil.z;
    s.x /= vp.stencil2.x;
    vec2 suv = s * 0.5 + 0.5;
    if (suv.x < 0.0 || suv.y < 0.0 || suv.x > 1.0 || suv.y > 1.0) {
      discard;
    }
    shape *= texture(stencilTex, suv).r;
  }
  if (shape <= 0.0) {
    discard;
  }

  ivec2 px = ivec2(clamp(scr, vec2(0.0), vp.region.xy - 1.0));
  float scene = texelFetch(depthTex, px, 0).r;
  float depth = dot(wpos - vp.viewO.xyz, vp.viewF.xyz);
  if (depth - scene > vp.viewF.w * max(1.0, abs(scene))) {
    discard;
  }

  vec3 V = (vp.viewO.w > 0.5) ? normalize(vp.viewO.xyz - wpos) : -vp.viewF.xyz;
  float facing = dot(normalize(wnor), V);
  if (facing <= 0.0) {
    discard;
  }

  fragColor = vec4(1.0, 1.0, 1.0, shape * dab.y * smoothstep(0.0, 0.25, facing));
}
"""

_COVERAGE_FRAG = """
void main()
{
  fragColor = vec4(1.0);
}
"""


def _create(name, inputs, outs, vert, frag, constants=(), samplers=(), view_ubo=False):
    info = gpu.types.GPUShaderCreateInfo()
    if view_ubo:
        info.typedef_source(VIEW_STRUCT)
        info.uniform_buf(0, "ViewParams", "vp")
    for kind, ident in constants:
        info.push_constant(kind, ident)
    for slot, ident in samplers:
        info.sampler(slot, "FLOAT_2D", ident)
    for slot, (kind, ident) in enumerate(inputs):
        info.vertex_in(slot, kind, ident)
    if outs:
        iface = gpu.types.GPUStageInterfaceInfo(name + "_iface")
        for kind, ident in outs:
            iface.smooth(kind, ident)
        info.vertex_out(iface)
    info.fragment_out(0, "VEC4", "fragColor")
    info.vertex_source(vert)
    info.fragment_source(frag)
    return gpu.shader.create_from_info(info)


def depth_shader():
    if "depth" not in _cache:
        _cache["depth"] = _create("ps_depth", [("VEC3", "pos")], [("VEC3", "wpos")],
                                  _DEPTH_VERT, _DEPTH_FRAG, view_ubo=True)
    return _cache["depth"]


def dab_shader():
    if "dab" not in _cache:
        _cache["dab"] = _create(
            "ps_dab", [("VEC3", "pos"), ("VEC2", "uv"), ("VEC3", "nor")],
            [("VEC3", "wpos"), ("VEC3", "wnor")], _UV_VERT, _DAB_FRAG,
            # brush = (x, y, radius, hardness), dab = (angle, flow, unused, unused)
            (("VEC4", "brush"), ("VEC4", "dab")),
            samplers=[(0, "depthTex"), (1, "tipTex"), (2, "stencilTex")], view_ubo=True)
    return _cache["dab"]


def coverage_shader():
    if "coverage" not in _cache:
        _cache["coverage"] = _create(
            "ps_coverage", [("VEC3", "pos"), ("VEC2", "uv"), ("VEC3", "nor")],
            [("VEC3", "wpos"), ("VEC3", "wnor")], _UV_VERT, _COVERAGE_FRAG)
    return _cache["coverage"]
