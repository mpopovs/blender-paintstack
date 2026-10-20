"""GLSL shaders for projection painting. Uniforms are packed into vec4s so the
push-constant block stays within the 128 bytes every GPU backend guarantees."""

import gpu

_cache = {}

# viewO = (view origin, is_perspective), viewF = (view forward, depth bias)
_VIEW_CONSTANTS = (("MAT4", "viewProj"), ("VEC4", "viewO"), ("VEC4", "viewF"))

_DEPTH_VERT = """
void main()
{
  wpos = pos;
  gl_Position = viewProj * vec4(pos, 1.0);
}
"""

_DEPTH_FRAG = """
void main()
{
  fragColor = vec4(dot(wpos - viewO.xyz, viewF.xyz), 0.0, 0.0, 1.0);
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
# tested against the brush footprint, the depth prepass and the surface angle.
_DAB_FRAG = """
void main()
{
  vec4 clip = viewProj * vec4(wpos, 1.0);
  if (clip.w <= 0.0) {
    discard;
  }
  vec2 ndc = clip.xy / clip.w;
  vec2 scr = (ndc * 0.5 + 0.5) * region.xy;
  float t = distance(scr, brush.xy) / brush.z;
  if (t >= 1.0) {
    discard;
  }

  ivec2 px = ivec2(clamp(scr, vec2(0.0), region.xy - 1.0));
  float scene = texelFetch(depthTex, px, 0).r;
  float depth = dot(wpos - viewO.xyz, viewF.xyz);
  if (depth - scene > viewF.w * max(1.0, abs(scene))) {
    discard;
  }

  vec3 V = (viewO.w > 0.5) ? normalize(viewO.xyz - wpos) : -viewF.xyz;
  float facing = dot(normalize(wnor), V);
  if (facing <= 0.0) {
    discard;
  }

  float falloff = 1.0 - smoothstep(min(brush.w, 0.999), 1.0, t);
  fragColor = vec4(1.0, 1.0, 1.0, falloff * region.z * smoothstep(0.0, 0.25, facing));
}
"""

_COVERAGE_FRAG = """
void main()
{
  fragColor = vec4(1.0);
}
"""


def _create(name, inputs, outs, vert, frag, constants=(), samplers=()):
    info = gpu.types.GPUShaderCreateInfo()
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
                                  _DEPTH_VERT, _DEPTH_FRAG, _VIEW_CONSTANTS)
    return _cache["depth"]


def dab_shader():
    if "dab" not in _cache:
        _cache["dab"] = _create(
            "ps_dab", [("VEC3", "pos"), ("VEC2", "uv"), ("VEC3", "nor")],
            [("VEC3", "wpos"), ("VEC3", "wnor")], _UV_VERT, _DAB_FRAG,
            # brush = (x, y, radius, hardness), region = (width, height, flow, unused)
            _VIEW_CONSTANTS + (("VEC4", "brush"), ("VEC4", "region")),
            samplers=[(0, "depthTex")])
    return _cache["dab"]


def coverage_shader():
    if "coverage" not in _cache:
        _cache["coverage"] = _create(
            "ps_coverage", [("VEC3", "pos"), ("VEC2", "uv"), ("VEC3", "nor")],
            [("VEC3", "wpos"), ("VEC3", "wnor")], _UV_VERT, _COVERAGE_FRAG)
    return _cache["coverage"]
