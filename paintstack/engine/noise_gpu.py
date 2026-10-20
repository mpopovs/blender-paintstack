"""fBm value noise on the GPU, using the same permutation and lattice tables as
generators.value_noise so results match the CPU fallback."""

import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from .projector import make_texture, read_texture

_shader = None

_VERT = """
void main()
{
  gl_Position = vec4(pos, 0.0, 1.0);
}
"""

_FRAG = """
int P(int i)
{
  return int(texelFetch(permTex, ivec2(i, 0), 0).r);
}

float V(int i)
{
  return texelFetch(valsTex, ivec2(i, 0), 0).r;
}

float vnoise(vec3 p)
{
  vec3 fl = floor(p);
  vec3 f = p - fl;
  ivec3 i = ivec3(fl) & 255;
  vec3 u = f * f * (3.0 - 2.0 * f);
  int hx0 = P(i.x);
  int hx1 = P(i.x + 1);
  float layer[2];
  for (int dz = 0; dz < 2; dz++) {
    float row[2];
    for (int dy = 0; dy < 2; dy++) {
      float c0 = V(P(P(hx0 + i.y + dy) + i.z + dz));
      float c1 = V(P(P(hx1 + i.y + dy) + i.z + dz));
      row[dy] = mix(c0, c1, u.x);
    }
    layer[dz] = mix(row[0], row[1], u.y);
  }
  return mix(layer[0], layer[1], u.z);
}

void main()
{
  vec3 p = texelFetch(posTex, ivec2(gl_FragCoord.xy), 0).xyz * params.x;
  float total = 0.0, amp = 0.5, norm = 0.0, freq = 1.0;
  for (int o = 0; o < octaves; o++) {
    total += amp * vnoise(p * freq);
    norm += amp;
    amp *= 0.5;
    freq *= 2.0;
  }
  fragColor = vec4(total / norm, 0.0, 0.0, 1.0);
}
"""


def _get_shader():
    global _shader
    if _shader is None:
        info = gpu.types.GPUShaderCreateInfo()
        info.push_constant("VEC4", "params")  # x = scale
        info.push_constant("INT", "octaves")
        info.sampler(0, "FLOAT_2D", "posTex")
        info.sampler(1, "FLOAT_2D", "permTex")
        info.sampler(2, "FLOAT_2D", "valsTex")
        info.vertex_in(0, "VEC2", "pos")
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source(_VERT)
        info.fragment_source(_FRAG)
        _shader = gpu.shader.create_from_info(info)
    return _shader


def fbm(pos, scale, octaves, perm, vals):
    """pos: (h, w, 3) float32. Raises if no GPU context is available."""
    h, w = pos.shape[:2]
    rgba = np.concatenate([pos, np.ones((h, w, 1), np.float32)], -1)
    sh = _get_shader()
    tex_pos = make_texture(rgba, "RGBA32F")
    tex_perm = make_texture(perm.reshape(1, -1), "R32F")
    tex_vals = make_texture(vals.reshape(1, -1), "R32F")
    off = gpu.types.GPUOffScreen(w, h, format="RGBA32F")
    try:
        batch = batch_for_shader(sh, "TRI_FAN", {"pos": ((-1, -1), (1, -1), (1, 1), (-1, 1))})
        with off.bind():
            gpu.state.blend_set("NONE")
            sh.bind()
            sh.uniform_float("params", (scale, 0.0, 0.0, 0.0))
            sh.uniform_int("octaves", octaves)
            sh.uniform_sampler("posTex", tex_pos)
            sh.uniform_sampler("permTex", tex_perm)
            sh.uniform_sampler("valsTex", tex_vals)
            batch.draw(sh)
            from .projector import _white
            for name in ("posTex", "permTex", "valsTex"):
                sh.uniform_sampler(name, _white())
        return read_texture(off.texture_color).astype(np.float32)
    finally:
        off.free()


def fbm_target(gc, pos_tex, size, scale, octaves, perm, vals):
    """fBm of a positions texture into a new RGBA32F gpu_compositor.Target (red),
    staying on the GPU (for GPU masks)."""
    from .gpu_compositor import Target
    from .projector import _white
    sh = _get_shader()
    w, h = size
    out = Target(w, h, "RGBA32F")
    tex_perm = make_texture(perm.reshape(1, -1), "R32F")
    tex_vals = make_texture(vals.reshape(1, -1), "R32F")
    batch = batch_for_shader(sh, "TRI_FAN", {"pos": ((-1, -1), (1, -1), (1, 1), (-1, 1))})
    with out.fb.bind():
        gpu.state.blend_set("NONE")
        gpu.state.viewport_set(0, 0, w, h)
        sh.bind()
        sh.uniform_float("params", (scale, 0.0, 0.0, 0.0))
        sh.uniform_int("octaves", octaves)
        sh.uniform_sampler("posTex", pos_tex)
        sh.uniform_sampler("permTex", tex_perm)
        sh.uniform_sampler("valsTex", tex_vals)
        batch.draw(sh)
        for name in ("posTex", "permTex", "valsTex"):
            sh.uniform_sampler(name, _white())
    return out
