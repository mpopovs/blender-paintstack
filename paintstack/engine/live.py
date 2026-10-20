"""GPU live stroke display.

During a stroke, the touched regions of the output maps are written straight
into the GPU textures the viewport samples (`gpu.texture.from_image` shares
Blender's own texture), including every mip level, so the viewport updates
without re-uploading whole images. The CPU images are written once, when the
stroke ends, which also makes Blender rebuild the textures from them.

Backends differ in whether writing to an sRGB texture encodes the value, so the
behaviour is measured once with a tiny probe instead of being assumed."""

import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from .projector import make_texture

_shader = None
_batch = None
_srgb_encodes = None  # True if the backend sRGB-encodes writes to sRGB textures

_FRAG = """
void main()
{
  fragColor = texelFetch(src, ivec2(gl_FragCoord.xy) - ivec2(origin.xy), 0);
}
"""


def _copy_shader():
    global _shader, _batch
    if _shader is None:
        info = gpu.types.GPUShaderCreateInfo()
        info.push_constant("VEC4", "origin")
        info.sampler(0, "FLOAT_2D", "src")
        info.vertex_in(0, "VEC2", "pos")
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source("void main()\n{\n  gl_Position = vec4(pos, 0.0, 1.0);\n}\n")
        info.fragment_source(_FRAG)
        _shader = gpu.shader.create_from_info(info)
        _batch = batch_for_shader(_shader, "TRI_FAN", {"pos": ((-1, -1), (1, -1), (1, 1), (-1, 1))})
    return _shader


def _blit(tex, mip, x, y, data):
    """Copy an (h, w, 4) float array into texture mip level at (x, y)."""
    h, w = data.shape[:2]
    src = make_texture(data, "RGBA16F")
    fb = gpu.types.GPUFrameBuffer(color_slots={"texture": tex, "mip": mip})
    sh = _copy_shader()
    with fb.bind():
        gpu.state.blend_set("NONE")
        gpu.state.viewport_set(x, y, w, h)
        sh.bind()
        sh.uniform_float("origin", (x, y, 0.0, 0.0))
        sh.uniform_sampler("src", src)
        _batch.draw(sh)


def srgb_encodes():
    """Probe once, on a real image texture like the ones strokes write to: does
    drawing into an 8-bit sRGB texture encode linear -> sRGB on this backend?"""
    global _srgb_encodes
    if _srgb_encodes is None:
        import bpy
        img = bpy.data.images.new("PS_srgb_probe", 4, 4, alpha=True)
        try:
            img.colorspace_settings.name = "sRGB"
            tex = gpu.texture.from_image(img)
            _blit(tex, 0, 0, 0, np.full((4, 4, 4), 0.5, np.float32))
            stored = np.array(tex.read().to_list(), np.float32).reshape(4, 4, 4)[0, 0, 0]
            stored = stored / 255.0 if stored > 1.5 else stored
            _srgb_encodes = bool(stored > 0.6)  # 0.5 encodes to ~0.735
        finally:
            bpy.data.images.remove(img)
    return _srgb_encodes


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


# Mip levels refreshed during a stroke: 0..6 (down to 1/64 scale). Coarser levels
# average large parts of the texture, change imperceptibly per stroke, and are
# rebuilt from the CPU image when the stroke ends.
LIVE_MIPS = 7


def aligned(region, shape):
    """Region grown so every live mip level's block starts and ends on whole
    texels: (y0, y1, x0, x1, levels). shape: the image's (height, width)."""
    y0, y1, x0, x1 = region
    h, w = shape
    levels = min(LIVE_MIPS, int(np.log2(min(h, w))) + 1)
    f = 1 << (levels - 1)
    return ((y0 // f) * f, min(h, -(-y1 // f) * f), (x0 // f) * f, min(w, -(-x1 // f) * f), levels)


def push(image, rgba, region):
    """Write region (y0, y1, x0, x1) of a full-resolution RGBA array (the image's
    stored values) into the image's GPU texture and its first mip levels."""
    ay0, ay1, ax0, ax1, levels = aligned(region, rgba.shape[:2])
    push_block(image, rgba[ay0:ay1, ax0:ax1], ay0, ax0, levels)


def push_block(image, block, y0, x0, levels):
    """Write an aligned RGBA block (from `aligned`) at (y0, x0) into the image's
    GPU texture and its first `levels` mip levels."""
    tex = gpu.texture.from_image(image)
    convert = tex.format.startswith("SRGB") and srgb_encodes()
    block = np.ascontiguousarray(block, np.float32)
    for mip in range(levels):
        if mip:
            if block.shape[0] < 2 or block.shape[1] < 2:
                break
            block = (block[0::2, 0::2] + block[1::2, 0::2] + block[0::2, 1::2] + block[1::2, 1::2]) * 0.25
        data = block
        if convert:
            data = np.concatenate([srgb_to_linear(block[..., :3]), block[..., 3:]], -1)
        try:
            _blit(tex, mip, x0 >> mip, y0 >> mip, np.ascontiguousarray(data, np.float32))
        except Exception:
            if mip == 0:
                raise
            break  # texture has no further mip levels


def read_level(image, mip=0):
    """Sample a mip level of the image's GPU texture (for tests / diagnostics)."""
    tex = gpu.texture.from_image(image)
    w, h = max(1, tex.width >> mip), max(1, tex.height >> mip)
    info = gpu.types.GPUShaderCreateInfo()
    info.sampler(0, "FLOAT_2D", "t")
    info.push_constant("VEC4", "lod")
    info.vertex_in(0, "VEC2", "pos")
    info.fragment_out(0, "VEC4", "o")
    info.vertex_source("void main()\n{\n  gl_Position = vec4(pos, 0.0, 1.0);\n}\n")
    info.fragment_source("void main()\n{\n  o = textureLod(t, gl_FragCoord.xy / lod.yz, lod.x);\n}\n")
    sh = gpu.shader.create_from_info(info)
    batch = batch_for_shader(sh, "TRI_FAN", {"pos": ((-1, -1), (1, -1), (1, 1), (-1, 1))})
    off = gpu.types.GPUOffScreen(w, h, format="RGBA32F")
    try:
        with off.bind():
            sh.bind()
            sh.uniform_sampler("t", tex)
            sh.uniform_float("lod", (float(mip), float(w), float(h), 0.0))
            batch.draw(sh)
            release_samplers()
        return np.array(off.texture_color.read().to_list(), np.float32).reshape(h, w, 4)
    finally:
        off.free()


def release_samplers():
    """After sampling an image's GPU texture, bind a harmless texture to slot 0.
    Otherwise the GPU state keeps a pointer to the image's texture, and deleting
    the image later makes the next texture operation crash Blender."""
    from .projector import _white
    sh = _copy_shader()
    sh.bind()
    sh.uniform_sampler("src", _white())
