"""Image <-> numpy helpers. Arrays are (height, width, 4) float32, bottom row first,
matching Blender's pixel layout and OpenGL texture space."""

import bpy
import numpy as np

PREFIX = "PS_"


def read(img):
    w, h = img.size
    arr = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(arr)
    return arr.reshape(h, w, 4)


def write(img, arr):
    img.pixels.foreach_set(np.ascontiguousarray(arr, np.float32).ravel())
    img.update()


def new_image(name, size, color, *, float_buffer=False, srgb=False, alpha=1.0):
    """size: texels per side, or (width, height)."""
    w, h = (size, size) if isinstance(size, int) else size
    img = bpy.data.images.new(PREFIX + name, w, h, alpha=True, float_buffer=float_buffer)
    img.colorspace_settings.name = "sRGB" if srgb else "Non-Color"
    img.generated_color = (*color[:3], alpha)
    return img


def clone_image(img, name=None):
    """Pixel-exact copy. Image.copy() regenerates unsaved generated images from their
    settings (losing painted pixels), so copy the buffer explicitly."""
    w, h = img.size
    new = bpy.data.images.new(name or img.name, w, h, alpha=True, float_buffer=img.is_float)
    new.colorspace_settings.name = img.colorspace_settings.name
    write(new, read(img))
    return new


def rgba(rgb, alpha):
    """Stack an (h, w, 3) array and an (h, w) alpha into RGBA."""
    return np.concatenate([rgb, alpha[..., None]], axis=-1)


def _neighbors(a):
    """The 8 neighbours of every texel (edges clamp)."""
    p = np.pad(a, [(1, 1), (1, 1)] + [(0, 0)] * (a.ndim - 2), mode="edge")
    h, w = a.shape[:2]
    return [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]


def grow_mask(mask, coverage, iterations):
    """Spread a scalar mask outwards into texels outside UV coverage, so strokes
    reach across island borders and bilinear filtering does not show seams."""
    outside = ~coverage
    for _ in range(iterations):
        grown = np.maximum.reduce(_neighbors(mask))
        mask = np.where(outside, np.maximum(mask, grown), mask)
    return mask


def dilate(arr, coverage, iterations):
    """Edge padding: fill texels outside UV coverage with the average of filled
    neighbours, growing one texel per iteration."""
    arr = arr.copy()
    filled = coverage.copy()
    for _ in range(iterations):
        if filled.all():
            break
        nb_val = _neighbors(arr * filled[..., None])
        nb_cnt = sum(n.astype(np.float32) for n in _neighbors(filled))
        total = sum(nb_val)
        grow = (~filled) & (nb_cnt > 0)
        arr[grow] = total[grow] / nb_cnt[grow][..., None]
        filled = filled | grow
    return arr
