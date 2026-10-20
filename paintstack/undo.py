"""Stroke undo/redo. Blender's own undo does not track pixels written from
Python, so strokes keep before/after copies of the images they touched."""

import bpy

from . import compositor, pixels

MAX_STEPS = 20

_undo = []
_redo = []


def push(mat, before):
    """before: {image_name: array} captured at stroke start."""
    _undo.append((mat.name, before))
    del _undo[:-MAX_STEPS]
    _redo.clear()


def _swap(src, dst):
    if not src:
        return False
    mat_name, state = src.pop()
    current = {}
    for name, arr in state.items():
        img = bpy.data.images.get(name)
        if img is None:
            continue
        current[name] = pixels.read(img)
        pixels.write(img, arr)
    dst.append((mat_name, current))
    mat = bpy.data.materials.get(mat_name)
    if mat is not None:
        compositor.composite(mat)
    return True


def undo():
    return _swap(_undo, _redo)


def redo():
    return _swap(_redo, _undo)


def clear():
    _undo.clear()
    _redo.clear()
