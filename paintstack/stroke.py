"""One brush stroke: spaces dabs along the cursor path, renders them through the
projector and applies the resulting stroke mask to every target channel."""

import numpy as np

from . import compositor, pixels, undo
from .channels import enabled_channels

SEAM_GROW = 3  # texels the stroke spreads past UV island borders


class Target:
    """One image the stroke writes: a layer channel (RGBA) or a layer mask."""

    def __init__(self, image, value, is_mask):
        self.image = image
        self.value = np.asarray(value, np.float32)
        self.is_mask = is_mask
        self.orig = pixels.read(image)
        self.current = self.orig.copy()


class Stroke:
    def __init__(self, projector, view, mat, brush, mirror=None):
        """view: (width, height, perspective_matrix, view_matrix, is_perspective).
        mirror: optional callable (x, y) -> (x, y) or None for symmetry dabs."""
        self.projector = projector
        self.mat = mat
        self.ts = ts = mat.paintstack
        self.layer = layer = ts.active_layer
        self.brush = brush
        self.mirror = mirror
        self.last = None
        self.dirty = False

        projector.begin_view(*view)
        projector.clear_mask()
        self.coverage = projector.coverage()

        erase = brush.mode == "ERASE"
        self.targets = []
        if layer.edit_target == "MASK" or layer.type == "FILL":
            if layer.mask_image is None:
                from . import ops
                ops.add_mask(mat, layer)
            self.targets.append(Target(layer.mask_image, 0.0 if erase else 1.0, True))
            self.channels = enabled_channels(ts)
        else:
            from . import ops
            self.channels = []
            for ch in enabled_channels(ts):
                lc = layer.channel(ch.id)
                if lc is None or not lc.enabled or not (erase or getattr(brush, "use_" + ch.key)):
                    continue
                if lc.image is None:
                    ops.ensure_layer_image(mat, layer, lc)
                self.targets.append(Target(lc.image, brush.value(ch), False))
                self.channels.append(ch)
        self.erase = erase

        # Layers below the active one never change during the stroke, so they
        # are flattened once; layers above are read once and re-applied on top.
        idx = ts.active_layer_index
        self.below, self.above = {}, {}
        for ch in self.channels:
            base = compositor.base_array(ch, ts.res)
            entries = [compositor.layer_entry(l, ch) for l in reversed(ts.layers[idx + 1:])]
            self.below[ch.id] = compositor.composite_entries(base, [e for e in entries if e])
            above = [compositor.layer_entry(l, ch) for l in reversed(ts.layers[:idx])]
            self.above[ch.id] = [e for e in above if e]

        undo.push(mat, {t.image.name: t.orig.copy() for t in self.targets})

    # Input

    def _dab(self, x, y, pressure):
        b = self.brush
        flow = b.flow * (pressure if b.use_pressure else 1.0)
        dabs = [(x, y, b.radius, b.hardness, flow)]
        if self.mirror:
            m = self.mirror(x, y)
            if m:
                dabs.append((m[0], m[1], b.radius, b.hardness, flow))
        self.projector.dabs(dabs)
        self.dirty = True

    def add_point(self, x, y, pressure=1.0):
        if self.last is None:
            self._dab(x, y, pressure)
            self.last = (x, y)
            return
        step = max(self.brush.radius * self.brush.spacing, 1.0)
        lx, ly = self.last
        dist = ((x - lx) ** 2 + (y - ly) ** 2) ** 0.5
        n = int(dist // step)
        for i in range(1, n + 1):
            t = i * step / dist
            self._dab(lx + (x - lx) * t, ly + (y - ly) * t, pressure)
        if n:
            t = n * step / dist
            self.last = (lx + (x - lx) * t, ly + (y - ly) * t)

    # Output

    def update(self):
        """Apply the accumulated stroke mask and refresh the composited outputs."""
        if not self.dirty:
            return
        self.dirty = False
        m = self.projector.read_mask()
        m = pixels.grow_mask(m, self.coverage, SEAM_GROW) * self.brush.opacity

        overrides = {}
        for t in self.targets:
            o = t.orig
            if t.is_mask:
                val = o[..., :3] + (t.value - o[..., :3]) * m[..., None]
                t.current = pixels.rgba(val, np.ones_like(m))
            elif self.erase:
                t.current = pixels.rgba(o[..., :3], o[..., 3] * (1.0 - m))
            else:
                # Colour is blended "over" the existing content so coverage and
                # colour stay consistent (un-premultiplied RGBA).
                a_new = o[..., 3] + m * (1.0 - o[..., 3])
                safe = np.maximum(a_new, 1e-6)[..., None]
                rgb = (t.value * m[..., None] + o[..., :3] * (o[..., 3] * (1.0 - m))[..., None]) / safe
                t.current = pixels.rgba(np.where(a_new[..., None] > 0, rgb, o[..., :3]), a_new)
            pixels.write(t.image, t.current)
            overrides[t.image.name] = t.current

        for ch in self.channels:
            entry = compositor.layer_entry(self.layer, ch, overrides)
            base = self.below[ch.id]
            if entry:
                base = compositor.mix(base, entry)
            compositor.write_output(self.ts, ch, compositor.composite_entries(base, self.above[ch.id]))

    def finish(self):
        self.update()
