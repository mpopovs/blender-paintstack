"""One brush stroke: spaces dabs along the cursor path, renders them through the
projector and applies the resulting stroke mask to every target channel."""

import math

import numpy as np

from . import brushes, compositor, pixels, stack, undo
from .channels import enabled_channels

SEAM_GROW = 3  # texels the stroke spreads past UV island borders


class Target:
    """One image the stroke writes. mode: MASK (opaque grey mask), OVER (RGBA
    content painted over what is there) or ERASE (remove RGBA coverage)."""

    def __init__(self, image, value, mode):
        self.image = image
        self.value = np.asarray(value, np.float32)
        self.mode = mode
        self.orig = pixels.read(image)
        self.current = self.orig.copy()


class Stroke:
    def __init__(self, projector, view, mat, brush, mirror=None, seed=None):
        """view: (width, height, perspective_matrix, view_matrix, is_perspective).
        mirror: optional callable (x, y) -> (x, y) or None for symmetry dabs."""
        self.projector = projector
        self.mat = mat
        self.ts = ts = mat.paintstack
        self.layer = layer = ts.active_layer
        self.brush = brush
        self.mirror = mirror
        self.rng = np.random.default_rng(seed)
        self.last = None  # last dab position
        self.pen = None  # lazy-mouse position (= cursor when lazy mouse is off)
        self.direction = 0.0
        self.dirty = False

        projector.begin_view(*view)
        projector.set_tip(brushes.tip_array(brush))
        if brush.use_stencil and brush.stencil_image is not None:
            projector.set_stencil(brushes.image_luminance(brush.stencil_image), brush.stencil_x,
                                  brush.stencil_y, brush.stencil_size, brush.stencil_rotation)
        projector.clear_mask()
        self.coverage = projector.coverage()

        from . import ops
        erase = brush.mode == "ERASE"
        self.targets = []
        if layer.edit_target == "MASK" or layer.type in ("FILL", "FOLDER", "FILTER"):
            effect = layer.active_mask_effect
            if (effect is None or effect.type != "PAINT") and layer.mask_effects:
                # Painting under generators would be hidden by them: paint on top instead.
                effect = ops.add_mask_effect(mat, layer, "PAINT")
            if effect is not None and effect.type == "PAINT":
                if effect.image is None:
                    ops.ensure_effect_image(mat, layer, effect)
                v = 0.0 if erase else 1.0
                self.targets.append(Target(effect.image, (v, v, v), "OVER"))
            else:
                if layer.mask_image is None:
                    ops.add_mask(mat, layer)
                self.targets.append(Target(layer.mask_image, 0.0 if erase else 1.0, "MASK"))
            layer.edit_target = "MASK"
            self.channels = enabled_channels(ts)
        else:
            self.channels = []
            for ch in enabled_channels(ts):
                lc = layer.channel(ch.id)
                if lc is None or not lc.enabled or not (erase or getattr(brush, "use_" + ch.key)):
                    continue
                if lc.image is None:
                    ops.ensure_layer_image(mat, layer, lc)
                self.targets.append(Target(lc.image, brush.value(ch), "ERASE" if erase else "OVER"))
                self.channels.append(ch)

        # Everything except the active layer is unchanged during the stroke: its
        # entries and masks are cached. For a root-level layer the stack below it is
        # also flattened once, so updates only re-blend the active layer and above.
        self.cache = {}
        self.tree = stack.tree(ts)
        self.below = None
        # Anchors make masks depend on other channels' live results: recomposite
        # everything (in anchor order) on each update instead of caching.
        self.anchors = bool(compositor.anchor_channels(ts))
        roots = [node[0].uid for node in self.tree]
        if layer.uid in roots and layer.type in ("PAINT", "FILL") and not self.anchors:
            i = roots.index(layer.uid)
            self.above = self.tree[:i]
            self.below = {ch.id: compositor.composite_nodes(compositor.base_array(ts, ch),
                                                            self.tree[i + 1:], ch, cache=self.cache)
                          for ch in self.channels}

        undo.push(mat, {t.image.name: t.orig.copy() for t in self.targets})

    # Input

    def _dab(self, x, y, pressure):
        b, rng = self.brush, self.rng
        radius = b.radius * (1.0 + b.size_jitter * 0.9 * (rng.random() * 2.0 - 1.0))
        flow = b.flow * (pressure if b.use_pressure else 1.0) * (1.0 - b.flow_jitter * rng.random())
        angle = b.angle + b.angle_jitter * math.pi * (rng.random() * 2.0 - 1.0)
        if b.follow_stroke:
            angle += self.direction
        if b.scatter > 0.0:
            r = b.radius * b.scatter * math.sqrt(rng.random())
            a = rng.random() * 2.0 * math.pi
            x, y = x + r * math.cos(a), y + r * math.sin(a)
        dabs = [(x, y, radius, b.hardness, flow, angle)]
        if self.mirror:
            m = self.mirror(x, y)
            if m:
                dabs.append((m[0], m[1], radius, b.hardness, flow, math.pi - angle))
        self.projector.dabs(dabs)
        self.dirty = True

    def add_point(self, x, y, pressure=1.0, lazy=True):
        """Feed a cursor position. With lazy mouse, the brush trails the cursor on a
        string of length lazy_radius and only moves when the string is taut."""
        r = self.brush.lazy_radius if lazy else 0
        if self.pen is None or r <= 0:
            self.pen = (x, y)
        else:
            px, py = self.pen
            d = math.hypot(x - px, y - py)
            if d <= r:
                return
            k = (d - r) / d
            self.pen = (px + (x - px) * k, py + (y - py) * k)
        x, y = self.pen

        if self.last is None:
            self._dab(x, y, pressure)
            self.last = (x, y)
            return
        step = max(self.brush.radius * self.brush.spacing, 1.0)
        lx, ly = self.last
        dist = math.hypot(x - lx, y - ly)
        if dist > 1e-6:
            self.direction = math.atan2(y - ly, x - lx)
        n = int(dist // step)
        for i in range(1, n + 1):
            t = i * step / dist
            self._dab(lx + (x - lx) * t, ly + (y - ly) * t, pressure)
        if n:
            t = n * step / dist
            self.last = (lx + (x - lx) * t, ly + (y - ly) * t)

    def line(self, x0, y0, x1, y1, pressure=1.0):
        """A straight line of dabs (Shift+click), ignoring lazy mouse."""
        self.add_point(x0, y0, pressure, lazy=False)
        self.add_point(x1, y1, pressure, lazy=False)

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
            if t.mode == "MASK":
                val = o[..., :3] + (t.value - o[..., :3]) * m[..., None]
                t.current = pixels.rgba(val, np.ones_like(m))
            elif t.mode == "ERASE":
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

        if self.anchors:
            # Any channel can depend on the painted one through an anchor mask.
            results = compositor.composite_all(self.ts, None, overrides)
            for ch in enabled_channels(self.ts):
                compositor.write_output(self.ts, ch, results[ch.id])
            if self.ts.view_mode == "MASK":
                compositor.write_mask_view(self.mat)
            return

        # Drop the active layer's cached entries and mask: its images just changed.
        uid = self.layer.uid
        for key in [k for k in self.cache if k[1] == uid]:
            del self.cache[key]
        for ch in self.channels:
            if self.below is not None:
                entry = compositor.layer_entry(self.layer, ch, overrides, self.cache)
                base = self.below[ch.id]
                if entry:
                    base = compositor.mix(base, entry)
                out = compositor.composite_nodes(base, self.above, ch, overrides, self.cache)
            else:
                out = compositor.composite_nodes(compositor.base_array(self.ts, ch), self.tree, ch,
                                                 overrides, self.cache)
            compositor.write_output(self.ts, ch, out)
        if self.ts.view_mode == "MASK":
            compositor.write_mask_view(self.mat, self.cache)

    def finish(self):
        self.update()
