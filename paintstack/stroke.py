"""One brush stroke: spaces dabs along the cursor path, renders them through the
projector and applies the resulting stroke mask to every target channel."""

import math

import numpy as np

from . import brushes, compositor, hardware, pixels, stack, undo
from .channels import enabled_channels

SEAM_GROW = 3  # texels the stroke spreads past UV island borders


class Target:
    """One image the stroke writes. mode: MASK (opaque grey mask), OVER (RGBA
    content painted over what is there) or ERASE (remove RGBA coverage)."""

    def __init__(self, image, value, mode, cache=None):
        self.image = image
        self.value = np.asarray(value, np.float32)
        self.mode = mode
        self.orig = compositor.read(image, None, cache)  # shared, never modified
        self.current = self.orig.copy()


class Stroke:
    def __init__(self, projector, view, mat, brush, mirror=None, seed=None, px_scale=1.0, live=None):
        """view: (width, height, perspective_matrix, view_matrix, is_perspective).
        mirror: optional callable (x, y) -> (x, y) or None for symmetry dabs.
        px_scale: physical pixels per UI pixel (2 on Retina); brush sizes are UI pixels.
        live: write stroke regions straight into the viewport's GPU textures
        (None = the add-on preference)."""
        compositor.sync_outputs(mat)  # strokes start from the CPU images
        self.px_scale = px_scale
        self.live = hardware.live_preview_enabled() if live is None else live
        self.live_channels = set()
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
                                  brush.stencil_y, brush.stencil_size * px_scale, brush.stencil_rotation)
        projector.clear_mask()
        self.coverage = projector.coverage()

        from . import ops
        erase = brush.mode == "ERASE"
        self.full_update = False
        self.cache = compositor.state(mat).cache
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
                self.targets.append(Target(effect.image, (v, v, v), "OVER", self.cache))
            else:
                if layer.mask_image is None:
                    # A new black mask hides the layer everywhere, not just under
                    # the brush: the first update must recomposite the whole texture.
                    ops.add_mask(mat, layer)
                    self.full_update = True
                self.targets.append(Target(layer.mask_image, 0.0 if erase else 1.0, "MASK", self.cache))
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
                self.targets.append(Target(lc.image, brush.value(ch), "ERASE" if erase else "OVER",
                                           self.cache))
                self.channels.append(ch)

        # Only the touched texture regions are recomposited; every other layer's
        # entries and masks come from the material's persistent cache.
        self.tree = stack.tree(ts)
        self.anchors = bool(compositor.anchor_channels(ts))
        self.pad = compositor.filter_reach(ts)
        self.outputs = {}

        undo.push(mat, {t.image.name: t.orig for t in self.targets})

    # Input

    def _dab(self, x, y, pressure):
        b, rng = self.brush, self.rng
        radius = b.radius * self.px_scale * (1.0 + b.size_jitter * 0.9 * (rng.random() * 2.0 - 1.0))
        flow = b.flow * (pressure if b.use_pressure else 1.0) * (1.0 - b.flow_jitter * rng.random())
        angle = b.angle + b.angle_jitter * math.pi * (rng.random() * 2.0 - 1.0)
        if b.follow_stroke:
            angle += self.direction
        if b.scatter > 0.0:
            r = b.radius * self.px_scale * b.scatter * math.sqrt(rng.random())
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
        r = self.brush.lazy_radius * self.px_scale if lazy else 0
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
        step = max(self.brush.radius * self.px_scale * self.brush.spacing, 1.0)
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

    def _apply_targets(self, m, region):
        """Write the stroke mask m (covering region) into every target's current pixels."""
        y0, y1, x0, x1 = region
        for t in self.targets:
            o = t.orig[y0:y1, x0:x1]
            cur = t.current[y0:y1, x0:x1]
            if t.mode == "MASK":
                cur[..., :3] = o[..., :3] + (t.value - o[..., :3]) * m[..., None]
                cur[..., 3] = 1.0
            elif t.mode == "ERASE":
                cur[..., :3] = o[..., :3]
                cur[..., 3] = o[..., 3] * (1.0 - m)
            else:
                # Colour is blended "over" the existing content so coverage and
                # colour stay consistent (un-premultiplied RGBA).
                a_new = o[..., 3] + m * (1.0 - o[..., 3])
                safe = np.maximum(a_new, 1e-6)[..., None]
                rgb = (t.value * m[..., None] + o[..., :3] * (o[..., 3] * (1.0 - m))[..., None]) / safe
                cur[..., :3] = np.where(a_new[..., None] > 0, rgb, o[..., :3])
                cur[..., 3] = a_new
            if not t.image.is_float:
                # 8-bit images store 1/255 steps: quantise now so the live outputs,
                # the cache and the saved image all agree.
                np.round(cur * 255.0, out=cur)
                cur /= 255.0

    def _output(self, ch):
        """Full RGBA output array, kept for the stroke and pasted into per region."""
        if ch.id not in self.outputs:
            self.outputs[ch.id] = pixels.read(self.ts.output(ch.id)).copy()
        return self.outputs[ch.id]

    def update(self):
        """Apply the stroke mask in the regions touched since the last update and
        refresh the composited outputs there."""
        if not self.dirty:
            return
        self.dirty = False
        res = self.ts.res
        grow = SEAM_GROW + 1
        regions = self.projector.take_regions(margin=grow)
        if self.pad is None or self.anchors or self.full_update:
            regions = [(0, res, 0, res)]  # warp filters / anchors / new masks: whole texture
            self.full_update = False
        if not regions:
            return
        self._composite_regions(regions)

    def _composite_regions(self, regions):
        res = self.ts.res
        grow = SEAM_GROW + 1
        overrides = {t.image.name: t.current for t in self.targets}
        channels = enabled_channels(self.ts) if self.anchors else self.channels

        for region in regions:
            y0, y1, x0, x1 = region
            # Work on a padded region so filters and seam growth see real neighbours,
            # then keep only the inner part.
            p = grow + (self.pad or 0)
            wy0, wy1, wx0, wx1 = max(0, y0 - p), min(res, y1 + p), max(0, x0 - p), min(res, x1 + p)
            work = (wy0, wy1, wx0, wx1)
            m = self.projector.read_mask(work)
            m = pixels.grow_mask(m, self.coverage[wy0:wy1, wx0:wx1], SEAM_GROW) * self.brush.opacity
            self._apply_targets(m, work)

            full = work == (0, res, 0, res)
            if self.anchors:
                results = compositor.composite_all(self.ts, channels, overrides, self.cache,
                                                   None if full else work)
            else:
                results = {ch.id: compositor.composite_nodes(
                    compositor.base_array(self.ts, ch, work), self.tree, ch, overrides, self.cache, work)
                    for ch in channels}
            iy, ix = slice(y0 - wy0, y1 - wy0), slice(x0 - wx0, x1 - wx0)
            for ch in channels:
                self._output(ch)[y0:y1, x0:x1, :3] = compositor.finalize(ch, results[ch.id][iy, ix])

        if self.live:
            try:
                from .engine import live
                for ch in channels:
                    for region in regions:
                        live.push(self.ts.output(ch.id), self._output(ch), region)
                    self.live_channels.add(ch.id)
            except Exception as e:  # unsupported backend / format: fall back for good
                print(f"PaintStack: GPU live preview disabled ({e})")
                self.live = False
        if not self.live:
            for ch in channels:
                pixels.write(self.ts.output(ch.id), self._output(ch))
        if self.ts.view_mode == "MASK":
            compositor.write_mask_view(self.mat, self.cache, overrides)

    def finish(self):
        self.update()
        # The live preview only touched GPU textures: store the outputs once.
        for cid in self.live_channels:
            pixels.write(self.ts.output(cid), self.outputs[cid])
        self.live_channels.clear()
        # Layer images are written once, at the end of the stroke; the cache keeps
        # the new pixels so the next composite does not have to read them back.
        uid = self.layer.uid
        compositor.invalidate(self.mat, uid)
        for t in self.targets:
            pixels.write(t.image, t.current)
            self.cache[("px", t.image.name)] = t.current
