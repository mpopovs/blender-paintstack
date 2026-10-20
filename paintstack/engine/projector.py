"""Screen-to-UV projection painting on the GPU.

A stroke is accumulated into a single-channel "stroke mask" in texture space.
The CPU side then uses that mask to write every target channel at once, which
is what gives multi-channel (PBR) painting in a single stroke."""

import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from . import shaders


def mesh_arrays(obj, depsgraph, material):
    """World-space positions, normals and UVs per triangle corner, restricted to
    the faces that use `material`."""
    eval_obj = obj.evaluated_get(depsgraph)
    me = eval_obj.to_mesh()
    try:
        if me.uv_layers.active is None:
            raise RuntimeError(f"'{obj.name}' has no UV map")
        me.calc_loop_triangles()
        n_tri, n_loop = len(me.loop_triangles), len(me.loops)

        tris = np.empty(n_tri * 3, np.int32)
        me.loop_triangles.foreach_get("loops", tris)
        mat_idx = np.empty(n_tri, np.int32)
        me.loop_triangles.foreach_get("material_index", mat_idx)
        slots = [i for i, m in enumerate(eval_obj.material_slots) if m.material == material]
        if slots and len(eval_obj.material_slots) > 1:
            tris = tris.reshape(-1, 3)[np.isin(mat_idx, slots)].ravel()

        vidx = np.empty(n_loop, np.int32)
        me.loops.foreach_get("vertex_index", vidx)
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        uv = np.empty(n_loop * 2, np.float32)
        me.uv_layers.active.data.foreach_get("uv", uv)
        nor = np.empty(n_loop * 3, np.float32)
        if hasattr(me, "corner_normals"):
            me.corner_normals.foreach_get("vector", nor)
        else:  # Blender < 4.1
            me.calc_normals_split()
            me.loops.foreach_get("normal", nor)
    finally:
        eval_obj.to_mesh_clear()

    mw = np.array(obj.matrix_world, np.float32)
    nm = np.linalg.inv(mw[:3, :3]).T
    pos = co.reshape(-1, 3)[vidx[tris]] @ mw[:3, :3].T + mw[:3, 3]
    nor = nor.reshape(-1, 3)[tris] @ nm.T
    nor /= np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-8)
    return (np.ascontiguousarray(pos, np.float32), np.ascontiguousarray(nor, np.float32),
            np.ascontiguousarray(uv.reshape(-1, 2)[tris], np.float32))


def cells_to_regions(occ, res, margin=0):
    """Bounding rectangles (y0, y1, x0, x1) in texels of the 8-connected clusters of
    True cells in a coarse occupancy grid covering the texture."""
    cells = occ.shape[0]
    cell = res / cells
    regions = []
    seen = np.zeros_like(occ)
    for cy, cx in zip(*np.nonzero(occ)):
        if seen[cy, cx]:
            continue
        stack, ys, xs = [(cy, cx)], [], []
        seen[cy, cx] = True
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if 0 <= ny < cells and 0 <= nx < cells and occ[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        regions.append((max(0, int(min(ys) * cell) - margin), min(res, int((max(ys) + 1) * cell) + margin),
                        max(0, int(min(xs) * cell) - margin), min(res, int((max(xs) + 1) * cell) + margin)))
    return regions


def make_texture(arr, fmt):
    """GPU texture from a float array of shape (h, w) or (h, w, c)."""
    h, w = arr.shape[:2]
    data = np.ascontiguousarray(arr, np.float32).ravel()
    return gpu.types.GPUTexture((w, h), format=fmt, data=gpu.types.Buffer("FLOAT", data.size, data))


_white_tex = None


def _white():
    global _white_tex
    if _white_tex is None:
        _white_tex = make_texture(np.ones((1, 1), np.float32), "R16F")
    return _white_tex


_buffer_path = None  # how buffer_array last succeeded (reported by Diagnostics)


def buffer_array(buf, count):
    """Flat float32 array of the first `count` values of a gpu.types.Buffer.

    Fast paths view the buffer's memory; each is checked for size, because some
    Blender versions (5.1) expose only part of a multi-dimensional readback
    buffer to numpy (one float per pixel instead of four)."""
    global _buffer_path
    try:
        buf.dimensions = count  # flat view of the whole buffer
    except Exception:  # noqa: BLE001
        pass
    for path, fn in (("asarray", lambda: np.asarray(buf, np.float32).ravel()),
                     ("frombuffer", lambda: np.frombuffer(buf, np.float32))):
        try:
            arr = fn()
        except (TypeError, ValueError, BufferError):
            continue
        if arr.size == count:
            _buffer_path = path
            return arr
    _buffer_path = "to_list"
    return np.array(buf.to_list(), np.float32).ravel()[:count]


def read_texture(tex, channels=1):
    h, w = tex.height, tex.width
    # Every texture read back here is RGBA (offscreens and render targets).
    arr = buffer_array(tex.read(), h * w * 4)
    return arr.reshape(h, w, 4)[..., :channels].squeeze()


class Projector:
    def __init__(self, obj, depsgraph, material, res):
        pos, nor, uv = mesh_arrays(obj, depsgraph, material)
        self._setup(pos, nor, uv, res, self.make_key(obj, material, res))

    @classmethod
    def from_arrays(cls, pos, nor, uv, res):
        """Projector for raw per-corner arrays (used by diagnostics and tests)."""
        p = cls.__new__(cls)
        p._setup(np.ascontiguousarray(pos, np.float32), np.ascontiguousarray(nor, np.float32),
                 np.ascontiguousarray(uv, np.float32), res, None)
        return p

    def _setup(self, pos, nor, uv, res, key):
        self.res = res
        self.key = key
        self.depth_batch = batch_for_shader(shaders.depth_shader(), "TRIS", {"pos": pos})
        self.uv_batch = batch_for_shader(shaders.dab_shader(), "TRIS",
                                         {"pos": pos, "uv": uv, "nor": nor})
        # Half float: blendable on every Metal GPU (32-bit float blending is not
        # guaranteed), and half the readback bandwidth.
        self.mask = gpu.types.GPUOffScreen(res, res, format="RGBA16F")
        self.depth = None
        self.view = None
        self._coverage = None
        # Per-triangle data for finding the texture regions a dab can touch.
        self._tri_pos = pos.reshape(-1, 3, 3)
        uvt = uv.reshape(-1, 3, 2)
        self._tri_uv_min = uvt.min(1)
        self._tri_uv_max = uvt.max(1)
        # Facing uses the same (corner) normals as the dab shader, not the winding,
        # so flipped winding or custom normals cannot disagree with what is painted.
        self._tri_normal = nor.reshape(-1, 3, 3).sum(1)
        self._pending = []

    @staticmethod
    def make_key(obj, material, res):
        return (obj.name, material.name, res, tuple(map(tuple, obj.matrix_world)),
                len(obj.data.vertices), len(obj.data.polygons))

    def free(self):
        self.mask.free()
        if self.depth:
            self.depth.free()

    # View

    def begin_view(self, width, height, perspective_matrix, view_matrix, is_perspective,
                   depth_bias=0.002):
        """Capture the view for the next stroke and render its depth prepass."""
        inv = view_matrix.inverted()
        origin = inv.translation
        forward = -(inv.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()
        self.view = {
            "viewProj": np.array(perspective_matrix, np.float32),
            "viewO": (*origin, 1.0 if is_perspective else 0.0),
            "viewF": (*forward, depth_bias),
            "size": (width, height),
        }
        self.tip = None
        self.stencil = None
        self._upload_view()
        self._screen_boxes(width, height, perspective_matrix, origin, forward, is_perspective)
        self._pending = []
        if self.depth is None or (self.depth.width, self.depth.height) != (width, height):
            if self.depth:
                self.depth.free()
            self.depth = gpu.types.GPUOffScreen(width, height, format="RGBA32F")

        sh = shaders.depth_shader()
        with self.depth.bind():
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(1e9, 0.0, 0.0, 1.0), depth=1.0)
            gpu.state.depth_test_set("LESS_EQUAL")
            gpu.state.depth_mask_set(True)
            sh.bind()
            sh.uniform_block("vp", self.ubo)
            self.depth_batch.draw(sh)
            gpu.state.depth_test_set("NONE")
            gpu.state.depth_mask_set(False)

    def _screen_boxes(self, width, height, persp, origin, forward, is_perspective):
        """Screen-space bounding box and facing of every triangle for this view."""
        vp = np.array(persp, np.float32)
        p = self._tri_pos
        clip = p @ vp[:3, :3].T + vp[:3, 3]
        w = p @ vp[3, :3] + vp[3, 3]
        visible = (w > 1e-6).all(1)
        w = np.where(w > 1e-6, w, 1.0)
        sx = (clip[..., 0] / w * 0.5 + 0.5) * width
        sy = (clip[..., 1] / w * 0.5 + 0.5) * height
        centre = p.mean(1)
        view = (np.asarray(origin, np.float32) - centre) if is_perspective \
            else -np.asarray(forward, np.float32)[None, :]
        facing = (self._tri_normal * view).sum(1) > 0
        self._scr_min = np.stack([sx.min(1), sy.min(1)], 1)
        self._scr_max = np.stack([sx.max(1), sy.max(1)], 1)
        self._tri_ok = visible & facing

    def regions_for(self, dabs, margin=0):
        """Texture regions (y0, y1, x0, x1) that dabs [(x, y, radius), ...] can touch,
        as a few rectangles (one per cluster of touched UV area)."""
        d = np.asarray(dabs, np.float32).reshape(-1, 3)
        if not len(d):
            return []
        lo = d[:, :2] - d[:, 2:3]
        hi = d[:, :2] + d[:, 2:3]
        hit = ((self._scr_min[None] <= hi[:, None]) & (self._scr_max[None] >= lo[:, None])).all(-1).any(0)
        hit &= self._tri_ok
        if not hit.any():
            return []
        res = self.res
        cells = 64
        x0 = np.clip(np.floor(self._tri_uv_min[hit, 0] * cells), 0, cells - 1).astype(np.int64)
        y0 = np.clip(np.floor(self._tri_uv_min[hit, 1] * cells), 0, cells - 1).astype(np.int64)
        x1 = np.clip(np.floor(self._tri_uv_max[hit, 0] * cells), 0, cells - 1).astype(np.int64)
        y1 = np.clip(np.floor(self._tri_uv_max[hit, 1] * cells), 0, cells - 1).astype(np.int64)
        # Mark the cells covered by each triangle's UV box (2D difference array).
        diff = np.zeros((cells + 1, cells + 1), np.int64)
        np.add.at(diff, (y0, x0), 1)
        np.add.at(diff, (y0, x1 + 1), -1)
        np.add.at(diff, (y1 + 1, x0), -1)
        np.add.at(diff, (y1 + 1, x1 + 1), 1)
        occ = diff.cumsum(0).cumsum(1)[:cells, :cells] > 0
        return cells_to_regions(occ, res, margin)

    def take_regions(self, margin=0):
        """Regions touched by the dabs since the last call (see regions_for)."""
        pending, self._pending = self._pending, []
        return self.regions_for(pending, margin)

    def _upload_view(self):
        v = self.view
        w, h = v["size"]
        st = self.stencil
        data = np.concatenate([
            v["viewProj"].T.ravel(),  # GLSL matrices are column-major
            v["viewO"], v["viewF"],
            (w, h, 1.0 if self.tip is not None else 0.0, 1.0 if st else 0.0),
            st["transform"] if st else (0.0, 0.0, 1.0, 0.0),
            (st["aspect"] if st else 1.0, 0.0, 0.0, 0.0),
        ]).astype(np.float32)
        if getattr(self, "ubo", None) is None:
            self.ubo = gpu.types.GPUUniformBuf(data.tobytes())
        else:
            self.ubo.update(data.tobytes())

    def set_tip(self, tip):
        """tip: (n, n) float array (brush-local, bottom row first) or None for the round falloff."""
        # R16F: linearly filterable on every backend (R32F filtering is optional on Metal).
        self.tip = None if tip is None else make_texture(tip, "R16F")
        self._upload_view()

    def set_stencil(self, image, x, y, half_height, rotation):
        """image: (h, w) float array, or None to disable. Placement is in region pixels."""
        if image is None:
            self.stencil = None
        else:
            h, w = image.shape[:2]
            self.stencil = {"tex": make_texture(image, "R16F"), "aspect": w / h,
                            "transform": (x, y, max(half_height, 1.0), rotation)}
        self._upload_view()

    # Stroke mask

    def clear_mask(self):
        with self.mask.bind():
            gpu.state.active_framebuffer_get().clear(color=(0.0, 0.0, 0.0, 0.0))

    def dabs(self, dabs):
        """Render brush dabs [(x, y, radius, hardness, flow, angle), ...] into the stroke mask."""
        if not dabs:
            return
        sh = shaders.dab_shader()
        white = _white()
        boxes = self.regions_for([(d[0], d[1], d[2]) for d in dabs], margin=2) if self.view else []
        if not boxes:
            return  # no visible, front-facing triangle under any dab: nothing to draw
        by0 = min(b[0] for b in boxes)
        by1 = max(b[1] for b in boxes)
        bx0 = min(b[2] for b in boxes)
        bx1 = max(b[3] for b in boxes)
        with self.mask.bind():
            # Only rasterize the part of the texture these dabs can reach.
            gpu.state.scissor_test_set(True)
            gpu.state.scissor_set(bx0, by0, bx1 - bx0, by1 - by0)
            gpu.state.blend_set("ALPHA")
            gpu.state.depth_test_set("NONE")
            gpu.state.face_culling_set("NONE")
            sh.bind()
            sh.uniform_block("vp", self.ubo)
            sh.uniform_sampler("depthTex", self.depth.texture_color)
            sh.uniform_sampler("tipTex", self.tip or white)
            sh.uniform_sampler("stencilTex", self.stencil["tex"] if self.stencil else white)
            for d in dabs:
                x, y, radius, hardness, flow = d[:5]
                self._pending.append((x, y, radius))
                angle = d[5] if len(d) > 5 else 0.0
                sh.uniform_float("brush", (x, y, max(radius, 0.5), hardness))
                sh.uniform_float("dab", (angle, flow, 0.0, 0.0))
                self.uv_batch.draw(sh)
            gpu.state.blend_set("NONE")
            # Leave no pointers to textures that are freed with this projector
            # (or with a deleted stencil / tip image) in the GPU state.
            for name in ("depthTex", "tipTex", "stencilTex"):
                sh.uniform_sampler(name, white)
            gpu.state.scissor_test_set(False)

    def read_mask(self, region=None):
        """Stroke mask (res, res), or just a region (y0, y1, x0, x1) of it."""
        if region is None:
            return read_texture(self.mask.texture_color)
        y0, y1, x0, x1 = region
        with self.mask.bind():
            buf = gpu.state.active_framebuffer_get().read_color(x0, y0, x1 - x0, y1 - y0, 4, 0, "FLOAT")
        arr = buffer_array(buf, (y1 - y0) * (x1 - x0) * 4)
        return arr.reshape(y1 - y0, x1 - x0, 4)[..., 0]

    def coverage(self):
        """Boolean (res, res) map of texels covered by the mesh's UV islands."""
        if self._coverage is None:
            off = gpu.types.GPUOffScreen(self.res, self.res, format="RGBA8")
            try:
                with off.bind():
                    gpu.state.active_framebuffer_get().clear(color=(0.0, 0.0, 0.0, 0.0))
                    sh = shaders.coverage_shader()
                    sh.bind()
                    self.uv_batch.draw(sh)
                self._coverage = read_texture(off.texture_color) > 0.5
            finally:
                off.free()
        return self._coverage
