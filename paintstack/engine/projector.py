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


def read_texture(tex, channels=1):
    buf = tex.read()
    try:
        arr = np.asarray(buf, np.float32)
    except (TypeError, ValueError):
        arr = np.array(buf.to_list(), np.float32)
    h, w = tex.height, tex.width
    return arr.reshape(h, w, -1)[..., :channels].squeeze()


class Projector:
    def __init__(self, obj, depsgraph, material, res):
        self.res = res
        pos, nor, uv = mesh_arrays(obj, depsgraph, material)
        self.key = self.make_key(obj, material, res)
        self.depth_batch = batch_for_shader(shaders.depth_shader(), "TRIS", {"pos": pos})
        self.uv_batch = batch_for_shader(shaders.dab_shader(), "TRIS",
                                         {"pos": pos, "uv": uv, "nor": nor})
        self.mask = gpu.types.GPUOffScreen(res, res, format="RGBA32F")
        self.depth = None
        self.view = None
        self._coverage = None

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
            "viewProj": perspective_matrix,
            "viewO": (*origin, 1.0 if is_perspective else 0.0),
            "viewF": (*forward, depth_bias),
            "size": (width, height),
        }
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
            self._set_view(sh)
            self.depth_batch.draw(sh)
            gpu.state.depth_test_set("NONE")
            gpu.state.depth_mask_set(False)

    def _set_view(self, sh):
        sh.uniform_float("viewProj", self.view["viewProj"])
        sh.uniform_float("viewO", self.view["viewO"])
        sh.uniform_float("viewF", self.view["viewF"])

    # Stroke mask

    def clear_mask(self):
        with self.mask.bind():
            gpu.state.active_framebuffer_get().clear(color=(0.0, 0.0, 0.0, 0.0))

    def dabs(self, dabs):
        """Render brush dabs [(x, y, radius, hardness, flow), ...] into the stroke mask."""
        if not dabs:
            return
        sh = shaders.dab_shader()
        w, h = self.view["size"]
        with self.mask.bind():
            gpu.state.blend_set("ALPHA")
            gpu.state.depth_test_set("NONE")
            gpu.state.face_culling_set("NONE")
            sh.bind()
            self._set_view(sh)
            sh.uniform_sampler("depthTex", self.depth.texture_color)
            for x, y, radius, hardness, flow in dabs:
                sh.uniform_float("brush", (x, y, max(radius, 0.5), hardness))
                sh.uniform_float("region", (w, h, flow, 0.0))
                self.uv_batch.draw(sh)
            gpu.state.blend_set("NONE")

    def read_mask(self):
        return read_texture(self.mask.texture_color)

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
