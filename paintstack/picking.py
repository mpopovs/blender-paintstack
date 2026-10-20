"""Click on the model to pick an ID colour for a Color Selection mask."""

import bpy
import numpy as np
from bpy.types import Operator
from bpy_extras import view3d_utils
from mathutils import Vector
from mathutils.geometry import barycentric_transform, intersect_point_tri

from . import ops, pixels


def uv_at_ray(obj, depsgraph, origin, direction):
    """UV coordinate where a world-space ray first hits obj, or None."""
    mwi = obj.matrix_world.inverted()
    o = mwi @ origin
    d = (mwi.to_3x3() @ direction).normalized()
    hit, loc, _normal, face = obj.ray_cast(o, d, depsgraph=depsgraph)
    if not hit:
        return None
    eval_obj = obj.evaluated_get(depsgraph)
    me = eval_obj.to_mesh()
    try:
        uvl = me.uv_layers.active
        if uvl is None:
            return None
        me.calc_loop_triangles()
        poly = np.empty(len(me.loop_triangles), np.int32)
        me.loop_triangles.foreach_get("polygon_index", poly)
        best = None
        for ti in np.nonzero(poly == face)[0]:
            tri = me.loop_triangles[int(ti)]
            co = [me.vertices[me.loops[l].vertex_index].co.copy() for l in tri.loops]
            uv = [Vector((*uvl.data[l].uv, 0.0)) for l in tri.loops]
            if intersect_point_tri(loc, *co) is not None or best is None:
                best = barycentric_transform(loc, *co, *uv)
        return None if best is None else (best.x, best.y)
    finally:
        eval_obj.to_mesh_clear()


def sample(img, uv, grid=(1, 1)):
    """Colour of a texture-set image at a mesh UV. grid: the texture set's UDIM
    (columns, rows); images hold the tiles side by side (see PS_TextureSet.size)."""
    px = pixels.read(img)
    h, w = px.shape[:2]
    cols, rows = grid
    u = uv[0] / cols if cols > 1 else uv[0] % 1.0
    v = uv[1] / rows if rows > 1 else uv[1] % 1.0
    x = int(np.clip(u * w, 0, w - 1))
    y = int(np.clip(v * h, 0, h - 1))
    return tuple(float(c) for c in px[y, x, :3])


def _id_effect(context):
    mat, ts = ops.texture_set(context)
    if mat is None or ts.active_layer is None:
        return None, None
    e = ts.active_layer.active_mask_effect
    if e is None or e.type != "ID_MASK" or ts.baked_map("ID") is None:
        return None, None
    return ts, e


class PAINTSTACK_OT_id_pick(Operator):
    """Click on the model to select its ID colour (Esc to cancel)"""
    bl_idname = "paintstack.id_pick"
    bl_label = "Pick ID Color"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (_id_effect(context)[1] is not None and context.area is not None
                and context.area.type == "VIEW_3D")

    def invoke(self, context, event):
        self.area = context.area
        self.region = next(r for r in context.area.regions if r.type == "WINDOW")
        self.rv3d = context.area.spaces.active.region_3d
        context.window.cursor_modal_set("EYEDROPPER")
        context.workspace.status_text_set("PaintStack: click the model to pick an ID colour · Esc cancel")
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _end(self, context, result):
        context.window.cursor_modal_restore()
        context.workspace.status_text_set(None)
        return result

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            return self._end(context, {"CANCELLED"})
        if event.type != "LEFTMOUSE" or event.value != "PRESS":
            return {"RUNNING_MODAL"} if event.type in {"LEFTMOUSE", "MOUSEMOVE"} else {"PASS_THROUGH"}
        r = self.region
        co = (event.mouse_x - r.x, event.mouse_y - r.y)
        if not (0 <= co[0] < r.width and 0 <= co[1] < r.height):
            return {"RUNNING_MODAL"}
        ts, effect = _id_effect(context)
        if effect is None:
            return self._end(context, {"CANCELLED"})
        origin = view3d_utils.region_2d_to_origin_3d(r, self.rv3d, co)
        direction = view3d_utils.region_2d_to_vector_3d(r, self.rv3d, co)
        uv = uv_at_ray(context.active_object, context.evaluated_depsgraph_get(), origin, direction)
        if uv is None:
            self.report({"INFO"}, "Missed the model")
            return {"RUNNING_MODAL"}
        effect.id_color = sample(ts.baked_map("ID"), uv, ts.grid)
        self.area.tag_redraw()
        return self._end(context, {"FINISHED"})


classes = (PAINTSTACK_OT_id_pick,)
