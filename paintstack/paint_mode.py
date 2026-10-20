"""Interactive paint mode: a modal operator that owns LMB in the 3D viewport and
passes everything else through, so orbit/zoom/pan keep working while painting."""

import math

import bpy
import gpu
from bpy.types import Operator
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader

from . import ops, undo
from .engine.projector import Projector
from .stroke import Stroke

UPDATE_INTERVAL = 1.0 / 15.0

_projector = None  # reused between strokes while the mesh and view setup are unchanged


def _window_region(area):
    return next((r for r in area.regions if r.type == "WINDOW"), None)


def _over_ui(area, x, y):
    """True if the window-space point is over a sidebar/header/toolbar region."""
    for r in area.regions:
        if r.type != "WINDOW" and r.width > 1 and r.height > 1:
            if r.x <= x < r.x + r.width and r.y <= y < r.y + r.height:
                return True
    return False


def _draw_cursor(op):
    if bpy.context.region != op.region or op.mouse is None:
        return
    brush = bpy.context.scene.paintstack_brush
    x, y = op.mouse
    pts = [(x + brush.radius * math.cos(a), y + brush.radius * math.sin(a))
           for a in (i * 2.0 * math.pi / 48 for i in range(49))]
    inner = brush.radius * brush.hardness
    pts_inner = [(x + inner * math.cos(a), y + inner * math.sin(a))
                 for a in (i * 2.0 * math.pi / 48 for i in range(49))]
    sh = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    sh.bind()
    color = (1.0, 0.35, 0.35, 0.9) if brush.mode == "ERASE" else (1.0, 1.0, 1.0, 0.9)
    sh.uniform_float("color", color)
    batch_for_shader(sh, "LINE_STRIP", {"pos": pts}).draw(sh)
    sh.uniform_float("color", (*color[:3], 0.35))
    batch_for_shader(sh, "LINE_STRIP", {"pos": pts_inner}).draw(sh)
    gpu.state.blend_set("NONE")


class PAINTSTACK_OT_paint_mode(Operator):
    """Paint all enabled PBR channels of the active layer at once (Esc to exit)"""
    bl_idname = "paintstack.paint_mode"
    bl_label = "Start Painting"

    running = False

    @classmethod
    def poll(cls, context):
        mat, ts = ops.texture_set(context)
        return (mat is not None and ts.active_layer is not None and not cls.running
                and context.area is not None and context.area.type == "VIEW_3D")

    def invoke(self, context, event):
        self.area = context.area
        self.region = _window_region(context.area)
        self.rv3d = context.area.spaces.active.region_3d
        self.obj = context.active_object
        self.mat = self.obj.active_material
        self.stroke = None
        self.mouse = None
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        wm = context.window_manager
        self._timer = wm.event_timer_add(UPDATE_INTERVAL, window=context.window)
        self._draw = bpy.types.SpaceView3D.draw_handler_add(_draw_cursor, (self,), "WINDOW", "POST_PIXEL")
        wm.modal_handler_add(self)
        type(self).running = True
        context.workspace.status_text_set(
            "PaintStack: LMB paint · [ ] radius · X paint/erase · Ctrl+Z undo · Esc exit")
        return {"RUNNING_MODAL"}

    def _finish(self, context):
        if self.stroke:
            self.stroke.finish()
            self.stroke = None
        context.window_manager.event_timer_remove(self._timer)
        bpy.types.SpaceView3D.draw_handler_remove(self._draw, "WINDOW")
        context.workspace.status_text_set(None)
        type(self).running = False
        self.area.tag_redraw()
        return {"FINISHED"}

    def _local(self, event):
        return event.mouse_x - self.region.x, event.mouse_y - self.region.y

    def _projector(self, context):
        global _projector
        res = self.mat.paintstack.res
        key = Projector.make_key(self.obj, self.mat, res)
        if _projector is None or _projector.key != key:
            if _projector:
                _projector.free()
            _projector = Projector(self.obj, context.evaluated_depsgraph_get(), self.mat, res)
        return _projector

    def _mirror_fn(self, context):
        """Map a screen point to its mirror across the object's local X axis."""
        obj, region, rv3d = self.obj, self.region, self.rv3d
        mw = obj.matrix_world
        mwi = mw.inverted()
        dg = context.evaluated_depsgraph_get()

        def mirror(x, y):
            origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, (x, y))
            direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, (x, y))
            o = mwi @ origin
            d = (mwi.to_3x3() @ direction).normalized()
            hit, loc, *_ = obj.ray_cast(o, d, depsgraph=dg)
            if not hit:
                return None
            loc.x = -loc.x
            return view3d_utils.location_3d_to_region_2d(region, rv3d, mw @ loc)
        return mirror

    def _begin_stroke(self, context, x, y, pressure):
        brush = context.scene.paintstack_brush
        r = self.region
        view = (r.width, r.height, self.rv3d.perspective_matrix.copy(),
                self.rv3d.view_matrix.copy(), self.rv3d.is_perspective)
        try:
            self.stroke = Stroke(self._projector(context), view, self.mat, brush,
                                 self._mirror_fn(context) if brush.mirror_x else None)
        except RuntimeError as e:
            self.report({"ERROR"}, str(e))
            return
        self.stroke.add_point(x, y, pressure)

    def modal(self, context, event):
        if self.region is None or context.active_object != self.obj:
            return self._finish(context)
        brush = context.scene.paintstack_brush
        self.area.tag_redraw()

        if event.type == "TIMER":
            if self.stroke:
                self.stroke.update()
            return {"PASS_THROUGH"}

        x, y = self._local(event)
        inside = (0 <= x < self.region.width and 0 <= y < self.region.height
                  and not _over_ui(self.area, event.mouse_x, event.mouse_y))
        self.mouse = (x, y) if inside else None

        if event.type == "MOUSEMOVE":
            if self.stroke:
                self.stroke.add_point(x, y, event.pressure or 1.0)
                return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}

        if event.type == "LEFTMOUSE":
            if event.value == "PRESS" and inside and not (event.alt or event.ctrl):
                self._begin_stroke(context, x, y, event.pressure or 1.0)
                return {"RUNNING_MODAL"}
            if event.value == "RELEASE" and self.stroke:
                self.stroke.finish()
                self.stroke = None
                return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}

        if self.stroke:
            return {"RUNNING_MODAL"}

        if event.value == "PRESS" and inside:
            if event.type == "ESC":
                return self._finish(context)
            if event.type == "LEFT_BRACKET":
                brush.radius = max(1, int(brush.radius / 1.15))
                return {"RUNNING_MODAL"}
            if event.type == "RIGHT_BRACKET":
                brush.radius = int(brush.radius * 1.15) + 1
                return {"RUNNING_MODAL"}
            if event.type == "X" and not (event.ctrl or event.alt or event.shift):
                brush.mode = "PAINT" if brush.mode == "ERASE" else "ERASE"
                return {"RUNNING_MODAL"}
            if event.type == "Z" and event.ctrl:
                undo.redo() if event.shift else undo.undo()
                return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}


classes = (PAINTSTACK_OT_paint_mode,)
