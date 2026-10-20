"""Interactive paint mode: a modal operator that owns LMB in the 3D viewport and
passes everything else through, so orbit/zoom/pan keep working while painting."""

import math

import bpy
import gpu
from bpy.types import Operator
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader

from . import hardware, ops, undo
from .engine.projector import Projector
from .stroke import Stroke

UPDATE_INTERVAL = 1.0 / 15.0
def status_text():
    mod = "⌘" if hardware.is_mac() else "Ctrl+"
    return ("PaintStack: LMB paint · Shift+LMB line · F / [ ] radius · X paint/erase · "
            f"S+move/wheel/{mod}wheel stencil · C channel view · M material · {mod}Z undo · Esc exit")

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


def _circle(x, y, r, n=48):
    return [(x + r * math.cos(a), y + r * math.sin(a)) for a in (i * 2.0 * math.pi / n for i in range(n + 1))]


def stencil_corners(brush, aspect):
    """Screen-space corners of the stencil quad (counter-clockwise from bottom-left)."""
    h = brush.stencil_size * hardware.pixel_size()
    w = h * aspect
    c, s = math.cos(brush.stencil_rotation), math.sin(brush.stencil_rotation)
    out = []
    for dx, dy in ((-w, -h), (w, -h), (w, h), (-w, h)):
        out.append((brush.stencil_x + c * dx - s * dy, brush.stencil_y + s * dx + c * dy))
    return out


def _draw_stencil(brush):
    img = brush.stencil_image
    if not (brush.use_stencil and img):
        return
    try:
        tex = gpu.texture.from_image(img)
    except Exception:
        return
    aspect = img.size[0] / max(img.size[1], 1)
    sh = gpu.shader.from_builtin("IMAGE_COLOR")
    batch = batch_for_shader(sh, "TRI_FAN", {"pos": stencil_corners(brush, aspect),
                                             "texCoord": ((0, 0), (1, 0), (1, 1), (0, 1))})
    gpu.state.blend_set("ALPHA")
    sh.bind()
    sh.uniform_sampler("image", tex)
    sh.uniform_float("color", (1.0, 1.0, 1.0, 0.3))
    batch.draw(sh)
    gpu.state.blend_set("NONE")
    # Do not leave the stencil image's texture bound: deleting the image would
    # leave the GPU state pointing at freed memory.
    from .engine import live
    live.release_samplers()


def _draw_cursor(op):
    if bpy.context.region != op.region:
        return
    brush = bpy.context.scene.paintstack_brush
    _draw_stencil(brush)
    if op.mouse is None:
        return
    x, y = op.resize_center or op.mouse
    scale = hardware.pixel_size()
    radius = brush.radius * scale
    sh = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    sh.bind()
    color = (1.0, 0.35, 0.35, 0.9) if brush.mode == "ERASE" else (1.0, 1.0, 1.0, 0.9)
    sh.uniform_float("color", color)
    batch_for_shader(sh, "LINE_STRIP", {"pos": _circle(x, y, radius)}).draw(sh)
    sh.uniform_float("color", (*color[:3], 0.35))
    if brush.tip == "ROUND":
        batch_for_shader(sh, "LINE_STRIP", {"pos": _circle(x, y, radius * brush.hardness)}).draw(sh)
    else:
        a = brush.angle
        tick = [(x, y), (x + radius * math.cos(a), y + radius * math.sin(a))]
        batch_for_shader(sh, "LINES", {"pos": tick}).draw(sh)
    pen = op.stroke.pen if op.stroke else None
    if pen and brush.lazy_radius > 0:
        batch_for_shader(sh, "LINES", {"pos": [pen, (x, y)]}).draw(sh)
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
        self.last_end = None  # end of the previous stroke, for Shift+click lines
        self.stencil_grab = False
        self.resize_center = None  # F: drag to set the brush radius
        self.resize_orig = 0
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        brush = context.scene.paintstack_brush
        if brush.use_stencil and brush.stencil_x == 400.0 and brush.stencil_y == 300.0:
            brush.stencil_x, brush.stencil_y = self.region.width / 2, self.region.height / 2

        wm = context.window_manager
        self._timer = wm.event_timer_add(UPDATE_INTERVAL, window=context.window)
        self._draw = bpy.types.SpaceView3D.draw_handler_add(_draw_cursor, (self,), "WINDOW", "POST_PIXEL")
        wm.modal_handler_add(self)
        type(self).running = True
        context.workspace.status_text_set(status_text())
        return {"RUNNING_MODAL"}

    def _finish(self, context):
        self._end_stroke()
        context.window_manager.event_timer_remove(self._timer)
        bpy.types.SpaceView3D.draw_handler_remove(self._draw, "WINDOW")
        context.workspace.status_text_set(None)
        type(self).running = False
        self.area.tag_redraw()
        return {"FINISHED"}

    def _end_stroke(self):
        if self.stroke:
            self.stroke.finish()
            self.last_end = self.stroke.pen
            self.stroke = None

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

    def _begin_stroke(self, context):
        brush = context.scene.paintstack_brush
        r = self.region
        view = (r.width, r.height, self.rv3d.perspective_matrix.copy(),
                self.rv3d.view_matrix.copy(), self.rv3d.is_perspective)
        try:
            self.stroke = Stroke(self._projector(context), view, self.mat, brush,
                                 self._mirror_fn(context) if brush.mirror_x else None,
                                 px_scale=hardware.pixel_size())
        except RuntimeError as e:
            self.report({"ERROR"}, str(e))
            self.stroke = None
        return self.stroke

    def _stencil_event(self, event, brush, x, y):
        """S held: move with the mouse, wheel scales, Ctrl+wheel rotates."""
        if event.type == "MOUSEMOVE" and self.mouse_prev is not None:
            brush.stencil_x += x - self.mouse_prev[0]
            brush.stencil_y += y - self.mouse_prev[1]
            return True
        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            sign = 1 if event.type == "WHEELUPMOUSE" else -1
            if event.ctrl or event.oskey:
                brush.stencil_rotation += sign * math.radians(5)
            else:
                brush.stencil_size *= 1.08 ** sign
            return True
        return False

    def _resize_event(self, event, brush, x, y):
        """F mode: the radius follows the cursor's distance from where F was pressed.
        Click / Enter / F confirms, Esc / right click cancels."""
        cx, cy = self.resize_center
        if event.type == "MOUSEMOVE":
            brush.radius = max(1, int(math.hypot(x - cx, y - cy) / hardware.pixel_size()))
        elif event.value == "PRESS" and event.type in {"LEFTMOUSE", "RET", "NUMPAD_ENTER", "F"}:
            self.resize_center = None
        elif event.value == "PRESS" and event.type in {"ESC", "RIGHTMOUSE"}:
            brush.radius = self.resize_orig
            self.resize_center = None
        return {"RUNNING_MODAL"}

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
        self.mouse_prev = self.mouse
        self.mouse = (x, y) if inside else None

        if self.resize_center is not None:
            return self._resize_event(event, brush, x, y)

        if event.type == "S" and not (event.ctrl or event.alt or event.shift or event.oskey):
            self.stencil_grab = event.value == "PRESS" and brush.use_stencil
            return {"RUNNING_MODAL"}
        if self.stencil_grab and inside and self._stencil_event(event, brush, x, y):
            return {"RUNNING_MODAL"}

        if event.type == "MOUSEMOVE":
            if self.stroke:
                self.stroke.add_point(x, y, event.pressure or 1.0)
                return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}

        if event.type == "LEFTMOUSE":
            if event.value == "PRESS" and inside and not (event.alt or event.ctrl):
                start = self.last_end if event.shift and self.last_end else None
                if self._begin_stroke(context):
                    if start:
                        self.stroke.line(*start, x, y, event.pressure or 1.0)
                    else:
                        self.stroke.add_point(x, y, event.pressure or 1.0)
                return {"RUNNING_MODAL"}
            if event.value == "RELEASE" and self.stroke:
                self._end_stroke()
                return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}

        if self.stroke:
            return {"RUNNING_MODAL"}

        if event.value == "PRESS" and inside:
            ts = self.mat.paintstack
            if event.type == "ESC":
                return self._finish(context)
            if event.type == "LEFT_BRACKET":
                brush.radius = max(1, int(brush.radius / 1.15))
                return {"RUNNING_MODAL"}
            if event.type == "RIGHT_BRACKET":
                brush.radius = int(brush.radius * 1.15) + 1
                return {"RUNNING_MODAL"}
            plain = not (event.ctrl or event.alt or event.shift or event.oskey)
            if event.type == "F" and plain:
                # Start with the cursor on the current circle, like Blender's F.
                r = brush.radius * hardware.pixel_size()
                self.resize_center = (x - r, y)
                self.resize_orig = brush.radius
                return {"RUNNING_MODAL"}
            if event.type == "X" and plain:
                brush.mode = "PAINT" if brush.mode == "ERASE" else "ERASE"
                return {"RUNNING_MODAL"}
            if event.type == "C" and plain:
                ops.cycle_view(ts)
                self.report({"INFO"}, f"View: {ts.view_mode.replace('_', ' ').title()}")
                return {"RUNNING_MODAL"}
            if event.type == "M" and plain:
                ops.cycle_view(ts, reset=True)
                return {"RUNNING_MODAL"}
            if event.type == "Z" and (event.ctrl or event.oskey):  # Cmd+Z on macOS
                undo.redo() if event.shift else undo.undo()
                return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}


classes = (PAINTSTACK_OT_paint_mode,)
