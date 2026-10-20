"""Mesh-map baker. Uses Cycles: a temporary emission material outputs the wanted
quantity (AO, pointiness, position, ...) and an EMIT bake writes it into the
texture set's UV space. With a high poly set, maps are baked from it onto the
low poly (selected-to-active), like Substance Painter's "bake mesh maps"."""

import colorsys

import bpy
import numpy as np
from bpy.types import Operator

from . import compositor, generators, material, ops, pixels
from .props import BAKE_MAPS

# Curvature is post-processed with the world normal and position maps, so bake those first.
BAKE_ORDER = ["NORMAL", "POSITION", "WORLD_NORMAL", "AO", "THICKNESS", "CURVATURE", "ID"]


def baked_image(mat, map_type):
    ts = mat.paintstack
    img = ts.baked_map(map_type)
    if img is not None:
        if tuple(img.size) != ts.size:
            img.scale(*ts.size)
        return img
    entry = ts.baked.add()
    entry.map_type = map_type
    entry.image = pixels.new_image(f"{mat.name}_bake_{map_type.lower()}", ts.size, (0.0, 0.0, 0.0),
                                   float_buffer=True)
    return entry.image


# Shader networks

def id_color(index):
    """Well-separated colour for the index-th material: golden-ratio hue steps keep
    any two materials at least ~0.24 apart in hue."""
    hue = (0.1 + index * 0.618033988749895) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.75, 0.9)


def material_order(obj):
    """Unique materials of obj in slot order; their index picks the ID colour."""
    seen = []
    for slot in obj.material_slots:
        if slot.material is not None and slot.material not in seen:
            seen.append(slot.material)
    return seen


def id_mode(settings, source):
    if settings.id_source != "AUTO":
        return settings.id_source
    if getattr(source.data, "color_attributes", None):
        return "VERTEX_COLOR"
    if len([s for s in source.material_slots if s.material]) > 1:
        return "MATERIAL"
    return "MESH_PART"


def _source_network(nt, map_type, settings, mode, color):
    """Nodes producing the baked quantity; returns the output socket."""
    n, link = nt.nodes, nt.links.new
    if map_type in ("AO", "THICKNESS"):
        ao = n.new("ShaderNodeAmbientOcclusion")
        ao.samples = 16
        ao.only_local = True
        ao.inside = map_type == "THICKNESS"
        ao.inputs["Distance"].default_value = (settings.ao_distance if map_type == "AO"
                                               else settings.thickness_distance)
        return ao.outputs["AO"]
    if map_type == "CURVATURE":
        return n.new("ShaderNodeNewGeometry").outputs["Pointiness"]
    if map_type == "POSITION":
        return n.new("ShaderNodeTexCoord").outputs["Generated"]
    if map_type == "WORLD_NORMAL":
        # In the mesh's own (object) space, matching the Position map, so curvature and
        # tri-planar projection stay correct for rotated objects and move with them.
        geo = n.new("ShaderNodeNewGeometry")
        xf = n.new("ShaderNodeVectorTransform")
        xf.vector_type = "NORMAL"
        xf.convert_from, xf.convert_to = "WORLD", "OBJECT"
        vm = n.new("ShaderNodeVectorMath")
        vm.operation = "MULTIPLY_ADD"
        vm.inputs[1].default_value = (0.5, 0.5, 0.5)
        vm.inputs[2].default_value = (0.5, 0.5, 0.5)
        link(geo.outputs["Normal"], xf.inputs[0])
        link(xf.outputs[0], vm.inputs[0])
        return vm.outputs["Vector"]
    if map_type == "ID":
        if mode == "MATERIAL":
            rgb = n.new("ShaderNodeRGB")
            rgb.outputs[0].default_value = (*color, 1.0)
            return rgb.outputs[0]
        if mode == "VERTEX_COLOR":
            return n.new("ShaderNodeVertexColor").outputs["Color"]
        geo = n.new("ShaderNodeNewGeometry")
        hsv = n.new("ShaderNodeCombineColor")
        hsv.mode = "HSV"
        hsv.inputs[1].default_value = 0.75
        hsv.inputs[2].default_value = 0.9
        link(geo.outputs["Random Per Island"], hsv.inputs[0])
        return hsv.outputs["Color"]
    raise ValueError(map_type)


def _bake_material(name, image, map_type, settings, mode, color=None):
    m = bpy.data.materials.new(name)
    if not m.node_tree:
        m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    if map_type == "NORMAL":
        shader = nt.nodes.new("ShaderNodeBsdfDiffuse")
    else:
        shader = nt.nodes.new("ShaderNodeEmission")
        shader.inputs["Strength"].default_value = 1.0
        nt.links.new(_source_network(nt, map_type, settings, mode, color), shader.inputs["Color"])
    nt.links.new(shader.outputs[0], out.inputs["Surface"])
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = image
    nt.nodes.active = tex
    return m


class _BakeEnvironment:
    """Temporarily swaps materials, selection, visibility and render settings,
    and puts everything back afterwards."""

    def __init__(self, context, low, high, cage=None, grid=(1, 1)):
        self.context, self.low, self.high, self.cage = context, low, high, cage
        self.slots, self.appended, self.temp_mats = [], [], []
        self.grid = grid
        self.uv = None  # (previous active UV layer name, temporary atlas UV layer name)

    def __enter__(self):
        ctx = self.context
        scene = ctx.scene
        vl = ctx.view_layer
        self.render = (scene.render.engine, scene.cycles.samples, scene.cycles.device)
        self.selection = [o for o in vl.objects if o.select_get()]
        self.active = vl.objects.active
        self.vis = [(o, o.hide_get(), o.hide_viewport, o.hide_render, o.hide_select)
                    for o in (self.low, self.high, self.cage) if o]
        if ctx.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        scene.render.engine = "CYCLES"
        for o, *_ in self.vis:
            o.hide_set(False)
            o.hide_viewport = o.hide_render = o.hide_select = False
        for o in vl.objects:
            o.select_set(False)
        for o in (self.low, self.high):
            if o:
                o.select_set(True)
        vl.objects.active = self.low
        if self.grid != (1, 1):
            self._atlas_uvs()
        return self

    def _atlas_uvs(self):
        """UDIM: bake through a temporary UV map scaled into the tile layout of the
        texture set's images (u / columns, v / rows), the active one while baking."""
        uvs = self.low.data.uv_layers
        src = uvs.active
        if src is None:
            return
        n = len(src.data)
        co = np.empty(n * 2, np.float32)
        src.data.foreach_get("uv", co)
        co = (co.reshape(-1, 2) / np.array(self.grid, np.float32)).ravel()
        prev = src.name
        tmp = uvs.new(name="PS_bake_atlas", do_init=False)
        tmp.data.foreach_set("uv", co)
        uvs.active = tmp
        self.uv = (prev, tmp.name)

    def assign(self, obj, choose):
        """Override every material slot of obj with choose(original_material)."""
        if not obj.material_slots:
            obj.data.materials.append(None)
            self.appended.append(obj)
        for i, slot in enumerate(obj.material_slots):
            orig = slot.material
            link = slot.link
            slot.link = "OBJECT"
            self.slots.append((obj, i, link, slot.material))
            slot.material = choose(orig)

    def release_slots(self):
        for obj, i, link, obj_level in reversed(self.slots):
            slot = obj.material_slots[i]
            slot.material = obj_level
            slot.link = link
        self.slots.clear()
        for obj in self.appended:
            obj.data.materials.pop()
        self.appended.clear()
        for m in self.temp_mats:
            bpy.data.materials.remove(m)
        self.temp_mats.clear()

    def __exit__(self, *exc):
        self.release_slots()
        if self.uv is not None:
            uvs = self.low.data.uv_layers
            prev, tmp = self.uv
            if prev in uvs:
                uvs.active = uvs[prev]
            if tmp in uvs:
                uvs.remove(uvs[tmp])
            self.uv = None
        scene, vl = self.context.scene, self.context.view_layer
        scene.render.engine, scene.cycles.samples, scene.cycles.device = self.render
        for o, hidden, hv, hr, hs in self.vis:
            o.hide_viewport, o.hide_render, o.hide_select = hv, hr, hs
            o.hide_set(hidden)
        for o in vl.objects:
            o.select_set(o in self.selection)
        vl.objects.active = self.active
        return False


def _normalized(dev, covered):
    """Scale deviations so the 97th percentile over covered texels maps to 1."""
    sample = np.abs(dev[covered]) if covered.any() else np.abs(dev)
    scale = max(float(np.percentile(sample, 97)) if sample.size else 0.0, 1e-6)
    return np.clip(dev / scale, -1.0, 1.0)


def geometric_curvature(normal, position, covered):
    """Surface curvature from baked world normals and positions: along each texture
    axis, (dN . dP) / |dP|^2 is the normal-curvature in that direction. Texels next
    to uncovered ones are left at zero, so island borders do not read as edges."""
    curv = np.zeros(normal.shape[:2], np.float32)
    ok = covered.copy()
    for axis in (0, 1):
        dn = np.gradient(normal, axis=axis)
        dp = np.gradient(position, axis=axis)
        curv += (dn * dp).sum(-1) / np.maximum((dp * dp).sum(-1), 1e-12)
        ok &= np.roll(covered, 1, axis) & np.roll(covered, -1, axis)
    return np.where(ok, curv, 0.0)


def _postprocess_curvature(img, ts):
    """Normalise pointiness around 0.5, and when world normal and position maps are
    baked, combine it with their (high-poly detailed) geometric curvature."""
    px = pixels.read(img)
    covered = px[..., 3] > 0.5
    dev = _normalized(px[..., 0] - 0.5, covered)

    wn, pos = ts.baked_map("WORLD_NORMAL"), ts.baked_map("POSITION")
    if wn is not None and pos is not None and tuple(wn.size) == tuple(pos.size) == tuple(img.size):
        n = pixels.read(wn)[..., :3] * 2.0 - 1.0
        p = pixels.read(pos)[..., :3]
        geo = _normalized(geometric_curvature(n, p, covered), covered)
        dev = np.clip(0.35 * dev + 0.65 * geo, -1.0, 1.0)

    curv = 0.5 + 0.5 * dev
    pixels.write(img, pixels.rgba(np.stack([curv] * 3, -1), px[..., 3]))


class BakeJob:
    """Bakes maps one at a time so a modal operator can keep the UI alive between
    maps. start() / step() until it returns False / finish()."""

    def __init__(self, context, low, mat, maps):
        self.context, self.low, self.mat = context, low, mat
        self.ts = mat.paintstack
        s = self.settings = self.ts.bake
        self.high = s.high_poly if s.high_poly not in (None, low) else None
        self.cage = s.cage_object if self.high and s.cage_object not in (None, low, self.high) else None
        self.reports = []
        self.maps = [m for m in BAKE_ORDER if m in maps]
        if "NORMAL" in self.maps and self.high is None:
            self.maps.remove("NORMAL")
            self.reports.append(({"INFO"}, "Normal map skipped: set a High Poly object to bake it"))
        self.done = []
        self.env = None

    @property
    def next_label(self):
        labels = {ident: label for ident, label, _ in BAKE_MAPS}
        return labels[self.maps[len(self.done)]] if len(self.done) < len(self.maps) else ""

    def start(self):
        self.dummy = bpy.data.images.new("PS_bake_dummy", 8, 8, float_buffer=True)
        self.env = _BakeEnvironment(self.context, self.low, self.high, self.cage, self.ts.grid).__enter__()

    def step(self):
        """Bake the next map. Returns True while more maps remain."""
        if len(self.done) == len(self.maps):
            return False
        map_type = self.maps[len(self.done)]
        env, s, mat, high = self.env, self.settings, self.mat, self.high
        source = high or self.low
        mode = id_mode(s, source)
        img = baked_image(mat, map_type)
        order = material_order(source)

        def make(image, src_mat=None):
            color = None
            if mode == "MATERIAL":
                color = id_color(order.index(src_mat) if src_mat in order else len(order))
            m = _bake_material("PS_bake_tmp", image, map_type, s, mode, color)
            env.temp_mats.append(m)
            return m

        per_material = map_type == "ID" and mode == "MATERIAL"
        target, other = make(img), make(self.dummy)
        if per_material and not high:
            env.assign(self.low, lambda m: make(img if m == mat else self.dummy, m))
        else:
            env.assign(self.low, lambda m: target if m == mat else other)
        if high:
            env.assign(high, (lambda m: make(self.dummy, m)) if per_material else (lambda m: target))

        self.context.scene.cycles.samples = s.samples if map_type in ("AO", "THICKNESS") else 4
        kwargs = dict(
            type="NORMAL" if map_type == "NORMAL" else "EMIT",
            normal_space="TANGENT",
            use_selected_to_active=high is not None,
            cage_extrusion=s.cage_extrusion,
            max_ray_distance=s.max_ray_distance,
            margin=s.margin,
            use_clear=True,
            target="IMAGE_TEXTURES",
        )
        if self.cage:
            kwargs.update(use_cage=True, cage_object=self.cage.name)
        try:
            bpy.ops.object.bake(**kwargs)
        finally:
            env.release_slots()
        if map_type == "CURVATURE":
            _postprocess_curvature(img, self.ts)
        img.pack()
        self.done.append(map_type)
        return len(self.done) < len(self.maps)

    def finish(self):
        if self.env is not None:
            self.env.__exit__(None, None, None)
            self.env = None
            bpy.data.images.remove(self.dummy)
        if self.done:
            generators.invalidate()
            material.build(self.mat)
            compositor.composite(self.mat)
            labels = {ident: label for ident, label, _ in BAKE_MAPS}
            self.reports.append(({"INFO"}, "Baked " + ", ".join(labels[m] for m in self.done)))
        return self.reports


def bake(context, low, mat, maps):
    """Bake the given map types synchronously. Returns a list of (level, message) reports."""
    job = BakeJob(context, low, mat, maps)
    if not job.maps:
        return job.reports
    job.start()
    try:
        while job.step():
            pass
    finally:
        reports = job.finish()
    return reports


def _selected_maps(ts):
    return [m for m in BAKE_ORDER if getattr(ts.bake, "use_" + m.lower())]


class PAINTSTACK_OT_bake(Operator):
    """Bake mesh maps (normal, AO, curvature, thickness, position, ...) used by generators.
    Maps bake one at a time; press Esc to stop after the current map"""
    bl_idname = "paintstack.bake"
    bl_label = "Bake Mesh Maps"

    @classmethod
    def poll(cls, context):
        return ops.texture_set(context)[0] is not None

    def _report_all(self, reports):
        for level, msg in reports:
            self.report(level, msg)

    def execute(self, context):
        mat, ts = ops.texture_set(context)
        try:
            reports = bake(context, context.active_object, mat, _selected_maps(ts))
        except RuntimeError as e:
            self.report({"ERROR"}, f"Bake failed: {e}")
            return {"CANCELLED"}
        self._report_all(reports)
        return {"FINISHED"}

    def invoke(self, context, event):
        mat, ts = ops.texture_set(context)
        self.job = BakeJob(context, context.active_object, mat, _selected_maps(ts))
        if not self.job.maps:
            self._report_all(self.job.reports)
            return {"CANCELLED"}
        self.job.start()
        wm = context.window_manager
        wm.progress_begin(0, len(self.job.maps))
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        self._status(context)
        return {"RUNNING_MODAL"}

    def _status(self, context):
        n = len(self.job.done)
        context.workspace.status_text_set(
            f"PaintStack: baking {self.job.next_label} ({n + 1}/{len(self.job.maps)}) · Esc to stop")
        for area in context.screen.areas:
            area.tag_redraw()

    def _end(self, context, cancelled=False):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        wm.progress_end()
        context.workspace.status_text_set(None)
        self._report_all(self.job.finish())
        if cancelled:
            self.report({"WARNING"}, "Bake stopped")
        return {"CANCELLED"} if cancelled and not self.job.done else {"FINISHED"}

    def modal(self, context, event):
        if event.type == "ESC" and event.value == "PRESS":
            return self._end(context, cancelled=True)
        if event.type != "TIMER":
            return {"RUNNING_MODAL"}
        try:
            more = self.job.step()
        except RuntimeError as e:
            self.report({"ERROR"}, f"Bake failed: {e}")
            return self._end(context, cancelled=True)
        context.window_manager.progress_update(len(self.job.done))
        if not more:
            return self._end(context)
        self._status(context)
        return {"RUNNING_MODAL"}


classes = (PAINTSTACK_OT_bake,)
