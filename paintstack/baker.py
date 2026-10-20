"""Mesh-map baker. Uses Cycles: a temporary emission material outputs the wanted
quantity (AO, pointiness, position, ...) and an EMIT bake writes it into the
texture set's UV space. With a high poly set, maps are baked from it onto the
low poly (selected-to-active), like Substance Painter's "bake mesh maps"."""

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
        if tuple(img.size) != (ts.res, ts.res):
            img.scale(ts.res, ts.res)
        return img
    entry = ts.baked.add()
    entry.map_type = map_type
    entry.image = pixels.new_image(f"{mat.name}_bake_{map_type.lower()}", ts.res, (0.0, 0.0, 0.0),
                                   float_buffer=True)
    return entry.image


# Shader networks

def _source_network(nt, map_type, settings, has_vcol):
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
        geo = n.new("ShaderNodeNewGeometry")
        vm = n.new("ShaderNodeVectorMath")
        vm.operation = "MULTIPLY_ADD"
        vm.inputs[1].default_value = (0.5, 0.5, 0.5)
        vm.inputs[2].default_value = (0.5, 0.5, 0.5)
        link(geo.outputs["Normal"], vm.inputs[0])
        return vm.outputs["Vector"]
    if map_type == "ID":
        if has_vcol:
            return n.new("ShaderNodeVertexColor").outputs["Color"]
        geo = n.new("ShaderNodeNewGeometry")
        hsv = n.new("ShaderNodeCombineColor")
        hsv.mode = "HSV"
        hsv.inputs[1].default_value = 0.75
        hsv.inputs[2].default_value = 0.9
        link(geo.outputs["Random Per Island"], hsv.inputs[0])
        return hsv.outputs["Color"]
    raise ValueError(map_type)


def _bake_material(name, image, map_type, settings, has_vcol):
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
        nt.links.new(_source_network(nt, map_type, settings, has_vcol), shader.inputs["Color"])
    nt.links.new(shader.outputs[0], out.inputs["Surface"])
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = image
    nt.nodes.active = tex
    return m


class _BakeEnvironment:
    """Temporarily swaps materials, selection, visibility and render settings,
    and puts everything back afterwards."""

    def __init__(self, context, low, high):
        self.context, self.low, self.high = context, low, high
        self.slots, self.appended, self.temp_mats = [], [], []

    def __enter__(self):
        ctx = self.context
        scene = ctx.scene
        vl = ctx.view_layer
        self.render = (scene.render.engine, scene.cycles.samples, scene.cycles.device)
        self.selection = [o for o in vl.objects if o.select_get()]
        self.active = vl.objects.active
        self.vis = [(o, o.hide_get(), o.hide_viewport, o.hide_render, o.hide_select)
                    for o in (self.low, self.high) if o]
        if ctx.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        scene.render.engine = "CYCLES"
        for o, *_ in self.vis:
            o.hide_set(False)
            o.hide_viewport = o.hide_render = o.hide_select = False
        for o in vl.objects:
            o.select_set(False)
        for o, *_ in self.vis:
            o.select_set(True)
        vl.objects.active = self.low
        return self

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


def bake(context, low, mat, maps):
    """Bake the given map types. Returns a list of (level, message) reports."""
    ts = mat.paintstack
    settings = ts.bake
    high = settings.high_poly if settings.high_poly not in (None, low) else None
    reports = []
    maps = [m for m in BAKE_ORDER if m in maps]
    if "NORMAL" in maps and high is None:
        maps.remove("NORMAL")
        reports.append(({"INFO"}, "Normal map skipped: set a High Poly object to bake it"))
    if not maps:
        return reports

    source = high or low
    has_vcol = bool(getattr(source.data, "color_attributes", None))
    dummy = bpy.data.images.new("PS_bake_dummy", 8, 8, float_buffer=True)
    wm = context.window_manager
    wm.progress_begin(0, len(maps))
    try:
        with _BakeEnvironment(context, low, high) as env:
            for i, map_type in enumerate(maps):
                img = baked_image(mat, map_type)
                target = _bake_material("PS_bake_target", img, map_type, settings, has_vcol)
                other = _bake_material("PS_bake_other", dummy, map_type, settings, has_vcol)
                env.temp_mats += [target, other]
                env.assign(low, lambda m: target if m == mat else other)
                if high:
                    env.assign(high, lambda m: target)
                context.scene.cycles.samples = settings.samples if map_type in ("AO", "THICKNESS") else 4
                bpy.ops.object.bake(
                    type="NORMAL" if map_type == "NORMAL" else "EMIT",
                    normal_space="TANGENT",
                    use_selected_to_active=high is not None,
                    cage_extrusion=settings.cage_extrusion,
                    max_ray_distance=settings.max_ray_distance,
                    margin=settings.margin,
                    use_clear=True,
                    target="IMAGE_TEXTURES",
                )
                env.release_slots()
                if map_type == "CURVATURE":
                    _postprocess_curvature(img, ts)
                img.pack()
                wm.progress_update(i + 1)
    finally:
        wm.progress_end()
        bpy.data.images.remove(dummy)

    generators.invalidate()
    material.build(mat)
    compositor.composite(mat)
    labels = {ident: label for ident, label, _ in BAKE_MAPS}
    reports.append(({"INFO"}, "Baked " + ", ".join(labels[m] for m in maps)))
    return reports


class PAINTSTACK_OT_bake(Operator):
    """Bake mesh maps (normal, AO, curvature, thickness, position, ...) used by generators"""
    bl_idname = "paintstack.bake"
    bl_label = "Bake Mesh Maps"

    @classmethod
    def poll(cls, context):
        return ops.texture_set(context)[0] is not None

    def execute(self, context):
        mat, ts = ops.texture_set(context)
        s = ts.bake
        maps = [m for m in BAKE_ORDER if getattr(s, "use_" + m.lower())]
        try:
            reports = bake(context, context.active_object, mat, maps)
        except RuntimeError as e:
            self.report({"ERROR"}, f"Bake failed: {e}")
            return {"CANCELLED"}
        for level, msg in reports:
            self.report(level, msg)
        return {"FINISHED"}


classes = (PAINTSTACK_OT_bake,)
