"""Phase 2 headless test: mesh-map baking (incl. high -> low), generators, mask stack.

    python tests/test_phase2.py
"""

import os
import sys
import tempfile

import bpy
import gpu
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _env import fresh_scene  # noqa: E402  (also sets up the GPU)

import paintstack  # noqa: E402
from paintstack import compositor, generators, masks, pixels  # noqa: E402
from paintstack.engine.projector import Projector  # noqa: E402
from paintstack.stroke import Stroke  # noqa: E402
from test_headless import W, H, check, front_view  # noqa: E402


def setup():
    fresh_scene()
    bpy.ops.mesh.primitive_cube_add(size=2)
    low = bpy.context.active_object
    low.name = "Low"
    # Smart-UV so the six faces do not overlap (the default cube UVs are a cross, fine too).
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.mesh.primitive_cube_add(size=2)
    high = bpy.context.active_object
    high.name = "High"
    bev = high.modifiers.new("Bevel", "BEVEL")
    bev.width, bev.segments = 0.15, 4
    sub = high.modifiers.new("Sub", "SUBSURF")
    sub.levels = sub.render_levels = 2
    high.hide_set(True)

    bpy.context.view_layer.objects.active = low
    low.select_set(True)
    with bpy.context.temp_override(active_object=low, object=low):
        bpy.ops.paintstack.init()
    mat = low.active_material
    mat.paintstack.resolution = "256"
    return low, high, mat


def main():
    low, high, mat = setup()
    ts = mat.paintstack
    ts.bake.high_poly = high
    ts.bake.samples = 16
    ts.bake.use_id = True
    engine_before = bpy.context.scene.render.engine
    n_mats_before = len(bpy.data.materials)

    with bpy.context.temp_override(active_object=low, object=low):
        res = bpy.ops.paintstack.bake()
    check(res == {"FINISHED"}, "bake operator finished")

    for m in ("NORMAL", "AO", "CURVATURE", "THICKNESS", "POSITION", "WORLD_NORMAL", "ID"):
        check(ts.baked_map(m) is not None, f"{m} baked")

    # Environment restored
    check(bpy.context.scene.render.engine == engine_before, "render engine restored")
    check(len(bpy.data.materials) == n_mats_before, "temporary bake materials removed")
    check(low.material_slots[0].material == mat and low.material_slots[0].link == "DATA",
          "low poly material slot restored")
    check(high.hide_get(), "high poly hidden again")
    check(bpy.context.view_layer.objects.active == low, "active object restored")

    proj = Projector(low, bpy.context.evaluated_depsgraph_get(), mat, ts.res)
    cov = proj.coverage()

    normal = pixels.read(ts.baked_map("NORMAL"))[..., :3][cov]
    check(np.allclose(np.median(normal, 0), (0.5, 0.5, 1.0), atol=0.05), "normal map mostly flat")
    check((normal[:, 2] < 0.95).mean() > 0.05, f"bevels visible in normal map "
          f"({(normal[:, 2] < 0.95).mean():.0%} tilted texels)")

    ao = pixels.read(ts.baked_map("AO"))[..., 0][cov]
    check(0.5 < ao.mean() <= 1.0 and ao.min() >= 0.0, f"AO plausible (mean {ao.mean():.2f})")

    pos = pixels.read(ts.baked_map("POSITION"))[..., :3][cov]
    check(pos.min() > -0.05 and pos.max() < 1.05 and np.ptp(pos, 0).min() > 0.8,
          "position map spans the bounding box")

    wn = pixels.read(ts.baked_map("WORLD_NORMAL"))[..., :3][cov] * 2 - 1
    check(np.abs(np.linalg.norm(wn, axis=1).mean() - 1.0) < 0.1, "world normals unit length")

    curv = pixels.read(ts.baked_map("CURVATURE"))[..., 0][cov]
    check(abs(np.median(curv) - 0.5) < 0.1 and (curv > 0.65).mean() > 0.02,
          f"curvature: flat ~0.5, convex edges bright ({(curv > 0.65).mean():.0%})")

    # Material uses the baked normal
    names = {n.name for n in mat.node_tree.nodes}
    check("PS_Baked Normal" in names and "PS_Normal Map" in names, "baked normal wired into material")

    # Generators
    noise = generators.fbm(np.random.default_rng(0).random((1000, 3)) * 10, 4)
    check(0.0 <= noise.min() and noise.max() <= 1.0 and noise.std() > 0.05, "fbm noise in range")

    with bpy.context.temp_override(active_object=low, object=low):
        bpy.ops.paintstack.layer_add(type="FILL")
        layer = ts.active_layer
        layer.name = "Edge Wear"
        layer.channel("BASE_COLOR").color = (0.9, 0.9, 0.95)
        layer.channel("METALLIC").value = 1.0
        bpy.ops.paintstack.effect_add(type="EDGE_WEAR")
    edge = masks.evaluate(layer)[cov]
    check(0.01 < edge.mean() < 0.6, f"edge wear covers some, not all ({edge.mean():.0%})")
    curv_full = pixels.read(ts.baked_map("CURVATURE"))[..., 0][cov]
    check(edge[curv_full > 0.7].mean() > edge[np.abs(curv_full - 0.5) < 0.05].mean() + 0.2,
          "edge wear concentrates on convex edges")

    compositor.composite(mat)
    metal = pixels.read(ts.output("METALLIC"))[..., 0][cov]
    check(np.allclose(metal, edge, atol=0.02), "edge wear drives the metallic output")

    # Filters on the stack
    sharp = masks.evaluate(layer)
    with bpy.context.temp_override(active_object=low, object=low):
        bpy.ops.paintstack.effect_add(type="BLUR")
    blurred = masks.evaluate(layer)
    check(np.abs(np.diff(blurred, axis=1)).mean() < np.abs(np.diff(sharp, axis=1)).mean(),
          "blur smooths the mask")
    with bpy.context.temp_override(active_object=low, object=low):
        bpy.ops.paintstack.effect_add(type="LEVELS")
    layer.active_mask_effect.in_max = 0.01
    lv = masks.evaluate(layer)[cov]
    check(lv.mean() > edge.mean(), "levels expands the mask")
    layer.active_mask_effect.enabled = False

    # Dirt / AO / position / thickness generators all evaluate
    for kind in ("DIRT", "AO", "POSITION", "THICKNESS", "CURVATURE", "NOISE"):
        with bpy.context.temp_override(active_object=low, object=low):
            bpy.ops.paintstack.effect_add(type=kind)
        e = layer.active_mask_effect
        v = generators.evaluate(e, ts)
        check(v is not None and v.shape == (256, 256) and 0 <= v.min() and v.max() <= 1,
              f"{kind} generator evaluates")
        e.enabled = False

    # Painting the mask of a layer with generators adds a Paint effect on top
    n_effects = len(layer.mask_effects)
    b = bpy.context.scene.paintstack_brush
    b.radius, b.flow, b.hardness = 60, 1.0, 1.0
    s = Stroke(proj, front_view(), mat, b)
    s.add_point(W / 2, H / 2)
    s.finish()
    top = layer.mask_effects[-1]
    check(len(layer.mask_effects) == n_effects + 1 and top.type == "PAINT", "paint effect added on top")
    check(pixels.read(top.image)[..., 3].max() > 0.9, "mask stroke landed in the paint effect")
    check(masks.evaluate(layer).max() > 0.99, "painted area fully revealed")

    # Export uses AO in packed maps
    d = tempfile.mkdtemp()
    with bpy.context.temp_override(active_object=low, object=low):
        bpy.ops.paintstack.export(directory=d, preset="UNREAL", padding=0)
    orm = pixels.read(bpy.data.images.load(os.path.join(d, f"{mat.name}_OcclusionRoughnessMetallic.png")))
    ao_full = pixels.read(ts.baked_map("AO"))[..., 0]
    check(np.abs(orm[..., 0][cov] - ao_full[cov]).mean() < 0.01, "ORM red channel = baked AO")

    # UI icons exist
    icons = bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items.keys()
    from paintstack import ui
    used = set(ui.EFFECT_ICONS.values()) | {"TEXTURE", "RENDER_STILL", "CHECKMARK", "BLANK1", "ERROR",
                                             "ADD", "REMOVE", "TRIA_UP", "TRIA_DOWN", "INFO"}
    check(not (used - set(icons)), f"all UI icons valid {sorted(used - set(icons))}")

    print("FAILED" if check.failed else "ALL PASSED")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
