"""Phase 2 completion test: ID by material, colour selection by ray pick, cage
bakes, stepwise / cancelled bake jobs, generator cache speed, AO preview.

    python tests/test_phase2_extras.py
"""

import os
import sys
import time

import bpy
import gpu
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _env import fresh_scene  # noqa: E402  (also sets up the GPU)

import paintstack  # noqa: E402
from paintstack import baker, generators, masks, picking, pixels  # noqa: E402
from paintstack.engine.projector import Projector  # noqa: E402
from test_headless import check  # noqa: E402


def setup():
    fresh_scene()
    bpy.ops.mesh.primitive_cube_add(size=2)
    low = bpy.context.active_object
    low.name = "Low"
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")

    # High poly with two materials: "Paint" everywhere, "Trim" on the +Z face.
    bpy.ops.mesh.primitive_cube_add(size=2)
    high = bpy.context.active_object
    high.name = "High"
    paint, trim = bpy.data.materials.new("Paint"), bpy.data.materials.new("Trim")
    high.data.materials.append(paint)
    high.data.materials.append(trim)
    for poly in high.data.polygons:
        poly.material_index = 1 if poly.normal.z > 0.5 else 0
    bev = high.modifiers.new("Bevel", "BEVEL")
    bev.width, bev.segments = 0.1, 3
    high.hide_set(True)

    bpy.context.view_layer.objects.active = low
    low.select_set(True)
    with bpy.context.temp_override(active_object=low, object=low):
        bpy.ops.paintstack.init()
    mat = low.active_material
    mat.paintstack.resolution = "256"
    return low, high, mat, paint, trim


def main():
    low, high, mat, paint, trim = setup()
    ts = mat.paintstack
    s = ts.bake
    s.high_poly = high
    s.samples = 8
    for m in ("normal", "ao", "curvature", "thickness", "position", "world_normal", "id"):
        setattr(s, "use_" + m, m in ("ao", "position", "id", "world_normal"))

    check(baker.id_mode(s, high) == "MATERIAL", "AUTO picks per-material ID for a multi-material high poly")

    # Stepwise job, cancelled after the first map: state must still be restored.
    job = baker.BakeJob(bpy.context, low, mat, ["AO", "ID"])
    job.start()
    first = job.next_label
    more = job.step()
    job.finish()
    check(more and job.done == ["AO"] and first == "Ambient Occlusion", "job stops cleanly after one map")
    check(low.material_slots[0].material == mat and not [m for m in bpy.data.materials
                                                         if m.name.startswith("PS_bake")],
          "cancelled job restored materials")
    check(high.hide_get(), "cancelled job re-hid the high poly")

    # Full bake (synchronous path) of the remaining maps
    with bpy.context.temp_override(active_object=low, object=low):
        check(bpy.ops.paintstack.bake() == {"FINISHED"}, "bake operator finished")

    cov = Projector(low, bpy.context.evaluated_depsgraph_get(), mat, ts.res).coverage()
    idm = pixels.read(ts.baked_map("ID"))[..., :3]
    c_paint, c_trim = np.array(baker.id_color(0)), np.array(baker.id_color(1))
    check(np.linalg.norm(c_paint - c_trim) > 0.3, "neighbouring material ID colours are distinct")
    near_paint = np.linalg.norm(idm[cov] - c_paint, axis=1) < 0.02
    near_trim = np.linalg.norm(idm[cov] - c_trim, axis=1) < 0.02
    check(near_paint.mean() > 0.6 and 0.08 < near_trim.mean() < 0.3,
          f"ID map holds material colours (paint {near_paint.mean():.0%}, trim {near_trim.mean():.0%})")

    # Pick by ray: from above the cube hits the +Z face, which is "Trim"
    dg = bpy.context.evaluated_depsgraph_get()
    uv = picking.uv_at_ray(low, dg, Vector((0.2, 0.1, 5)), Vector((0, 0, -1)))
    check(uv is not None, "ray pick found a UV")
    picked = np.array(picking.sample(ts.baked_map("ID"), uv))
    check(np.linalg.norm(picked - c_trim) < 0.02, "picked colour is the Trim material's ID")
    check(picking.uv_at_ray(low, dg, Vector((5, 5, 5)), Vector((0, 0, 1))) is None, "ray miss returns None")

    # Colour-selection mask from the picked colour
    with bpy.context.temp_override(active_object=low, object=low):
        bpy.ops.paintstack.layer_add(type="FILL")
        layer = ts.active_layer
        bpy.ops.paintstack.effect_add(type="ID_MASK")
    layer.active_mask_effect.id_color = picked
    m = masks.evaluate(layer)[cov]
    check(abs(m.mean() - near_trim.mean()) < 0.03, f"colour selection matches the Trim area ({m.mean():.0%})")

    # Cage object: same topology as the low poly, pushed outwards
    cage = low.copy()
    cage.data = low.data.copy()
    cage.name = "Cage"
    cage.scale = (1.15, 1.15, 1.15)
    bpy.context.collection.objects.link(cage)
    s.cage_object = cage
    for m in ("ao", "position", "id", "world_normal"):
        setattr(s, "use_" + m, False)
    s.use_normal = True
    with bpy.context.temp_override(active_object=low, object=low):
        check(bpy.ops.paintstack.bake() == {"FINISHED"}, "cage bake finished")
    n = pixels.read(ts.baked_map("NORMAL"))[..., :3][cov]
    check(np.allclose(np.median(n, 0), (0.5, 0.5, 1.0), atol=0.05) and (n[:, 2] < 0.95).mean() > 0.03,
          "cage-baked normal map: flat faces, tilted bevels")

    # Generator cache: changing Level reuses noise and source
    s.use_curvature = True
    s.use_normal = False
    s.cage_object = None
    with bpy.context.temp_override(active_object=low, object=low):
        bpy.ops.paintstack.bake()
    ts.resolution = "1024"
    with bpy.context.temp_override(active_object=low, object=low):
        bpy.ops.paintstack.effect_add(type="EDGE_WEAR")
    e = layer.active_mask_effect
    generators.invalidate()
    t0 = time.perf_counter()
    generators.evaluate(e, ts)
    cold = time.perf_counter() - t0
    e.level = 0.6
    t0 = time.perf_counter()
    generators.evaluate(e, ts)
    warm = time.perf_counter() - t0
    check(warm < cold / 3, f"level change is cheap ({cold * 1000:.0f} ms cold, {warm * 1000:.0f} ms warm)")

    # GPU noise matches the CPU fallback exactly (same tables, same float32 maths)
    from paintstack.engine import noise_gpu
    pos = np.random.default_rng(1).random((128, 128, 3)).astype(np.float32)
    perm, vals = generators._table(5)
    gpu_n = noise_gpu.fbm(pos, 9.0, 4, perm, vals)
    check(np.abs(gpu_n - generators.fbm(pos * 9.0, 4, 5)).max() < 1e-4, "GPU noise == CPU noise")

    # AO preview in the material
    ts.preview_ao = True
    names = {n.name for n in mat.node_tree.nodes}
    check({"PS_Baked AO", "PS_AO Multiply"} <= names, "AO multiply nodes added")
    ts.preview_ao = False
    check("PS_AO Multiply" not in {n.name for n in mat.node_tree.nodes}, "AO multiply removed")

    print("FAILED" if check.failed else "ALL PASSED")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
