"""Phase 4 headless test: grunge library, filter layers, anchor points, built-in and
saved smart materials / smart masks (asset library round trip).

    python tests/test_phase4.py
"""

import math
import os
import sys

import bpy
import gpu
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _env import fresh_scene  # noqa: E402  (also sets up the GPU)

import paintstack  # noqa: E402
from paintstack import compositor, filters, generators, grunge, masks, pixels, smart  # noqa: E402
from paintstack.channels import BY_ID  # noqa: E402
from paintstack.engine.projector import Projector  # noqa: E402
from paintstack.stroke import Stroke  # noqa: E402
from test_headless import W, H, check, front_view  # noqa: E402
from test_phase3 import override  # noqa: E402


def make_object(kind="cube"):
    if kind == "cube":
        bpy.ops.mesh.primitive_cube_add(size=2)
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16)
    obj = bpy.context.active_object
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")
    with override(obj):
        bpy.ops.paintstack.init()
    mat = obj.active_material
    mat.paintstack.resolution = "256"
    s = mat.paintstack.bake
    s.samples = 4
    for m in ("normal", "thickness", "id"):
        setattr(s, "use_" + m, False)
    with override(obj):
        bpy.ops.paintstack.bake()
    return obj, mat


def L(ts, uid):
    """Look a layer up by uid. References to collection items go stale when the
    collection grows, so tests must not hold them across layer_add."""
    return next(l for l in ts.layers if l.uid == uid)


def index_of(ts, uid):
    return next(i for i, l in enumerate(ts.layers) if l.uid == uid)


def out(ts, cid):
    return pixels.read(ts.output(cid))[..., :3]


def main():
    fresh_scene()

    # --- Grunge library
    for kind, _, _ in grunge.GRUNGE_TYPES:
        a = grunge.grunge_array(kind, 0, 256)
        wrapped = np.roll(a, 128, axis=(0, 1))  # the old tile border is now in the middle
        mid_jump = np.abs(np.diff(wrapped[:, 126:130], axis=1)).mean()
        typical = np.abs(np.diff(a, axis=1)).mean()
        check(0 <= a.min() and a.max() <= 1 and a.std() > 0.02 and mid_jump < typical * 3 + 0.02,
              f"grunge {kind} is seamless ({mid_jump:.3f} vs {typical:.3f})")
    img = grunge.make_image("RUST", 1)
    check(img.packed_file is not None and img is grunge.make_image("RUST", 1), "grunge image packed and reused")

    obj, mat = make_object()
    ts = mat.paintstack
    cov = Projector(obj, bpy.context.evaluated_depsgraph_get(), mat, ts.res).coverage()

    # --- Grunge mask effect (tri-planar because Position / World Normal are baked)
    with override(obj):
        bpy.ops.paintstack.layer_add(type="FILL")
        bpy.ops.paintstack.effect_add(type="GRUNGE")
    layer = ts.active_layer
    e = layer.active_mask_effect
    e.grunge = "RUST"
    m = masks.evaluate(layer)[cov]
    check(0.05 < m.mean() < 0.95, f"grunge mask covers part of the mesh ({m.mean():.0%})")
    with override(obj):
        bpy.ops.paintstack.layer_remove()

    # --- Filter layers
    base_uid = ts.layers[-1].uid
    with override(obj):
        bpy.ops.paintstack.layer_add(type="FILL")
    pattern_uid = ts.active_layer.uid
    pattern = ts.active_layer
    pattern.channel("BASE_COLOR").texture = grunge.make_image("CELLS", 0)
    pattern.channel("BASE_COLOR").color = (1, 1, 1)
    compositor.composite(mat)
    before = out(ts, "BASE_COLOR")

    with override(obj):
        bpy.ops.paintstack.layer_add(type="FILTER")
    filt = ts.active_layer  # no layers are added while this reference is used
    filt.filter_type = "BLUR"
    filt.filter_radius = 6
    compositor.composite(mat)
    blurred = out(ts, "BASE_COLOR")
    check(np.abs(np.diff(blurred, axis=1)).mean() < np.abs(np.diff(before, axis=1)).mean() * 0.6,
          "blur filter smooths the stack below")
    filt.filter_type = "SHARPEN"
    compositor.composite(mat)
    check(np.abs(np.diff(out(ts, "BASE_COLOR"), axis=1)).mean() > np.abs(np.diff(before, axis=1)).mean(),
          "sharpen filter adds contrast")
    filt.filter_type = "WARP"
    filt.filter_amount = 1.0
    compositor.composite(mat)
    warped = out(ts, "BASE_COLOR")
    check(not np.allclose(warped, before, atol=0.02) and abs(warped.mean() - before.mean()) < 0.05,
          "warp filter displaces without changing the overall tone")
    filt.opacity = 0.0
    compositor.composite(mat)
    check(np.allclose(out(ts, "BASE_COLOR"), before, atol=1e-3), "filter at opacity 0 does nothing")
    filt.opacity = 1.0

    # HSL on a known colour: hide the pattern, base red
    L(ts, pattern_uid).visible = False
    base = L(ts, base_uid)
    base.channel("BASE_COLOR").color = (1.0, 0.0, 0.0)
    filt.filter_type = "HSL"
    filt.hue = 1 / 3
    compositor.composite(mat)
    check(np.allclose(out(ts, "BASE_COLOR")[5, 5], (0, 1, 0), atol=0.02), "hue +1/3 turns red into green")
    filt.hue, filt.saturation = 0.0, -1.0
    compositor.composite(mat)
    c = out(ts, "BASE_COLOR")[5, 5]
    check(np.ptp(c) < 0.02, "saturation -1 makes it grey")
    filt.saturation, filt.lightness = 0.0, 0.5
    base.channel("ROUGHNESS").value = 0.4
    compositor.composite(mat)
    check(abs(out(ts, "ROUGHNESS")[5, 5, 0] - 0.7) < 0.01, "lightness on a grey channel (0.4 -> 0.7)")
    filt.channel("ROUGHNESS").enabled = False
    compositor.composite(mat)
    check(abs(out(ts, "ROUGHNESS")[5, 5, 0] - 0.4) < 0.01, "filter respects per-channel toggles")
    with override(obj):
        bpy.ops.paintstack.layer_remove()  # filter
    ts.active_layer_index = index_of(ts, pattern_uid)
    with override(obj):
        bpy.ops.paintstack.layer_remove()  # pattern
    L(ts, base_uid).channel("BASE_COLOR").color = (0.5, 0.5, 0.5)

    # --- Anchor points: a mask driven by painted height
    paint_uid = next(l for l in ts.layers if l.type == "PAINT").uid
    L(ts, paint_uid).is_anchor = True
    ts.active_layer_index = index_of(ts, paint_uid)
    with override(obj):
        bpy.ops.paintstack.layer_add(type="FILL")
        bpy.ops.paintstack.effect_add(type="ANCHOR")
    rust = ts.active_layer
    for lc in rust.channels:  # colour only, so the painted height stays visible
        lc.enabled = lc.channel == "BASE_COLOR"
    rust.channel("BASE_COLOR").color = (0.6, 0.2, 0.05)
    with override(obj):
        check(bpy.ops.paintstack.effect_set_anchor(uid=paint_uid) == {"FINISHED"}, "anchor chosen")
    eff = ts.active_layer.active_mask_effect
    eff.anchor_channel, eff.level, eff.softness = "HEIGHT", 0.4, 0.05  # height above ~0.6
    compositor.composite(mat)
    check(np.allclose(out(ts, "BASE_COLOR")[cov].mean(0), (0.5, 0.5, 0.5), atol=0.01),
          "no painted height -> anchor mask empty")

    # Paint height on the anchor layer; the rust layer above follows live
    ts.active_layer_index = index_of(ts, paint_uid)
    b = bpy.context.scene.paintstack_brush
    for ch in ("base_color", "metallic", "roughness"):
        setattr(b, "use_" + ch, False)
    b.use_height, b.height = True, 1.0
    b.radius, b.flow, b.hardness, b.opacity = 40, 1.0, 1.0, 1.0
    proj = Projector(obj, bpy.context.evaluated_depsgraph_get(), mat, ts.res)
    s = Stroke(proj, front_view(), mat, b, seed=0)
    s.add_point(W / 2, H / 2)
    s.finish()
    height = out(ts, "HEIGHT")[..., 0]
    col = out(ts, "BASE_COLOR")
    raised = height > 0.9
    check(raised.sum() > 50 and np.allclose(col[raised].mean(0), (0.6, 0.2, 0.05), atol=0.03),
          "anchor mask follows the painted height (live during the stroke)")
    check(np.allclose(col[cov & (height < 0.55)].mean(0), (0.5, 0.5, 0.5), atol=0.02),
          "anchor mask stays empty elsewhere")
    live = col.copy()
    compositor.composite(mat)
    check(np.allclose(live, out(ts, "BASE_COLOR"), atol=1e-3), "anchor stroke update == full composite")

    # --- Built-in smart materials and masks
    n_layers = len(ts.layers)
    for name, specs in smart.SMART_MATERIALS.items():
        with override(obj):
            r = bpy.ops.paintstack.smart_apply(kind="SMART_MATERIAL", name=name)
        folder = ts.active_layer
        ok = r == {"FINISHED"} and folder.type == "FOLDER" and folder.name == name
        kids = [l for l in ts.layers if l.parent_uid == folder.uid]
        check(ok and len(kids) == len(specs), f"smart material '{name}' applied ({len(kids)} layers)")
        rgb = out(ts, "BASE_COLOR")[cov]
        check(np.isfinite(rgb).all(), f"'{name}' composites")
        with override(obj):
            bpy.ops.paintstack.layer_remove()
    check(len(ts.layers) == n_layers, "removing the smart material folders restores the stack")

    with override(obj):
        bpy.ops.paintstack.smart_apply(kind="SMART_MATERIAL", name="Painted Steel")
    folder_uid = ts.active_layer.uid
    kids = {l.name: l for l in ts.layers if l.parent_uid == folder_uid}
    paint_mask = masks.evaluate(kids["Paint"])[cov]
    check(0.3 < paint_mask.mean() < 0.99, f"Painted Steel: paint is worn off on edges ({1 - paint_mask.mean():.0%})")

    ts.active_layer_index = len(ts.layers) - 1  # a root layer, so the fill lands outside folders
    with override(obj):
        bpy.ops.paintstack.layer_add(type="FILL")
    target_uid = ts.active_layer.uid
    target = ts.active_layer
    for name, specs in smart.SMART_MASKS.items():
        target.mask_effects.clear()
        with override(obj):
            r = bpy.ops.paintstack.smart_apply(kind="SMART_MASK", name=name)
        mv = masks.evaluate(target)
        check(r == {"FINISHED"} and len(target.mask_effects) == len(specs) and mv is not None
              and np.isfinite(mv).all(), f"smart mask '{name}' applied")

    # --- Save a smart material (with painted content) and a smart mask, then apply them
    # to a different mesh in a fresh file.
    ts.active_layer_index = index_of(ts, folder_uid)
    with override(obj):
        bpy.ops.paintstack.layer_add(type="PAINT")  # inside the folder
    painted = ts.active_layer
    b.use_base_color, b.use_height, b.base_color = True, False, (0.0, 1.0, 0.0)
    s = Stroke(proj, front_view(), mat, b, seed=0)
    s.add_point(W / 2, H / 2)
    s.finish()
    painted_px = pixels.read(painted.channel("BASE_COLOR").image)
    ts.active_layer_index = index_of(ts, folder_uid)
    with override(obj):
        r = bpy.ops.paintstack.smart_save(kind="SMART_MATERIAL", name="Test Steel")
    path = smart.user_assets(smart.KIND_MATERIAL).get("Test Steel")
    check(r == {"FINISHED"} and path and os.path.isfile(path), "smart material saved to the library")

    ts.active_layer_index = index_of(ts, target_uid)
    L(ts, target_uid).mask_effects.clear()
    with override(obj):
        bpy.ops.paintstack.smart_apply(kind="SMART_MASK", name="Edge Wear")
        r = bpy.ops.paintstack.smart_save(kind="SMART_MASK", name="Test Wear")
    mask_path = smart.user_assets(smart.KIND_MASK).get("Test Wear")
    check(r == {"FINISHED"} and mask_path and os.path.isfile(mask_path), "smart mask saved to the library")

    with bpy.data.libraries.load(path) as (data_from, data_to):
        data_to.materials = data_from.materials
    lib_mat = data_to.materials[0]
    check(lib_mat.asset_data is not None and "PaintStack" in [t.name for t in lib_mat.asset_data.tags],
          "saved file holds a tagged material asset")
    bpy.data.materials.remove(lib_mat)
    libs = [l for l in bpy.context.preferences.filepaths.asset_libraries if l.name == smart.LIBRARY_NAME]
    check(len(libs) == 1, "PaintStack asset library registered in the preferences")

    # Fresh file, different mesh
    fresh_scene()
    obj2, mat2 = make_object("sphere")
    ts2 = mat2.paintstack
    with override(obj2):
        r = bpy.ops.paintstack.smart_apply(kind="SMART_MATERIAL", name="Test Steel", path=path)
    top = ts2.active_layer
    kids2 = [l for l in ts2.layers if l.parent_uid == top.uid]
    check(r == {"FINISHED"} and top.type == "FOLDER" and len(kids2) == 4, "saved smart material applied")
    uids = [l.uid for l in ts2.layers]
    check(len(uids) == len(set(uids)), "applied layers have unique uids")
    p2 = next(l for l in kids2 if l.type == "PAINT")
    p2_uid = p2.uid
    loaded = pixels.read(p2.channel("BASE_COLOR").image)
    check(np.allclose(loaded, painted_px, atol=0.01), "painted content travelled with the asset")
    check(generators.baked(ts2, "CURVATURE") is not None and masks.evaluate(
        next(l for l in kids2 if l.name == "Paint")) is not None, "generators re-bind to the new mesh's bakes")
    check(not [m for m in bpy.data.materials if m.name.startswith("Test Steel")],
          "temporary library material cleaned up")
    with override(obj2):
        bpy.ops.paintstack.layer_add(type="FILL")
        r = bpy.ops.paintstack.smart_apply(kind="SMART_MASK", name="Test Wear", path=mask_path)
    check(r == {"FINISHED"} and [e.type for e in ts2.active_layer.mask_effects] == ["EDGE_WEAR"],
          "saved smart mask applied")

    # Duplicate keeps painted pixels (Image.copy() would regenerate them blank)
    ts2.active_layer_index = index_of(ts2, p2_uid)
    with override(obj2):
        bpy.ops.paintstack.layer_duplicate()
    dup = pixels.read(ts2.active_layer.channel("BASE_COLOR").image)
    check(np.allclose(dup, painted_px, atol=0.01), "duplicated layer keeps painted pixels")

    # Clean up the user library
    os.remove(path)
    os.remove(mask_path)

    print("FAILED" if check.failed else "ALL PASSED")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
