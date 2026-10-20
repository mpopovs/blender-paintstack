"""Phase 3 headless test: brush tips, rotation, stencil, dynamics, lazy mouse,
lines, presets, normal channel, fill textures, folders and channel views.

    python tests/test_phase3.py
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
from paintstack import brushes, compositor, masks, ops, pixels, stack  # noqa: E402
from paintstack.engine.projector import Projector  # noqa: E402
from paintstack.stroke import Stroke  # noqa: E402
from test_headless import W, H, check, front_view  # noqa: E402


def override(obj):
    return bpy.context.temp_override(active_object=obj, object=obj)


def setup():
    fresh_scene()
    # A plane facing the camera: UV space maps linearly onto the screen.
    bpy.ops.mesh.primitive_plane_add(size=2, rotation=(math.pi / 2, 0, 0))
    obj = bpy.context.active_object
    with override(obj):
        bpy.ops.paintstack.init()
    mat = obj.active_material
    mat.paintstack.resolution = "256"
    return obj, mat


def dab_mask(proj, tip=None, angle=0.0, radius=30, stencil=None):
    proj.begin_view(*front_view())
    proj.set_tip(tip)
    if stencil is not None:
        proj.set_stencil(*stencil)
    proj.clear_mask()
    proj.dabs([(W / 2, H / 2, radius, 1.0, 1.0, angle)])
    return proj.read_mask() > 0.5


def bbox(m):
    ys, xs = np.nonzero(m)
    return xs.max() - xs.min() + 1, ys.max() - ys.min() + 1


def image_from(arr, name):
    h, w = arr.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    rgba = arr if arr.shape[-1] == 4 else np.concatenate([arr, np.ones((h, w, 1), np.float32)], -1)
    img.pixels.foreach_set(rgba.astype(np.float32).ravel())
    return img


def main():
    obj, mat = setup()
    ts = mat.paintstack
    b = bpy.context.scene.paintstack_brush
    proj = Projector(obj, bpy.context.evaluated_depsgraph_get(), mat, ts.res)

    # --- Tips
    for t, _, _ in brushes.TIPS:
        b.tip = t
        arr = brushes.tip_array(b)
        ok = (arr is None) if t in ("ROUND", "IMAGE") else (arr.shape == (128, 128) and arr.max() > 0.5)
        check(ok, f"tip {t} generated")
    b.hardness = 1.0
    round_m = dab_mask(proj)
    b.tip = "SQUARE"
    square_m = dab_mask(proj, brushes.tip_array(b))
    ratio = square_m.sum() / round_m.sum()
    check(abs(ratio - 4 / math.pi) < 0.08, f"square tip area / round = {ratio:.2f} (4/pi = 1.27)")
    rot_m = dab_mask(proj, brushes.tip_array(b), angle=math.radians(45))
    w0, w45 = bbox(square_m)[0], bbox(rot_m)[0]
    check(abs(w45 / w0 - math.sqrt(2)) < 0.12, f"45 deg rotation widens the square by sqrt2 ({w45 / w0:.2f})")
    check(abs(rot_m.sum() / square_m.sum() - 1) < 0.05, "rotation keeps the tip area")

    # --- Stencil: left half black, right half white, centred on the dab
    sten = np.zeros((64, 64), np.float32)
    sten[:, 32:] = 1.0
    sm = dab_mask(proj, None, stencil=(sten, W / 2, H / 2, 100.0, 0.0))
    left, right = sm[:, :120].sum(), sm[:, 136:].sum()
    check(left == 0 and right > 100, f"stencil masks the left half (left {left}, right {right})")
    sm_rot = dab_mask(proj, None, stencil=(sten, W / 2, H / 2, 100.0, math.pi))
    check(sm_rot[:, 136:].sum() == 0 and sm_rot[:, :120].sum() > 100, "rotated stencil flips the side")

    # --- Dynamics are seeded and change the result
    b.tip = "ROUND"
    b.radius, b.flow, b.opacity, b.spacing = 20, 1.0, 1.0, 0.25
    b.size_jitter, b.scatter = 0.8, 1.0
    ts.active_layer_index = 0
    results = []
    for seed in (1, 1, 2):
        s = Stroke(proj, front_view(), mat, b, seed=seed)
        for i in range(10):
            s.add_point(120 + i * 16, 200)
        results.append(proj.read_mask())
        undo_restore = s.targets
        s.finish()
        from paintstack import undo
        undo.undo()
    check(np.array_equal(results[0], results[1]), "same seed -> identical jittered stroke")
    check(not np.array_equal(results[0], results[2]), "different seed -> different jittered stroke")
    b.size_jitter = b.scatter = 0.0

    # --- Lazy mouse: small moves within the string do not paint further
    b.lazy_radius = 50
    s = Stroke(proj, front_view(), mat, b, seed=0)
    s.add_point(150, 200)
    first = proj.read_mask().sum()
    s.add_point(180, 200)  # 30 px < 50 px string
    check(proj.read_mask().sum() == first and s.pen == (150, 200), "lazy mouse holds the brush back")
    s.add_point(260, 200)  # 110 px: brush moves 60 px towards the cursor
    check(abs(s.pen[0] - 210) < 1e-6, f"lazy mouse trails by the string length (pen at {s.pen[0]:.0f})")
    s.finish()
    undo.undo()
    b.lazy_radius = 0

    # --- Straight line
    s = Stroke(proj, front_view(), mat, b, seed=0)
    s.line(130, 200, 270, 200)
    m = proj.read_mask() > 0.5
    s.finish()
    undo.undo()
    cols = np.nonzero(m.any(0))[0]
    check(cols.max() - cols.min() > 180 and m[:, 128].any(), "Shift+click line covers the whole segment")

    # --- Presets
    check(brushes.apply_preset(b, "Dirt Splatter") and b.tip == "SPLATTER" and b.scatter > 0,
          "built-in preset applied")
    name = brushes.save_preset(b, "Test Brush")
    check("Test Brush" in brushes.user_presets(), "user preset saved")
    b.tip = "ROUND"
    brushes.apply_preset(b, "Test Brush")
    check(b.tip == "SPLATTER", "user preset loaded")
    check(brushes.delete_preset(name) and "Test Brush" not in brushes.user_presets(), "user preset deleted")
    brushes.apply_preset(b, "Basic Hard")

    # --- Normal channel + fill texture (UV) with normal-map combine
    ts.use_normal = True
    res = ts.res
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32) / 64
    bumps = np.stack([0.5 + 0.3 * np.sin(xx * 2 * np.pi), 0.5 + 0.3 * np.cos(yy * 2 * np.pi),
                      np.full_like(xx, 0.8)], -1)
    with override(obj):
        bpy.ops.paintstack.layer_add(type="FILL")
    nfill = ts.active_layer
    nfill.name = "Normal detail"
    for lc in nfill.channels:
        lc.enabled = lc.channel == "NORMAL"
    nfill.channel("NORMAL").texture = image_from(bumps, "bumps")
    nfill.tiling = 2.0
    compositor.composite(mat)
    out_n = pixels.read(ts.output("NORMAL"))[..., :3]
    dec = out_n * 2 - 1
    check(np.allclose(np.linalg.norm(dec, axis=-1), 1.0, atol=0.02), "normal output is unit length")
    check(dec[..., 0].std() > 0.1, "normal detail from the fill texture")
    # Tiling 2: texture repeats every half of the UV range
    check(np.allclose(out_n[:, : res // 2], out_n[:, res // 2:], atol=0.02), "UV tiling repeats the texture")
    names = {n.name for n in mat.node_tree.nodes}
    check("PS_Normal" in names and "PS_Normal Map" in names, "normal channel wired into the material")

    # Paint flat normals with Replace over the detail
    with override(obj):
        bpy.ops.paintstack.layer_add(type="PAINT")
    flat = ts.active_layer
    flat.channel("NORMAL").blend_mode = "NORMAL"
    for ch in ("base_color", "metallic", "roughness", "height"):
        setattr(b, "use_" + ch, False)
    b.use_normal = True
    b.normal = (0.5, 0.5, 1.0)
    b.radius = 30
    hit = dab_mask(proj, radius=20)  # core of the stroke, away from its soft edge
    s = Stroke(proj, front_view(), mat, b, seed=0)
    s.add_point(W / 2, H / 2)
    s.finish()
    out_n = pixels.read(ts.output("NORMAL"))[..., :3]
    check(np.allclose(out_n[hit], (0.5, 0.5, 1.0), atol=0.03), "painted flat normal erases detail")

    # --- Folders
    with override(obj):
        bpy.ops.paintstack.layer_add(type="FOLDER")
    folder = ts.active_layer
    folder_uid = folder.uid
    with override(obj):
        bpy.ops.paintstack.layer_add(type="FILL")  # goes inside the expanded folder
    inner = ts.active_layer
    inner.channel("BASE_COLOR").color = (0.0, 0.0, 1.0)
    check(inner.parent_uid == folder_uid, "new layer created inside the active folder")
    compositor.composite(mat)
    col = pixels.read(ts.output("BASE_COLOR"))
    check(np.allclose(col[5, 5, :3], (0, 0, 1), atol=0.01), "folder content composites")
    folder = next(l for l in ts.layers if l.uid == folder_uid)
    folder.opacity = 0.0
    compositor.composite(mat)
    check(not np.allclose(pixels.read(ts.output("BASE_COLOR"))[5, 5, :3], (0, 0, 1), atol=0.05),
          "folder opacity 0 hides its contents")
    folder.opacity = 0.5
    compositor.composite(mat)
    below = (0.5, 0.5, 0.5)
    check(np.allclose(pixels.read(ts.output("BASE_COLOR"))[5, 5, :3],
                      [(c + t) / 2 for c, t in zip(below, (0, 0, 1))], atol=0.02),
          "folder opacity 0.5 fades its contents")
    folder.opacity = 1.0

    # Paint into a layer inside the folder: the stroke path must match a full composite
    ts.active_layer_index = next(i for i, l in enumerate(ts.layers) if l.uid == inner.uid)
    with override(obj):
        bpy.ops.paintstack.layer_add(type="PAINT")
    painted = ts.active_layer
    check(painted.parent_uid == folder_uid, "paint layer added next to sibling inside the folder")
    b.use_base_color, b.use_normal = True, False
    b.base_color = (1.0, 1.0, 0.0)
    s = Stroke(proj, front_view(), mat, b, seed=0)
    s.add_point(W / 2, H / 2)
    s.finish()
    live = pixels.read(ts.output("BASE_COLOR"))[..., :3].copy()
    compositor.composite(mat)
    full = pixels.read(ts.output("BASE_COLOR"))[..., :3]
    check(np.allclose(live, full, atol=1e-3), "stroke inside a folder == full recomposite")
    check(np.allclose(full[hit].mean(0), (1, 1, 0), atol=0.03), "stroke inside folder visible")

    # Moving out of and into folders
    idx = ts.active_layer_index
    order_before = [l.name for l in ts.layers]
    check(stack.move(ts, idx, "UP") and ts.active_layer.parent_uid == "", "move up out of the folder")
    check(stack.move(ts, ts.active_layer_index, "DOWN") and ts.active_layer.parent_uid == folder_uid,
          "move down into the folder")
    blocks_ok = all(stack.block_end(ts, i) <= len(ts.layers) for i in range(len(ts.layers)))
    check(blocks_ok and [l.name for l in ts.layers] == order_before, "folder blocks stay contiguous")

    # Folder mask hides its content where black
    ts.active_layer_index = next(i for i, l in enumerate(ts.layers) if l.uid == folder_uid)
    with override(obj):
        bpy.ops.paintstack.mask_add(white=False)
    compositor.composite(mat)
    col = pixels.read(ts.output("BASE_COLOR"))[..., :3]
    check(not np.allclose(col[hit].mean(0), (1, 1, 0), atol=0.1), "black folder mask hides folder contents")

    # --- Channel / mask views
    with override(obj):
        bpy.ops.paintstack.view_cycle()
    check(ts.view_mode == "BASE_COLOR", "C cycles to the first channel")
    nodes = mat.node_tree.nodes
    active_out = [n for n in nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output]
    check(len(active_out) == 1 and active_out[0].name == "PS_View Output", "channel view output active")
    ts.view_mode = "MASK"
    compositor.composite(mat)
    mv = pixels.read(bpy.data.images[f"PS_{mat.name}_maskview"])[..., 0]
    check(np.allclose(mv, masks.evaluate(ts.active_layer), atol=0.01), "mask view shows the active mask")
    with override(obj):
        bpy.ops.paintstack.view_cycle(reset=True)
    active_out = [n for n in mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output]
    check(ts.view_mode == "MATERIAL" and active_out and not active_out[0].name.startswith("PS_"),
          "M returns to the material output")

    # Removing a folder removes its contents
    n_before = len(ts.layers)
    with override(obj):
        bpy.ops.paintstack.layer_remove()
    check(len(ts.layers) == n_before - 3 and all(l.parent_uid != folder_uid for l in ts.layers),
          "deleting a folder deletes its contents")

    # Group / ungroup
    ts.active_layer_index = 0
    uid0 = ts.layers[0].uid
    with override(obj):
        bpy.ops.paintstack.layer_group()
    check(ts.active_layer.type == "FOLDER" and ts.layers[1].parent_uid == ts.active_layer.uid,
          "group wraps the layer in a folder")
    with override(obj):
        bpy.ops.paintstack.layer_ungroup()
    check(ts.layers[0].uid == uid0 and ts.layers[0].parent_uid == "", "ungroup restores the layer")

    # --- Tri-planar fill: on an axis-aligned plane it must match UV projection
    s_ = ts.bake
    for m in ("normal", "ao", "curvature", "thickness", "id"):
        setattr(s_, "use_" + m, False)
    s_.use_position = s_.use_world_normal = True
    with override(obj):
        bpy.ops.paintstack.bake()
    checker = np.kron(np.indices((4, 4)).sum(0) % 2, np.ones((16, 16)))[..., None].repeat(3, -1)
    from paintstack import fills
    with override(obj):
        bpy.ops.paintstack.layer_add(type="FILL")
    tri = ts.active_layer
    tri.channel("BASE_COLOR").texture = image_from(checker.astype(np.float32), "checker")
    uv_img = fills.sample(ts, tri, tri.channel("BASE_COLOR").texture)
    tri.projection = "TRIPLANAR"
    tp_img = fills.sample(ts, tri, tri.channel("BASE_COLOR").texture, allow_triplanar=True)
    inner = (slice(8, -8), slice(8, -8))
    agree = (np.abs(uv_img[inner] - tp_img[inner]) < 0.1).mean()
    check(agree > 0.95, f"tri-planar matches UV on an axis-aligned plane ({agree:.0%} texels)")

    print("FAILED" if check.failed else "ALL PASSED")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
