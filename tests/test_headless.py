"""Headless integration test. Run with the `bpy` pip module (Blender as a Python
module) and a software OpenGL/EGL driver:

    pip install bpy numpy
    python tests/test_headless.py
"""

import math
import os
import sys
import tempfile

import bpy
import gpu
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _env import fresh_scene  # noqa: E402  (also sets up the GPU)

import paintstack  # noqa: E402
from paintstack import compositor, export, pixels, undo  # noqa: E402
from paintstack.engine.projector import Projector  # noqa: E402
from paintstack.stroke import Stroke  # noqa: E402

W = H = 400


def front_view():
    """Perspective camera at (0, -5, 0) looking at the origin along +Y."""
    cam = Matrix.Translation((0, -5, 0)) @ Matrix.Rotation(math.pi / 2, 4, "X")
    view = cam.inverted()
    f, n, fa = 1 / math.tan(math.radians(25)), 0.1, 100.0
    proj = Matrix(((f, 0, 0, 0), (0, f, 0, 0),
                   (0, 0, (fa + n) / (n - fa), 2 * fa * n / (n - fa)), (0, 0, -1, 0)))
    return (W, H, proj @ view, view, True)


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        check.failed = True


check.failed = False


def center_texel_mask(projector, view, radius=30):
    projector.begin_view(*view)
    projector.clear_mask()
    projector.dabs([(W / 2, H / 2, radius, 1.0, 1.0)])
    return projector.read_mask() > 0.5


def main():
    fresh_scene()
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24)
    obj = bpy.context.active_object
    dg = bpy.context.evaluated_depsgraph_get()

    # Texture set creation
    with bpy.context.temp_override(active_object=obj, object=obj):
        bpy.ops.paintstack.init()
    mat = obj.active_material
    ts = mat.paintstack
    ts.resolution = "512"
    check(ts.initialized and len(ts.layers) == 2, "texture set initialised with 2 layers")
    check(all(ts.output(c) is not None for c in ("BASE_COLOR", "ROUGHNESS", "HEIGHT")), "outputs created")
    names = {n.name for n in mat.node_tree.nodes}
    check("PS_Base Color" in names and "PS_Bump" in names, "material nodes built")

    # Fill layer values reach the output
    base = ts.layers[1]
    base.channel("BASE_COLOR").color = (1.0, 0.0, 0.0)
    base.channel("ROUGHNESS").value = 0.8
    compositor.composite(mat)
    out = pixels.read(ts.output("BASE_COLOR"))
    check(np.allclose(out[0, 0, :3], (1, 0, 0), atol=0.01), "fill colour composited")
    check(abs(pixels.read(ts.output("ROUGHNESS"))[5, 5, 0] - 0.8) < 0.01, "fill roughness composited")

    # Multi-channel stroke on the paint layer
    projector = Projector(obj, dg, mat, ts.res)
    hit = center_texel_mask(projector, front_view())
    check(hit.sum() > 100, f"dab covers texels ({hit.sum()})")

    ts.active_layer_index = 0
    brush = bpy.context.scene.paintstack_brush
    brush.base_color = (0.0, 0.0, 1.0)
    brush.roughness = 0.1
    brush.use_height = True
    brush.height = 1.0
    brush.flow = brush.opacity = 1.0
    brush.hardness = 1.0
    brush.radius = 30
    s = Stroke(projector, front_view(), mat, brush)
    s.add_point(W / 2, H / 2)
    s.finish()
    col = pixels.read(ts.output("BASE_COLOR"))
    rough = pixels.read(ts.output("ROUGHNESS"))
    check(np.allclose(col[hit][:, :3].mean(0), (0, 0, 1), atol=0.02), "stroke painted base colour")
    check(abs(rough[hit][:, 0].mean() - 0.1) < 0.02, "same stroke painted roughness")
    check(abs(pixels.read(ts.output("HEIGHT"))[hit][:, 0].mean() - 1.0) < 0.02, "same stroke painted height")
    far = ~pixels.grow_mask(hit.astype(np.float32), np.zeros_like(hit), 6).astype(bool)
    check(np.allclose(col[far][:, :3], (1, 0, 0), atol=0.01), "texels outside the brush untouched")

    # Back of the sphere must not receive paint (occlusion / facing)
    back = Matrix.Rotation(math.pi, 4, "Z")
    w, h, pm, vm, persp = front_view()
    back_hit = center_texel_mask(projector, (w, h, pm @ back, vm @ back, persp))
    check(back_hit.sum() > 100 and not (hit & back_hit).any(), "front and back dabs do not overlap")

    # Undo / redo
    undo.undo()
    check(np.allclose(pixels.read(ts.output("BASE_COLOR"))[hit][:, :3].mean(0), (1, 0, 0), atol=0.02),
          "undo restored stroke")
    undo.redo()
    check(np.allclose(pixels.read(ts.output("BASE_COLOR"))[hit][:, :3].mean(0), (0, 0, 1), atol=0.02),
          "redo re-applied stroke")

    # Erase
    brush.mode = "ERASE"
    s = Stroke(projector, front_view(), mat, brush)
    s.add_point(W / 2, H / 2)
    s.finish()
    check(np.allclose(pixels.read(ts.output("BASE_COLOR"))[hit][:, :3].mean(0), (1, 0, 0), atol=0.02),
          "erase revealed layer below")
    brush.mode = "PAINT"

    # Fill layer + painted mask (the core Substance workflow)
    with bpy.context.temp_override(active_object=obj, object=obj):
        bpy.ops.paintstack.layer_add(type="FILL")
    fill = ts.active_layer
    fill.channel("BASE_COLOR").color = (0.0, 1.0, 0.0)
    compositor.composite(mat)
    check(np.allclose(pixels.read(ts.output("BASE_COLOR"))[0, 0, :3], (0, 1, 0), atol=0.01),
          "new fill layer covers everything")
    s = Stroke(projector, front_view(), mat, brush)  # painting a fill layer => black mask, reveal
    s.add_point(W / 2, H / 2)
    s.finish()
    col = pixels.read(ts.output("BASE_COLOR"))
    check(fill.mask_image is not None, "fill layer got a mask")
    check(np.allclose(col[hit][:, :3].mean(0), (0, 1, 0), atol=0.02), "mask stroke revealed fill")
    check(np.allclose(col[far][:, :3], (1, 0, 0), atol=0.01), "masked-out area shows layers below")

    # Blend modes
    fill.channel("BASE_COLOR").blend_mode = "MULTIPLY"
    compositor.composite(mat)
    check(np.allclose(pixels.read(ts.output("BASE_COLOR"))[hit][:, :3].mean(0), (0, 0, 0), atol=0.02),
          "multiply blend (red x green = black)")

    # Export
    d = tempfile.mkdtemp()
    with bpy.context.temp_override(active_object=obj, object=obj):
        for preset in ("BLENDER", "UNREAL", "UNITY", "GLTF"):
            bpy.ops.paintstack.export(directory=os.path.join(d, preset), preset=preset)
    files = sorted(f for _, _, fs in os.walk(d) for f in fs)
    check(any("OcclusionRoughnessMetallic" in f for f in files), "Unreal ORM exported")
    check(any("MaskMap" in f for f in files), "Unity mask map exported")
    orm = bpy.data.images.load(os.path.join(d, "UNREAL", f"{mat.name}_OcclusionRoughnessMetallic.png"))
    orm_px = pixels.read(orm)
    check(abs(orm_px[0, 0, 1] - 0.8) < 0.01, "ORM green = roughness")

    # Save / reload keeps painted pixels
    blend = os.path.join(d, "test.blend")
    mat_name = mat.name
    bpy.ops.wm.save_as_mainfile(filepath=blend, copy=True)
    if bpy.app.background:
        bpy.ops.wm.open_mainfile(filepath=blend)
        mat = bpy.data.materials[mat_name]
    else:
        # In the Blender app, reloading the file from a running script frees the
        # window and screen the test runner is using (and crashes Blender), so
        # read the saved material back from the file instead.
        with bpy.data.libraries.load(blend, link=False) as (src, dst):
            dst.materials = [mat_name]
        mat = dst.materials[0]
    lc = mat.paintstack.layers[1].channel("BASE_COLOR")
    check(lc.image is not None and lc.image.packed_file is not None, "layer image packed in .blend")
    reread = pixels.read(mat.paintstack.output("BASE_COLOR"))
    check(reread.shape == (512, 512, 4), "output image reloads")

    print("\nFiles:", files)
    print("FAILED" if check.failed else "ALL PASSED")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
