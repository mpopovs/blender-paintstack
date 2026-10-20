"""Phase 3 demo: brush presets (splatter, scratches, square), a stencil, a folder and
a tri-planar normal-detail fill, rendered with Cycles.

    python tests/demo_phase3.py docs/phase3.png
"""

import math
import os
import sys

import bpy
import gpu
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
gpu.init()

import paintstack  # noqa: E402
from paintstack import brushes, compositor  # noqa: E402
from paintstack.engine.projector import Projector  # noqa: E402
from paintstack.stroke import Stroke  # noqa: E402
from test_headless import W, H, front_view  # noqa: E402
from test_phase3 import image_from, override  # noqa: E402

out = os.path.abspath(sys.argv[-1] if sys.argv[-1].endswith(".png") else "phase3.png")
bpy.ops.wm.read_factory_settings(use_empty=True)
paintstack.register()
bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32)
obj = bpy.context.active_object
bpy.ops.object.shade_smooth()
with override(obj):
    bpy.ops.paintstack.init()
mat = obj.active_material
ts = mat.paintstack
ts.resolution = "1024"
ts.use_normal = True
s = ts.bake
for m in ("normal", "ao", "curvature", "thickness", "id"):
    setattr(s, "use_" + m, False)
with override(obj):
    bpy.ops.paintstack.bake()

base = ts.layers[-1]
base.channel("BASE_COLOR").color = (0.62, 0.66, 0.70)
base.channel("ROUGHNESS").value = 0.35
base.channel("METALLIC").value = 1.0

# Hammered-metal normal detail, tri-planar so it has no UV seams
yy, xx = np.mgrid[0:128, 0:128].astype(np.float32) / 128
h = sum(np.sin((xx * a + yy * b) * 2 * np.pi * f) for a, b, f in ((1, 0, 6), (0, 1, 7), (0.7, 0.7, 5)))
gx, gy = np.gradient(h)
nrm = np.stack([-gx * 0.6, -gy * 0.6, np.ones_like(h)], -1)
nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
with override(obj):
    bpy.ops.paintstack.layer_add(type="FILL")
detail = ts.active_layer
detail.name = "Hammered"
for lc in detail.channels:
    lc.enabled = lc.channel == "NORMAL"
detail.channel("NORMAL").texture = image_from(nrm * 0.5 + 0.5, "hammered")
detail.projection = "TRIPLANAR"
detail.tiling = 2.0

# Paint inside a folder
with override(obj):
    bpy.ops.paintstack.layer_add(type="FOLDER")
    bpy.ops.paintstack.layer_add(type="PAINT")
proj = Projector(obj, bpy.context.evaluated_depsgraph_get(), mat, ts.res)
b = bpy.context.scene.paintstack_brush
b.use_height = False


def stroke(points, color, rough, metal, seed):
    b.base_color, b.roughness, b.metallic = color, rough, metal
    st = Stroke(proj, front_view(), mat, b, seed=seed)
    for p in points:
        st.add_point(*p)
    st.finish()


# Stencil: bold stripes, painted in red through it
stripes = (np.indices((64, 128))[1] // 16 % 2).astype(np.float32)
brushes.apply_preset(b, "Basic Hard")
b.radius, b.flow = 70, 1.0
b.use_stencil = True
b.stencil_image = image_from(np.repeat(stripes[..., None], 3, -1), "stripes")
b.stencil_x, b.stencil_y, b.stencil_size, b.stencil_rotation = 200, 245, 32, math.radians(-12)
stroke([(130 + i * 10, 245) for i in range(15)], (0.75, 0.05, 0.04), 0.4, 0.0, 1)
b.use_stencil = False

brushes.apply_preset(b, "Dirt Splatter")
b.radius = 40
stroke([(120 + i * 12, 150 + 25 * math.sin(i / 2)) for i in range(14)], (0.08, 0.06, 0.04), 0.9, 0.0, 2)

brushes.apply_preset(b, "Scratches")
b.radius = 45
stroke([(150 + i * 12, 232 + i * 3) for i in range(9)], (0.95, 0.95, 0.97), 0.15, 1.0, 3)

brushes.apply_preset(b, "Square")
b.radius, b.spacing = 9, 0.3
stroke([(200 + 62 * math.cos(t / 9), 200 + 62 * math.sin(t / 9)) for t in range(0, 57)],
       (1.0, 0.75, 0.2), 0.3, 1.0, 4)
compositor.composite(mat)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 32
scene.render.resolution_x = scene.render.resolution_y = 480
bpy.ops.object.camera_add(location=(0, -5, 0), rotation=(math.pi / 2, 0, 0))
scene.camera = bpy.context.active_object
scene.camera.data.angle = math.radians(27)
for loc, energy in (((-3, -3, 4), 700), ((3, -1, 1), 250)):
    bpy.ops.object.light_add(type="AREA", location=loc)
    light = bpy.context.active_object
    light.data.energy, light.data.size = energy, 3
    light.rotation_euler = (Vector((0, 0, 0)) - light.location).to_track_quat("-Z", "Y").to_euler()
world = bpy.data.worlds.new("W")
world.color = (0.15, 0.15, 0.17)
scene.world = world
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print("rendered", out)
