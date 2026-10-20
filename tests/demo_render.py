"""Paints a few multi-channel strokes on a sphere headlessly and renders the result.

    python tests/demo_render.py out.png
"""

import math
import os
import sys

import bpy
import gpu
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _env import fresh_scene  # noqa: E402  (also sets up the GPU)

import paintstack  # noqa: E402
from paintstack import compositor  # noqa: E402
from paintstack.engine.projector import Projector  # noqa: E402
from paintstack.stroke import Stroke  # noqa: E402
from test_headless import W, H, front_view  # noqa: E402

out = os.path.abspath(sys.argv[-1] if sys.argv[-1].endswith(".png") else "demo.png")
fresh_scene()
bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32)
obj = bpy.context.active_object
bpy.ops.object.shade_smooth()
with bpy.context.temp_override(active_object=obj, object=obj):
    bpy.ops.paintstack.init()
mat = obj.active_material
ts = mat.paintstack
ts.resolution = "1024"
ts.bump_strength = 0.5
base = ts.layers[1]
base.channel("BASE_COLOR").color = (0.75, 0.72, 0.68)
base.channel("ROUGHNESS").value = 0.7
compositor.composite(mat)

proj = Projector(obj, bpy.context.evaluated_depsgraph_get(), mat, ts.res)
b = bpy.context.scene.paintstack_brush
b.flow, b.hardness, b.radius = 0.8, 0.5, 18
b.use_height = True

def paint(points, color, rough, metal, height):
    b.base_color, b.roughness, b.metallic, b.height = color, rough, metal, height
    s = Stroke(proj, front_view(), mat, b)
    for p in points:
        s.add_point(*p)
    s.finish()

# Gold metallic swirl
paint([(200 + 110 * math.cos(t / 10) * (t / 60), 200 + 110 * math.sin(t / 10) * (t / 60)) for t in range(0, 60)],
      (1.0, 0.77, 0.34), 0.2, 1.0, 1.0)
# Glossy red zig-zag, recessed
paint([(110 + i * 6, 120 + (30 if (i // 5) % 2 else 0) + (i % 5) * (6 if (i // 5) % 2 == 0 else -6)) for i in range(32)],
      (0.75, 0.05, 0.05), 0.05, 0.0, 0.2)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 24
scene.render.resolution_x = scene.render.resolution_y = 480
bpy.ops.object.camera_add(location=(0, -5, 0), rotation=(math.pi / 2, 0, 0))
scene.camera = bpy.context.active_object
scene.camera.data.angle = math.radians(50)
bpy.ops.object.light_add(type="AREA", location=(-3, -3, 4))
light = bpy.context.active_object
light.data.energy = 900
light.data.size = 3
light.rotation_euler = (Vector((0, 0, 0)) - light.location).to_track_quat("-Z", "Y").to_euler()
world = bpy.data.worlds.new("W")
world.color = (0.05, 0.05, 0.06)
scene.world = world
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print("rendered", out)
