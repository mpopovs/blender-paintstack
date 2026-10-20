"""Phase 4 demo: the same built-in smart materials applied to four Suzannes; each
adapts to its own baked maps. Rendered with Cycles.

    python tests/demo_phase4.py docs/smart_materials.png
"""

import math
import os
import sys

import bpy
import gpu
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
gpu.init()

import paintstack  # noqa: E402
from test_phase3 import override  # noqa: E402

out = os.path.abspath(sys.argv[-1] if sys.argv[-1].endswith(".png") else "smart_materials.png")
bpy.ops.wm.read_factory_settings(use_empty=True)
paintstack.register()

NAMES = ["Painted Steel", "Rusty Iron", "Worn Gold", "Dusty Plastic"]
for i, name in enumerate(NAMES):
    bpy.ops.mesh.primitive_monkey_add(location=((i - 1.5) * 2.3, 0, 0), rotation=(0, 0, math.radians(-20)))
    obj = bpy.context.active_object
    mod = obj.modifiers.new("Sub", "SUBSURF")
    mod.levels = mod.render_levels = 2
    bpy.ops.object.modifier_apply(modifier="Sub")
    bpy.ops.object.shade_smooth()
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(island_margin=0.01)
    bpy.ops.object.mode_set(mode="OBJECT")
    with override(obj):
        bpy.ops.paintstack.init()
    ts = obj.active_material.paintstack
    ts.resolution = "1024"
    s = ts.bake
    s.samples = 16
    s.use_normal = s.use_id = False
    s.ao_distance = 0.5
    with override(obj):
        bpy.ops.paintstack.bake()
        bpy.ops.paintstack.smart_apply(kind="SMART_MATERIAL", name=name)
    print("applied", name)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 48
scene.render.resolution_x, scene.render.resolution_y = 1200, 420
bpy.ops.object.camera_add(location=(0, -9.5, 1.2))
cam = bpy.context.active_object
cam.rotation_euler = (Vector((0, 0, 0.15)) - cam.location).to_track_quat("-Z", "Y").to_euler()
cam.data.angle = math.radians(52)
scene.camera = cam
for loc, energy, size in (((-4, -6, 6), 2500, 6), ((6, -2, 3), 900, 4), ((0, 5, 4), 900, 5)):
    bpy.ops.object.light_add(type="AREA", location=loc)
    light = bpy.context.active_object
    light.data.energy, light.data.size = energy, size
    light.rotation_euler = (Vector((0, 0, 0)) - light.location).to_track_quat("-Z", "Y").to_euler()
world = bpy.data.worlds.new("W")
world.color = (0.12, 0.12, 0.14)
scene.world = world
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print("rendered", out)
