"""Bakes all mesh maps for Suzanne (subdivided high poly -> low poly) and writes
a contact sheet of the maps plus a generator-driven render.

    python tests/demo_bake.py docs/
"""

import math
import os
import sys

import bpy
import gpu
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
gpu.init()

import paintstack  # noqa: E402
from paintstack import compositor, pixels  # noqa: E402

out_dir = os.path.abspath(sys.argv[-1] if os.path.isdir(sys.argv[-1]) else ".")
RES = 512

bpy.ops.wm.read_factory_settings(use_empty=True)
paintstack.register()
bpy.ops.mesh.primitive_monkey_add()
low = bpy.context.active_object
bpy.ops.object.shade_smooth()
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(island_margin=0.01)
bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.mesh.primitive_monkey_add()
high = bpy.context.active_object
high.modifiers.new("Sub", "SUBSURF").levels = 3
high.modifiers["Sub"].render_levels = 3
bpy.ops.object.shade_smooth()
high.hide_set(True)
high.hide_render = True
bpy.context.view_layer.objects.active = low
low.select_set(True)

with bpy.context.temp_override(active_object=low, object=low):
    bpy.ops.paintstack.init()
mat = low.active_material
ts = mat.paintstack
ts.resolution = str(RES)
ts.bake.high_poly = high
ts.bake.use_id = True
ts.bake.ao_distance = 0.6
ts.bake.thickness_distance = 0.4
with bpy.context.temp_override(active_object=low, object=low):
    bpy.ops.paintstack.bake()

# Contact sheet: 4 x 2 tiles
names = ["NORMAL", "AO", "CURVATURE", "THICKNESS", "POSITION", "WORLD_NORMAL", "ID"]
tile = 256
sheet = np.full((tile * 2, tile * 4, 4), 0.12, np.float32)
sheet[..., 3] = 1.0
for i, n in enumerate(names):
    px = pixels.read(ts.baked_map(n))
    px = px[:: RES // tile, :: RES // tile]
    rgb = px[..., :3] if n not in ("AO", "CURVATURE", "THICKNESS") else np.repeat(px[..., :1], 3, -1)
    r, c = 1 - i // 4, i % 4
    sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile, :3] = np.clip(rgb, 0, 1) ** (1 / 2.2)
img = bpy.data.images.new("sheet", tile * 4, tile * 2, alpha=True)
img.pixels.foreach_set(sheet.ravel())
img.filepath_raw = os.path.join(out_dir, "bakes.png")
img.file_format = "PNG"
img.save()
for n in names:
    a = pixels.read(ts.baked_map(n))[..., :3]
    print(f"{n:13s} mean={a.mean():.3f} min={a.min():.3f} max={a.max():.3f}")

# Smart material from generators: painted steel with edge wear and dirt
base = ts.layers[-1]
base.channel("BASE_COLOR").color = (0.13, 0.25, 0.45)
base.channel("ROUGHNESS").value = 0.45
with bpy.context.temp_override(active_object=low, object=low):
    bpy.ops.paintstack.layer_add(type="FILL")
    wear = ts.active_layer
    wear.name = "Edge Wear"
    wear.channel("BASE_COLOR").color = (0.75, 0.75, 0.78)
    wear.channel("METALLIC").value = 1.0
    wear.channel("ROUGHNESS").value = 0.25
    wear.channel("HEIGHT").enabled = False
    bpy.ops.paintstack.effect_add(type="EDGE_WEAR")
    wear.active_mask_effect.level = 0.45
    bpy.ops.paintstack.layer_add(type="FILL")
    dirt = ts.active_layer
    dirt.name = "Dirt"
    dirt.channel("BASE_COLOR").color = (0.18, 0.12, 0.07)
    dirt.channel("ROUGHNESS").value = 0.9
    dirt.channel("METALLIC").enabled = False
    dirt.channel("HEIGHT").enabled = False
    bpy.ops.paintstack.effect_add(type="DIRT")
    dirt.active_mask_effect.level = 0.4
compositor.composite(mat)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 32
scene.render.resolution_x = scene.render.resolution_y = 480
bpy.ops.object.camera_add(location=(1.6, -3.4, 0.9))
cam = bpy.context.active_object
cam.rotation_euler = (Vector((0, 0, 0.1)) - cam.location).to_track_quat("-Z", "Y").to_euler()
scene.camera = cam
bpy.ops.object.light_add(type="AREA", location=(-2.5, -3, 3))
key = bpy.context.active_object
key.data.energy, key.data.size = 500, 3
key.rotation_euler = (Vector((0, 0, 0)) - key.location).to_track_quat("-Z", "Y").to_euler()
bpy.ops.object.light_add(type="AREA", location=(3, 1, 2))
rim = bpy.context.active_object
rim.data.energy, rim.data.size = 250, 2
rim.rotation_euler = (Vector((0, 0, 0)) - rim.location).to_track_quat("-Z", "Y").to_euler()
world = bpy.data.worlds.new("W")
world.color = (0.06, 0.06, 0.07)
scene.world = world
scene.render.filepath = os.path.join(out_dir, "generators.png")
bpy.ops.render.render(write_still=True)
print("done")
