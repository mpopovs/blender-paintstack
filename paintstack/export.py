"""Export the composited channels as texture files, packed per target engine."""

import os

import bpy
import numpy as np
from bpy.props import EnumProperty, IntProperty, StringProperty
from bpy.types import Operator

from . import compositor, generators, ops, pixels
from .channels import BY_ID, enabled_channels

PRESETS = [
    ("BLENDER", "Blender / Generic", "BaseColor, Metallic, Roughness, Normal, Height, Emission"),
    ("UNREAL", "Unreal Engine", "BaseColor, Normal (DirectX), OcclusionRoughnessMetallic"),
    ("UNITY", "Unity (URP/HDRP)", "BaseColor, Normal (OpenGL), MaskMap (Metal, AO, Detail, Smoothness)"),
    ("GLTF", "glTF", "BaseColor, Normal (OpenGL), MetallicRoughness (G=Roughness, B=Metal)"),
]


def normal_from_height(height, strength):
    """Tangent-space normal map from a height map via central differences."""
    h = height * strength * height.shape[0] / 256.0
    dx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5
    dy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5
    n = np.stack([-dx, -dy, np.ones_like(h)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


def combine_normals(base, detail):
    """Whiteout blend of two tangent-space normal maps (0..1 encoded, OpenGL)."""
    b = base * 2.0 - 1.0
    d = detail * 2.0 - 1.0
    n = np.stack([b[..., 0] + d[..., 0], b[..., 1] + d[..., 1], b[..., 2] * d[..., 2]], -1)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    return n * 0.5 + 0.5


def gray(maps, cid):
    """Scalar map for a channel, or its default if the channel is disabled."""
    if cid in maps:
        return maps[cid][..., 0]
    return None


def build_maps(ts, preset, strength):
    """{suffix: (rgb_or_rgba array, srgb)} for one preset."""
    maps = compositor.composite_all(ts)
    one = np.ones(ts.shape, np.float32)
    zero = np.zeros(ts.shape, np.float32)
    metal = gray(maps, "METALLIC")
    rough = gray(maps, "ROUGHNESS")
    metal = zero if metal is None else metal
    rough = one * BY_ID["ROUGHNESS"].default[0] if rough is None else rough
    height = gray(maps, "HEIGHT")
    ao_map = generators.baked(ts, "AO")
    ao = one if ao_map is None else ao_map[..., 0]
    baked_normal = generators.baked(ts, "NORMAL")

    out = {}
    if "BASE_COLOR" in maps:
        out["BaseColor"] = (maps["BASE_COLOR"], True)
    # The Normal channel already starts from the baked normal.
    normal = compositor.encode_normal(compositor.decode_normal(maps["NORMAL"])) if "NORMAL" in maps \
        else baked_normal
    if height is not None:
        detail = normal_from_height(height, strength)
        normal = detail if normal is None else combine_normals(normal, detail)
    if normal is not None:
        if preset == "UNREAL":  # DirectX: green down
            normal = normal.copy()
            normal[..., 1] = 1.0 - normal[..., 1]
        out["Normal"] = (normal, False)

    if preset == "BLENDER":
        out["Metallic"] = (np.stack([metal] * 3, -1), False)
        out["Roughness"] = (np.stack([rough] * 3, -1), False)
        if height is not None:
            out["Height"] = (np.stack([height] * 3, -1), False)
        if ao_map is not None:
            out["AmbientOcclusion"] = (np.stack([ao] * 3, -1), False)
    elif preset == "UNREAL":
        out["OcclusionRoughnessMetallic"] = (np.stack([ao, rough, metal], -1), False)
    elif preset == "UNITY":
        out["MaskMap"] = (np.stack([metal, ao, zero, 1.0 - rough], -1), False)
    elif preset == "GLTF":
        out["MetallicRoughness"] = (np.stack([zero, rough, metal], -1), False)
        if ao_map is not None:
            out["Occlusion"] = (np.stack([ao] * 3, -1), False)
    if "EMISSION" in maps:
        out["Emission"] = (maps["EMISSION"], True)
    return out


def split_tiles(arr, ts):
    """[(tile, array)] for each UDIM tile of a texture-set-sized array."""
    from .props import tile_cell
    res = ts.res
    out = []
    for tile in ts.tiles:
        c, r = tile_cell(tile)
        out.append((tile, arr[r * res:(r + 1) * res, c * res:(c + 1) * res]))
    return out


def save_png(path, arr, srgb):
    h, w = arr.shape[:2]
    if arr.shape[-1] == 3:
        arr = pixels.rgba(arr, np.ones((h, w), np.float32))
    img = bpy.data.images.new("PS_export_tmp", w, h, alpha=True)
    try:
        img.colorspace_settings.name = "sRGB" if srgb else "Non-Color"
        img.pixels.foreach_set(np.clip(arr, 0.0, 1.0).astype(np.float32).ravel())
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
    finally:
        bpy.data.images.remove(img)


class PAINTSTACK_OT_export(Operator):
    """Export the texture set's maps for a game engine or renderer"""
    bl_idname = "paintstack.export"
    bl_label = "Export Textures"

    directory: StringProperty(subtype="DIR_PATH")
    preset: EnumProperty(name="Preset", items=PRESETS, default="BLENDER")
    padding: IntProperty(name="Padding", min=0, max=64, default=8,
                         description="Texels of edge padding around UV islands")
    normal_strength: IntProperty(name="Normal Strength", min=1, max=100, default=10)

    @classmethod
    def poll(cls, context):
        return ops.texture_set(context)[0] is not None

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        mat, ts = ops.texture_set(context)
        compositor.sync_outputs(mat)
        directory = bpy.path.abspath(self.directory)
        os.makedirs(directory, exist_ok=True)
        maps = build_maps(ts, self.preset, self.normal_strength / 10.0)

        coverage = None
        if self.padding:
            try:
                from .engine.projector import Projector
                p = Projector(context.active_object, context.evaluated_depsgraph_get(), mat, ts.res)
                coverage = p.coverage()
                p.free()
            except Exception as e:  # no GPU context (e.g. background mode)
                self.report({"WARNING"}, f"Edge padding skipped: {e}")

        base = bpy.path.clean_name(mat.name)
        tiles = ts.tiles
        udim = ts.grid != (1, 1) or tiles != [1001]
        files = 0
        for suffix, (arr, srgb) in maps.items():
            if coverage is not None:
                arr = pixels.dilate(arr, coverage, self.padding)
            if not udim:
                save_png(os.path.join(directory, f"{base}_{suffix}.png"), arr, srgb)
                files += 1
                continue
            # UDIM: one <name>.<tile>.png per tile, the naming every DCC and engine reads.
            for tile, part in split_tiles(arr, ts):
                save_png(os.path.join(directory, f"{base}_{suffix}.{tile}.png"), part, srgb)
                files += 1
        self.report({"INFO"}, f"Exported {files} files ({len(maps)} maps) to {directory}")
        return {"FINISHED"}


classes = (PAINTSTACK_OT_export,)
