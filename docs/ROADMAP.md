# Roadmap

| Phase | Status | Scope |
|---|---|---|
| 0. GPU spike | ✅ | UV-space projection of brush dabs, depth prepass, read back |
| 1. MVP | ✅ v0.1 | Texture sets, paint and fill layers, blend modes, masks, multi-channel strokes, undo, export presets |
| 2. Masks and baking | ✅ v0.2.1 | Mesh-map baker (normal high→low, AO, curvature, thickness, position, world normal, ID; cage object; non-blocking); generators (edge wear, dirt, curvature, AO, thickness, position gradient, 3D noise on the GPU); Color Selection from the ID map; a mask stack with paint, fill, levels and blur |
| 3. Painter UX | ✅ v0.3 | Brush tips (built-in and image) with rotation and jitter, a screen-space stencil, a brush preset library, layer folders, a channel and mask isolation view, a Normal channel with whiteout blending, fill textures (UV and tri-planar), lazy mouse, line strokes |
| 4. Smart assets | next | Smart materials and smart masks (save or load a layer stack as an asset), Asset Browser integration, a CC0 grunge library, filters (blur, HSL, warp, sharpen), anchor points |
| 5. Performance | | GPU compositor (keep layers as GPU textures, composite in GLSL), dirty-rectangle updates, tiled 4K/8K and UDIM, an optional C++ extension |

## Known limitations (v0.3)

- Painting the Normal channel uses a fixed normal value (for example, flat to erase detail). Painting normals from the brush tip's shape is not implemented yet.
- Tri-planar projection is not available for the Normal channel, which would need tangent-space reorientation. Normal fills use UV projection.
- Folders are always pass-through. Isolated folders, where blend modes inside the folder do not see the layers below, are not supported yet.
- Brush tips are sampled from a single texture with no mipmaps, so very small dabs from a detailed image tip can alias.

- The UI only updates between maps, not while a single map is baking. A 2K AO bake at 32 samples takes seconds to minutes, depending on the mesh and CPU.
- Generator thresholds and the final blending still run on the CPU. Noise runs on the GPU. Changing Level or Softness takes about 15 ms at 1K, but the full layer composite after it scales with the number of layers.

- The compositor runs on the CPU (numpy) and writes full images. 2K is interactive; at 4K, live preview while painting will lag.
- One UV map (the active one) and one tile; no UDIM yet.
- Height is the only relief channel. Normals are derived from it in the material (Bump node) and on export.
- Blender's global undo does not cover pixels. Use stroke undo (Ctrl+Z inside paint mode).
- The projector reuses its mesh data while the vertex and face counts and the transform stay the same. Exit and restart paint mode after editing a mesh's shape (for example, changes made by modifiers).
