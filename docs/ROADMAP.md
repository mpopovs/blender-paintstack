# Roadmap

| Phase | Status | Scope |
|---|---|---|
| 0. GPU spike | ✅ | UV-space projection of brush dabs, depth prepass, read back |
| 1. MVP | ✅ v0.1 | Texture sets, paint and fill layers, blend modes, masks, multi-channel strokes, undo, export presets |
| 2. Masks and baking | next | Mesh-map baker (normal high→low, AO, curvature, thickness, position, world normal, ID); generators (edge wear, dirt, position gradient); a mask stack with levels and blur |
| 3. Painter UX | | Brush alphas and stencils, a brush preset library, layer folders, a channel isolation view, painting a normal channel (RNM blending), lazy mouse, line strokes |
| 4. Smart assets | | Smart materials and smart masks (save or load a layer stack as an asset), Asset Browser integration, a CC0 grunge library, filters (blur, HSL, warp, sharpen), anchor points |
| 5. Performance | | GPU compositor (keep layers as GPU textures, composite in GLSL), dirty-rectangle updates, tiled 4K/8K and UDIM, an optional C++ extension |

## Known limitations (v0.1)

- The compositor runs on the CPU (numpy) and writes full images. 2K is interactive; at 4K, live preview while painting will lag.
- One UV map (the active one) and one tile; no UDIM yet.
- Height is the only relief channel. Normals are derived from it in the material (Bump node) and on export.
- Blender's global undo does not cover pixels. Use stroke undo (Ctrl+Z inside paint mode).
- The projector reuses its mesh data while the vertex and face counts and the transform stay the same. Exit and restart paint mode after editing a mesh's shape (for example, changes made by modifiers).
