# Roadmap

| Phase | Status | Scope |
|---|---|---|
| 0. GPU spike | ✅ | UV-space projection of brush dabs, depth prepass, read back |
| 1. MVP | ✅ v0.1 | Texture sets, paint and fill layers, blend modes, masks, multi-channel strokes, undo, export presets |
| 2. Masks and baking | ✅ v0.2.1 | Mesh-map baker (normal high→low, AO, curvature, thickness, position, world normal, ID; cage object; non-blocking); generators (edge wear, dirt, curvature, AO, thickness, position gradient, 3D noise on the GPU); Color Selection from the ID map; a mask stack with paint, fill, levels and blur |
| 3. Painter UX | ✅ v0.3 | Brush tips (built-in and image) with rotation and jitter, a screen-space stencil, a brush preset library, layer folders, a channel and mask isolation view, a Normal channel with whiteout blending, fill textures (UV and tri-planar), lazy mouse, line strokes |
| 4. Smart assets | ✅ v0.4 | Built-in and saved smart materials and smart masks (as `.blend` assets in an Asset Browser library), a procedural seamless grunge library, filter layers (blur, sharpen, HSL, warp), anchor points |
| 5. Performance | ✅ v0.5 | Region (dirty-rectangle) strokes with partial GPU readback; a persistent compositor cache with per-layer and per-channel invalidation and a memory budget; a below-checkpoint; cover culling; sparse and in-place blending; deferred layer writes. 3–42× faster (see the README) |
| 6. Apple Silicon | ✅ v0.6 | Metal-safe formats, a GPU live stroke preview (regions and mips), multithreaded compositing, unified-memory cache sizing, Retina and Mac input, on-machine diagnostics, a GPU state crash fix |
| 7. GPU compositor | ✅ v0.7 | GPU blending, folders and filters checked against the CPU; display without uploads plus idle sync; Auto backend with CPU fallback; a Mac test kit and report |
| 8. Scale | next | UDIM texture sets, 8K (half-float GPU-resident layers), masks and generators on the GPU, and tuning from the M2 test report |

## Phase 7 plan (formerly the Phase 6 plan; step 1 shipped in v0.6)

**Why it is not part of Phase 5:**
- **UDIM** changes the data model: every layer image becomes a tiled image, and the projector, compositor, exports and bakes all need to work per tile. That is a phase of its own, not an optimisation.
- **A GPU compositor** was planned for Phase 5, but this development environment only has software OpenGL, so its benefit cannot be measured here. Shipping it unmeasured would be guesswork.

The next steps, in order of expected payoff:

1. **GPU display during strokes.** About 100 ms of every 2K stroke update is `Image.pixels.foreach_set` rewriting whole output images, because Blender's Python API has no partial image update. Drawing the stroke regions into GPU textures, with a viewport preview bound to the material, would remove that cost. The CPU images would then be written once, at the end of the stroke.
2. **A GPU compositor** for full composites and filters: keep decoded layers as GPU textures (the cache becomes VRAM) and blend in GLSL. Generators already run their noise on the GPU.
3. **UDIM**: a texture set gets a list of tiles. Each tile has its own projector, uses a UV offset, and gets its own region scheduling. Images become `source='TILED'`, and export writes a `<name>.<tile>.png` per tile.
4. **8K**: float32 numpy arrays need about 1 GB per RGBA image at 8K. That needs float16 storage in the cache, or step 2.

## Known limitations (v0.7)

- Masks, generators and layer entries are still prepared on the CPU. They are cached, but the first composite of a large stack, or a change to a mask, does CPU work before the GPU blends.
- The GPU compositor accumulates in half floats. Viewport and 8-bit outputs are unaffected (within one 8-bit step). Exports always use the full-precision CPU path.
- Stacks with Anchor masks always use the CPU compositor.

## Known limitations (v0.6)

- I could not test on Metal from the Linux development environment. Every Metal-relevant path is covered by **Diagnostics**, so run it once on the Mac.
- Mip levels coarser than 1/64 are refreshed when the stroke ends, not live. They only show at extreme zoom-out.
- On a software-rendered GPU (no hardware acceleration), stroke speed is limited by the emulated GPU. Benchmarks from such machines understate the gains on real GPUs.

## Known limitations (v0.5)

- About 100 ms of each 2K stroke update writes whole output images, because Blender has no partial update (see Phase 6 step 1).
- Big strokes cost in proportion to the area they cover. A stroke across half the texture recomposites half the texture.
- Low-poly meshes with large triangles produce large dab regions, since regions come from triangle UV boxes.
- Editing a layer image outside PaintStack (Blender's own texture paint) is not detected. Press **Refresh**, which now also drops the cache.
- Warp filters and anchor masks still force whole-texture stroke updates.

## Known limitations (v0.4)

- Smart assets are applied from the PaintStack panel. Dragging a saved asset from the Asset Browser onto an object assigns it as a separate material and texture set; it does not merge into the current stack.
- Saved asset files have no thumbnail. Blender renders material previews only in an interactive session.
- With anchors in the stack, each stroke update recomposites every channel. This is correct, but slower on large stacks.
- Filter layers run on the CPU. Blur and Warp at 4K take noticeable time per change.

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
