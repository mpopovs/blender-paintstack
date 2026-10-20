# PaintStack on a MacBook Pro (Apple Silicon: M1, M2, M3 and later)

## Install
1. Use **Blender 4.2 or newer, Apple Silicon build** (the "macOS – Apple Silicon" download, not Intel). Blender draws with **Metal** on these Macs.
2. In Blender, go to **Edit → Preferences → Get Extensions → ⌄ → Install from Disk…** and choose `paintstack-<version>.zip`.
3. In **Preferences → Add-ons → PaintStack**, click **Run Diagnostics** once. It tests every GPU feature PaintStack uses against a CPU reference on your machine, and saves a report to the text datablock **PaintStack Diagnostics** (open it in a Text Editor area). Every line should say `PASS`. If any line says `FAIL`, keep the report: PaintStack falls back to safer paths, and the report shows what differs on your machine.

## What PaintStack does differently on Apple Silicon
| Topic | What happens |
|---|---|
| **Metal** | All shaders are built through Blender's cross-backend shader API. Textures that are blended into or filtered use half-float formats, which every Metal GPU supports (32-bit float blending and filtering are optional on Metal). Whether drawing into an sRGB texture encodes the values is measured on your Mac, not assumed. |
| **Unified memory** | CPU and GPU share RAM. With the compositor cache set to **Auto** it uses 15% of your memory (about 2.4 GB on a 16 GB machine, 4.8 GB on 32 GB). You can change this in the add-on preferences. |
| **Performance cores** | Large blends are split across the **performance cores**: 4 on an M2, 6–8 on an M2 Pro, 8 on an M2 Max. Efficiency cores are left out, so they cannot hold the others up. |
| **Live stroke preview** | While you paint, only the touched parts of the textures, and their first 7 mip levels, are written straight into the GPU textures the viewport shows. The full images are written once, when you release the brush. You can turn this off in the preferences (**GPU Live Stroke Preview**). |
| **Retina** | Brush radius, lazy mouse and stencil size are in UI points, so the brush looks the same size as Blender's own brushes on a 2× display. |

## Trackpad and keyboard
| Action | Keys |
|---|---|
| Paint | Click and drag (one finger on a Magic Trackpad, or a pen tablet) |
| Orbit / pan / zoom while in paint mode | Two-finger drag, Shift + two-finger drag, pinch (trackpad gestures pass through) |
| Brush radius | **F**, then move and click. Or **[** / **]** |
| Undo / redo stroke | **⌘Z** / **⌘⇧Z** |
| Straight line | **⇧-click** |
| Stencil | Hold **S** and move. Scroll scales it and **⌘-scroll** rotates it |
| Channel view / material | **C** / **M** |
| Exit paint mode | **Esc** |

Tips:
- **Preferences → Input → Emulate 3 Button Mouse** lets you orbit with **⌥ + drag**. PaintStack never paints while ⌥ is held.
- For pressure, connect a Wacom-style tablet, or use an iPad with Apple Pencil through **Sidecar**. Turn on **Pressure → Flow** in the Brush panel.
- On 16 GB machines, prefer 2K texture sets. 4K works, but each full 4K RGBA float layer costs 256 MB of the shared memory.

## Reporting performance
Run **Diagnostics** and copy the timing lines from the report: noise GPU vs CPU, 2K blend with 1 thread vs N threads, and 50 dabs plus region readback. They show what your M2 actually does, and they guide the next optimisation step: moving the full compositor to the GPU.

## Testing PaintStack on your Mac
The **test kit** (`paintstack-<version>-testkit.zip`) contains the add-on source, all test suites and a runner.

1. Unzip it, for example into `~/Downloads/paintstack-<version>-testkit`.
2. Open **Terminal** and run:
   ```sh
   sh ~/Downloads/paintstack-<version>-testkit/tests/run_on_mac.sh
   ```
   If Blender is not at `/Applications/Blender.app`, add its path: `sh …/run_on_mac.sh /path/to/Blender.app`.
   From a git clone, run it from the repository folder: `sh tests/run_on_mac.sh`. Add `"" --no-bench` to skip the benchmark: `sh tests/run_on_mac.sh "" --no-bench`.
3. Each step (every test suite, then Diagnostics, then the benchmarks) opens its own Blender window. The window may look frozen while that step runs, and it closes by itself. The Terminal prints `OK`, `FAILED` or `BLENDER CRASHED` for each step.
4. Send back **`~/Desktop/paintstack_report.txt`**. Its first line says `RESULT: ALL PASSED` or how many steps failed. It also contains the Diagnostics (Metal checks) and the CPU and GPU benchmarks. If Blender crashed, the report names the step, shows the last test lines it printed and the Python stack at the crash. Per-step logs are in `~/Desktop/paintstack_report_steps/`.

The run uses `--factory-startup`, so your preferences, installed add-ons and open files are not touched. It loads PaintStack from the test kit, not from your installed copy.

### Manual checklist (things only a person can check)
Install the add-on zip, open the default cube scene, select the cube, then click **PaintStack → Create Texture Set** and **Start Painting**:
- [ ] The brush circle has the same on-screen size as Blender's own brushes at the same radius (Retina).
- [ ] Painting updates the viewport while you drag, with no visible lag spikes at 2K.
- [ ] **⌘Z** undoes the last stroke and **⌘⇧Z** redoes it (Blender's own undo should not trigger).
- [ ] **F**, then moving and clicking, resizes the brush.
- [ ] Two-finger scroll orbits, pinch zooms, and ⇧ + two-finger drag pans, all while in paint mode.
- [ ] With a pen or Sidecar, **Pressure → Flow** changes stroke strength.
- [ ] Dragging a layer's **Opacity** slider updates the viewport smoothly. The GPU compositor is in use if **System** shows `Metal` and the preferences say Auto.
- [ ] Saving, closing and reopening the .blend keeps everything you painted.
- [ ] Dragging a generator's **Level** slider (e.g. Edge Wear) updates the viewport smoothly at 2K.
- [ ] UDIM: on a mesh with UVs in tiles 1001 and 1002, click the **UV** button next to **UDIM**. A stroke across the tile border paints both tiles, and **Export Textures** writes `.1001` and `.1002` files.
