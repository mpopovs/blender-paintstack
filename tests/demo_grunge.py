"""Contact sheet of the grunge library, each texture tiled 2 x 2 to show it is seamless.

    python tests/demo_grunge.py docs/grunge.png
"""

import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paintstack import grunge  # noqa: E402

out = os.path.abspath(sys.argv[-1] if sys.argv[-1].endswith(".png") else "grunge.png")
tiles = [np.tile(grunge.grunge_array(k, 0, 256), (2, 2)) for k, _, _ in grunge.GRUNGE_TYPES]
rows = [np.concatenate(tiles[i:i + 4], 1) for i in (0, 4)]
sheet = np.concatenate(rows[::-1], 0)  # Blender images start at the bottom row
img = bpy.data.images.new("sheet", sheet.shape[1], sheet.shape[0])
img.pixels.foreach_set(np.dstack([sheet] * 3 + [np.ones_like(sheet)]).astype(np.float32).ravel())
img.filepath_raw = out
img.file_format = "PNG"
img.save()
print("saved", out, "order:", ", ".join(label for _, label, _ in grunge.GRUNGE_TYPES))
