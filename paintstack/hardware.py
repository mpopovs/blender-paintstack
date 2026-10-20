"""Hardware awareness: GPU backend (Metal on macOS), unified memory size,
performance-core count, display scale, and a shared thread pool.

Apple Silicon notes
-------------------
* Blender draws with Metal there. Every shader is built through
  GPUShaderCreateInfo (cross-compiled per backend); textures that are blended
  into or filtered use half-float formats, which Metal supports on all GPUs.
* CPU and GPU share memory, so cache budgets are derived from total RAM.
* numpy releases the GIL in its loops, so big array work is split across the
  performance cores (M2: 4, M2 Pro: 6-8, M2 Max: 8) with a thread pool.
* Retina displays have 2 physical pixels per UI pixel; brush sizes are UI pixels."""

import os
import platform as _platform
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_info = {}
_pool = None
# Arrays smaller than this many elements are not worth splitting across threads.
PARALLEL_MIN = 1 << 20


def is_mac():
    return _platform.system() == "Darwin"


def _sysctl(name):
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=2)
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def total_ram():
    if "ram" not in _info:
        ram = _sysctl("hw.memsize") if is_mac() else None
        if ram is None:
            try:
                ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            except (ValueError, OSError, AttributeError):
                ram = 8 << 30
        _info["ram"] = ram
    return _info["ram"]


def performance_cores():
    """Cores worth using for parallel numpy: performance cores on Apple Silicon
    (efficiency cores would only straggle), all cores elsewhere."""
    if "cores" not in _info:
        cores = _sysctl("hw.perflevel0.physicalcpu") if is_mac() else None
        _info["cores"] = max(1, cores or os.cpu_count() or 1)
    return _info["cores"]


def gpu_backend():
    """'METAL', 'OPENGL', 'VULKAN' or 'NONE' (no GPU context)."""
    try:
        import gpu
        return gpu.platform.backend_type_get()
    except Exception:
        return "NONE"


def gpu_name():
    try:
        import gpu
        return f"{gpu.platform.vendor_get()} {gpu.platform.renderer_get()}"
    except Exception:
        return "unknown"


def pixel_size():
    """Physical pixels per UI pixel (2 on Retina displays)."""
    try:
        import bpy
        return float(bpy.context.preferences.system.pixel_size)
    except Exception:
        return 1.0


def live_preview_enabled():
    try:
        import bpy
        prefs = bpy.context.preferences.addons.get(__package__)
        return bool(getattr(prefs.preferences, "gpu_live_preview", True)) if prefs else True
    except Exception:
        return True


def auto_cache_budget_mb():
    """Default compositor cache: 15% of (unified) memory, between 512 MB and 6 GB."""
    return int(min(6144, max(512, total_ram() * 0.15 / (1 << 20))))


# Threads

def threads():
    try:
        import bpy
        prefs = bpy.context.preferences.addons.get(__package__)
        n = getattr(prefs.preferences, "threads", 0) if prefs else 0
    except Exception:
        n = 0
    return n if n > 0 else performance_cores()


def pool():
    global _pool
    n = threads()
    if _pool is None or _pool._max_workers != n:
        if _pool is not None:
            _pool.shutdown(wait=False)
        _pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="paintstack")
    return _pool


def row_chunks(rows, parts):
    """Split range(rows) into about `parts` contiguous (start, stop) chunks."""
    edges = np.linspace(0, rows, parts + 1).astype(int)
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def parallel_rows(fn, rows, size):
    """Run fn(y0, y1) over row chunks, in parallel when the work is big enough.
    More chunks than threads, so slower cores do not hold the others up."""
    n = threads()
    if n <= 1 or size < PARALLEL_MIN:
        fn(0, rows)
        return
    list(pool().map(lambda c: fn(*c), row_chunks(rows, n * 3)))


def summary():
    import bpy
    return {
        "blender": bpy.app.version_string,
        "os": f"{_platform.system()} {_platform.release()} ({_platform.machine()})",
        "gpu_backend": gpu_backend(),
        "gpu": gpu_name(),
        "ram_gb": round(total_ram() / (1 << 30), 1),
        "threads": threads(),
        "pixel_size": pixel_size(),
        "auto_cache_mb": auto_cache_budget_mb(),
    }
