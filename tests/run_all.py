"""Run every PaintStack test suite, the Diagnostics self-test and the benchmark,
and write one report file.

On a Mac (Metal), from a Terminal:
    sh tests/run_on_mac.sh [/Applications/Blender.app]
(each step runs in its own Blender process, so a crash only loses that step and
the report says where it happened)

One step in its own process (what run_on_mac.sh does):
    <blender> --factory-startup --python tests/run_all.py -- --only STEP --out DIR
STEP is "info", a suite name, "diagnostics", "bench-cpu" or "bench-gpu"; the
result goes to DIR/STEP.txt, a native crash's Python stack to DIR/STEP.fault.log.

Any Blender build (a window opens, runs everything, and closes):
    <blender> --factory-startup --python tests/run_all.py -- [--report FILE] [--no-bench] [--keep-open]

With Blender as a Python module (Linux CI, software OpenGL):
    python tests/run_all.py [--report FILE] [--no-bench]
"""

import contextlib
import importlib
import io
import os
import platform
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import bpy  # noqa: E402

SUITES = ["test_headless", "test_phase2", "test_phase2_extras", "test_phase3", "test_phase4",
          "test_phase5", "test_phase6", "test_phase7"]


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    opts = {"report": None, "bench": True, "keep_open": False, "only": None, "out": None}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--report" and i + 1 < len(argv):
            opts["report"] = argv[i + 1]
            i += 1
        elif a in ("--only", "--out") and i + 1 < len(argv):
            opts[a[2:]] = argv[i + 1]
            i += 1
        elif a == "--no-bench":
            opts["bench"] = False
        elif a == "--keep-open":
            opts["keep_open"] = True
        i += 1
    if opts["report"] is None:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        opts["report"] = os.path.join(desktop if os.path.isdir(desktop) else os.path.expanduser("~"),
                                      "paintstack_report.txt")
    return opts


class Tee(io.StringIO):
    def __init__(self, real):
        super().__init__()
        self.real = real

    def write(self, s):
        self.real.write(s)
        self.real.flush()  # the last line printed before a native crash must reach the log
        return super().write(s)


def run_suite(name):
    import test_headless
    from paintstack import compositor
    test_headless.check.failed = False
    # Suites 1-6 test behaviour, so pin the CPU compositor and immediate image
    # writes; test_phase7 switches backends itself.
    if name != "test_phase7":
        compositor.BACKEND_OVERRIDE = "CPU"
        compositor.FORCE_SYNC = True
    out = Tee(sys.__stdout__)
    t0 = time.perf_counter()
    crashed = None
    with contextlib.redirect_stdout(out):
        try:
            mod = importlib.import_module(name)
            mod.main()
        except SystemExit:
            pass
        except Exception:
            crashed = traceback.format_exc()
            print(crashed)
    compositor.BACKEND_OVERRIDE = None
    compositor.FORCE_SYNC = False
    compositor.DEFER_OVERRIDE = None
    lines = out.getvalue().splitlines()
    passed = sum(l.startswith("PASS") for l in lines)
    failed = [l for l in lines if l.startswith("FAIL")]
    return {"name": name, "passed": passed, "failed": failed, "crashed": crashed,
            "seconds": time.perf_counter() - t0}


def run(opts):
    from paintstack import diagnostics, hardware
    report = [f"PaintStack test report  {time.strftime('%Y-%m-%d %H:%M')}",
              f"Blender {bpy.app.version_string}  ({'background' if bpy.app.background else 'UI'})",
              f"{platform.platform()}  python {platform.python_version()}"]
    from _env import fresh_scene
    fresh_scene()
    for k, v in hardware.summary().items():
        report.append(f"  {k}: {v}")
    report.append("")

    total_fail = 0
    for name in SUITES:
        res = run_suite(name)
        status = "OK" if not res["failed"] and not res["crashed"] else "FAILED"
        report.append(f"[{status}] {name}: {res['passed']} passed, {len(res['failed'])} failed "
                      f"({res['seconds']:.0f} s)")
        report += [f"    {l}" for l in res["failed"]]
        if res["crashed"]:
            report += ["    CRASH:"] + [f"      {l}" for l in res["crashed"].splitlines()]
        total_fail += len(res["failed"]) + bool(res["crashed"])
        _write(opts["report"], report)

    report.append("")
    try:
        fresh_scene()
        diag = diagnostics.run()
        report += ["Diagnostics:"] + [f"  {l}" for l in diag.lines]
        total_fail += diag.failures
    except Exception:
        report += ["Diagnostics CRASHED:"] + [f"  {l}" for l in traceback.format_exc().splitlines()]
        total_fail += 1
    _write(opts["report"], report)

    if opts["bench"]:
        import json
        import bench
        report += ["", "Benchmark (ms):"]
        backends = ["CPU"] + (["GPU"] if hardware.gpu_is_hardware() else [])
        for backend in backends:
            try:
                with contextlib.redirect_stdout(Tee(sys.__stdout__)):
                    rows = bench.main([1024, 2048], backend)
                report += [f"  {json.dumps(r)}" for r in rows]
            except Exception:
                report += [f"  {backend} benchmark CRASHED:"] + \
                          [f"    {l}" for l in traceback.format_exc().splitlines()]
            _write(opts["report"], report)

    report.insert(0, f"RESULT: {'ALL PASSED' if total_fail == 0 else f'{total_fail} FAILURE(S)'}")
    _write(opts["report"], report)
    print("\n".join(report))
    print(f"\nReport written to {opts['report']}")
    return total_fail


def _header():
    from paintstack import hardware
    lines = [f"Blender {bpy.app.version_string}  ({'background' if bpy.app.background else 'UI'})",
             f"{platform.platform()}  python {platform.python_version()}"]
    return lines + [f"  {k}: {v}" for k, v in hardware.summary().items()]


def run_step(step):
    """One step, for a fresh Blender process. Returns (ok, report lines)."""
    from _env import fresh_scene
    from paintstack import hardware
    fresh_scene()
    if step == "info":
        return True, _header()
    if step in SUITES:
        res = run_suite(step)
        ok = not res["failed"] and not res["crashed"]
        lines = [f"[{'OK' if ok else 'FAILED'}] {step}: {res['passed']} passed, "
                 f"{len(res['failed'])} failed ({res['seconds']:.0f} s)"]
        lines += [f"    {l}" for l in res["failed"]]
        if res["crashed"]:
            lines += ["    ERROR:"] + [f"      {l}" for l in res["crashed"].splitlines()]
        return ok, lines
    if step == "diagnostics":
        from paintstack import diagnostics
        try:
            diag = diagnostics.run()
        except Exception:
            return False, ["Diagnostics ERROR:"] + [f"  {l}" for l in traceback.format_exc().splitlines()]
        return diag.failures == 0, ["Diagnostics:"] + [f"  {l}" for l in diag.lines]
    if step in ("bench-cpu", "bench-gpu"):
        import json
        import bench
        backend = step[-3:].upper()
        if backend == "GPU" and not hardware.gpu_is_hardware():
            return True, ["Benchmark GPU: skipped (software GPU)"]
        try:
            with contextlib.redirect_stdout(Tee(sys.__stdout__)):
                rows = bench.main([1024, 2048], backend)
        except Exception:
            return False, [f"Benchmark {backend} ERROR:"] + \
                          [f"  {l}" for l in traceback.format_exc().splitlines()]
        return True, [f"Benchmark {backend} (ms):"] + [f"  {json.dumps(r)}" for r in rows]
    return False, [f"Unknown step: {step}"]


def _step_main(opts):
    step, out = opts["only"], opts["out"] or HERE
    os.makedirs(out, exist_ok=True)
    try:
        ok, lines = run_step(step)
    except Exception:
        ok, lines = False, [f"[FAILED] {step} ERROR:"] + \
                           [f"    {l}" for l in traceback.format_exc().splitlines()]
    _write(os.path.join(out, step + ".txt"), [f"STATUS {'OK' if ok else 'FAILED'}"] + lines)
    return 0 if ok else 1


def _enable_faulthandler(opts):
    """A segfault in Blender/GPU code leaves the Python stack (which test line) behind."""
    import faulthandler
    if opts["only"]:
        out = opts["out"] or HERE
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, opts["only"] + ".fault.log")
    else:
        path = os.path.splitext(opts["report"])[0] + ".fault.log"
    try:
        f = open(path, "w")
        faulthandler.enable(file=f, all_threads=True)
        _enable_faulthandler.file = f  # keep it open for the whole run
    except OSError:
        pass


def _write(path, lines):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError as e:
        print(f"Could not write the report to {path}: {e}")


def _ui_context():
    wm = bpy.context.window_manager
    win = wm.windows[0]
    area = next((a for a in win.screen.areas if a.type == "VIEW_3D"), win.screen.areas[0])
    region = next((r for r in area.regions if r.type == "WINDOW"), None)
    return dict(window=win, screen=win.screen, area=area, region=region)


def main():
    opts = _args()
    _enable_faulthandler(opts)
    work = (lambda: _step_main(opts)) if opts["only"] else (lambda: run(opts))
    if bpy.app.background or not bpy.context.window_manager.windows:
        fails = work()
        sys.exit(1 if fails else 0)

    def in_ui():
        # Runs from the event loop, so the window and its GPU (Metal) context exist.
        with bpy.context.temp_override(**_ui_context()):
            work()
            if not opts["keep_open"]:
                # Factory-startup session: nothing to save, so skip the "save changes?"
                # prompt that would otherwise keep the window open.
                bpy.context.preferences.view.use_save_prompt = False
                bpy.ops.wm.quit_blender()
        return None
    bpy.app.timers.register(in_ui, first_interval=1.0)


if __name__ == "__main__":
    main()
