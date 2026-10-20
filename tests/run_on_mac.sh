#!/bin/sh
# Run all PaintStack tests, Diagnostics and the benchmark in your own Blender on
# macOS (Metal), then write ~/Desktop/paintstack_report.txt.
#
#   sh tests/run_on_mac.sh                      # uses /Applications/Blender.app
#   sh tests/run_on_mac.sh ~/Apps/Blender.app   # or another Blender
#   sh tests/run_on_mac.sh "" --no-bench        # skip the (slow) benchmark
#   sh tests/run_on_mac.sh "" --diag            # only Diagnostics (GPU checks), ~30 s
#
# Every step opens its own Blender window with a fresh factory setup (your
# preferences, add-ons and open files are not touched) and closes it when done.
# If Blender crashes in a step, the remaining steps still run and the report
# says which step crashed and the last test line it reached.
HERE="$(cd "$(dirname "$0")" && pwd)"
APP="${1:-/Applications/Blender.app}"
[ $# -gt 0 ] && shift
BLENDER="$APP/Contents/MacOS/Blender"
if [ ! -x "$BLENDER" ]; then
  echo "Blender not found at: $APP"
  echo "Pass the path to Blender.app, e.g.: sh $0 /Applications/Blender.app"
  exit 1
fi
BENCH=1
DIAG_ONLY=0
for a in "$@"; do
  [ "$a" = "--no-bench" ] && BENCH=0
  [ "$a" = "--diag" ] && DIAG_ONLY=1
done

REPORT="$HOME/Desktop/paintstack_report.txt"
[ -d "$HOME/Desktop" ] || REPORT="$HOME/paintstack_report.txt"
OUT="${REPORT%.txt}_steps"
rm -rf "$OUT"
mkdir -p "$OUT"

STEPS="info test_headless test_phase2 test_phase2_extras test_phase3 test_phase4 test_phase5 test_phase6 test_phase7 diagnostics"
[ $BENCH = 1 ] && STEPS="$STEPS bench-cpu bench-gpu"
[ $DIAG_ONLY = 1 ] && STEPS="info diagnostics"

echo "Testing with: $("$BLENDER" --version 2>/dev/null | head -1)"
echo "Each step opens a Blender window that closes by itself. Logs: $OUT"
FAILS=0
BODY="$OUT/body.txt"
: > "$BODY"
for STEP in $STEPS; do
  printf '%-20s ' "$STEP"
  "$BLENDER" --factory-startup --python "$HERE/run_all.py" -- --only "$STEP" --out "$OUT" \
    > "$OUT/$STEP.log" 2>&1
  CODE=$?
  if [ -f "$OUT/$STEP.txt" ]; then
    STATUS="$(head -1 "$OUT/$STEP.txt" | cut -d' ' -f2)"
    echo "$STATUS"
    [ "$STATUS" = "OK" ] || FAILS=$((FAILS + 1))
    tail -n +2 "$OUT/$STEP.txt" >> "$BODY"
  else
    echo "BLENDER CRASHED (exit code $CODE)"
    FAILS=$((FAILS + 1))
    {
      echo "[CRASHED] $STEP: Blender exited with code $CODE before finishing"
      echo "    Last output:"
      tail -n 25 "$OUT/$STEP.log" | sed 's/^/      /'
      if [ -s "$OUT/$STEP.fault.log" ]; then
        echo "    Python stack at the crash:"
        sed 's/^/      /' "$OUT/$STEP.fault.log" | head -n 60
      fi
    } >> "$BODY"
  fi
  echo >> "$BODY"
done

{
  if [ $FAILS = 0 ]; then echo "RESULT: ALL PASSED"; else echo "RESULT: $FAILS STEP(S) FAILED"; fi
  echo "PaintStack test report  $(date '+%Y-%m-%d %H:%M')"
  echo "macOS $(sw_vers -productVersion 2>/dev/null)  $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  echo
  cat "$BODY"
} > "$REPORT"

echo
head -1 "$REPORT"
echo "Full report: $REPORT  (please send this file back)"
