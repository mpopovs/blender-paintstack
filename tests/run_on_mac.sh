#!/bin/sh
# Run all PaintStack tests, Diagnostics and the benchmark in your own Blender on
# macOS (Metal), then write ~/Desktop/paintstack_report.txt.
#
#   sh tests/run_on_mac.sh                      # uses /Applications/Blender.app
#   sh tests/run_on_mac.sh ~/Apps/Blender.app   # or another Blender
#
# A Blender window opens with a fresh factory setup (your preferences, add-ons and
# open files are not touched), runs for a few minutes and closes by itself.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
APP="${1:-/Applications/Blender.app}"
BLENDER="$APP/Contents/MacOS/Blender"
if [ ! -x "$BLENDER" ]; then
  echo "Blender not found at: $APP"
  echo "Pass the path to Blender.app, e.g.: sh $0 /Applications/Blender.app"
  exit 1
fi
REPORT="$HOME/Desktop/paintstack_report.txt"
echo "Testing with: $("$BLENDER" --version 2>/dev/null | head -1)"
echo "A Blender window will open and close by itself when the tests finish..."
"$BLENDER" --factory-startup --python "$HERE/run_all.py" -- --report "$REPORT" "$@" || true
echo
if [ -f "$REPORT" ]; then
  head -1 "$REPORT"
  echo "Full report: $REPORT  (please send this file back)"
else
  echo "No report was written - please copy the Terminal output above."
fi
