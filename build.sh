#!/usr/bin/env sh
# Package the add-on as an installable Blender extension zip, plus a test kit
# (add-on source + test suites + Mac runner) for testing on another machine.
set -e
cd "$(dirname "$0")"
version=$(sed -n 's/^version = "\(.*\)"/\1/p' paintstack/blender_manifest.toml)
mkdir -p dist
rm -f "dist/paintstack-$version.zip" "dist/paintstack-$version-testkit.zip"
(cd paintstack && zip -qr "../dist/paintstack-$version.zip" . -x '__pycache__/*' '*/__pycache__/*')
kit="paintstack-$version-testkit"
rm -rf "dist/$kit" && mkdir -p "dist/$kit"
cp -R paintstack tests README.md "dist/$kit/"
mkdir -p "dist/$kit/docs" && cp docs/MAC.md docs/ROADMAP.md "dist/$kit/docs/"
find "dist/$kit" -name __pycache__ -prune -exec rm -rf {} +
(cd dist && zip -qr "$kit.zip" "$kit" && rm -rf "$kit")
echo "dist/paintstack-$version.zip"
echo "dist/$kit.zip"
