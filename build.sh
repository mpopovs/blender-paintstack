#!/usr/bin/env sh
# Package the add-on as an installable Blender extension zip.
set -e
cd "$(dirname "$0")"
version=$(sed -n 's/^version = "\(.*\)"/\1/p' paintstack/blender_manifest.toml)
mkdir -p dist
rm -f "dist/paintstack-$version.zip"
(cd paintstack && zip -qr "../dist/paintstack-$version.zip" . -x '__pycache__/*' '*/__pycache__/*')
echo "dist/paintstack-$version.zip"
