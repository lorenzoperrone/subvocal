#!/usr/bin/env bash
# Packages the already-built synapse Metal addon (synapse/build-metal/*.node) into a tarball
# named after this machine's Node ABI version, ready to attach as a GitHub Release asset.
#
# Native addons are ABI-locked to the Node major version they were built against (N-API's
# NODE_MODULE_VERSION) — a prebuilt only helps users on the SAME ABI. install.sh looks for a
# release asset matching the ABI it detects at install time and falls back to a full local
# build (see install.sh) if none matches.
#
# Usage: run after `npm run build:metal` inside synapse/ (from repo root):
#   ./scripts/package-prebuilt.sh
# Output: release-assets/subvocal-synapse-metal-node<ABI>-darwin-arm64.tar.gz (+ .sha256)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE_FILE="$ROOT/synapse/build-metal/subvocal_ffi_metal.node"

if [ ! -f "$NODE_FILE" ]; then
  echo "error: $NODE_FILE not found — run 'npm run build:metal -w synapse' first" >&2
  exit 1
fi

ARCH="$(uname -m)"
if [ "$ARCH" != "arm64" ]; then
  echo "error: this project targets Apple Silicon (arm64) only, detected '$ARCH'" >&2
  exit 1
fi

ABI="$(node -p 'process.versions.modules')"
OUT_DIR="$ROOT/release-assets"
NAME="subvocal-synapse-metal-node${ABI}-darwin-arm64"
STAGE="$(mktemp -d)"

mkdir -p "$OUT_DIR"
mkdir -p "$STAGE/build-metal"
cp "$NODE_FILE" "$STAGE/build-metal/subvocal_ffi_metal.node"

tar -czf "$OUT_DIR/$NAME.tar.gz" -C "$STAGE" build-metal
shasum -a 256 "$OUT_DIR/$NAME.tar.gz" | awk '{print $1}' > "$OUT_DIR/$NAME.tar.gz.sha256"
rm -rf "$STAGE"

echo "Wrote $OUT_DIR/$NAME.tar.gz"
echo "  Node ABI: $ABI (this file only loads on Node runtimes with the same ABI)"
echo "  sha256:   $(cat "$OUT_DIR/$NAME.tar.gz.sha256")"
echo ""
echo "Next: attach both files to a GitHub Release (tag matching package.json's version) so"
echo "install.sh can find them at:"
echo "  https://github.com/lorenzoperrone/subvocal/releases/download/<version>/$NAME.tar.gz"
