#!/usr/bin/env bash
# Packages a fully self-contained, ready-to-run Subvocal bundle: everything already built
# (encode/synapse dist, the compiled synapse Metal addon, utter's TUI dist) plus the
# already-installed node_modules — so the end user only needs a matching Node.js version and
# the GGUF model files. No Xcode CLT, no cmake, no npm install/ci.
#
# Deliberately reuses the CURRENT node_modules as-is rather than re-resolving one from scratch
# in the staging copy: a fresh `npm ci`/`npm install` here can fail on stale-lockfile drift or
# strict peer-dependency resolution that an older/original install tolerated (hit both while
# building this script — see doc/epics/EPIC-M17 for specifics). Reusing what's already known to
# work sidesteps that whole class of problems, at the cost of also bundling devDependencies
# (no pruning — correctness over a smaller zip).
#
# The bundle is pinned to the exact Node version running THIS script (native addons — synapse's
# own, plus tree-sitter/canvas/etc.'s prebuilt binaries already resolved in node_modules — are
# ABI-locked). install.sh looks for it first, matched by Node ABI, before falling back to a
# full source build.
#
# Usage (from repo root, after a full `npm install`, `npm run build`, `npm run build:metal -w
# synapse`, and `npm run build -w utter/packages/coding-agent` — i.e. after a normal install.sh
# source-build run, or manual equivalent):
#   ./scripts/package-full-bundle.sh
# Output: release-assets/subvocal-macos-arm64-node<version>.zip (+ .sha256)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log()  { printf '▸ %s\n' "$1"; }
die()  { printf '✗ %s\n' "$1" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] || die "This bundle targets macOS/arm64."

# ---- Preflight: everything must already be built and installed --------------------------

REQUIRED=(
  "node_modules/@subvocal/encode"
  "synapse/build-metal/subvocal_ffi_metal.node"
  "synapse/dist/index.js"
  "encode/dist/index.js"
  "utter/packages/coding-agent/dist/cli.js"
  "utter/packages/agent/dist/index.js"
  "utter/packages/ai/dist/index.js"
  "utter/packages/tui/dist/index.js"
)
for f in "${REQUIRED[@]}"; do
  [ -e "$f" ] || die "$f missing — run 'npm install', 'npm run build', 'npm run build:metal -w synapse', and 'npm run build -w utter/packages/coding-agent' first."
done
log "All required build artifacts + node_modules present."

NODE_VERSION="$(node -p 'process.version.slice(1)')"
NODE_ABI="$(node -p 'process.versions.modules')"
NAME="subvocal-macos-arm64-node${NODE_VERSION}"
OUT_DIR="$ROOT/release-assets"
STAGE="$(mktemp -d)/subvocal"

log "Staging a runtime-only copy (Node $NODE_VERSION, ABI $NODE_ABI)..."
mkdir -p "$STAGE"

# Source + already-built dist, minus CMake's intermediate build files (only the compiled .node
# is needed at runtime, not CMakeFiles/Makefile/etc.) and minus node_modules (copied separately
# below, as-is, from the currently-installed tree rather than reinstalled).
rsync -a --exclude=node_modules --exclude=build-cpu --exclude=build-metal --exclude=.DS_Store \
  encode synapse utter bin "$STAGE/"
mkdir -p "$STAGE/synapse/build-metal"
cp synapse/build-metal/subvocal_ffi_metal.node "$STAGE/synapse/build-metal/"
for f in package.json LICENSE NOTICE.md README.md ARCHITECTURE.md; do
  [ -e "$f" ] && cp "$f" "$STAGE/"
done

# The already-working node_modules: root (hoisted deps + npm workspaces' symlinks to
# encode/synapse/utter) plus each workspace's own local node_modules where present. -a
# preserves the workspace symlinks (e.g. node_modules/@subvocal/encode -> ../../encode), which
# resolve correctly since encode/ sits at the same relative path in the staged copy.
log "Copying the existing node_modules (this is the slow part — it's ~300+ MB)..."
rsync -a node_modules "$STAGE/"
for ws in encode synapse utter; do
  [ -d "$ws/node_modules" ] && rsync -a "$ws/node_modules" "$STAGE/$ws/"
done

echo "$NODE_VERSION" > "$STAGE/.node-version-required"

# ---- Zip it up, preserving symlinks -------------------------------------------------------

mkdir -p "$OUT_DIR"
log "Zipping (this is also slow — hundreds of MB of node_modules)..."
(cd "$(dirname "$STAGE")" && zip -ry -q "$OUT_DIR/$NAME.zip" "$(basename "$STAGE")")
shasum -a 256 "$OUT_DIR/$NAME.zip" | awk '{print $1}' > "$OUT_DIR/$NAME.zip.sha256"
rm -rf "$(dirname "$STAGE")"

log "Wrote $OUT_DIR/$NAME.zip ($(du -h "$OUT_DIR/$NAME.zip" | cut -f1))"
log "  Requires Node $NODE_VERSION (ABI $NODE_ABI) — install.sh matches on this before falling back to a source build."
log "  sha256: $(cat "$OUT_DIR/$NAME.zip.sha256")"
echo ""
echo "Next: attach both files to a GitHub Release (tag matching package.json's version) so"
echo "install.sh can find them at:"
echo "  https://github.com/lorenzoperrone/subvocal/releases/download/<version>/$NAME.zip"
