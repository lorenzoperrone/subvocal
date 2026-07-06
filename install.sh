#!/usr/bin/env bash
# Subvocal one-shot installer for Apple Silicon Macs.
#
# What it does, in order:
#   1. Check for a full self-contained bundle (everything already built, incl. node_modules)
#      matching this machine's exact Node version on a GitHub Release — if found, extract it
#      and skip straight to step 4. No toolchain needed for this path at all.
#   2. Otherwise, verify Xcode Command Line Tools, cmake, Node >= 22.19 (offers to install
#      missing pieces via Homebrew, always asking first), then npm install.
#   3. Fetch a prebuilt synapse Metal addon matching this machine's Node ABI from a GitHub
#      Release asset; if none matches, fall back to a full local build (clone llama.cpp +
#      apply subvocal-patches + compile via cmake-js — the suffix-tree drafter's source files
#      are vendored in synapse/vendor/, no separate ik_llama.cpp checkout needed). Then build
#      encode + synapse's TS wrapper + utter's TUI.
#   4. Check for the two GGUF checkpoints under models/; if missing, ask before downloading
#      them from Hugging Face (large files — several GB each).
#
# Safe to re-run: every step checks whether its work is already done before repeating it.
#
# Env overrides:
#   SUBVOCAL_SKIP_MODEL_DOWNLOAD=1   skip step 4 entirely (bring your own GGUF files)
#   SUBVOCAL_MODEL_12B_HF_REPO / _FILE, SUBVOCAL_MODEL_E2B_HF_REPO / _FILE
#     override the Hugging Face repo/file names used for auto-download — the defaults below
#     match bin/subvocal's existing default model paths and were confirmed reachable on
#     Hugging Face as of 2026-07-06 (~6.7 GB / ~2.6 GB respectively); re-check if they 404,
#     upstream repos can be renamed or taken down.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

log()  { printf '▸ %s\n' "$1"; }
warn() { printf '⚠️  %s\n' "$1" >&2; }
die()  { printf '✗ %s\n' "$1" >&2; exit 1; }

confirm() {
  # confirm "prompt" — returns 0 (yes) or 1 (no). Defaults to no on non-interactive shells.
  local reply
  if [ ! -t 0 ]; then return 1; fi
  read -r -p "$1 [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]]
}

github_slug() {
  # Turn package.json's repository.url (https://github.com/x/y.git or git@github.com:x/y.git)
  # into "x/y"; empty or containing "<" (still the README placeholder) means "not usable yet".
  local url
  url="$(node -p "(require('./package.json').repository || {}).url || ''")"
  printf '%s' "$url" | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##'
}

die_need_manual_install() {
  # die_need_manual_install "<thing>" "<brew formula>" — the one case this script can't take
  # care of itself: if Homebrew isn't there to do it with, print the official Homebrew install
  # command (https://brew.sh) rather than a useless "brew install X" the user can't run yet.
  local thing="$1" formula="$2"
  if command -v brew >/dev/null 2>&1; then
    die "$thing is required. Install it: brew install $formula — then re-run this script."
  else
    warn "$thing is required, and Homebrew isn't installed either."
    warn "Install Homebrew first (see https://brew.sh):"
    warn '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    die "Then: brew install $formula — and re-run this script."
  fi
}

[ "$(uname -s)" = "Darwin" ] || die "Subvocal's native binding only builds on macOS (Metal backend)."
[ "$(uname -m)" = "arm64" ] || die "Subvocal targets Apple Silicon (arm64); detected $(uname -m)."

if ! command -v node >/dev/null 2>&1; then
  warn "Node.js not found."
  if command -v brew >/dev/null 2>&1 && confirm "Install Node via Homebrew now?"; then
    brew install node
  else
    die_need_manual_install "Node.js >= 22.19" "node"
  fi
fi

BUNDLE_OK=0
SYNAPSE_NODE="$ROOT/synapse/build-metal/subvocal_ffi_metal.node"

# ---- 1. Full self-contained bundle: zero toolchain, zero build, if one matches ------------

if [ -f "$ROOT/node_modules/@subvocal/encode/package.json" ] && [ -f "$SYNAPSE_NODE" ]; then
  log "Already set up (node_modules + synapse addon present), skipping bundle/build entirely."
  BUNDLE_OK=1
else
  REPO_SLUG="$(github_slug)"
  if [ -n "$REPO_SLUG" ] && [[ "$REPO_SLUG" != *"<"* ]]; then
    NODE_VERSION="$(node -p 'process.version.slice(1)')"
    PKG_VERSION="$(node -p "require('./package.json').version")"
    BUNDLE_NAME="subvocal-macos-arm64-node${NODE_VERSION}.zip"
    BUNDLE_URL="https://github.com/${REPO_SLUG}/releases/download/v${PKG_VERSION}/${BUNDLE_NAME}"

    log "Looking for a full prebuilt bundle (exact Node ${NODE_VERSION} match) at $BUNDLE_URL ..."
    TMP_DIR="$(mktemp -d)"
    TMP_ZIP="$TMP_DIR/bundle.zip"
    if curl -fsSL -o "$TMP_ZIP" "$BUNDLE_URL" 2>/dev/null; then
      unzip -q "$TMP_ZIP" -d "$TMP_DIR/extracted"
      # The zip contains one top-level "subvocal/" directory (see package-full-bundle.sh) —
      # merge its contents into $ROOT rather than nesting a copy inside it.
      rsync -a "$TMP_DIR/extracted/subvocal/" "$ROOT/"
      if [ -f "$SYNAPSE_NODE" ]; then
        log "Full bundle installed — no build needed, no toolchain required."
        BUNDLE_OK=1
      else
        warn "Bundle downloaded but looks incomplete — falling back to a source build."
      fi
    else
      log "No matching bundle found for Node ${NODE_VERSION} — falling back to a source build."
    fi
    rm -rf "$TMP_DIR"
  fi
fi

# ---- 2. Toolchain checks (only needed for the source-build path) -------------------------

if [ "$BUNDLE_OK" -ne 1 ]; then
  if ! xcode-select -p >/dev/null 2>&1; then
    warn "Xcode Command Line Tools not found."
    if confirm "Run 'xcode-select --install' now? (opens a GUI installer; re-run this script after it finishes)"; then
      xcode-select --install
      exit 0
    else
      die "Xcode Command Line Tools are required. Install manually: xcode-select --install"
    fi
  fi

  if ! command -v cmake >/dev/null 2>&1; then
    warn "cmake not found."
    if command -v brew >/dev/null 2>&1 && confirm "Install cmake via Homebrew now?"; then
      brew install cmake
    else
      die_need_manual_install "cmake" "cmake"
    fi
  fi

  NODE_OK=0
  NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
  NODE_MINOR="$(node -p 'process.versions.node.split(".")[1]')"
  if [ "$NODE_MAJOR" -gt 22 ] || { [ "$NODE_MAJOR" -eq 22 ] && [ "$NODE_MINOR" -ge 19 ]; }; then
    NODE_OK=1
  fi
  if [ "$NODE_OK" -ne 1 ]; then
    warn "Node.js >= 22.19 not found (or too old)."
    if command -v brew >/dev/null 2>&1 && confirm "Install Node via Homebrew now?"; then
      brew install node
    else
      die_need_manual_install "Node.js >= 22.19" "node"
    fi
  fi

  log "Platform checks passed: macOS/arm64, Xcode CLT, cmake, Node $(node -v)."

  log "Installing workspace dependencies (npm install)..."
  npm install

  # ---- 2b. synapse native addon: prebuilt fetch, fallback to full local build ------------

  if [ -f "$SYNAPSE_NODE" ]; then
    log "synapse Metal addon already built, skipping."
  else
    NODE_ABI="$(node -p 'process.versions.modules')"
    PKG_VERSION="$(node -p "require('./package.json').version")"
    REPO_SLUG="$(github_slug)"
    ASSET="subvocal-synapse-metal-node${NODE_ABI}-darwin-arm64.tar.gz"
    ASSET_URL="https://github.com/${REPO_SLUG}/releases/download/v${PKG_VERSION}/${ASSET}"

    PREBUILT_OK=0
    if [ -n "$REPO_SLUG" ] && [[ "$REPO_SLUG" != *"<"* ]]; then
      log "Looking for a prebuilt addon (Node ABI ${NODE_ABI}) at $ASSET_URL ..."
      TMP_DIR="$(mktemp -d)"
      TMP_TARBALL="$TMP_DIR/prebuilt.tar.gz"
      if curl -fsSL -o "$TMP_TARBALL" "$ASSET_URL" 2>/dev/null; then
        mkdir -p "$ROOT/synapse"
        tar -xzf "$TMP_TARBALL" -C "$ROOT/synapse"
        rm -rf "$TMP_DIR"
        if [ -f "$SYNAPSE_NODE" ]; then
          log "Prebuilt addon installed."
          PREBUILT_OK=1
        fi
      else
        rm -rf "$TMP_DIR"
      fi
    fi

    if [ "$PREBUILT_OK" -ne 1 ]; then
      log "No matching prebuilt found — building the Metal addon locally (this compiles llama.cpp, takes a while)."

      if [ ! -d "$ROOT/llama.cpp" ]; then
        log "Cloning llama.cpp..."
        git clone --depth 1 https://github.com/ggml-org/llama.cpp "$ROOT/llama.cpp"
      fi

      for patch in "$ROOT"/subvocal-patches/llama-cpp-upstream/*.patch; do
        [ -e "$patch" ] || continue
        tag="SUBVOCAL-PATCH-$(basename "$patch" | grep -oE '^[0-9]+')"
        if grep -rq "$tag" "$ROOT/llama.cpp/src" "$ROOT/llama.cpp/include" 2>/dev/null; then
          log "$(basename "$patch") already applied, skipping."
        else
          log "Applying $(basename "$patch")..."
          git -C "$ROOT/llama.cpp" apply "$patch"
        fi
      done

      log "Building llama.cpp (Metal)..."
      cmake -B "$ROOT/llama.cpp/build" -S "$ROOT/llama.cpp" -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
      cmake --build "$ROOT/llama.cpp/build" --target llama -j "$(sysctl -n hw.ncpu)"

      log "Compiling the synapse addon (cmake-js)..."
      npm run build:metal -w synapse
    fi
  fi

  log "Building @subvocal/synapse (TS) + @subvocal/encode..."
  npm run build:ts -w synapse
  npm run build:encode

  log "Building the utter TUI (utter/packages/coding-agent)..."
  (cd utter/packages/coding-agent && npm run build)
fi

# ---- 3. GGUF model checkpoints -------------------------------------------------------------

if [ "${SUBVOCAL_SKIP_MODEL_DOWNLOAD:-0}" = "1" ]; then
  log "SUBVOCAL_SKIP_MODEL_DOWNLOAD=1, skipping model download step."
else
  MODEL_12B_REPO="${SUBVOCAL_MODEL_12B_HF_REPO:-unsloth/gemma-4-12B-it-qat-GGUF}"
  MODEL_12B_FILE="${SUBVOCAL_MODEL_12B_FILE:-gemma-4-12B-it-qat-UD-Q4_K_XL.gguf}"
  MODEL_E2B_REPO="${SUBVOCAL_MODEL_E2B_HF_REPO:-unsloth/gemma-4-E2B-it-qat-GGUF}"
  MODEL_E2B_FILE="${SUBVOCAL_MODEL_E2B_FILE:-gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf}"

  fetch_model() {
    local repo="$1" file="$2" dest_dir="$3"
    local dest="$dest_dir/$file"
    [ -f "$dest" ] && { log "$file already present, skipping."; return; }

    local url="https://huggingface.co/${repo}/resolve/main/${file}"
    local size_bytes
    size_bytes="$(curl -fsIL "$url" 2>/dev/null | tr -d '\r' | grep -i '^content-length:' | tail -1 | awk '{print $2}')"
    local size_human="unknown size"
    if [ -n "${size_bytes:-}" ]; then
      size_human="$(( (size_bytes + 500000000) / 1000000000 )) GB (approx.)"
    fi

    warn "Model not found: $file ($size_human) from huggingface.co/$repo"
    if confirm "Download it now into models/$(basename "$dest_dir")/ ?"; then
      mkdir -p "$dest_dir"
      curl -L --progress-bar -o "$dest" "$url"
      log "Downloaded $file."
    else
      warn "Skipped. To fetch it manually:"
      warn "  curl -L -o '$dest' '$url'"
      warn "  (mkdir -p '$dest_dir' first, or set SUBVOCAL_LOCAL_MODEL to any GGUF file's path instead)"
    fi
  }

  # bin/subvocal expects models/<org>-<repo>/<file> (org/repo joined with "-", e.g.
  # "unsloth-gemma-4-12B-it-qat-GGUF") — NOT basename(repo), which would drop the org prefix
  # and put the download where bin/subvocal would never look for it.
  fetch_model "$MODEL_12B_REPO" "$MODEL_12B_FILE" "$ROOT/models/${MODEL_12B_REPO//\//-}"
  fetch_model "$MODEL_E2B_REPO" "$MODEL_E2B_FILE" "$ROOT/models/${MODEL_E2B_REPO//\//-}"
fi

# ---- 4. Register the global `subvocal` command --------------------------------------------

# package.json declares "bin": {"subvocal": "bin/subvocal"} — `npm link` turns that into a
# global symlink so the user just types `subvocal` from anywhere, instead of ./bin/subvocal
# relative to this directory. Idempotent (safe to re-run: recreates the same symlink).
if npm link >/dev/null 2>&1; then
  log "'subvocal' command linked globally."
else
  warn "Could not link 'subvocal' globally (often a permissions issue with npm's global prefix)."
  warn "  Run 'npm link' yourself once that's sorted, or just use ./bin/subvocal directly."
fi

if command -v subvocal >/dev/null 2>&1; then
  log "Setup complete. Run 'subvocal' to start."
else
  log "Setup complete. Run ./bin/subvocal to start."
fi
