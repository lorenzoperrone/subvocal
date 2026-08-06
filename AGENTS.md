# Subvocal — Agent Instructions

## Project overview
Subvocal is a tensor-native local developer agent: in-process GGUF inference (CPU + GPU) with AST-aware preprocessing, logit-level steering, speculative decoding, and dual-brain orchestration.

## Directory structure
```
subvocal/
├── skills/                     # Agent skills (tool-agnostic, symlinked into .opencode/)
├── synapse/                 # @subvocal/synapse — C++ N-API bindings for llama.cpp
├── encode/                     # @subvocal/encode — preprocessor (AST, intent, KV cache)
├── utter/                      # Upstream pi agent (gitignored — own git history)
├── ik_llama.cpp/               # CPU engine fork (gitignored — own git history)
├── llama.cpp/                  # GPU engine fork (gitignored — own git history)
├── subvocal-patches/           # Patch registry for engine modifications
├── doc/                        # Epics, substories, task roadmap
├── poc/                        # Python proof-of-concept scripts (REST era, pre-FFI)
└── test-rag/                   # RAG experiments (gitignored)
```

## Package dependency graph
```
@subvocal/synapse          (no deps)
    ↑
@subvocal/encode              (depends on synapse via file:../synapse)
    ↑
utter/coding-agent            (depends on synapse via file:../../synapse)
    └── local-provider/       (wires FFI as in-process inference backend)
```

## Key rules

### Upstream repos
- `ik_llama.cpp/`, `llama.cpp/` are gitignored at the monorepo level. Each has its own git history.
- Never commit changes from these repos into the monorepo.
- Engine modifications go through the patch system (see `subvocal-patches/README.md`).

### Engine patches
- All modifications to `ik_llama.cpp/` and `llama.cpp/` must use the patch convention: `SUBVOCAL-PATCH-NNN-BEGIN/END` tags, `.md` documentation, `.patch` export, INDEX.md registry entry.
- When pulling upstream, use skill `subvocal-sync-upstream`.

### Building
```bash
npm install                    # from repo root
npm run build:ffi              # cmake-js compile CPU
npm run build:encode           # tsc
npm test                       # vitest for both packages
```

### utter integration
- The local provider lives in `utter/packages/coding-agent/src/core/local-provider/`.
- It imports `@subvocal/synapse` (resolved via symlink in utter's node_modules).
- The `@subvocal/ffi-binder` name in old imports refers to `@earendil-works/pi-ai` — use the latter directly.
- Never modify utter package.json dependencies without updating the symlink.

### Agent Loop (Epic 4.1)
- `encode/src/agentLoop.ts` — `AgentLoop` class orchestrates multi-turn coding sessions.
- Uses `preprocess()` (small CPU model) for intent + AST tagging, then GPU model for generation.
- Incremental decode via `decodeAppend()` preserves KV cache across turns (O(n), not O(n²)).
- Integration into utter local-provider is complete: `AgentLoop` is wired into `utter/packages/coding-agent/src/core/local-provider/conversation.ts` running inside a worker thread.

### Cross-Platform Strategy & Single-Trunk Policy
- **Single Trunk (`main`)**: Never create long-lived OS-specific branches (`mac-dev`, `linux-dev`, `windows-dev`). All OS platforms share `main` as the single source of truth.
- **Dynamic Platform Dispatch**: Platform differentiation is handled at runtime (`process.platform === 'darwin' | 'linux' | 'win32'`) and compile-time (`SUBVOCAL_BACKEND=cpu|gpu|metal`).
- **Separate Binary Build Outputs**: Native C++ binaries output to platform-specific directories (`build-metal/` on Mac, `build-gpu/` on Linux CUDA, `build-cpu/` on CPU, `build-win/` on Windows) to prevent build clobbering across environments.
- **Short-Lived Feature Branches**: Hardware-specific experimental features must be developed on short-lived feature branches and merged into `main` behind conditional runtime/compile flags.

### TypeScript style
- All code is ESM (`"type": "module"`).
- No `any`, no dynamic imports, top-level imports only.
- Tests use vitest.
