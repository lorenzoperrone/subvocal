# Subvocal patches — INDEX (Mac / Metal port)

Registry of Subvocal patches applied to this Mac port's engine checkout. Unlike the Linux
project (which forks both `ik_llama.cpp` and `llama.cpp`), the Mac port builds **only the
Metal `llama.cpp` backend** — see `doc/epics/EPIC-M1-metal-backend.md` and
`feedback_avoid_ik_llama_cpu_backend`. So only the llama.cpp range (101-199) is relevant here.

Patches live as **uncommitted working-tree edits** in the vendored `../llama.cpp/` checkout
(which has its own git), wrapped in `SUBVOCAL-PATCH-NNN-BEGIN/END` region markers. This registry
is the durable, version-controlled record: the `.patch` files are exported from the actual Mac
checkout (`git -C llama.cpp diff <files>`) so the patches can be re-applied after an upstream
sync or a fresh checkout, even if the working-tree edits are lost.

## llama.cpp upstream (Metal, range 101-199)
| ID  | Slug             | Status  | Date       | Files                                                                                     | Doc | Patch |
|-----|------------------|---------|------------|-------------------------------------------------------------------------------------------|-----|-------|
| 101 | partial-forward  | applied | 2026-05-24 | `include/llama.h`, `src/llama-cparams.h`, `src/llama-context.{h,cpp}`, `src/models/qwen3moe.cpp` | [101-partial-forward.md](llama-cpp-upstream/101-partial-forward.md) | [.patch](llama-cpp-upstream/101-partial-forward.patch) |
| 102 | compact-swa-iswa | applied | 2026-07-01 | `src/llama-kv-cache-iswa.cpp`                                                              | [102-compact-swa-iswa.md](llama-cpp-upstream/102-compact-swa-iswa.md) | [.patch](llama-cpp-upstream/102-compact-swa-iswa.patch) |

> Patch 102 (compact SWA cache under `LLAMA_KV_SWA_OFFLOAD`) applied to the Mac checkout and
> verified on Metal 2026-07-01: SWA KV cache 40960 MiB → 480 MiB at 128k, total KV @128k
> ~43 GiB → ~2.5 GiB, coherent 12B generation. Without it the SWA cache allocates the full
> context and 128k is unusable. See `project_kv_real_size_swa_fullsize` and EPIC-M3.

## ik_llama.cpp (CPU engine) — not built on Mac
Not applicable: the Mac port does not build the ik_llama.cpp CPU backend. As of 2026-07-06 it
isn't even needed as a source-only checkout anymore — the 4 files the Metal backend used to pull
from `ik_llama.cpp/common/` (suffix-tree.{cpp,h}, log.{cpp,h}, unmodified upstream, MIT) plus
nlohmann/json are vendored directly in `synapse/vendor/` (see its `ATTRIBUTION.md`).

## Status legend
- **applied** — currently active in the `llama.cpp/` source tree, built into `synapse/build-metal`.
- **superseded** / **reverted** — see the patch's `.md`. NEVER reuse an ID.

## Re-applying after an upstream sync / fresh checkout
```
cd llama.cpp
git apply ../subvocal-patches/llama-cpp-upstream/101-partial-forward.patch
git apply ../subvocal-patches/llama-cpp-upstream/102-compact-swa-iswa.patch
# then rebuild:
cmake --build build --target llama && cmake --build ../synapse/build-metal
```

## Next available IDs
- llama-cpp-upstream: **103**
