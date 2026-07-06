# Patch 102 — Compact SWA cache with ISWA split

**Engine**: llama.cpp upstream
**Status**: applied
**Date**: 2026-06-28

## What

When `LLAMA_KV_SWA_OFFLOAD=1` is set, forces the SWA KV cache to use only the sliding window size (`n_swa` tokens, ~1500 cells) instead of the full context size. The full context is already available via the base (non-SWA) KV cache in CPU RAM.

## Why

Without this patch, the SWA KV cache is allocated for `ctx_size` cells even though sliding-window models like Gemma 4 only look at the last `n_swa` (1024) tokens per SWA layer. At ctx=8192, the SWA cache alone consumed 1536 MiB VRAM (8192 cells × 25 layers). With the patch, it's fixed at ~288 MiB regardless of context size.

This enables 64k context with all 30 layers on GPU within 16 GB VRAM.

## Files modified
- `src/llama-kv-cache-iswa.cpp` — moved `offload_swa`/`offload_base` definition before `swa_full` check, added override

## Compatibility risk
- Public API: none — controlled by existing `LLAMA_KV_SWA_OFFLOAD` env var
- ABI: none
- Performance impact: zero when ISWA is off. When ISWA is on, SWA cache uses ~15% of the VRAM it would otherwise use.

## Test / validation
- Build: `cmake --build . --target llama` in llama.cpp/build
- Test: loaded Gemma 4 26B-A4B at ctx=65536 with 30 layers GPU. VRAM fixed at 14.9 GB regardless of context (4k → 64k). SWA cache: 288 MiB fixed.
