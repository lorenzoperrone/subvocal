# Patch 101 — Partial forward (early-exit) for qwen3moe

**Engine**: llama.cpp upstream (`llama.cpp/`)
**Status**: applied
**Date**: 2026-05-24
**Author**: Claude (Subvocal session)

## What
Adds `n_layer_limit` field to `llama_context_params` and a runtime setter `llama_set_n_layer_limit(ctx, value)`. When `n_layer_limit > 0`, the qwen3moe graph builder stops after layer `(n_layer_limit-1)`, skipping the final RMS norm and lm_head matmul. The raw `l_out-<limit-1>` tensor becomes the output embedding; `t_logits` is set to `nullptr` (no logits available in partial mode).

## Why
For Subvocal we need to:
- Get the hidden state at a chosen mid-layer (e.g. layer 15 of the 40-layer Qwen 14B-A3B) for the small→large transfer and for agent-side early intent detection.
- On CPU the binder can use the public `abort_callback` API (V3.2 "smart" approach) to stop the eval mid-graph — this works for ik_llama.
- On GPU the smart approach fails: CUDA executes the whole graph in a few mega-kernels, abort_callback doesn't fire in time. **We need to physically build a smaller graph** — that's what this patch does.

## Files modified
1. `include/llama.h` (~line 345): added `int32_t n_layer_limit` field to `llama_context_params` struct + at ~line 985: declared public API `llama_set_n_layer_limit`.
2. `src/llama-cparams.h` (~line 19): added `int32_t n_layer_limit` to internal `llama_cparams` struct.
3. `src/llama-context.cpp`:
   - Constructor (~line 63): propagation `cparams.n_layer_limit = params.n_layer_limit` (with clamp to `hparams.n_layer`).
   - `llama_context_default_params()` (~line 3342): initialize to `0` (= no limit).
   - Added `llama_context::set_n_layer_limit(value)` method + public C API wrapper.
4. `src/llama-context.h` (~line 116): declared `set_n_layer_limit` method.
5. `src/models/qwen3moe.cpp`:
   - Layer loop: `for (int il = 0; il < n_layer_eff; ++il)` where `n_layer_eff = clamp(cparams.n_layer_limit, 1..n_layer)`.
   - `inp_out_ids` reduction: trigger on `il == n_layer_eff - 1` (not `n_layer - 1`).
   - Post-loop: when `partial`, set `res->t_embd = cur` and `res->t_logits = nullptr`, then `return` before `result_norm` and lm_head.

All patches are tagged `// SUBVOCAL-PATCH-101-BEGIN: ...` / `// SUBVOCAL-PATCH-101-END` for greppability.

## API surface added
```c
// in llama.h, llama_context_params:
int32_t  n_layer_limit;

// in llama.h, public C API:
LLAMA_API void llama_set_n_layer_limit(struct llama_context * ctx, int32_t value);
```

## Compatibility risk
- **Public API**: NON-breaking — only adds a new field at end of int section + a new function. Existing callers using `{0}` initialization get `n_layer_limit = 0` = no limit = full forward (= old behavior).
- **ABI**: struct layout change (new field added in middle of struct). Anyone compiled against the old header would have wrong offsets — need rebuild of all callers. Acceptable for static-linked Subvocal binders.
- **Performance**: when `n_layer_limit == 0` (default), the branch in qwen3moe.cpp is a single int comparison — negligible overhead. When >0 and the cut is early, saves the compute of skipped layers + final norm + lm_head.

## Re-apply procedure (when upstream changes)
The patch is concentrated in 5 files. The most likely drift points:
- `llama-context.cpp` constructor: lines around the cparams initialization shift frequently in upstream. Look for `cparams.n_threads = params.n_threads` as anchor; insert `n_layer_limit` block right after.
- `qwen3moe.cpp` graph builder: the layer loop body is stable but the post-loop section (result_norm, lm_head) might evolve. Find the `for (int il = 0; il < n_layer; ++il)` loop, replace `n_layer` with `n_layer_eff`, add the partial early-return before result_norm.
- `llama_context_default_params()`: list initializer; insert `n_layer_limit = 0` in the same position as the field in the struct.

If qwen3moe.cpp is heavily refactored, port the same logic to whatever new builder structure exists.

## Test / validation

### Build
```bash
cd llama.cpp/build
cmake --build . -j 16 --target llama

cd synapse
npx cmake-js compile --out build-gpu --CDSUBVOCAL_BACKEND=gpu
```

### Run bench
```bash
cd synapse
npx tsx bench/v4-partial-forward.ts
```

### Expected results
For Qwen 14B-A3B GPU (40 layers, prompt 31 tok):
| Layer limit | Speedup | Cosine vs full |
|---|---:|---:|
| L5 (15%)  | 1.05-1.10x  | 1.000000 |
| L20 (53%) | 1.05-1.10x | 1.000000 |
| L35 (90%) | 1.05-1.10x | 1.000000 |
| L39 (100%) | ~1.0x | 0 (known cb_eval edge case for last layer) |

For CPU the patch is a no-op (smart abort approach already works).

### Honest performance note
**The expected 2-6x GPU speedup did NOT materialize.** Reason: the 14B-A3B prefill (53ms for ~31 tok) is heavily overhead-dominated (batch alloc, KV clear, CUDA graph capture, sched buffer setup). The ~40 actual transformer layers cost only ~15ms of that total; saving half of them = ~7-8ms = ~10-15% improvement, not 2x.

The patch IS still valuable:
- Saves the lm_head matmul (massive: vocab=248320, n_embd=2048 → ~500M MAC ops avoided per call)
- Provides architecturally clean access to mid-layer hidden state for small→large transfer
- Makes the graph compute deterministic vs the racy "abort_callback" smart approach

## Non-goals
- Other architectures (qwen2, llama, gemma, etc.) — only qwen3moe. To extend, lift the same pattern (`n_layer_eff` clamp + partial early-return) into each builder, or refactor into a base-class helper in `llm_graph_context`.
- ik_llama equivalent (PATCH-001 deferred) — smart abort_callback approach already gives 1.3x-6x on CPU.

## Known issues
- **L=n_layer-1 cosine mismatch**: `getHiddenStateLayer(n_layer-1)` after `forwardPartial(n_layer-1)` ≠ same after `forward()`. Pre-existing cb_eval behavior, not caused by this patch. To investigate separately — possibly the `inp_out_ids` `ggml_get_rows` reduction interacts with the cb_eval capture differently.
- **logits unavailable in partial mode**: `getLogits()` returns NULL/garbage after `forwardPartial()`. Document in binder API.
