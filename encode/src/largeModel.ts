/**
 * largeModel.ts
 *
 * Singleton ModelGPU instance for subvocal-large (Gemma 4 26B-A4B on CUDA).
 *
 * Call initLargeModel(path) at harness boot before using SpeculativeDecoder.
 * The GPU model is the verifier brain in the speculative decoding loop:
 *   CPU (subvocal-small) drafts → GPU (subvocal-large) verifies in one batch.
 *
 * Model: Gemma 4 26B-A4B (MoE, 128 experts top-8, 30 layers, hidden 2816).
 * Fits entirely in VRAM at Q4_K_XL (13.26 GiB) on RTX 4070 Ti SUPER (16 GiB).
 */

import { ModelGPU, type ModelOptions } from '@subvocal/synapse';
import { activeProfile } from './modelProfile.js';

let _model: ModelGPU | null = null;

export interface LargeModelConfig extends ModelOptions {
  /** Path to the GGUF model file (Gemma 4 26B-A4B Q4_K_XL). */
  modelPath: string;
}

/**
 * Load subvocal-large onto the GPU and initialize the in-process binding.
 * Must be called before creating a SpeculativeDecoder.
 * Safe to call multiple times — frees the previous model if already loaded.
 *
 * KV tier wiring (see ModelProfile.kvTiers for the full 2-vs-3-tier explanation):
 *   hot  — SWA KV in VRAM (~fixed). Active when noKvOffload=true; controlled by the
 *           LLAMA_KV_SWA_OFFLOAD env var read by the patched llama C++ code.
 *   warm — the live global KV of the current context, in system RAM. cparams.offload_kqv=false
 *           (noKvOffload option). ctx = kvTiers.warm.contextSize.
 *   cold — the disk KV *checkpoint cache* (encode/src/kvColdStore.ts), owned by AgentLoop, NOT
 *           by this loader. It is not a model-load option: it saves/restores KV snapshots at
 *           the state-serialization layer, not a live context paged to SSD. (Live SSD KV paging
 *           via LLAMA_KV_DISK_PATH is a separate upstream feature that is not ported anywhere.)
 *
 * On Apple Silicon hot and warm share the one unified memory pool (no separate VRAM); the ISWA
 * split still routes SWA-layer KV access appropriately via llama.cpp internals.
 */
export function initLargeModel(config: LargeModelConfig): void {
  if (_model) {
    _model.free();
    _model = null;
  }

  const tiers = activeProfile.kvTiers;
  const iswa = activeProfile.largeOpts.noKvOffload ?? false;

  // Hot tier: pin SWA KV to VRAM via env var (read by getenv() in llama-kv-cache-iswa.cpp).
  if (iswa) {
    process.env.LLAMA_KV_SWA_OFFLOAD = '1';
  } else {
    delete process.env.LLAMA_KV_SWA_OFFLOAD;
  }

  const { modelPath, ...opts } = config;
  const contextSize = opts.contextSize ?? tiers.warm.contextSize;
  _model = new ModelGPU(modelPath, {
    contextSize,
    threads: opts.threads ?? 4,
    threadsBatch: opts.threadsBatch ?? opts.threads ?? 4,
    gpuLayers: opts.gpuLayers ?? 999,
    noKvOffload: iswa,
    ...opts,
  });

  const cold = tiers.cold
    ? `  cold=SSD checkpoint cache @ ${tiers.cold.diskPath} (${(tiers.cold.budgetBytes / 1073741824).toFixed(0)} GiB)`
    : '';
  const capped = contextSize !== tiers.warm.contextSize ? ` (capped from ${tiers.warm.contextSize})` : '';
  console.log(`[largeModel] loaded: ${modelPath}`);
  console.log(`[largeModel] KV tiers: hot=${tiers.hot.approxMib}MiB/vram  warm/ram ctx=${contextSize}${capped}${iswa ? '  (ISWA split active)' : ''}${cold}`);
}

/**
 * Returns the live ModelGPU instance. Throws if initLargeModel() was not called.
 */
export function getLargeModel(): ModelGPU {
  if (!_model) {
    throw new Error(
      'subvocal-large is not initialized. Call initLargeModel({ modelPath, gpuLayers }) before using SpeculativeDecoder.',
    );
  }
  return _model;
}

/**
 * Release the model and free its VRAM. After this call, getLargeModel() throws
 * until initLargeModel() is called again.
 */
export function freeLargeModel(): void {
  if (_model) {
    _model.free();
    _model = null;
  }
}
