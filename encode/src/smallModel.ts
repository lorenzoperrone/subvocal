/**
 * smallModel.ts
 *
 * Singleton instance for subvocal-small (Gemma 4 E2B).
 *
 * Call initSmallModel(path) once at harness boot before using intentRouter or
 * tokenize(). All preprocessor functions share this single in-process model
 * instance — no REST, no serialization, no HTTP overhead.
 *
 * Backend is profile-driven (ModelProfile.smallBackend): 'cpu' loads ModelCPU
 * (ik_llama.cpp static, the Linux default); 'gpu' loads ModelGPU — required on Mac,
 * where the CPU addon is deliberately not built and everything model-shaped goes
 * through Metal (feedback_avoid_ik_llama_cpu_backend).
 *
 * Model: Gemma 4 E2B (dense, 35 layers, hidden 1536, vocab 262144, 2.43 GiB Q4_K_XL).
 */

import { ModelCPU, ModelGPU, type BaseModel, type ModelOptions } from '@subvocal/synapse';
import { activeProfile } from './modelProfile.js';

let _model: BaseModel | null = null;

export interface SmallModelConfig extends ModelOptions {
  /** Path to the GGUF model file (Gemma 4 E2B Q4_K_XL). */
  modelPath: string;
}

/**
 * Load subvocal-small and initialize the in-process binding.
 * Must be called before routeIntent(), tokenize(), or assembleTensorPayload().
 * Safe to call multiple times — frees the previous model if already loaded.
 */
export function initSmallModel(config: SmallModelConfig): void {
  if (_model) {
    _model.free();
    _model = null;
  }
  const { modelPath, ...opts } = config;
  const backend = activeProfile.smallBackend ?? 'cpu';
  // Profile smallOpts are the base (Mac needs gpuLayers 999 for Metal); caller opts win.
  const resolved: ModelOptions = {
    ...activeProfile.smallOpts,
    threadsBatch: opts.threadsBatch ?? opts.threads ?? activeProfile.smallOpts.threads,
    ...opts,
  };
  _model = backend === 'gpu'
    ? new ModelGPU(modelPath, resolved)
    : new ModelCPU(modelPath, resolved);
  console.log(`🧠 subvocal-small loaded (${backend}): ${modelPath}`);
}

/**
 * Returns the live small-model instance. Throws if initSmallModel() was not called.
 */
export function getSmallModel(): BaseModel {
  if (!_model) {
    throw new Error(
      'subvocal-small is not initialized. Call initSmallModel({ modelPath }) before using the preprocessor.',
    );
  }
  return _model;
}

/**
 * Release the model and free its RAM. After this call, getSmallModel() throws
 * until initSmallModel() is called again.
 */
export function freeSmallModel(): void {
  if (_model) {
    _model.free();
    _model = null;
  }
}
