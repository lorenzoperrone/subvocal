/**
 * intentWorker.ts — worker_threads entry for off-main-thread intent classification.
 *
 * 2026-07 audit item: `IncrementalIntentClassifier`'s model classification is a synchronous
 * native forward() (~99ms on Metal, M11.2) that blocks the main JS thread — noticeable only
 * if the user resumes typing inside that window, but the REPL's keystroke echo shares that
 * thread. This worker owns its OWN small-model instance (the GGUF weights are mmap'd, so the
 * 2.4 GiB is shared with the main thread's instance via the page cache — the marginal cost is
 * just this context's KV + compute buffers at a deliberately small contextSize), so the main
 * thread never blocks on classification.
 *
 * Protocol (see WorkerIntentClassifier in workerIntent.ts):
 *   in :  { id: number, text: string }
 *   out: { type: 'ready' } | { type: 'result', id, result: IntentResult } | { type: 'error', id, error }
 */

import { parentPort, workerData } from 'node:worker_threads';
import { ModelCPU, ModelGPU, type BaseModel } from '@subvocal/synapse';
import { activeProfile } from './modelProfile.js';
import { classifyIntentWith } from './intentRouter.js';

const { modelPath } = workerData as { modelPath: string };

// Same backend rule as smallModel.ts (Mac has no CPU addon — everything via Metal). Small
// context on purpose: classification prompts are ~60 tokens; 2048 keeps the buffers tiny.
const backend = activeProfile.smallBackend ?? 'cpu';
const opts = { ...activeProfile.smallOpts, contextSize: 2048 };
const model: BaseModel = backend === 'gpu' ? new ModelGPU(modelPath, opts) : new ModelCPU(modelPath, opts);

parentPort!.postMessage({ type: 'ready' });

parentPort!.on('message', (msg: { id: number; text: string }) => {
  try {
    const result = classifyIntentWith(model, msg.text);
    parentPort!.postMessage({ type: 'result', id: msg.id, result });
  } catch (e) {
    parentPort!.postMessage({ type: 'error', id: msg.id, error: String(e) });
  }
});
