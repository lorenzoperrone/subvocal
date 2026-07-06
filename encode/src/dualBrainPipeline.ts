/**
 * dualBrainPipeline.ts
 *
 * Substory 18 — Dual-Brain Pipeline
 *
 * Overlap CPU pre-processing of the NEXT turn with GPU decode of the CURRENT
 * turn.  preprocess (intent + AST + compress) takes ~1s; GPU decode takes
 * ~2.5s.  By speculatively pre-computing turn N+1's context during turn N's
 * decode we eliminate the CPU gap between turns.
 *
 * Key constraint: no threads — async scheduling (setImmediate) keeps
 * everything on the main event loop.
 */

import { preprocess, type PreprocessInput, type PreprocessResult } from './index.js';

/**
 * Pre-compute the next turn's preprocess result while the current turn's
 * GPU decode runs.  Returns a promise that resolves when preprocess completes.
 */
export function precomputeNextTurn(
  input: PreprocessInput,
): Promise<PreprocessResult> {
  return new Promise(resolve => {
    setImmediate(async () => {
      const result = await preprocess(input);
      resolve(result);
    });
  });
}

/**
 * Pipeline controller: manages precomputed preprocess results.
 *
 * Usage in the agent loop:
 *   1. pipeline.schedule(input)   ← fire-and-forget, returns immediately
 *   2. GPU decode runs
 *   3. const result = await pipeline.take()  ← get precomputed (or null)
 */
export class PipelineController {
  private pending: Promise<PreprocessResult> | null = null;

  /** Start precomputing the next turn. Returns immediately. */
  schedule(input: PreprocessInput): void {
    this.pending = precomputeNextTurn(input);
  }

  /** Get the precomputed result. Awaits if not yet ready. */
  async take(): Promise<PreprocessResult | null> {
    const result = this.pending;
    this.pending = null;
    return result ?? null;
  }

  /** Cancel pending precomputation (result will be discarded). */
  cancel(): void {
    this.pending = null;
  }
}
