/**
 * workerIntent.ts — main-thread client for the worker-based intent classifier
 * (see intentWorker.ts for the why and the memory-cost analysis).
 *
 * Lifecycle: construct → worker loads its model in the background (isReady flips when done) →
 * classify() resolves off-thread → terminate() on session teardown. Any worker failure marks
 * the client dead permanently; callers are expected to fall back to the in-thread
 * routeIntent() (see the wiring in utter.ts) — classification is an optimization, never a
 * correctness dependency.
 */

import { Worker } from 'node:worker_threads';
import type { IntentResult } from './intentRouter.js';

export class WorkerIntentClassifier {
  private worker: Worker | null = null;
  private ready = false;
  private dead = false;
  private nextId = 1;
  private pending = new Map<number, { resolve: (r: IntentResult) => void; reject: (e: Error) => void }>();

  constructor(modelPath: string) {
    // Under tsx the source .ts is the real module; a compiled build ships .js next to this
    // file instead. Resolve whichever variant this module itself is running as.
    const ext = import.meta.url.endsWith('.ts') ? '.ts' : '.js';
    try {
      this.worker = new Worker(new URL(`./intentWorker${ext}`, import.meta.url), {
        workerData: { modelPath },
      });
    } catch (e) {
      this.dead = true;
      console.warn(`[workerIntent] failed to spawn worker (${(e as Error).message}) — falling back to in-thread classification`);
      return;
    }
    this.worker.on('message', (msg: { type: string; id?: number; result?: IntentResult; error?: string }) => {
      if (msg.type === 'ready') {
        this.ready = true;
        return;
      }
      const entry = msg.id !== undefined ? this.pending.get(msg.id) : undefined;
      if (!entry) return;
      this.pending.delete(msg.id!);
      if (msg.type === 'result' && msg.result) entry.resolve(msg.result);
      else entry.reject(new Error(msg.error ?? 'worker classification failed'));
    });
    const die = (why: string) => {
      this.dead = true;
      this.ready = false;
      for (const { reject } of this.pending.values()) reject(new Error(why));
      this.pending.clear();
    };
    this.worker.on('error', (e) => die(`worker error: ${e.message}`));
    this.worker.on('exit', (code) => { if (code !== 0) die(`worker exited (${code})`); });
  }

  /** True once the worker's model finished loading and no failure occurred. */
  get isReady(): boolean {
    return this.ready && !this.dead;
  }

  /** Classify off-thread. Reject = worker unavailable/failed; caller falls back in-thread. */
  classify(text: string): Promise<IntentResult> {
    if (!this.worker || this.dead) return Promise.reject(new Error('worker classifier unavailable'));
    const id = this.nextId++;
    return new Promise<IntentResult>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker!.postMessage({ id, text });
    });
  }

  async terminate(): Promise<void> {
    if (!this.worker) return;
    const w = this.worker;
    this.worker = null;
    this.dead = true;
    await w.terminate();
  }
}
