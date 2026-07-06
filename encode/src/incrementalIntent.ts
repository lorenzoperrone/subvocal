/**
 * incrementalIntent.ts
 *
 * Mechanism B from doc/research/predictive-prefill-while-typing.md: routeIntent() is cheap
 * (~15ms on the E2B small model) and the CPU is genuinely idle while the user is composing a
 * prompt. Unlike CPU-draft speculative decoding (project memory
 * project_cpu_draft_specdec_negative), there's no real-time GPU decode loop to keep pace with
 * here -- discarding a stale in-flight classification costs nothing.
 *
 * Wired to the REPL since M11.1: utter.ts feeds rl.line from a process.stdin 'keypress'
 * listener, and a settled classification is reused at submit via
 * PreprocessInput.precomputedIntent. Classification only actually runs once input is quiet
 * for debounceMs, so input superseded before the timer fires is never computed at all.
 * Note: the classification itself is synchronous on the main thread (~99ms on Metal with the
 * E2B model, M11.2) — acceptable because it fires only after a typing pause.
 */

import { routeIntent, routeIntentRegex, type IntentResult } from './intentRouter.js';

export interface IncrementalIntentOptions {
	/** Quiet period before classification actually runs. Default 150ms. */
	debounceMs?: number;
	/** Mirror of PreprocessInput.cpuOff -- use the regex fallback instead of the small model. */
	cpuOff?: boolean;
	/**
	 * Off-main-thread classifier (see workerIntent.ts). When provided (and cpuOff is false)
	 * the debounced classification runs here instead of the synchronous in-thread
	 * routeIntent(), so the main thread never blocks. A rejection falls back to the
	 * in-thread path transparently.
	 */
	classifyAsync?: (text: string) => Promise<IntentResult>;
}

export class IncrementalIntentClassifier {
	private readonly debounceMs: number;
	private readonly cpuOff: boolean;
	private readonly classifyAsync?: (text: string) => Promise<IntentResult>;
	private timer: ReturnType<typeof setTimeout> | null = null;
	private latest: IntentResult | null = null;
	private latestText = '';
	/** Monotonic feed sequence — an async result is accepted only if no newer feed exists. */
	private seq = 0;

	constructor(opts: IncrementalIntentOptions = {}) {
		this.debounceMs = opts.debounceMs ?? 150;
		this.cpuOff = opts.cpuOff ?? false;
		this.classifyAsync = opts.classifyAsync;
	}

	/**
	 * Feed the current partial prompt text. Cancels any pending classification scheduled by
	 * a previous, now-stale call and reschedules. Safe to call on every keystroke -- the
	 * actual classification only runs once typing pauses for debounceMs.
	 */
	feed(partialPrompt: string, onResult?: (result: IntentResult) => void): void {
		if (this.timer) clearTimeout(this.timer);
		const mySeq = ++this.seq;
		this.timer = setTimeout(() => {
			this.timer = null;
			if (!this.cpuOff && this.classifyAsync) {
				// Off-thread path: accept the result only if it is still the newest feed —
				// a slow worker round-trip must never overwrite a fresher classification.
				this.classifyAsync(partialPrompt)
					.catch(() => routeIntent(partialPrompt)) // worker down → in-thread fallback
					.then((result) => {
						if (mySeq !== this.seq) return; // superseded while in flight
						this.latestText = partialPrompt;
						this.latest = result;
						onResult?.(result);
					})
					.catch(() => { /* both paths failed — keep the previous classification */ });
				return;
			}
			this.latestText = partialPrompt;
			this.latest = this.cpuOff ? routeIntentRegex(partialPrompt) : routeIntent(partialPrompt);
			onResult?.(this.latest);
		}, this.debounceMs);
	}

	/** Most recent completed classification, or null if none has settled yet. */
	getLatest(): IntentResult | null {
		return this.latest;
	}

	/** The exact text the latest classification was computed from (staleness check for callers). */
	getLatestText(): string {
		return this.latestText;
	}

	/** Cancel any pending (not-yet-run) classification. */
	cancel(): void {
		if (this.timer) {
			clearTimeout(this.timer);
			this.timer = null;
		}
	}
}
