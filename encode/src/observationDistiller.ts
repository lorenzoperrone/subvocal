/**
 * observationDistiller.ts
 *
 * Epico 4.1.3 — Observation Distillation
 *
 * Compresses noisy build/test output into dense token signals and injects
 * them into the large GPU model's KV cache. A regex heuristic runs first
 * (~0ms); on miss, the small CPU model generates a compact summary.
 */

import { type BaseModel, sampleGreedy } from '@subvocal/synapse';

// ── Result ─────────────────────────────────────────────────────────────────────

export interface DistilledObservation {
	/** Compact tokens representing the error summary. NOT detokenized — raw token IDs. */
	tokens: Int32Array;
	/** Human-readable label for logging. */
	label: string;
	/** Estimated token savings (original token count - distilled count). */
	savedTokens: number;
}

// ── Regex patterns (first pass, before LLM) ────────────────────────────────────

const TS_ERROR_RE = /error TS(\d+): (.+)/;
const PY_ERROR_RE = /(\w+Error): (.+)/;
const ESLINT_RE = /\s+(\d+:\d+)\s+error\s+(.+?)\s+(\S+)/;
const JEST_RE = /● (.+)/;

// ── Heuristic distillation ─────────────────────────────────────────────────────

/**
 * Fast regex-based distillation that runs BEFORE the LLM.
 * Extracts error type, file, line, and message from common formats:
 *   - TypeScript compiler errors
 *   - Python tracebacks
 *   - ESLint output
 *   - jest/vitest failures
 *
 * Returns null if no pattern matched (fall back to distillObservation).
 */
export function heuristicDistill(
	output: string,
	model: BaseModel,
): DistilledObservation | null {
	let errorType = '';
	let nodeLabel = '';

	const ts = TS_ERROR_RE.exec(output);
	if (ts) {
		errorType = `TS${ts[1]}`;
		nodeLabel = ts[2].slice(0, 30).replace(/\s+/g, '_');
	}

	if (!errorType) {
		const py = PY_ERROR_RE.exec(output);
		if (py) {
			errorType = py[1];
			nodeLabel = py[2].slice(0, 30).replace(/\s+/g, '_');
		}
	}

	if (!errorType) {
		const es = ESLINT_RE.exec(output);
		if (es) {
			errorType = es[2];
			nodeLabel = `${es[1]}_${es[3].slice(0, 20).replace(/\s+/g, '_')}`;
		}
	}

	if (!errorType) {
		const je = JEST_RE.exec(output);
		if (je) {
			errorType = 'TestFail';
			nodeLabel = je[1].slice(0, 30).replace(/\s+/g, '_');
		}
	}

	if (!errorType) return null;

	const summary = `[ERR:${errorType}:${nodeLabel}]`;
	const tokens = model.tokenize(summary, false, true);
	const originalTokens = model.tokenize(output, true, false);
	return {
		tokens,
		label: summary,
		savedTokens: originalTokens.length - tokens.length,
	};
}

// ── LLM distillation ───────────────────────────────────────────────────────────

/**
 * Distill raw compiler/test output into a compact token signal.
 *
 * Uses the small CPU model with a specialised prompt that asks it to output
 * exactly 2–4 tokens: [TYPE]:[NODE]:[HINT].
 *
 * The output tokens are NOT detokenised — they are injected directly
 * into the GPU's KV cache via decodeAppend().
 */
export function distillObservation(
	output: string,
	model: BaseModel,
): DistilledObservation {
	const prompt = [
		'Distill this error output into EXACTLY this format: [TYPE]:[NODE]:[HINT]',
		'Output nothing else.',
		'---',
		output,
	].join('\n');

	const promptTokens = model.tokenize(prompt, true, true);
	const newlineToken = model.tokenize('\n', false, false)[0];

	model.forward(promptTokens);

	const generated: number[] = [];
	const maxTokens = 6;

	for (let i = 0; i < maxTokens; i++) {
		if (i > 0) {
			model.decodeAppend(Int32Array.of(generated[i - 1]));
		}

		const logits = model.getLogitsFast(); // zero-alloc shadow buffer, consumed synchronously
		const next = sampleGreedy(logits);

		if (next === newlineToken) break;
		generated.push(next);
	}

	const label = generated.length > 0 ? model.detokenize(Int32Array.from(generated)) : '';
	const originalTokens = model.tokenize(output, true, false);

	return {
		tokens: Int32Array.from(generated),
		label,
		savedTokens: originalTokens.length - generated.length,
	};
}
