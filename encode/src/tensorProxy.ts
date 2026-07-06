/**
 * tensorProxy.ts
 *
 * Epico 4.1.8 — Proxy Tensoriale for Subvocal.
 *
 * Filters massive HTTP API responses (4000+ lines of JSON) through the small
 * CPU model, extracting only the relevant fields/metrics as a compact token
 * signal for KV cache injection. Same principle as Observation Distillation,
 * but for HTTP API responses.
 *
 * Small responses (< 500 chars) bypass the model entirely — tokenised as-is.
 */

import { type ModelCPU, sampleGreedy } from '@subvocal/synapse';
import { type DistilledObservation } from './observationDistiller.js';

// ── Types ──────────────────────────────────────────────────────────────────────

export { type DistilledObservation };

// ── Config ─────────────────────────────────────────────────────────────────────

export interface TensorProxyConfig {
	/** Maximum characters of raw JSON to send to the model (truncate beyond). */
	maxInputChars: number;
	/** Fields to extract (empty = let the model decide). */
	extractFields?: string[];
}

const DEFAULT_MAX_INPUT_CHARS = 4000;
const SMALL_RESPONSE_THRESHOLD = 500;
const MAX_GENERATION_TOKENS = 128;

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Filter a large JSON API response through subvocal-small.
 * Returns distilled tokens ready for KV cache injection.
 *
 * If the response is small (under 500 chars), returns it tokenised as-is
 * (no distillation needed).
 *
 * @param response  Raw HTTP response body.
 * @param model     Small CPU model instance.
 * @param config    Optional extraction config.
 */
export function proxyFilter(
	response: string,
	model: ModelCPU,
	config?: Partial<TensorProxyConfig>,
): DistilledObservation {
	if (response.length < SMALL_RESPONSE_THRESHOLD) {
		const tokens = model.tokenize(response, true, true);
		return { tokens, label: response, savedTokens: 0 };
	}

	const maxChars = config?.maxInputChars ?? DEFAULT_MAX_INPUT_CHARS;
	const prompt = buildProxyPrompt(response.slice(0, maxChars), config?.extractFields);
	const promptTokens = model.tokenize(prompt, true, true);

	model.forward(promptTokens);

	return distillFromLogits(model, response.length);
}

/**
 * Build the distillation prompt for the small model.
 * Tells it to extract relevant fields and format them as:
 *   [API_RES]:[STATUS]:[HAS_DATA]:[KEY_FIELD=value]
 */
export function buildProxyPrompt(response: string, fields?: string[]): string {
	const fieldHint = fields && fields.length > 0
		? `Prefer these fields: ${fields.join(', ')}.\n`
		: '';

	return (
		'Extract from this API response ONLY the status code, if data exists, and the 2 most important values.\n' +
		'Output EXACTLY this format: [API_RES]:[HTTP_STATUS]:[HAS_DATA|NO_DATA]:[key=value]:[key=value]\n' +
		`${fieldHint}Nothing else.\n` +
		'---\n' +
		response
	);
}

// ── Private helpers ────────────────────────────────────────────────────────────

function distillFromLogits(model: ModelCPU, originalCharCount: number): DistilledObservation {
	const generated: number[] = [];

	for (let step = 0; step < MAX_GENERATION_TOKENS; step++) {
		if (step > 0) {
			model.decodeAppend(Int32Array.of(generated[generated.length - 1]));
		}

		const logits = model.getLogitsFast();
		const next = sampleGreedy(logits);
		generated.push(next);
	}

	const text = generated.length > 0 ? model.detokenize(generated) : '';
	const tokens = generated.length > 0 ? Int32Array.from(generated) : new Int32Array(0);

	// Rough estimate: original ~1 tok per 4 chars, distilled ~count of generated tokens
	const savedTokens = Math.max(0, Math.ceil(originalCharCount / 4) - generated.length);
	return { tokens, label: text, savedTokens };
}
