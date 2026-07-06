/**
 * contextDistiller.ts — M16.3: asymmetric context via the E2B as a long-window reader.
 *
 * The E2B's KV is ~14x cheaper per token than the 12B's (~23 vs ~336 KiB), so the small model
 * can afford to hold a context window several times larger than the big one. This module gives
 * the LONG context (old session history, big files) to the E2B and asks it to DISTILL: extract
 * only the facts, decisions, constraints and code snippets relevant to the CURRENT task. The
 * 12B then reasons over the compact distillate instead of a blind tail-truncation.
 *
 * This is NOT speculative decoding (the drafter needs symmetric context for its identical-output
 * guarantee) — it is a deliberately lossy compaction mechanism whose quality depends on the
 * E2B's faithfulness. Measured go/no-go bench: training/bench-distill.mjs (planted-fact
 * scenarios, 12B answers with full context vs distillate vs naive truncation).
 *
 * See doc/substories/M16-agent-robustness.md (Story M16.3).
 */

import type { BaseModel } from '@subvocal/synapse';
import { MacE2BProfile } from './modelProfile.js';

export interface DistillInput {
	/** The long context to compress: old turns, file contents, tool outputs. */
	context: string;
	/** The user's CURRENT task — distillation is task-directed, not a generic summary. */
	task: string;
	/** Max tokens to generate for the distillate (default 512). */
	maxOutputTokens?: number;
}

export interface DistillResult {
	distillate: string;
	inputTokens: number;
	outputTokens: number;
	elapsedMs: number;
}

const DISTILLER_SYSTEM_PROMPT =
	'You are a context distiller for a coding agent. You receive a long session history and the ' +
	"user's CURRENT TASK. Write a compact briefing with ONLY what is needed to do that task:\n" +
	'- decisions and constraints (quote exact numbers, names, paths verbatim)\n' +
	'- relevant code snippets VERBATIM\n' +
	'- relevant file paths\n' +
	'Do NOT invent anything. If something is not in the history, do not mention it. ' +
	'No introduction, no commentary — output only the briefing.';

/**
 * Task-directed distillation of a long context on a small model.
 *
 * The caller owns the model instance (typically an E2B loaded with a large-but-cheap context)
 * and its lifecycle; distill() resets the model's KV, so do not interleave with other uses of
 * the same instance mid-generation.
 */
export function distillContext(model: BaseModel, input: DistillInput): DistillResult {
	const maxOut = input.maxOutputTokens ?? 512;
	const userPrompt =
		`[Session history begins]\n${input.context}\n[Session history ends]\n\n` +
		`CURRENT TASK: ${input.task}\n\n` +
		'Briefing (only what the task needs):';

	const prompt = MacE2BProfile.buildPrompt({ systemPrompt: DISTILLER_SYSTEM_PROMPT, userPrompt });
	const t0 = performance.now();

	// Fresh sequence: the distiller fully owns the KV for this call.
	model.kvCacheSeqRemove(0, 0, -1);
	model.resetNPast(0);

	const promptTokens = model.tokenize(prompt, true, true);
	model.decodeAppend(promptTokens);

	const eot = new Set(MacE2BProfile.eotTokenIds);
	const out: number[] = [];
	for (let i = 0; i < maxOut; i++) {
		const top = model.getLogitsTopK(1)[0] as unknown as { id?: number } | number;
		const id = typeof top === 'object' && top !== null ? (top.id as number) : (top as number);
		if (eot.has(id)) break;
		out.push(id);
		model.decodeAppend(Int32Array.from([id]));
	}

	return {
		distillate: model.detokenize(Int32Array.from(out)).trim(),
		inputTokens: promptTokens.length,
		outputTokens: out.length,
		elapsedMs: performance.now() - t0,
	};
}
