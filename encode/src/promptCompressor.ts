/**
 * promptCompressor.ts — Substory 16: Compress verbose user prompts into a short
 * keyword label using the CPU small model.
 *
 * Epic M10: this used to also build a `tokens` field by re-tokenizing `label` right
 * back (a pointless detokenize-then-retokenize round-trip, not even guaranteed to
 * reproduce the same token sequence) for direct cross-model KV injection — but nothing
 * ever consumed it (the real consumer, agentLoop.ts's buildPrompt(), only used the
 * string). Dropped rather than wired up; see doc/epics/EPIC-M10-compress-prompt-token-injection.md
 * for the cross-model-injection option if this is revisited properly.
 */

import { getSmallModel } from './smallModel.js';
import { sampleGreedy } from '@subvocal/synapse';
import { activeProfile } from './modelProfile.js';

export interface CompressedPrompt {
	label: string;
}

export function compressPrompt(
	userPrompt: string,
	filePath: string,
): CompressedPrompt | null {
	try {
		const model = getSmallModel();
		const sysPrompt =
			'Reduce the coding request to 3-5 essential keywords. Output ONLY the keywords, space-separated, lowercase.';
		const userText = `Request: ${userPrompt}\nFile: ${filePath}`;
		const prompt = activeProfile.buildPrompt({ systemPrompt: sysPrompt, userPrompt: userText });

		const tokens = model.tokenize(prompt, true, true);
		model.forward(tokens);

		const generated: number[] = [];
		const stopIds = new Set(activeProfile.eotTokenIds);
		for (let i = 0; i < 20; i++) {
			if (i > 0) model.decodeAppend(Int32Array.of(generated[generated.length - 1]));
			const logits = model.getLogitsFast(); // zero-alloc shadow buffer, consumed synchronously
			const best = sampleGreedy(logits);
			if (stopIds.has(best)) break;
			generated.push(best);
		}

		const label = generated.length > 0
			? model.detokenize(Int32Array.from(generated)).trim()
			: userPrompt.slice(0, 40);

		return { label };
	} catch {
		return { label: userPrompt.slice(0, 40) };
	}
}
