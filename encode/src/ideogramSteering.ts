/**
 * ideogramSteering.ts — Substory 4.2
 *
 * Logit-level tool steering via ideogram tokens. Replaces text-based
 * <tool_call> parsing with direct token-ID recognition of AST markers.
 *
 * How it works:
 *   1. Fold markers ⊂ (FOLD_START) and ⊃ (FOLD_END) bracket edit targets.
 *   2. An ideogram token from the tagMap identifies the AST node.
 *   3. Code between fold markers is the replacement content.
 *   4. Edit is applied via astEditor.atomically.
 *
 * Also performs post-sample rejection: blocks tokens that would start
 * a new conversation turn or degenerate loop (Gemma-specific patterns).
 */

import { sampleGreedy } from '@subvocal/synapse';
import type { ModelGPU } from '@subvocal/synapse';
import {
	FOLD_START_TOKEN_ID, FOLD_END_TOKEN_ID, INSERT_MARKER_TOKEN_ID,
	RENAME_OP_TOKEN_ID, DELETE_OP_TOKEN_ID,
} from './foldTokens.js';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface IdeogramEdit {
	/** Token ID of the ideogram that identified the target AST node. */
	tokenId: number;
	/** Node label from the tagMap (e.g. "function_declaration:calculateTotal"). */
	nodeLabel: string;
	/**
	 * Generated payload text (detokenized code between ⊂ and ⊃, or between ⊕ and ⊃, or
	 * between ⊆ and ⊃). Semantics depend on `mode`: whole-node/inserted CODE for
	 * 'edit'/'insert', the new IDENTIFIER NAME for 'rename', ignored (no payload expected)
	 * for 'delete'.
	 */
	newCode: string;
	/**
	 * Edit mode: 'edit' for ⊂ (replace node — the ORIGINAL whole-node-replacement syntax,
	 * still the default when no op token follows the tag), 'insert' for ⊕ (insert after
	 * node), 'rename' for ⊆ (M15.5 — swap the node's own name, no body re-emission),
	 * 'delete' for ⊇ (M15.5 — remove the node entirely, no payload).
	 */
	mode: 'edit' | 'insert' | 'rename' | 'delete';
}

export interface SteeringConfig {
	/** The GPU model (for detokenizing code buffers). */
	model: ModelGPU;
	/** Map from token ID → node label, from preprocess(). */
	tagMap: Map<number, string>;
	/** Regex patterns that indicate a degenerate turn restart. */
	poisonPatterns?: RegExp[];
	/** Callback for streaming token output. Called with detokenized text. */
	onToken?: (text: string) => void;
}

type State = 'idle' | 'fold' | 'tag' | 'code' | 'insert' | 'tag-seen';

// ── Poison patterns (Gemma4-specific) ─────────────────────────────────────────
//
// Turn/channel markers are ALWAYS degenerate (the model hallucinating its own turn
// boundaries mid-generation). The tool_call patterns are only poison in EXCLUSIVE steering
// mode, where the prompt forbids tool calls — in hybrid mode tool calls are legal, and
// poisoning them corrupts the native protocol's CLOSING token `<tool_call|>` (its text starts
// with `<tool_call`, matching /<tool_call/i), which forces a malformed call the parser rejects.

/** Always-degenerate markers — poison in every mode. */
export const TURN_POISON: RegExp[] = [
	/<start_of_turn/i,
	/<end_of_turn/i,
	/<\|channel[|>]/,
];

/** Exclusive-steering poison: turn markers PLUS the tool-call bans. */
const DEFAULT_POISON: RegExp[] = [
	...TURN_POISON,
	/<tool_call/i,
	/<\/tool_call/i,
];

// ── State machine ─────────────────────────────────────────────────────────────

export class IdeogramSteering {
	private model: ModelGPU;
	private tagMap: Map<number, string>;
	private poison: RegExp[];
	private onToken?: (text: string) => void;
	private state: State = 'idle';
	private tagTokenId = -1;
	private tagLabel = '';
	private editMode: 'edit' | 'insert' | 'rename' | 'delete' = 'edit';
	private codeTokens: number[] = [];
	private accumulatedText = '';
	/** Collected edits since last reset. */
	edits: IdeogramEdit[] = [];

	constructor(config: SteeringConfig) {
		this.model = config.model;
		this.tagMap = config.tagMap;
		this.poison = config.poisonPatterns ?? DEFAULT_POISON;
		this.onToken = config.onToken;
	}

	/**
	 * Sample the next token with rejection of poison tokens.
	 * Feeds the token into the fold→tag→code→fold state machine.
	 * When an edit completes (⊃), it's added to this.edits[].
	 * Returns stop=true after an edit completes (caller should NOT continue decoding).
	 */
	sample(logits: Float32Array): { token: number; stop: boolean } {
		const MAX_REJECTS = 20;
		for (let i = 0; i < MAX_REJECTS; i++) {
			const token = sampleGreedy(logits);
			if (this.isPoison(token)) {
				logits[token] = -Infinity;
				continue;
			}
			if (this.onToken) {
				this.onToken(this.model.detokenize(Int32Array.of(token)));
			}
			const result = this.feed(token);
			return { token, stop: result.stop };
		}
		const token = sampleGreedy(logits);
		if (this.onToken) {
			this.onToken(this.model.detokenize(Int32Array.of(token)));
		}
		const result = this.feed(token);
		return { token, stop: result.stop };
	}

	/**
	 * Feed an externally-generated token through the state machine.
	 * Used by speculative decode where tokens come from the draft loop
	 * rather than being sampled inside IdeogramSteering itself.
	 */
	feedToken(token: number): { stop: boolean; edit: IdeogramEdit | null } {
		return this.feed(token);
	}

	/** Reset for a new turn. */
	reset(): void {
		this.state = 'idle';
		this.tagTokenId = -1;
		this.tagLabel = '';
		this.editMode = 'edit';
		this.codeTokens = [];
		this.accumulatedText = '';
		this.edits = [];
	}

	// ── Private ──────────────────────────────────────────────────────────────

	private feed(token: number): { stop: boolean; edit: IdeogramEdit | null } {
		switch (this.state) {
			case 'idle':
				if (token === FOLD_START_TOKEN_ID) {
					this.state = 'fold';
					return { stop: false, edit: null };
				}
				if (token === INSERT_MARKER_TOKEN_ID) {
					this.state = 'insert';
					return { stop: false, edit: null };
				}
				return { stop: false, edit: null };

			case 'insert': {
				if (token === INSERT_MARKER_TOKEN_ID) {
					this.state = 'insert';
					return { stop: false, edit: null };
				}
				if (token === FOLD_END_TOKEN_ID) {
					this.state = 'idle';
					return { stop: false, edit: null };
				}
				const insertLabel = this.tagMap.get(token);
				if (insertLabel) {
					this.tagTokenId = token;
					this.tagLabel = insertLabel;
					this.editMode = 'insert';
					this.state = 'code';
					this.codeTokens = [];
					return { stop: false, edit: null };
				}
				this.state = 'idle';
				return { stop: false, edit: null };
			}

			case 'fold':
				// Next token must be an ideogram from tagMap (the edit target).
				if (token === FOLD_START_TOKEN_ID) {
					// Double fold-start — restart
					this.state = 'fold';
					return { stop: false, edit: null };
				}
				if (token === FOLD_END_TOKEN_ID) {
					// Empty fold — skip
					this.state = 'idle';
					return { stop: false, edit: null };
				}
				const label = this.tagMap.get(token);
				if (label) {
					this.tagTokenId = token;
					this.tagLabel = label;
					this.editMode = 'edit'; // default; 'tag-seen' below may override via an op token
					this.state = 'tag-seen';
					this.codeTokens = [];
					return { stop: false, edit: null };
				}
				// Not an ideogram after fold-start — abort fold
				this.state = 'idle';
				return { stop: false, edit: null };

			// M15.5: one token of lookahead after the target tag. An op token here SWITCHES
			// the payload semantics (rename: short new-name payload; delete: no payload at
			// all); anything else is the FIRST token of a whole-node-replacement body (the
			// original ⊂<tag><code>⊃ syntax), so it must be pushed into codeTokens rather
			// than dropped — this state exists ONLY to peek at that one token; edit_mode
			// stays 'edit' unless an op token is seen.
			case 'tag-seen':
				if (token === RENAME_OP_TOKEN_ID) {
					this.editMode = 'rename';
					this.state = 'code';
					return { stop: false, edit: null };
				}
				if (token === DELETE_OP_TOKEN_ID) {
					this.editMode = 'delete';
					this.state = 'code';
					return { stop: false, edit: null };
				}
				if (token === FOLD_END_TOKEN_ID) {
					// Empty whole-node replacement (⊂tag⊃, no payload) — same as 'code' seeing
					// FOLD_END with zero accumulated tokens.
					this.edits.push({ tokenId: this.tagTokenId, nodeLabel: this.tagLabel, newCode: '', mode: this.editMode });
					this.state = 'idle';
					return { stop: true, edit: null };
				}
				// Not an op token — this IS the first content token of a whole-node body.
				this.codeTokens.push(token);
				this.state = 'code';
				return { stop: false, edit: null };

			case 'code':
				if (token === FOLD_END_TOKEN_ID) {
					const newCode = this.codeTokens.length > 0
						? this.model.detokenize(Int32Array.from(this.codeTokens))
						: '';
					this.edits.push({
						tokenId: this.tagTokenId,
						nodeLabel: this.tagLabel,
						newCode,
						mode: this.editMode,
					});
					this.state = 'idle';
					return { stop: true, edit: null };
				}
				if (token === FOLD_START_TOKEN_ID) {
					this.codeTokens.push(token);
					return { stop: false, edit: null };
				}
				if (token === INSERT_MARKER_TOKEN_ID) {
					this.codeTokens.push(token);
					return { stop: false, edit: null };
				}
				this.codeTokens.push(token);
				return { stop: false, edit: null };

			default:
				return { stop: false, edit: null };
		}
	}

	/**
	 * Check if a token would produce text that matches known poison patterns.
	 * We detokenize the accumulated text + new token and check against regex.
	 */
	private isPoison(token: number): boolean {
		// Quick check: if accumulated text is already fine, only check the new token
		const candidate = this.model.detokenize(Int32Array.of(token));
		const combined = this.accumulatedText + candidate;

		for (const re of this.poison) {
			if (re.test(combined)) return true;
		}
		// Token is safe — accumulate
		this.accumulatedText = combined;
		return false;
	}
}
