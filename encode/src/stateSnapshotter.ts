/**
 * stateSnapshotter.ts
 *
 * Epico 4.1 Fase C — State Snapshotting (macchina del tempo).
 *
 * Before the model generates a destructive action (edit, delete, shell command),
 * the AgentLoop takes a snapshot of the KV position + AST state. If the action fails
 * (syntax error, test failure, linter error), the KV is rewound to the snapshot point
 * and the AST state is restored — no text-based "undo" that wastes model context.
 *
 * KV cache: records the seq-0 position (`actionStartPos`) at snapshot time. Rollback
 * removes every token appended since — `kvCacheSeqRemove(0, actionStartPos, -1)` — and
 * resets n_past. O(1), no tensor copy.
 *
 * 2026-07 audit — WHY NOT kvCacheFork(): the previous design forked seq 0 into a backup
 * sequence on every snapshot(). That was wrong three ways, all verified against the real
 * Metal backend:
 *   1. CRASH: Fork() allocated a monotonically-increasing seq id that was never reused or
 *      reset. llama.cpp caps sequences at LLAMA_MAX_SEQ (256) with a hard GGML_ASSERT in
 *      seq_cp — so the ~256th edit in a long REPL session aborted the whole process
 *      (measured: fork #255 ok, #256 → assert failure → SIGABRT).
 *   2. LEAK: a fork keeps the copied cells alive under the backup seq. With enough live
 *      backups a full-context decodeAppend runs out of free cells and fails (measured).
 *   3. DEAD WEIGHT: rollback() never actually restored FROM the backup — it only tail-
 *      evicted seq 0 and reset n_past, which the position record alone already does. The
 *      fork was pure overhead even when perfectly balanced by commit()/rollback().
 * The position-only design below is what the rollback path always effectively used.
 *
 * AST: pushes { source, tagMap, injections } onto a stack. Rollback pops and returns it.
 *
 * NOTE on n_past ownership: this rewinds the MODEL's native n_past_ directly. AgentLoop
 * keeps its own JS `nPast` mirror, which this cannot see — callers that let the model
 * generate tokens BETWEEN snapshot() and rollback() must resync the loop themselves. The
 * REPL's edit/ideogram paths snapshot at the post-generation position and apply edits in
 * JS (no intervening decode), so rollback is a consistent no-op-or-rewind there.
 */

import type { BaseModel } from '@subvocal/synapse';
import type { TagInjection } from './astTagger.js';

// ── AST snapshot ───────────────────────────────────────────────────────────────

export interface ASTSnapshot {
	source: string;
	tagMap: Map<number, string>;
	injections: TagInjection[];
}

// ── Snapshot state ─────────────────────────────────────────────────────────────

interface SnapshotEntry {
	/** Position in seq 0 where the action started (n_past before the action). */
	actionStartPos: number;
	/** AST state at snapshot time. */
	ast: ASTSnapshot;
	/** First token ID of the proposed action (for penalization on failure). */
	actionTokenId: number | null;
}

// ── Implementation ─────────────────────────────────────────────────────────────

export class StateSnapshotter {
	private model: BaseModel;
	private stack: SnapshotEntry[] = [];

	constructor(model: BaseModel) {
		this.model = model;
	}

	/**
	 * Record a rewind point before the model acts. O(1) — no tensor copy, no seq fork.
	 *
	 * @param currentNPast  n_past position BEFORE the model starts generating the action.
	 * @param ast           Current AST state.
	 * @param actionTokenId First token of the action (for penalization). Null if unknown.
	 */
	snapshot(currentNPast: number, ast: ASTSnapshot, actionTokenId: number | null): void {
		this.stack.push({ actionStartPos: currentNPast, ast, actionTokenId });
	}

	/**
	 * Rollback to the last snapshot: remove every seq-0 token from the snapshot position to
	 * the end of the cache and reset n_past there. Returns the saved AST state (or null if
	 * the stack was empty).
	 *
	 * @param _tokenCount  Vestigial (2026-07 audit). The old design removed exactly
	 *                     [start, start+tokenCount); callers always passed 0, which removed
	 *                     NOTHING and silently left post-snapshot tokens in the KV. Rewinding
	 *                     to the end is what "restore to the snapshot" always meant, so the
	 *                     parameter is ignored — the whole tail is removed unconditionally.
	 */
	rollback(_tokenCount = 0): ASTSnapshot | null {
		const entry = this.stack.pop();
		if (!entry) return null;

		this.model.kvCacheSeqRemove(0, entry.actionStartPos, -1);
		this.model.resetNPast(entry.actionStartPos);

		return entry.ast;
	}

	/**
	 * Commit the last snapshot (action succeeded). Pops the stack without rewinding —
	 * a plain O(1) pop now that there is no backup sequence to release.
	 */
	commit(): void {
		this.stack.pop();
	}

	/**
	 * Apply a specific token penalty. Call after rollback with the action's
	 * first token ID to discourage the model from repeating the failed action.
	 */
	penalizeToken(tokenId: number, bias = -100.0): void {
		this.model.applyLogitBias([{ tokenId, bias }]);
	}

	get depth(): number {
		return this.stack.length;
	}

	clear(): void {
		this.stack.length = 0;
	}
}
