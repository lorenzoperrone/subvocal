/**
 * kvCacheManager.ts
 *
 * Substory 1.3: Integration layer between LineCRCCache and FFI KV cache.
 *
 * Bridges "which lines changed" (CRC) → "which token ranges to invalidate in
 * the KV cache".  Relies on the caller providing line-to-token boundary
 * information at setFile() time; if not available, invalidates the full file
 * rather than guessing token positions.
 *
 * CAUSAL-ATTENTION CONTRACT (2026-07 KV audit): the returned range always runs
 * from the FIRST changed token to the END of the file's token span, never a
 * mid-file hole. Every token AFTER an edit has KV computed while attending to
 * the pre-edit tokens, so "remove just the changed lines' tokens and re-prefill
 * those" — what an earlier version of this file's usage example suggested — is
 * unsound: the downstream tokens' KV would still encode the old content (and
 * mid-sequence position holes on top). Invalidate-to-end is the cheapest sound
 * granularity short of full-file.
 *
 * STATUS: not wired into any production path today. The TUI briefly constructed
 * one per turn (inert — a fresh instance has no baseline to diff against;
 * removed in the same audit). To wire it for real: persist ONE instance per
 * session, call setFile() after each successful prefill of a file block, and on
 * invalidation kvCacheSeqRemove + resetNPast + re-decode from `range.start`.
 *
 * Usage:
 *   const crCache = new LineCRCCache();
 *   const mgr = new KVCacheManager(crCache, (text) => model.tokenize(text));
 *
 *   // After successful GPU prefill:
 *   const tokens = model.tokenize(content);
 *   const boundaries = computeTokenBoundaries(content, tokens); // [[0,2],[3,5],...]
 *   await mgr.setFile(path, content, tokens, boundaries);
 *
 *   // On file change event:
 *   const result = await mgr.invalidateIfDirty(path, newContent);
 *   if (result?.dirty && result.tokenRanges.length > 0) {
 *     const r = result.tokenRanges[0]; // single range: first change → end of file span
 *     model.kvCacheSeqRemove(0, fileBasePos + r.start, -1);
 *     model.resetNPast(fileBasePos + r.start);
 *     // re-decode from r.start with the new content, then call setFile again
 *   }
 */

import { LineCRCCache } from './lineCRCCache.js';

/**
 * A range in token-space.  llama.cpp uses inclusive [p0, p1] semantics.
 * Pass to model.kvCacheSeqRemove(0, range.start, range.end).
 */
export interface TokenRange {
	start: number;
	end: number;
}

/**
 * Optional per-token boundary information.  Each entry maps a 0-based token
 * index to the 0-based byte offset of that token in the source content.
 * Used to convert line-change ranges to precise token-position ranges.
 */
export type TokenBoundaries = Map<number, number>;

export interface InvalidationResult {
	dirty: boolean;
	isStructureChanged: boolean;
	/**
	 * Token-position range(s) to remove from the KV cache. Empty when dirty===false;
	 * otherwise a SINGLE range from the first changed token to the end of the file's
	 * token span (causal attention makes anything downstream of an edit stale too —
	 * see the header comment). Kept as an array for API stability.
	 */
	tokenRanges: TokenRange[];
	changedLineCount: number;
}

interface TokenisedEntry {
	tokenIds: Int32Array;
	boundaries: TokenBoundaries;
	content: string;
}

/**
 * Build a map: line index (0-based) -> { start, end } token position
 * using the token byte-boundary map.  This is O(n) in lines.
 *
 * @param boundaries  tokenIdx -> byteOffset for each token.
 * @param content     Raw file content (used to locate line boundaries).
 * @param tokenIds    Token IDs (only used for length).
 */
function buildLineToTokenMap(
	boundaries: TokenBoundaries,
	content: string,
	tokenIds: Int32Array,
): Map<number, TokenRange> {
	// Cumulative line start offsets
	const lineByteOffset: number[] = [0];
	for (let i = 0; i < content.length; i++) {
		if (content.charCodeAt(i) === 10) {
			lineByteOffset.push(i + 1);
		}
	}
	// Trim trailing empty line
	while (lineByteOffset.length > 1 && lineByteOffset[lineByteOffset.length - 1] >= content.length) {
		lineByteOffset.pop();
	}

	const map = new Map<number, TokenRange>();
	const n = tokenIds.length;

	// Walk tokens in order; for each, find which line its byte offset falls in
	for (let t = 0; t < n; t++) {
		const byteOffset = boundaries.get(t) ?? 0;
		// Binary search for the line that contains this byte offset
		let lo = 0;
		let hi = lineByteOffset.length - 1;
		while (lo < hi) {
			const mid = (lo + hi + 1) >> 1;
			if (lineByteOffset[mid] <= byteOffset) lo = mid;
			else hi = mid - 1;
		}
		const lineIdx = lo;

		const existing = map.get(lineIdx);
		if (!existing) {
			map.set(lineIdx, { start: t, end: t });
		} else {
			existing.end = t;
		}
	}

	return map;
}

/**
 * Line-level CRC cache + token-position resolver for KV cache invalidation.
 */
export class KVCacheManager {
	private crCcache: LineCRCCache;
	private tokenize: (text: string) => Promise<Int32Array> | Int32Array;
	private tokenised = new Map<string, TokenisedEntry>();

	constructor(
		crCcache: LineCRCCache,
		tokenize: (text: string) => Promise<Int32Array> | Int32Array,
	) {
		this.crCcache = crCcache;
		this.tokenize = tokenize;
	}

	/**
	 * Store the baseline snapshot for a file after a successful GPU prefill.
	 *
	 * @param filePath   Stable cache key.
	 * @param content    File content (used for CRC snapshot and line-to-token mapping).
	 * @param tokenIds   Token IDs of content.
	 * @param boundaries Optional per-token byte offsets.  If not provided, only
	 *                   full-file invalidation is available on structural changes.
	 */
	async setFile(
		filePath: string,
		content: string,
		tokenIds: Int32Array,
		boundaries?: TokenBoundaries,
	): Promise<void> {
		this.crCcache.set(filePath, content);
		this.tokenised.set(filePath, {
			tokenIds,
			boundaries: boundaries ?? new Map(),
			content,
		});
	}

	/**
	 * Check whether a file changed since the last setFile() and return
	 * the token-position ranges to invalidate.  Does NOT modify the KV cache.
	 *
	 * @param filePath       File to check.
	 * @param currentContent New content on disk.
	 * @returns              InvalidationResult or null (no baseline).
	 */
	async invalidateIfDirty(
		filePath: string,
		currentContent: string,
	): Promise<InvalidationResult | null> {
		const delta = this.crCcache.checkDirty(filePath, currentContent);
		if (!delta) return null;

		const { changedRanges, isStructureChanged } = delta;

		if (!isStructureChanged && changedRanges.length === 0) {
			return { dirty: false, isStructureChanged: false, tokenRanges: [], changedLineCount: 0 };
		}

		const entry = this.tokenised.get(filePath);
		const changedLineCount = changedRanges.reduce((n, r) => n + r.end - r.start + 1, 0);

		// Structural edit (line insert/delete) — can't map old line indices to new token
		// positions; do full-file invalidation.
		if (!entry || isStructureChanged) {
			const tokenCount = entry?.tokenIds.length ?? 0;
			return {
				dirty: true,
				isStructureChanged,
				tokenRanges: tokenCount > 0 ? [{ start: 0, end: tokenCount - 1 }] : [],
				changedLineCount,
			};
		}

		// No boundary info — fall back to full-file invalidation
		if (entry.boundaries.size === 0) {
			return {
				dirty: true,
				isStructureChanged,
				tokenRanges: [{ start: 0, end: entry.tokenIds.length - 1 }],
				changedLineCount,
			};
		}

		// Find the FIRST changed token via the byte boundaries, then invalidate from there to
		// the end of the file's token span. (2026-07 KV audit: this used to return the merged
		// per-line ranges, inviting callers to punch mid-file holes in the KV — unsound under
		// causal attention, see the header comment.)
		const lineToToken = buildLineToTokenMap(entry.boundaries, entry.content, entry.tokenIds);

		let firstChangedToken = -1;
		for (const r of changedRanges) {
			for (let line = r.start; line <= r.end; line++) {
				const t = lineToToken.get(line);
				if (t !== undefined && (firstChangedToken < 0 || t.start < firstChangedToken)) {
					firstChangedToken = t.start;
				}
			}
		}
		const tokenRanges: TokenRange[] =
			firstChangedToken >= 0
				? [{ start: firstChangedToken, end: entry.tokenIds.length - 1 }]
				: [{ start: 0, end: entry.tokenIds.length - 1 }]; // no mappable line — full file

		return { dirty: true, isStructureChanged, tokenRanges, changedLineCount };
	}

	evict(filePath: string): void {
		this.crCcache.evict(filePath);
		this.tokenised.delete(filePath);
	}

	get size(): number {
		return this.crCcache.size;
	}
}