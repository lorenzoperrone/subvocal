/**
 * lineCRCCache.ts
 *
 * Substory 1.3: In-RAM CRC Cache & Dirty-State Checker.
 *
 * Maintains a per-file snapshot of line-level MurmurHash3 hashes so the
 * harness can detect which lines changed since the last GPU prefill and
 * issue targeted KV cache invalidations instead of rebuilding from scratch.
 */

import {
	buildLineHashSnapshot,
	computeLineDelta,
	type LineHashSnapshot,
} from './lineLevelCRC.js';

/**
 * Per-file cache entry.
 */
interface FileCacheEntry {
	/** The snapshot at the time of last GPU prefill. */
	snapshot: LineHashSnapshot;
	/** The file content at that time (used to map line ranges to token offsets later). */
	content: string;
}

/**
 * In-RAM line-level CRC cache.
 *
 * Maps file path -> { snapshot, content } so that on a file change event we
 * can compute the delta of dirty lines in O(n) (n = number of lines) and
 * return the ranges to invalidate in the KV cache.
 */
export class LineCRCCache {
	/** filePath -> cache entry */
	private cache = new Map<string, FileCacheEntry>();

	/**
	 * Store or update the cached snapshot for a file.
	 *
	 * Call this every time you prefill the GPU with a file's context so the
	 * next change event can cheaply compute the delta.
	 */
	set(filePath: string, content: string): void {
		this.cache.set(filePath, {
			snapshot: buildLineHashSnapshot(content),
			content,
		});
	}

	/**
	 * Check whether a file has changed since the last GPU prefill.
	 *
	 * @returns the changed line ranges (0-based indices) and a file-wide
	 *          "structure changed" boolean, or `null` if there is no cached baseline.
	 */
	checkDirty(filePath: string, currentContent: string): {
		changedRanges: Array<{ start: number; end: number }>;
		isStructureChanged: boolean;
	} | null {
		const entry = this.cache.get(filePath);
		if (!entry) return null;

		const currentSnap = buildLineHashSnapshot(currentContent);
		return computeLineDelta(entry.snapshot, currentSnap);
	}

	/**
	 * Convenience: check dirty state and update the cache in one call.
	 * Returns the delta result, or `null` if this is the first time we see the file.
	 */
	checkAndUpdate(filePath: string, currentContent: string): {
		changedRanges: Array<{ start: number; end: number }>;
		isStructureChanged: boolean;
	} | null {
		const delta = this.checkDirty(filePath, currentContent);
		this.set(filePath, currentContent);
		return delta;
	}

	/**
	 * Remove a file from the cache (e.g. on file deletion).
	 */
	evict(filePath: string): void {
		this.cache.delete(filePath);
	}

	/**
	 * Number of tracked files.
	 */
	get size(): number {
		return this.cache.size;
	}

	/**
	 * Clear all cached snapshots.
	 */
	clear(): void {
		this.cache.clear();
	}
}