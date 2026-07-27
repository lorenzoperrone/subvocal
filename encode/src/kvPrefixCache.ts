/**
 * kvPrefixCache.ts
 *
 * Substory 3.4 / Epic 3: KV Cache Prefix Sharing & Sequence Forking.
 *
 * Fingerprints common initial prompt prefixes (System Prompt + Codebase Context)
 * and manages sequence checkpoints via sequence forking / sequence removal.
 * When a new turn or session shares an exact prefix with an active checkpoint:
 *   - `restorePrefix()` forks the sequence or sets `nPast = prefixLength`
 *   - Prefill latency for the common prefix becomes 0ms.
 */

export interface CachedPrefix {
	hash: string;
	tokenLength: number;
	seqId: number;
	createdAt: number;
}

export function computePrefixHash(tokens: Int32Array | number[]): string {
	let h1 = 0xdeadbeef;
	let h2 = 0x41c6ce57;
	for (let i = 0; i < tokens.length; i++) {
		const k = tokens[i];
		h1 = Math.imul(h1 ^ k, 2654435761);
		h2 = Math.imul(h2 ^ k, 1597334677);
	}
	return (h1 >>> 0).toString(16) + (h2 >>> 0).toString(16);
}

export class KVPrefixCacheManager {
	private cache = new Map<string, CachedPrefix>();
	private nextSeqId = 1;

	/**
	 * Check if an exact prefix (by tokens) is cached.
	 */
	hasPrefix(tokens: Int32Array | number[]): boolean {
		const hash = computePrefixHash(tokens);
		return this.cache.has(hash);
	}

	/**
	 * Get cached prefix metadata if present.
	 */
	getPrefix(tokens: Int32Array | number[]): CachedPrefix | undefined {
		const hash = computePrefixHash(tokens);
		return this.cache.get(hash);
	}

	/**
	 * Store a sequence prefix checkpoint.
	 */
	savePrefix(tokens: Int32Array | number[], model: any, srcSeqId: number = 0): CachedPrefix {
		const hash = computePrefixHash(tokens);
		const tokenLength = tokens ? tokens.length : 0;

		// Clean up existing checkpoint for this exact prefix to avoid sequence leaks in VRAM
		const existing = this.cache.get(hash);
		if (existing && model && typeof model.kvCacheSeqRemove === 'function') {
			try { model.kvCacheSeqRemove(existing.seqId, 0, -1); } catch { /* ignore */ }
		}

		const seqId = this.nextSeqId++;

		if (model && typeof model.kvCacheFork === 'function') {
			try {
				model.kvCacheFork(srcSeqId, seqId, tokenLength);
			} catch (e) {
				console.warn('[kvPrefixCache] Warning: kvCacheFork failed:', e);
			}
		}

		const entry: CachedPrefix = {
			hash,
			tokenLength,
			seqId,
			createdAt: Date.now(),
		};

		this.cache.set(hash, entry);
		return entry;
	}

	/**
	 * Restore a cached sequence prefix into target sequence (default seq 0).
	 * Returns true if restored (0ms prefill for prefix), false if cache miss.
	 */
	restorePrefix(tokens: Int32Array | number[], model: any, targetSeqId: number = 0): boolean {
		const hash = computePrefixHash(tokens);
		const entry = this.cache.get(hash);
		if (!entry) return false;

		if (model) {
			try {
				if (typeof model.kvCacheFork === 'function') {
					model.kvCacheSeqRemove(targetSeqId, 0, -1);
					model.kvCacheFork(entry.seqId, targetSeqId, entry.tokenLength);
				}
				if (typeof model.resetNPast === 'function') {
					model.resetNPast(entry.tokenLength);
				}
			} catch (e) {
				console.warn('[kvPrefixCache] Warning: restorePrefix failed:', e);
				return false;
			}
		}

		return true;
	}

	/**
	 * Remove a cached prefix and release its sequence.
	 */
	evictPrefix(tokens: Int32Array | number[], model?: any): void {
		const hash = computePrefixHash(tokens);
		const entry = this.cache.get(hash);
		if (entry) {
			if (model && typeof model.kvCacheSeqRemove === 'function') {
				try { model.kvCacheSeqRemove(entry.seqId, 0, -1); } catch { /* ignore */ }
			}
			this.cache.delete(hash);
		}
	}

	/**
	 * Clear all cached prefixes.
	 */
	clear(model?: any): void {
		for (const entry of this.cache.values()) {
			if (model && typeof model.kvCacheSeqRemove === 'function') {
				try { model.kvCacheSeqRemove(entry.seqId, 0, -1); } catch { /* ignore */ }
			}
		}
		this.cache.clear();
	}

	get size(): number {
		return this.cache.size;
	}
}
