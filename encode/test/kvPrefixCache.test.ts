/**
 * kvPrefixCache.test.ts
 *
 * Substory 3.4 / Epic 3: Unit tests for KVPrefixCacheManager.
 * Pure unit tests (no model required).
 */

import { describe, it, expect } from 'vitest';
import { KVPrefixCacheManager, computePrefixHash } from '../src/kvPrefixCache.js';

describe('computePrefixHash', () => {
	it('generates consistent hashes for identical token arrays', () => {
		const tokens1 = new Int32Array([1, 150, 2400, 999]);
		const tokens2 = new Int32Array([1, 150, 2400, 999]);
		expect(computePrefixHash(tokens1)).toBe(computePrefixHash(tokens2));
	});

	it('generates different hashes for different token arrays', () => {
		const tokens1 = new Int32Array([1, 150, 2400, 999]);
		const tokens2 = new Int32Array([1, 150, 2400, 1000]);
		expect(computePrefixHash(tokens1)).not.toBe(computePrefixHash(tokens2));
	});
});

describe('KVPrefixCacheManager', () => {
	it('saves and checks prefix existence', () => {
		const mgr = new KVPrefixCacheManager();
		const tokens = new Int32Array([10, 20, 30, 40]);

		expect(mgr.hasPrefix(tokens)).toBe(false);

		const mockModel = {
			kvCacheFork: (src: number, dest: number, len: number) => {},
		};

		const entry = mgr.savePrefix(tokens, mockModel);
		expect(entry.tokenLength).toBe(4);
		expect(entry.seqId).toBe(1);
		expect(mgr.hasPrefix(tokens)).toBe(true);
		expect(mgr.size).toBe(1);
	});

	it('restores prefix with mock model', () => {
		const mgr = new KVPrefixCacheManager();
		const tokens = new Int32Array([100, 200, 300]);

		let forkedSrc = -1;
		let forkedDest = -1;
		let removedSeq = -1;
		let resetLen = -1;

		const mockModel = {
			kvCacheFork: (src: number, dest: number, len: number) => {
				forkedSrc = src;
				forkedDest = dest;
			},
			kvCacheSeqRemove: (seq: number, p0: number, p1: number) => {
				removedSeq = seq;
			},
			resetNPast: (n: number) => {
				resetLen = n;
			},
		};

		mgr.savePrefix(tokens, mockModel);
		const restored = mgr.restorePrefix(tokens, mockModel, 0);

		expect(restored).toBe(true);
		expect(removedSeq).toBe(0);
		expect(forkedSrc).toBe(1);
		expect(forkedDest).toBe(0);
		expect(resetLen).toBe(3);
	});

	it('evicts prefix correctly', () => {
		const mgr = new KVPrefixCacheManager();
		const tokens = new Int32Array([5, 15, 25]);
		mgr.savePrefix(tokens, null);

		expect(mgr.size).toBe(1);
		mgr.evictPrefix(tokens, null);
		expect(mgr.size).toBe(0);
		expect(mgr.hasPrefix(tokens)).toBe(false);
	});
});
