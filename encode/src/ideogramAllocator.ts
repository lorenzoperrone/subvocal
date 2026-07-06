/**
 * ideogramAllocator.ts — Epic M15's one-registry rule, made executable.
 *
 * TagRegistry.gemma4.json holds 12583 verified mono-token chars (byte-identical across the
 * E2B and 12B checkpoints — see EPIC-M4). Multiple M15 features allocate from it; a token
 * must NEVER mean two things in one session, so the registry is PARTITIONED by range:
 *
 *   [0, 2000)          — astTagger node tags (sequential per file; pre-M15 behavior, untouched)
 *   [2000, 10000)      — M15.2 content-addressed block anchors (murmur3-derived, collision-bumped)
 *   [10000, len-9)      — M15.4 per-session path tokens (sequential per session)
 *   [len-9, len)        — M15.6 static markers: 6 fixed Intent classes + 1 unchanged-file
 *                         breadcrumb marker + 2 bash pass/fail markers (see STATIC_TOKEN_RANGE
 *                         below); same char every session, on purpose — unlike the ranges
 *                         above, these carry a FIXED meaning the model must learn from one
 *                         system-prompt sentence, the same "prompt-taught opaque token" bet
 *                         that already works for node tags/path tokens (no LoRA needed, per
 *                         M4's showdown + M15's hybrid validation) — but here nothing is ever
 *                         echoed back or parsed, so a model that ignores/misreads the marker
 *                         just degrades to the no-signal baseline, never corrupts an edit.
 *                         NOTE: don't hand-pick "meaningful-looking" symbols (e.g. ⊤/⊥) for new
 *                         static markers without checking their registry INDEX first — ⊤/⊥
 *                         sit at indices 77/78, inside NODE_TAG_RANGE, so using them directly
 *                         would let a per-file AST tag and a static marker mean two different
 *                         things with the same char in one prompt. Always pull static markers
 *                         from STATIC_TOKEN_RANGE, whatever glyph ends up there.
 *
 * Content-addressing note (M15.2): `crcAnchorIndex` maps a block hash into the anchor range
 * deterministically, so unchanged content re-renders with the SAME token across turns and
 * sessions (KV/trie stability). Collisions and duplicate blocks bump forward within the
 * range — those anchors are only session-stable, which is acceptable (the session map is
 * state like tagMap; cold-store keys on text anyway).
 */

import fs from 'node:fs';
import { activeProfile } from './modelProfile.js';
import type { Intent } from './intentRouter.js';

export interface RegistryEntry {
	char: string;
	tokenId: number;
}

// Same per-profile loading as astTagger — token IDs differ between model families.
const REGISTRY: RegistryEntry[] = JSON.parse(fs.readFileSync(activeProfile.tagRegistryPath, 'utf-8'));

export const NODE_TAG_RANGE = { start: 0, end: 2000 } as const;
export const CRC_ANCHOR_RANGE = { start: 2000, end: 10000 } as const;
/** M15.6: 9 static markers reserved at the tail — 6 Intent classes + 1 breadcrumb marker + 2
 *  bash pass/fail markers. */
export const STATIC_TOKEN_RANGE = { start: REGISTRY.length - 9, end: REGISTRY.length } as const;
export const PATH_TOKEN_RANGE = { start: 10000, end: STATIC_TOKEN_RANGE.start } as const;

// M15.6 item 1: one static registry char per Intent class (fixed order = fixed slot, same
// char every session — this is a LABEL, not a per-file/per-session reference, so it must
// never move). `[∴ <char>]` replaces `[Intent: ADD_FEATURE]`'s 3-4 subword tokens with 1.
const INTENT_ORDER: readonly Intent[] = ['BUGFIX', 'REFACTOR', 'EXPLAIN', 'ADD_FEATURE', 'WRITE_TEST', 'UNKNOWN'];
const INTENT_CHAR: ReadonlyMap<Intent, string> = new Map(
	INTENT_ORDER.map((intent, i) => [intent, REGISTRY[STATIC_TOKEN_RANGE.start + i].char]),
);

/** The single registry char standing for `intent` in the compressed `[∴ <char>]` prompt line. */
export function intentChar(intent: Intent): string {
	return INTENT_CHAR.get(intent) ?? INTENT_CHAR.get('UNKNOWN')!;
}

/**
 * The `[∴ X]` char is opaque by construction — unlike a node tag or path token, nothing in the
 * SAME prompt ever pairs it with its meaning inline (there's no "X (BUGFIX)" moment), so a
 * system prompt that only says "X marks the category" without listing which char is which
 * leaves the model unable to resolve it. Callers MUST render this legend once, literally,
 * rather than describing the convention in the abstract.
 */
export function intentLegend(): string {
	return INTENT_ORDER.map((intent) => `${intentChar(intent)}=${intent}`).join(' ');
}

/** M15.6 item 3: bash result pass/fail markers — a fixed first line (`[<char>]`) ahead of the
 *  compacted output, so pass/fail is legible without re-reading extracted lines for exit-status
 *  clues. Deliberately just pass/fail: the harness's ToolResultMessage only carries `isError`
 *  (no exit code, no distinct timeout signal reaches this layer) — inventing a `<n>` count or a
 *  timeout marker here would be fabricating data the code doesn't actually have. */
export const BASH_PASS_CHAR = REGISTRY[STATIC_TOKEN_RANGE.start + 6].char;
export const BASH_FAIL_CHAR = REGISTRY[STATIC_TOKEN_RANGE.start + 7].char;

/** M15.6 item 4: marks a `read` result whose content is UNCHANGED and already in the model's
 *  context (M13.1's breadcrumb), paired with the path's own M15.4 token. */
export const UNCHANGED_FILE_CHAR = REGISTRY[STATIC_TOKEN_RANGE.end - 1].char;

export function registryEntry(index: number): RegistryEntry {
	return REGISTRY[index];
}

/** Every character the registry can inject — node tags, CRC anchors AND path tokens. */
const REGISTRY_CHARS: Set<string> = new Set(REGISTRY.map((e) => e.char));

/**
 * Strip registry ideogram characters (and a single following space, the injection separator)
 * from model-echoed text. Unlike the old fixed Greek+Math range, this covers the FULL registry
 * — M15.2 block anchors and M15.4 path tokens come from wider unicode blocks (CJK etc.) that a
 * range guess misses. Over-stripping is safe: detag CANDIDATES are always tried after the raw
 * text, so a corrupted candidate simply fails to match and the caller falls through.
 */
export function stripRegistryTags(s: string, dropFollowingSpace: boolean): string {
	let out = '';
	const chars = [...s];
	for (let i = 0; i < chars.length; i++) {
		if (REGISTRY_CHARS.has(chars[i])) {
			if (dropFollowingSpace && chars[i + 1] === ' ') i++; // consume the separator too
			continue;
		}
		out += chars[i];
	}
	return out;
}

/**
 * Deterministic base index for a content hash, inside the CRC anchor range.
 * Callers resolve collisions by bumping (see nextFreeAnchor).
 */
export function crcAnchorIndex(hash: number): number {
	const size = CRC_ANCHOR_RANGE.end - CRC_ANCHOR_RANGE.start;
	// murmur3 output is a signed-ish 32-bit int in JS land — normalize.
	return CRC_ANCHOR_RANGE.start + (Math.abs(hash | 0) % size);
}

/** Linear-probe from the hash's base index to the first entry not in `used`. */
export function nextFreeAnchor(hash: number, used: ReadonlySet<number>): { entry: RegistryEntry; index: number } {
	const size = CRC_ANCHOR_RANGE.end - CRC_ANCHOR_RANGE.start;
	let idx = crcAnchorIndex(hash);
	for (let i = 0; i < size; i++) {
		if (!used.has(idx)) return { entry: REGISTRY[idx], index: idx };
		idx++;
		if (idx >= CRC_ANCHOR_RANGE.end) idx = CRC_ANCHOR_RANGE.start;
	}
	// 8000 anchors exhausted in one render — practically unreachable; fall back to base.
	const base = crcAnchorIndex(hash);
	return { entry: REGISTRY[base], index: base };
}

/**
 * M15.4: per-session path registry. Paths allocate lazily on first touch and stay stable for
 * the session (replay reproduces the same allocation because touch ORDER reproduces).
 * Rendering convention: `char (path)` on first mention, bare `char` afterwards — the legend
 * lives inline, no prompt-layout hook needed.
 */
export class PathRegistry {
	private byPath = new Map<string, RegistryEntry>();
	private byChar = new Map<string, string>();
	private byTokenId = new Map<number, string>();
	private next = PATH_TOKEN_RANGE.start;
	private introduced = new Set<string>();

	/** Allocate (or fetch) the token for a path. Returns null when the range is exhausted. */
	tokenFor(path: string): RegistryEntry | null {
		const existing = this.byPath.get(path);
		if (existing) return existing;
		if (this.next >= PATH_TOKEN_RANGE.end) return null;
		const entry = REGISTRY[this.next++];
		this.byPath.set(path, entry);
		this.byChar.set(entry.char, path);
		this.byTokenId.set(entry.tokenId, path);
		return entry;
	}

	/** Render a path reference: `char (path)` the first time, bare `char` afterwards. */
	render(path: string): string {
		const entry = this.tokenFor(path);
		if (!entry) return path;
		if (this.introduced.has(path)) return entry.char;
		this.introduced.add(path);
		return `${entry.char} (${path})`;
	}

	/**
	 * Resolve a model-emitted string that may be (or start with) a path token back to the
	 * real path. MUST run BEFORE any detagging — detag would strip the token. Returns the
	 * input unchanged when no token matches.
	 */
	resolve(value: string): string {
		const trimmed = value.trim();
		if (trimmed.length === 0) return value;
		const direct = this.byChar.get(trimmed);
		if (direct) return direct;
		// Tolerate `char (path)` echoes and `char` with stray whitespace.
		const first = [...trimmed][0];
		const mapped = this.byChar.get(first);
		if (mapped) return mapped;
		return value;
	}

	get size(): number {
		return this.byPath.size;
	}
}
