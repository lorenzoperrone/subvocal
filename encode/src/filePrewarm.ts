/**
 * filePrewarm.ts
 *
 * Mechanism A from doc/research/predictive-prefill-while-typing.md: AST tagging and
 * multi-file-context resolution depend only on the file being edited, not on the user's
 * prompt text. Splitting them out of preprocess() into a standalone, cacheable step lets a
 * caller run this work as soon as a file is opened/focused -- before the user has typed a
 * request -- so preprocess() can reuse the result instead of recomputing it at submit time.
 *
 * Wired to the REPL's file-open events since M11.1: utter.ts's openFile() (the `:file`
 * command and the `--file` flag) calls FilePrewarmCache.warm(); preprocess() consumes the
 * cache via PreprocessInput.filePrewarmCache. An editor integration would call the same API.
 */

import { injectASTTags, detectLanguage, type SupportedLanguage, type TagInjection } from './astTagger.js';
import { resolveMultiFileContext, type MultiFileBlock } from './multiFileContext.js';

export interface FilePrewarmResult {
	filePath: string;
	/** Exact content this result was computed from -- used to detect staleness on lookup. */
	fileContent: string;
	lang: SupportedLanguage;
	taggedCode: string;
	tagMap: Map<number, string>;
	tagCount: number;
	injections: TagInjection[];
	multiFileBlocks?: MultiFileBlock[];
	computedAtMs: number;
}

export interface PrewarmOptions {
	/** Mirror of PreprocessInput.multiFile -- whether to resolve dependency files too. */
	multiFile?: boolean;
}

/** Run the file-dependent (prompt-independent) phases of preprocess() standalone. */
export function prewarmFile(
	filePath: string,
	fileContent: string,
	opts: PrewarmOptions = {},
): FilePrewarmResult {
	const lang = detectLanguage(filePath);
	const { taggedCode, tagMap, tagCount, injections } = injectASTTags(fileContent, lang);

	let multiFileBlocks: MultiFileBlock[] | undefined;
	if (opts.multiFile) {
		multiFileBlocks = resolveMultiFileContext(filePath, process.cwd());
	}

	return { filePath, fileContent, lang, taggedCode, tagMap, tagCount, injections, multiFileBlocks, computedAtMs: Date.now() };
}

/**
 * Per-session cache of prewarm results, keyed by filePath. A cached entry is valid only
 * while `fileContent` matches exactly -- any edit invalidates it for free, no CRC/hash
 * machinery needed since preprocess() always has the current fileContent to compare against.
 * 
 * Implements a simple LRU policy capped at 50 entries to prevent unbounded RAM growth
 * across long coding sessions.
 */
export class FilePrewarmCache {
	private entries = new Map<string, FilePrewarmResult>();
	private readonly maxSize = 50;

	/** Returns the cached result only if still valid for the given fileContent + multiFile need. */
	get(filePath: string, fileContent: string, needMultiFile = false): FilePrewarmResult | undefined {
		const entry = this.entries.get(filePath);
		if (!entry || entry.fileContent !== fileContent) return undefined;
		if (needMultiFile && entry.multiFileBlocks === undefined) return undefined;
		
		// LRU bump
		this.entries.delete(filePath);
		this.entries.set(filePath, entry);
		return entry;
	}

	set(result: FilePrewarmResult): void {
		this.entries.delete(result.filePath);
		this.entries.set(result.filePath, result);
		if (this.entries.size > this.maxSize) {
			const oldestKey = this.entries.keys().next().value;
			if (oldestKey !== undefined) {
				this.entries.delete(oldestKey);
			}
		}
	}

	/** Prewarm now and store the result -- what a future "file opened" caller would invoke. */
	warm(filePath: string, fileContent: string, opts: PrewarmOptions = {}): FilePrewarmResult {
		const result = prewarmFile(filePath, fileContent, opts);
		this.set(result);
		return result;
	}

	delete(filePath: string): void {
		this.entries.delete(filePath);
	}

	clear(): void {
		this.entries.clear();
	}
}
