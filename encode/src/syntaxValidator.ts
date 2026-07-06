/**
 * syntaxValidator.ts
 *
 * Early-rejection of model-generated code. Before a `write` or `edit` tool
 * commits content to disk, the resulting source is parsed with tree-sitter and
 * rejected if the grammar reports syntax errors (ERROR or MISSING nodes).
 *
 * This closes the loop the agent otherwise pays for at runtime: instead of the
 * large brain emitting broken code that only fails when later executed, the
 * harness hands back a precise diagnostic so the model can self-correct on the
 * next turn.
 *
 * Only languages with a compiled grammar are validated (TS/TSX/JS/JSX/Python).
 * Anything else passes through untouched — we never block a write we cannot
 * confidently judge.
 */

import { createRequire } from "module";

const require = createRequire(import.meta.url);

// biome-ignore lint/suspicious/noExplicitAny: native addon untyped
const Parser = require("tree-sitter") as any;
// biome-ignore lint/suspicious/noExplicitAny: native addon untyped
const TSMod = require("tree-sitter-typescript") as any;
const TSTypeScript = TSMod.typescript;
const TSTsx = TSMod.tsx;
// Pass the whole module to setLanguage(), NOT `.language` -- see the comment in
// astTagger.ts for why: `.language` ships frozen and without its own
// nodeTypeInfo, which crashes tree.rootNode later inside tree-sitter's own
// unmarshalNode. The whole module has both as sibling properties and isn't frozen.
// biome-ignore lint/suspicious/noExplicitAny: native addon untyped
const TSPython = require("tree-sitter-python") as any;

export type ValidatableLanguage = "typescript" | "tsx" | "python";

// biome-ignore lint/suspicious/noExplicitAny: tree-sitter Language objects are untyped
const RAW_GRAMMARS: Record<ValidatableLanguage, any> = {
	typescript: TSTypeScript,
	tsx: TSTsx,
	python: TSPython,
};

/**
 * tree-sitter grammars are versioned by a language ABI that must match the core
 * `tree-sitter` runtime. Installed versions can drift (e.g. tree-sitter@0.21 vs
 * tree-sitter-python@0.25), in which case `setLanguage` throws. We probe each
 * grammar once and cache whether it is usable; an unusable grammar degrades to
 * pass-through validation rather than crashing a file write.
 */
const grammarUsable = new Map<ValidatableLanguage, boolean>();

function isGrammarUsable(language: ValidatableLanguage): boolean {
	const cached = grammarUsable.get(language);
	if (cached !== undefined) return cached;
	let usable = false;
	try {
		const parser = new Parser();
		parser.setLanguage(RAW_GRAMMARS[language]);
		parser.parse("");
		usable = true;
	} catch {
		usable = false;
	}
	grammarUsable.set(language, usable);
	return usable;
}

/**
 * Map a file path to a grammar. Returns null when the extension has no compiled
 * grammar, signalling that validation should be skipped for that file.
 */
export function languageForPath(filePath: string): ValidatableLanguage | null {
	const lower = filePath.toLowerCase();
	const dot = lower.lastIndexOf(".");
	if (dot === -1) return null;
	const ext = lower.slice(dot);
	switch (ext) {
		case ".py":
		case ".pyi":
			return "python";
		case ".tsx":
		case ".jsx":
			return "tsx";
		case ".ts":
		case ".mts":
		case ".cts":
		case ".js":
		case ".mjs":
		case ".cjs":
			return "typescript";
		default:
			return null;
	}
}

export interface SyntaxError {
	/** 1-based line number. */
	line: number;
	/** 1-based column number. */
	column: number;
	/** "error" = unexpected/unparseable text, "missing" = a required token the grammar inserted. */
	kind: "error" | "missing";
	/** The node type the grammar expected or choked on. */
	nodeType: string;
}

export interface SyntaxValidationResult {
	ok: boolean;
	language: ValidatableLanguage | null;
	errors: SyntaxError[];
	/** True when validation was skipped (unknown extension or unusable grammar). */
	skipped?: boolean;
}

// biome-ignore lint/suspicious/noExplicitAny: tree-sitter SyntaxNode is untyped
type SyntaxNode = any;

/**
 * Parse `code` with the grammar for `language` and collect syntax errors.
 * Stops collecting after `maxErrors` to keep diagnostics readable.
 */
export function validateSyntax(
	code: string,
	language: ValidatableLanguage,
	maxErrors = 10,
): SyntaxValidationResult {
	if (!isGrammarUsable(language)) {
		return { ok: true, language, errors: [], skipped: true };
	}
	const parser = new Parser();
	parser.setLanguage(RAW_GRAMMARS[language]);
	const tree = parser.parse(code);

	if (!tree.rootNode.hasError) {
		return { ok: true, language, errors: [] };
	}

	const errors: SyntaxError[] = [];
	const stack: SyntaxNode[] = [tree.rootNode];
	while (stack.length > 0 && errors.length < maxErrors) {
		const node = stack.pop();
		// `isMissing` flags a token the grammar had to synthesize (e.g. a missing
		// `)` ). `type === "ERROR"` flags text the grammar could not parse at all.
		if (node.isMissing || node.type === "ERROR") {
			errors.push({
				line: node.startPosition.row + 1,
				column: node.startPosition.column + 1,
				kind: node.isMissing ? "missing" : "error",
				nodeType: node.type,
			});
			// An ERROR node's children are noise; don't descend into it.
			continue;
		}
		// Only descend into subtrees that actually contain an error.
		if (node.hasError) {
			for (let i = node.childCount - 1; i >= 0; i--) {
				stack.push(node.child(i));
			}
		}
	}

	// hasError was true but no ERROR/MISSING node surfaced (rare grammar edge
	// case): report a generic failure rather than silently passing.
	if (errors.length === 0) {
		errors.push({ line: 1, column: 1, kind: "error", nodeType: "unknown" });
	}

	return { ok: false, language, errors };
}

/**
 * Convenience wrapper used by tools: validate by file path, skipping files
 * whose extension has no grammar. Files we cannot judge always pass (`ok: true`).
 */
export function validateSyntaxForPath(filePath: string, code: string): SyntaxValidationResult {
	const language = languageForPath(filePath);
	if (language === null) {
		return { ok: true, language: null, errors: [] };
	}
	return validateSyntax(code, language);
}

/** Render errors as a compact, model-readable diagnostic string. */
export function formatSyntaxErrors(result: SyntaxValidationResult): string {
	if (result.ok) return "";
	const lines = result.errors.map((e) => {
		const what = e.kind === "missing" ? `missing ${e.nodeType}` : `syntax error near ${e.nodeType}`;
		return `  line ${e.line}:${e.column} — ${what}`;
	});
	return `Syntax check failed (${result.language}):\n${lines.join("\n")}`;
}
