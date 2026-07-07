/**
 * astEditor.ts
 *
 * Epico 4.1 Fase B — AST-aware code editing.
 *
 * Replaces code at a specific AST node (identified by a tag from the preprocessing
 * tagMap) with new code, using tree-sitter byte-range slicing. No search/replace
 * regex — the node is located by type+name via the parsed AST.
 *
 * Flow:
 *   1. Resolve target → node type + optional name (via tagMap)
 *   2. Re-parse source with tree-sitter
 *   3. Walk AST to find matching node
 *   4. String splice: replace byte range with newCode
 *   5. Re-parse result and validate syntax
 */

import { createRequire } from 'module';
import { injectASTTags, detectLanguage, type TagInjection } from './astTagger.js';
import { validateSyntax, type SyntaxError } from './syntaxValidator.js';

const require = createRequire(import.meta.url);
const Parser = require('tree-sitter') as any;
// biome-ignore lint/suspicious/noExplicitAny: native addon untyped
const TSTypeScript = require('tree-sitter-typescript').tsx as any;
// biome-ignore lint/suspicious/noExplicitAny: native addon untyped
const TSPython = require('tree-sitter-python') as any;

export type { SyntaxError };

// ── Edit input ─────────────────────────────────────────────────────────────────

export interface ASTEditInput {
	/** Token ID from the tagMap (the ideogram emitted by the model). */
	tokenId: number;
	/** New source code to replace the node's body with. */
	newCode: string;
	/** Original (untagged) source file content. */
	source: string;
	/** Programming language. */
	language: 'typescript' | 'python';
	/** Tag injection list from the same preprocess() call that produced this tagMap. */
	injections: TagInjection[];
}

export interface ASTEditResult {
	/** Modified source code. */
	newSource: string;
	/** True if the target node was found and replaced. */
	found: boolean;
	/** What was replaced. */
	replaced: { startIndex: number; endIndex: number; nodeType: string; nodeLabel: string } | null;
	/** Syntax errors in the result (empty = clean). */
	errors: SyntaxError[];
}

// ── Implementation ─────────────────────────────────────────────────────────────

export function editASTNode(input: ASTEditInput): ASTEditResult {
	// 1. Find the injection entry for this tokenId
	const injection = input.injections.find((inj) => inj.tokenId === input.tokenId);
	if (!injection) {
		return { newSource: input.source, found: false, replaced: null, errors: [] };
	}

	// 2. Parse the original source
	const parser = new Parser();
	parser.setLanguage(input.language === 'python' ? TSPython : TSTypeScript);
	const tree = parser.parse(input.source);

	// 3. Locate the node matching the injection label + startIndex
	const node = findNode(tree.rootNode, injection);
	if (!node) {
		return { newSource: input.source, found: false, replaced: null, errors: [] };
	}

	const startIndex: number = node.startIndex;
	const endIndex: number = node.endIndex;

	// 4. Byte-range string replacement
	const newSource =
		input.source.slice(0, startIndex) + input.newCode + input.source.slice(endIndex);

	// 5. Validate
	const result = validateSyntax(newSource, input.language);
	const errors = result.errors;

	return {
		newSource,
		found: true,
		replaced: { startIndex, endIndex, nodeType: node.type as string, nodeLabel: injection.label },
		errors,
	};
}

/**
 * Insert new code after the target AST node (instead of replacing it).
 * Uses `\n\n${newCode}` inserted immediately after the node's endIndex.
 * All node finding and validation is the same as editASTNode.
 */
export function insertAfterNode(input: ASTEditInput): ASTEditResult {
	const injection = input.injections.find((inj) => inj.tokenId === input.tokenId);
	if (!injection) {
		return { newSource: input.source, found: false, replaced: null, errors: [] };
	}

	const parser = new Parser();
	parser.setLanguage(input.language === 'python' ? TSPython : TSTypeScript);
	const tree = parser.parse(input.source);

	const node = findNode(tree.rootNode, injection);
	if (!node) {
		return { newSource: input.source, found: false, replaced: null, errors: [] };
	}

	const endIndex: number = node.endIndex;

	const newSource =
		input.source.slice(0, endIndex) + '\n\n' + input.newCode + input.source.slice(endIndex);

	const result = validateSyntax(newSource, input.language);
	const errors = result.errors;

	return {
		newSource,
		found: true,
		replaced: { startIndex: endIndex, endIndex, nodeType: node.type as string, nodeLabel: injection.label },
		errors,
	};
}

// ── M15.5: rename / delete ops ──────────────────────────────────────────────────

export interface ASTRenameInput {
	/** Token ID from the tagMap (the ideogram emitted by the model). */
	tokenId: number;
	/** The new identifier name (the rename op's payload — a short name, not a code body). */
	newName: string;
	/** Original (untagged) source file content. */
	source: string;
	language: 'typescript' | 'python';
	injections: TagInjection[];
}

/**
 * Rename the identifier that names the target node (e.g. a function/class declaration).
 *
 * LIMITATION (documented, not hidden): this is a whole-file, whole-word substitution of the
 * OLD name for occurrences that match the node's own declared identifier — not a scope-resolved
 * rename. It renames every standalone occurrence of the name in the file, including one that
 * happens to belong to an unrelated shadowed variable in a different scope. This is the same
 * limitation the story called out for a Python fallback (no lspShim wiring here), generalized
 * to both languages: a real scope resolver is a separate, larger piece of work. Good enough for
 * the common case this op targets — renaming a top-level function/class and its call sites in
 * the same file — and `validateSyntax` below still catches anything that breaks parsing.
 */
export function renameASTNode(input: ASTRenameInput): ASTEditResult {
	const injection = input.injections.find((inj) => inj.tokenId === input.tokenId);
	if (!injection) {
		return { newSource: input.source, found: false, replaced: null, errors: [] };
	}

	const parser = new Parser();
	parser.setLanguage(input.language === 'python' ? TSPython : TSTypeScript);
	const tree = parser.parse(input.source);

	const node = findNode(tree.rootNode, injection);
	if (!node || !node.firstNamedChild) {
		return { newSource: input.source, found: false, replaced: null, errors: [] };
	}

	const oldName = extractText(node.firstNamedChild);
	if (!oldName || !ID_RE.test(oldName)) {
		// Not a simple identifier (e.g. the first named child wasn't the name after all) —
		// refuse rather than guess at a risky whole-file substitution.
		return { newSource: input.source, found: false, replaced: null, errors: [] };
	}

	const newSource = oldName === input.newName
		? input.source // no-op rename
		: input.source.replace(new RegExp(`\\b${escapeRegExp(oldName)}\\b`, 'g'), input.newName);

	const result = validateSyntax(newSource, input.language);
	return {
		newSource,
		found: true,
		replaced: { startIndex: node.startIndex, endIndex: node.endIndex, nodeType: node.type as string, nodeLabel: injection.label },
		errors: result.errors,
	};
}

export interface ASTDeleteInput {
	/** Token ID from the tagMap (the ideogram emitted by the model). */
	tokenId: number;
	/** Original (untagged) source file content. */
	source: string;
	language: 'typescript' | 'python';
	injections: TagInjection[];
}

/** Remove the target node entirely, absorbing its own leading indentation and one trailing
 *  newline so deleting a top-level declaration doesn't leave a dangling blank line. */
export function deleteASTNode(input: ASTDeleteInput): ASTEditResult {
	const injection = input.injections.find((inj) => inj.tokenId === input.tokenId);
	if (!injection) {
		return { newSource: input.source, found: false, replaced: null, errors: [] };
	}

	const parser = new Parser();
	parser.setLanguage(input.language === 'python' ? TSPython : TSTypeScript);
	const tree = parser.parse(input.source);

	let node = findNode(tree.rootNode, injection);
	if (!node) {
		return { newSource: input.source, found: false, replaced: null, errors: [] };
	}
	// A `function_declaration`/`class_declaration` node does NOT include a wrapping `export` —
	// that's a separate parent node in the grammar. Deleting only the inner node leaves the
	// keyword dangling as a syntactically invalid fragment (caught by a live unit test before
	// this shipped). Absorb the wrapper's own range instead when present.
	if (node.parent && EXPORT_WRAPPER_TYPES.has(node.parent.type)) {
		node = node.parent;
	}

	let startIndex: number = node.startIndex;
	let endIndex: number = node.endIndex;
	// Absorb the node's own leading indentation (spaces/tabs back to the previous newline)...
	while (startIndex > 0 && (input.source[startIndex - 1] === ' ' || input.source[startIndex - 1] === '\t')) {
		startIndex--;
	}
	// ...and ALL trailing newlines (the node's own line ending PLUS any blank separator lines
	// after it), so the gap this leaves matches the gap that was BEFORE the node (one seam,
	// not two) instead of stacking into a double-blank line. Consuming just one newline here
	// (the first version of this fix) left exactly that double-blank artifact — caught by a
	// live unit test before shipping.
	while (input.source[endIndex] === '\n') endIndex++;

	const newSource = input.source.slice(0, startIndex) + input.source.slice(endIndex);

	const result = validateSyntax(newSource, input.language);
	return {
		newSource,
		found: true,
		replaced: { startIndex, endIndex, nodeType: node.type as string, nodeLabel: injection.label },
		errors: result.errors,
	};
}

const ID_RE = /^[A-Za-z_$][A-Za-z0-9_$]*$/;

function escapeRegExp(s: string): string {
	return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// ── Node lookup ────────────────────────────────────────────────────────────────

interface TreeNode {
	readonly type: string;
	readonly startIndex: number;
	readonly endIndex: number;
	readonly childCount: number;
	readonly firstNamedChild: TreeNode | null;
	readonly hasError: boolean;
	/** Not on the minimal interface above but present on the real tree-sitter node (same
	 *  pattern as `extractText`'s `.text` access) — used by `deleteASTNode` to detect and
	 *  absorb an `export`/`export default` wrapper around the target node. */
	readonly parent: TreeNode | null;
	child(i: number): TreeNode;
}

/** Node types that WRAP a declaration (export_statement, export_default_declaration) whose
 *  own range must be deleted too — otherwise deleting the inner node leaves a dangling,
 *  syntactically invalid `export` keyword behind. TypeScript/JS-specific; Python has no
 *  equivalent wrapper (its `export` doesn't exist as a keyword), so this is a no-op there. */
const EXPORT_WRAPPER_TYPES = new Set(['export_statement', 'export_default_declaration']);

function findNode(root: TreeNode, injection: TagInjection): TreeNode | null {
	const [nodeType, nodeName] = parseLabel(injection.label);

	// Priority is EXACT-position → EXACT-name → nearest-position (2026-07 audit). The old code
	// returned findNodeAtPosition()'s result unconditionally, and that helper returned the
	// CLOSEST node of the type with NO distance bound — so a stale injection.startIndex (an
	// earlier in-turn edit shifted every later offset, or astTagger and this parse disagree by a
	// byte) silently resolved to whatever same-type node happened to sit nearest, and the exact
	// name match below was never even tried. Result: an edit applied to the wrong function, no
	// error. Now the exact-position match only wins when it's actually exact; otherwise the name
	// match (unambiguous when the model named its target) is preferred over a fuzzy position guess.
	const exact = findNodeAtPosition(root, injection.startIndex, nodeType, /* exactOnly */ true);
	if (exact) return exact;

	// Strategy B: type + name match — reliable even when byte offsets have drifted.
	if (nodeName) {
		const byName = findNodeByName(root, nodeType, nodeName);
		if (byName) return byName;
	}

	// Strategy C (last resort): the nearest same-type node by position. Only reached when there
	// is no exact-position hit AND no name (or no name match) — a genuine best-effort guess.
	return findNodeAtPosition(root, injection.startIndex, nodeType, /* exactOnly */ false);
}

/**
 * Walk the AST for a node of `nodeType` at `targetOffset`.
 *   exactOnly=true  → return only a node whose startIndex matches within 1 byte (BOM/whitespace
 *                     tolerance), else null.
 *   exactOnly=false → return the nearest same-type node by startIndex (best-effort last resort).
 */
function findNodeAtPosition(
	root: TreeNode,
	targetOffset: number,
	nodeType: string,
	exactOnly: boolean,
): TreeNode | null {
	let best: TreeNode | null = null;
	const stack: TreeNode[] = [root];
	while (stack.length > 0) {
		const node = stack.pop()!;
		if (node.type === nodeType) {
			if (node.startIndex === targetOffset || Math.abs(node.startIndex - targetOffset) <= 1) {
				return node; // exact match (within tolerance)
			}
			if (!best || Math.abs(node.startIndex - targetOffset) < Math.abs(best.startIndex - targetOffset)) {
				best = node;
			}
		}
		for (let i = 0; i < node.childCount; i++) {
			stack.push(node.child(i));
		}
	}
	return exactOnly ? null : best;
}

function findNodeByName(root: TreeNode, nodeType: string, name: string): TreeNode | null {
	const stack: TreeNode[] = [root];
	while (stack.length > 0) {
		const node = stack.pop()!;
		if (node.type === nodeType && node.firstNamedChild) {
			const child = node.firstNamedChild;
			const childText = extractText(child);
			if (childText === name) return node;
		}
		for (let i = 0; i < node.childCount; i++) {
			stack.push(node.child(i));
		}
	}
	return null;
}

function extractText(node: TreeNode): string {
	// tree-sitter nodes don't expose .text on the TreeNode interface we defined,
	// but the actual TS objects do. We access it via a type assertion.
	return (node as any).text as string;
}

/** Parse "function_declaration:calculateTotal" → ["function_declaration", "calculateTotal"] */
function parseLabel(label: string): [string, string | null] {
	const idx = label.indexOf(':');
	if (idx === -1) return [label, null];
	return [label.slice(0, idx), label.slice(idx + 1)];
}
