/**
 * dependencyGraph.ts
 *
 * Epico 4.1.2 — In-Memory Dependency Graph for Subvocal.
 *
 * When the large model modifies a function/class signature, we proactively find
 * all files that import/export that symbol and inject [AFFECTED_TARGET] markers
 * into the model's context so it knows to update dependent files too.
 */

import { createRequire } from 'module';
import * as path from 'path';
import type { SupportedLanguage } from './astTagger.js';

const require = createRequire(import.meta.url);
const Parser = require('tree-sitter') as any;
const TSTypeScript = require('tree-sitter-typescript').tsx as any;
const TSPython = require('tree-sitter-python') as any;

// ── Types ─────────────────────────────────────────────────────────────────────

export interface DepNode {
	path: string;
	imports: Array<{ name: string; source: string }>;
	exports: string[];
}

export interface DependencyGraph {
	nodes: Map<string, DepNode>;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Build a dependency graph from a list of file contents.
 *
 * Parses each file with tree-sitter to extract import and export declarations.
 *
 * @param files     Array of { path, content } for each source file.
 * @param language  'typescript' | 'python'
 * @returns A DependencyGraph keyed by normalized (resolved) file paths.
 */
export function buildDependencyGraph(
	files: Array<{ path: string; content: string }>,
	language: 'typescript' | 'python',
): DependencyGraph {
	const nodes = new Map<string, DepNode>();
	const parser = new Parser();
	parser.setLanguage(language === 'python' ? TSPython : TSTypeScript);

	for (const file of files) {
		const normalizedPath = path.resolve(file.path);
		const tree = parser.parse(file.content);
		const root = tree.rootNode;

		const imports: Array<{ name: string; source: string }> = [];
		const exports: string[] = [];

		if (language === 'python') {
			extractPythonImports(root, imports);
			extractPythonExports(root, exports);
		} else {
			extractTypeScriptImports(root, imports);
			extractTypeScriptExports(root, exports);
		}

		nodes.set(normalizedPath, { path: normalizedPath, imports, exports });
	}

	return { nodes };
}

/**
 * Find all files that import a changed symbol from the given file.
 *
 * Resolves relative import paths against each importing file's directory,
 * strips known extensions (.ts, .tsx, .js, .jsx, .py, .mjs, .cjs), and
 * compares against the target file path (also extension-stripped).
 *
 * @param graph          The dependency graph.
 * @param filePath       Path of the file where the symbol was changed.
 * @param changedSymbol  The exported symbol name that changed.
 * @returns Deduplicated array of file paths that import the symbol from filePath.
 */
export function findAffectedFiles(
	graph: DependencyGraph,
	filePath: string,
	changedSymbol: string,
): string[] {
	const normalizedPath = path.resolve(filePath);
	const noExt = stripExtension(normalizedPath);
	const affected: string[] = [];

	for (const [nodePath, node] of graph.nodes) {
		if (nodePath === normalizedPath) continue;
		for (const imp of node.imports) {
			if (imp.name !== changedSymbol) continue;
			const resolved = resolveImportPath(nodePath, imp.source);
			if (resolved === noExt) {
				affected.push(nodePath);
				break;
			}
		}
	}

	return affected;
}

// ── TypeScript: import extraction ─────────────────────────────────────────────

function extractTypeScriptImports(
	root: any,
	out: Array<{ name: string; source: string }>,
): void {
	for (const node of collectNodesOfType(root, 'import_statement')) {
		const source = extractImportSource(node);
		if (!source) continue;

		const namedImports = findChild(node, 'named_imports');
		if (namedImports) {
			for (const spec of namedImports.namedChildren) {
				if (spec.type !== 'import_specifier') continue;
				const id = findChild(spec, 'identifier');
				if (id) out.push({ name: id.text, source });
			}
			continue;
		}

		const nsImport = findChild(node, 'namespace_import');
		if (nsImport) {
			const id = findChild(nsImport, 'identifier');
			if (id) out.push({ name: id.text, source });
			continue;
		}

		const importClause = findChild(node, 'import_clause');
		if (importClause) {
			const id = findChild(importClause, 'identifier');
			if (id) out.push({ name: id.text, source });
		}
	}
}

// ── TypeScript: export extraction ─────────────────────────────────────────────

function extractTypeScriptExports(root: any, out: string[]): void {
	for (const node of collectNodesOfType(root, 'export_statement')) {
		const exportClause = findChild(node, 'export_clause');
		if (exportClause) {
			for (const spec of exportClause.namedChildren) {
				if (spec.type !== 'export_specifier') continue;
				const id = findChild(spec, 'identifier');
				if (id) out.push(id.text);
			}
			continue;
		}

		for (const child of node.namedChildren) {
			if (child.type === 'function_declaration' || child.type === 'class_declaration') {
				const id = findChild(child, 'identifier');
				if (id) out.push(id.text);
			} else if (child.type === 'lexical_declaration' || child.type === 'variable_declaration') {
				for (const decl of collectNodesOfType(child, 'variable_declarator')) {
					const id = findChild(decl, 'identifier');
					if (id) out.push(id.text);
				}
			} else if (child.type === 'identifier') {
				out.push(child.text);
			}
		}
	}
}

// ── Python: import extraction ─────────────────────────────────────────────────

function extractPythonImports(
	root: any,
	out: Array<{ name: string; source: string }>,
): void {
	for (const node of collectNodesOfType(root, 'import_statement')) {
		for (const child of node.namedChildren) {
			if (child.type === 'dotted_name') {
				out.push({ name: child.text, source: child.text });
			}
		}
	}

	for (const node of collectNodesOfType(root, 'import_from_statement')) {
		// The first dotted_name after 'from' is the module source.
		const moduleName = findChild(node, 'dotted_name');
		if (!moduleName) continue;
		const source = moduleName.text;

		let seenModule = false;
		for (const child of node.namedChildren) {
			if (child.type === 'dotted_name') {
				if (!seenModule) { seenModule = true; continue; }
				out.push({ name: child.text, source });
			} else if (child.type === 'aliased_import') {
				const nameNode = findChild(child, 'name');
				if (nameNode) out.push({ name: nameNode.text, source });
			}
		}
	}
}

// ── Python: export extraction ─────────────────────────────────────────────────

function extractPythonExports(root: any, out: string[]): void {
	for (const node of root.namedChildren) {
		if (node.type === 'function_definition' || node.type === 'class_definition') {
			const name = findChild(node, 'identifier');
			if (name) out.push(name.text);
		}
	}
}

// ── Tree helpers ──────────────────────────────────────────────────────────────

function collectNodesOfType(root: any, type: string): any[] {
	const result: any[] = [];
	const stack: any[] = [root];
	while (stack.length > 0) {
		const node = stack.pop();
		if (node.type === type) result.push(node);
		for (let i = 0; i < node.childCount; i++) {
			stack.push(node.child(i));
		}
	}
	return result;
}

function findChild(node: any, type: string): any | null {
	for (let i = 0; i < node.childCount; i++) {
		const child = node.child(i);
		if (child.type === type) return child;
		const found = findChild(child, type);
		if (found) return found;
	}
	return null;
}

function extractImportSource(node: any): string | null {
	for (let i = 0; i < node.childCount; i++) {
		const child = node.child(i);
		if (child.type === 'string') {
			const text: string = child.text;
			if (text.length >= 2) return text.slice(1, -1);
			return text;
		}
	}
	return null;
}

// ── Path utilities ────────────────────────────────────────────────────────────

const STRIPPABLE_EXTS = new Set(['.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs', '.py']);

function stripExtension(filePath: string): string {
	const ext = path.extname(filePath);
	if (STRIPPABLE_EXTS.has(ext)) return filePath.slice(0, -ext.length);
	return filePath;
}

function resolveImportPath(importingFile: string, source: string): string {
	if (source.startsWith('.')) {
		const dir = path.dirname(importingFile);
		return stripExtension(path.resolve(dir, source));
	}
	return source;
}
