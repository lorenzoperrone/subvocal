/**
 * lspShim.ts
 *
 * Epico 4.1.4 — Direct LSP Telemetry (lightweight shim).
 *
 * Instead of connecting to a full LSP server via IPC (heavy, needs external deps,
 * suffers from 200-300ms debouncing), or shelling out to `tsc --noEmit` and regex-parsing
 * its terminal-formatted stdout, this shim drives the TypeScript compiler API directly
 * in-process via a cached `ts.LanguageService` — diagnostics come back as structured
 * `ts.Diagnostic[]` objects, no subprocess, no text round-trip.
 *
 * A `ts.LanguageService` per project directory is kept alive across calls; each call only
 * needs to bump the changed files' versions, so repeated calls re-check incrementally
 * instead of re-parsing the whole project.
 */

import * as ts from 'typescript';
import { readFileSync, statSync } from 'fs';
import { join } from 'path';
import { type SyntaxError } from './syntaxValidator.js';

export type { SyntaxError };

// ── Per-project LanguageService cache ───────────────────────────────────────────

interface ProjectService {
	service: ts.LanguageService;
	fileVersions: Map<string, string>;
	rootFiles: Set<string>;
	compilerOptions: ts.CompilerOptions;
}

const projectServices = new Map<string, ProjectService>();

function loadCompilerOptions(projectDir: string): { options: ts.CompilerOptions; rootFiles: string[] } {
	const configPath = join(projectDir, 'tsconfig.json');
	const configFile = ts.readConfigFile(configPath, (p) => readFileSync(p, 'utf-8'));
	const parsed = ts.parseJsonConfigFileContent(configFile.config, ts.sys, projectDir);
	return { options: parsed.options, rootFiles: parsed.fileNames };
}

function getOrCreateService(projectDir: string): ProjectService {
	const existing = projectServices.get(projectDir);
	if (existing) return existing;

	const { options, rootFiles } = loadCompilerOptions(projectDir);
	const fileVersions = new Map<string, string>();
	const rootFilesSet = new Set(rootFiles);

	const host: ts.LanguageServiceHost = {
		getScriptFileNames: () => Array.from(rootFilesSet),
		getScriptVersion: (fileName) => fileVersions.get(fileName) ?? '0',
		getScriptSnapshot: (fileName) => {
			if (!ts.sys.fileExists(fileName)) return undefined;
			return ts.ScriptSnapshot.fromString(readFileSync(fileName, 'utf-8'));
		},
		getCurrentDirectory: () => projectDir,
		getCompilationSettings: () => options,
		getDefaultLibFileName: (opts) => ts.getDefaultLibFilePath(opts),
		fileExists: ts.sys.fileExists,
		readFile: ts.sys.readFile,
		readDirectory: ts.sys.readDirectory,
		directoryExists: ts.sys.directoryExists,
		getDirectories: ts.sys.getDirectories,
	};

	const service = ts.createLanguageService(host, ts.createDocumentRegistry());
	const entry: ProjectService = { service, fileVersions, rootFiles: rootFilesSet, compilerOptions: options };
	projectServices.set(projectDir, entry);
	return entry;
}

/** Bump a file's version (mtime-based) so the LanguageService re-checks it on next call. */
function touchFile(entry: ProjectService, fileName: string): void {
	try {
		const mtime = statSync(fileName).mtimeMs.toString();
		entry.fileVersions.set(fileName, mtime);
	} catch {
		// File gone — leave version as-is; getScriptSnapshot will return undefined for it.
	}
	entry.rootFiles.add(fileName);
}

/**
 * Run TypeScript diagnostics on a project directory using a cached, in-process
 * `ts.LanguageService` — no subprocess, no stdout parsing.
 *
 * @param projectDir  Directory containing tsconfig.json.
 * @returns Array of structured syntax/type errors.
 */
export function typescriptDiagnostics(projectDir: string): SyntaxError[] {
	const entry = getOrCreateService(projectDir);

	// Re-check every known root file's version (cheap mtime stat) so edits since the last
	// call are picked up incrementally rather than requiring a full project re-parse.
	for (const fileName of entry.rootFiles) {
		touchFile(entry, fileName);
	}

	const errors: SyntaxError[] = [];
	for (const fileName of entry.rootFiles) {
		const diagnostics = [
			...entry.service.getSyntacticDiagnostics(fileName),
			...entry.service.getSemanticDiagnostics(fileName),
		];
		for (const diagnostic of diagnostics) {
			errors.push(diagnosticToSyntaxError(diagnostic));
		}
	}

	return errors;
}

function diagnosticToSyntaxError(diagnostic: ts.Diagnostic): SyntaxError {
	let line = 0;
	let column = 0;
	if (diagnostic.file && diagnostic.start !== undefined) {
		const pos = ts.getLineAndCharacterOfPosition(diagnostic.file, diagnostic.start);
		line = pos.line + 1;
		column = pos.character + 1;
	}
	// Matches the old regex-based behavior: every diagnostic was reported as kind 'error',
	// regardless of whether tsc printed it as "error" or "warning".
	return {
		line,
		column,
		kind: 'error',
		nodeType: `TS${diagnostic.code}`,
	};
}

// ── File-level quick check ─────────────────────────────────────────────────────

/**
 * Quick syntax-only check on a single file using tree-sitter.
 * For type checking, use typescriptDiagnostics() instead.
 */
// For quick file-level syntax checking, import { validateSyntax } from './syntaxValidator.js' directly.
