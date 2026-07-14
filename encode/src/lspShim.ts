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

import type * as tsType from 'typescript';
import { readFileSync, statSync } from 'fs';
import { join } from 'path';
import { createRequire } from 'module';
import { type SyntaxError } from './syntaxValidator.js';

const require = createRequire(import.meta.url);
let ts: typeof tsType | undefined;
function getTs(): typeof tsType {
	if (!ts) ts = require('typescript');
	return ts!;
}

export type { SyntaxError };

// ── Per-project LanguageService cache ───────────────────────────────────────────

interface ProjectService {
	service: tsType.LanguageService;
	fileVersions: Map<string, string>;
	rootFiles: Set<string>;
	compilerOptions: tsType.CompilerOptions;
}

const projectServices = new Map<string, ProjectService>();

function loadCompilerOptions(projectDir: string): { options: tsType.CompilerOptions; rootFiles: string[] } {
	const tsObj = getTs();
	const configPath = join(projectDir, 'tsconfig.json');
	const configFile = tsObj.readConfigFile(configPath, (p) => readFileSync(p, 'utf-8'));
	const parsed = tsObj.parseJsonConfigFileContent(configFile.config, tsObj.sys, projectDir);
	return { options: parsed.options, rootFiles: parsed.fileNames };
}

function getOrCreateService(projectDir: string): ProjectService {
	const existing = projectServices.get(projectDir);
	if (existing) return existing;

	const tsObj = getTs();
	const { options, rootFiles } = loadCompilerOptions(projectDir);
	const fileVersions = new Map<string, string>();
	const rootFilesSet = new Set(rootFiles);

	const host: tsType.LanguageServiceHost = {
		getScriptFileNames: () => Array.from(rootFilesSet),
		getScriptVersion: (fileName) => fileVersions.get(fileName) ?? '0',
		getScriptSnapshot: (fileName) => {
			if (!tsObj.sys.fileExists(fileName)) return undefined;
			return tsObj.ScriptSnapshot.fromString(readFileSync(fileName, 'utf-8'));
		},
		getCurrentDirectory: () => projectDir,
		getCompilationSettings: () => options,
		getDefaultLibFileName: (opts) => tsObj.getDefaultLibFilePath(opts),
		fileExists: tsObj.sys.fileExists,
		readFile: tsObj.sys.readFile,
		readDirectory: tsObj.sys.readDirectory,
		directoryExists: tsObj.sys.directoryExists,
		getDirectories: tsObj.sys.getDirectories,
	};

	const service = tsObj.createLanguageService(host, tsObj.createDocumentRegistry());
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

function diagnosticToSyntaxError(diagnostic: tsType.Diagnostic): SyntaxError {
	const tsObj = getTs();
	let line = 0;
	let column = 0;
	if (diagnostic.file && diagnostic.start !== undefined) {
		const pos = tsObj.getLineAndCharacterOfPosition(diagnostic.file, diagnostic.start);
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
