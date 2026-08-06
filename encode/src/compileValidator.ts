/**
 * compileValidator.ts
 *
 * Substory 4.3 — Compiler Validation Layer.
 *
 * Runs the authoritative language compiler on model-generated code *before*
 * committing it to the VFS. Tree-sitter (syntaxValidator.ts) catches structural
 * syntax errors; this module adds the language runtime's own checks:
 *   - Python (.py): `python -m py_compile` — CPython authoritative parse,
 *     catches SyntaxError, top-level `return`, duplicate keyword args, etc.
 *     Does NOT check imports (intentional — import resolution needs the real
 *     filesystem/venv, not a tmpfile).
 *   - TypeScript/JS (.ts .tsx .js): deferred to Substory 2.3 (Shadow VFS).
 *     Full `tsc --noEmit` needs the VFS flushed to disk with a tsconfig;
 *     isolated single-file tsc floods phantom "Cannot find module" errors.
 *     Tree-sitter (already run by Utter) covers TS syntax.
 *   - All other extensions: skipped (pass-through).
 */

import { spawnSync } from 'child_process';
import { writeFileSync, unlinkSync } from 'fs';
import { extname, join } from 'path';
import { tmpdir } from 'os';
import ts from 'typescript';

// ── Types ──────────────────────────────────────────────────────────────────────

export interface CompileResult {
	/** True when the compiler accepted the code (or check was skipped). */
	ok: boolean;
	/** Human-readable error summary for distillation. Empty when ok. */
	message: string;
	/** Which tool ran, or 'skipped' when no applicable compiler exists. */
	tool: 'py_compile' | 'tsc' | 'skipped';
	durationMs: number;
}

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Run the language compiler on `content` and return the result.
 *
 * @param filePath  Original file path (used only to detect language via extension).
 * @param content   Source code string to validate (written to tmpfile if needed).
 */
export function compileCheck(filePath: string, content: string): CompileResult {
	if (!filePath) return { ok: true, message: '', tool: 'skipped', durationMs: 0 };
	const safeContent = typeof content === 'string' ? content : String(content ?? '');
	const ext = extname(filePath).toLowerCase();

	if (ext === '.py' || ext === '.pyi') {
		return checkPython(safeContent);
	}

	if (ext === '.ts' || ext === '.tsx' || ext === '.js' || ext === '.jsx') {
		return checkTypeScript(filePath, safeContent);
	}

	return { ok: true, message: '', tool: 'skipped', durationMs: 0 };
}

// ── TypeScript ─────────────────────────────────────────────────────────────────

function checkTypeScript(filePath: string, content: string): CompileResult {
	const t0 = performance.now();
	const scriptTarget = ts.ScriptTarget.ESNext;
	const isTsx = filePath.endsWith('.tsx') || filePath.endsWith('.jsx');
	const scriptKind = isTsx ? ts.ScriptKind.TSX : ts.ScriptKind.TS;

	const sourceFile = ts.createSourceFile(filePath, content, scriptTarget, true, scriptKind);
	const parseDiagnostics = (sourceFile as any).parseDiagnostics as ts.Diagnostic[] | undefined;

	const durationMs = performance.now() - t0;

	if (parseDiagnostics && parseDiagnostics.length > 0) {
		const firstErr = parseDiagnostics[0];
		const message = formatTsDiagnostic(sourceFile, firstErr);
		return { ok: false, message, tool: 'tsc', durationMs };
	}

	return { ok: true, message: '', tool: 'tsc', durationMs };
}

function formatTsDiagnostic(sourceFile: ts.SourceFile, diagnostic: ts.Diagnostic): string {
	const text = ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n');
	if (diagnostic.start !== undefined) {
		const { line } = sourceFile.getLineAndCharacterOfPosition(diagnostic.start);
		return `TS${diagnostic.code}: ${text} (line ${line + 1})`;
	}
	return `TS${diagnostic.code}: ${text}`;
}

// ── Python ─────────────────────────────────────────────────────────────────────

function checkPython(content: string): CompileResult {
	const tmpPath = join(tmpdir(), `subvocal-pycheck-${Date.now()}-${process.pid}.py`);
	const t0 = performance.now();
	try {
		writeFileSync(tmpPath, content, 'utf-8');

		// Prefer python3 (the interpreter macOS actually ships; bare `python` is often absent),
		// fall back to `python`. 2026-07 audit: a missing interpreter sets result.error (ENOENT)
		// and result.status === null — the old `status === 0` check treated that as a COMPILE
		// FAILURE, so on any machine without the named binary EVERY Python edit was rolled back
		// and retried forever. A missing/unrunnable compiler must SKIP the check (tree-sitter
		// already covered structural syntax), never reject the code.
		let result = spawnSync('python3', ['-m', 'py_compile', tmpPath], { encoding: 'utf-8', timeout: 10_000 });
		if (result.error) {
			result = spawnSync('python', ['-m', 'py_compile', tmpPath], { encoding: 'utf-8', timeout: 10_000 });
		}

		const durationMs = performance.now() - t0;

		// Interpreter not found, killed by the timeout, or otherwise never produced a verdict.
		if (result.error || result.status === null) {
			return { ok: true, message: '', tool: 'skipped', durationMs };
		}

		if (result.status === 0) {
			return { ok: true, message: '', tool: 'py_compile', durationMs };
		}

		const raw = (result.stderr ?? '') + (result.stdout ?? '');
		const message = extractPyError(raw);
		return { ok: false, message, tool: 'py_compile', durationMs };
	} finally {
		try { unlinkSync(tmpPath); } catch { /* tmpfile may not exist on write failure */ }
	}
}

// Error patterns emitted by `python -m py_compile` stderr:
//   File "/tmp/...", line 5        ← line reference
//     def f(:                      ← bad line
//            ^                     ← caret
// SyntaxError: invalid syntax      ← or "return outside function", etc.

const PY_SYNTAX_RE = /SyntaxError:\s*(.+)/;
const PY_LINE_RE = /line\s+(\d+)/;

function extractPyError(stderr: string): string {
	const synErr = PY_SYNTAX_RE.exec(stderr);
	const lineNum = PY_LINE_RE.exec(stderr);
	const type = synErr ? `SyntaxError: ${synErr[1].trim()}` : 'py_compile error';
	const line = lineNum ? ` (line ${lineNum[1]})` : '';
	return `${type}${line}`;
}
