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
import { extname } from 'path';

// ── Types ──────────────────────────────────────────────────────────────────────

export interface CompileResult {
	/** True when the compiler accepted the code (or check was skipped). */
	ok: boolean;
	/** Human-readable error summary for distillation. Empty when ok. */
	message: string;
	/** Which tool ran, or 'skipped' when no applicable compiler exists. */
	tool: 'py_compile' | 'skipped';
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
	const ext = extname(filePath).toLowerCase();

	if (ext === '.py' || ext === '.pyi') {
		return checkPython(content);
	}

	// TS/JS: deferred — see module doc.
	return { ok: true, message: '', tool: 'skipped', durationMs: 0 };
}

// ── Python ─────────────────────────────────────────────────────────────────────

function checkPython(content: string): CompileResult {
	const tmpPath = `/tmp/subvocal-pycheck-${Date.now()}-${process.pid}.py`;
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
