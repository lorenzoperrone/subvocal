/**
 * compileValidator.test.ts
 *
 * Substory 4.3 — Unit tests for compileCheck().
 *
 * All tests are pure (no GPU, no models). python must be in PATH.
 */

import { describe, it, expect } from 'vitest';
import { compileCheck } from '../src/compileValidator.js';

// ── Python ─────────────────────────────────────────────────────────────────────

describe('compileCheck — Python (.py)', () => {
	it('accepts valid Python', () => {
		const result = compileCheck('/tmp/test.py', 'x = 1\n');
		expect(result.ok).toBe(true);
		expect(result.tool).toBe('py_compile');
		expect(result.message).toBe('');
		expect(result.durationMs).toBeGreaterThan(0);
	});

	it('accepts multi-line valid Python', () => {
		const code = [
			'def greet(name: str) -> str:',
			'    return f"hello {name}"',
			'',
			'result = greet("world")',
			'',
		].join('\n');
		const result = compileCheck('/tmp/test.py', code);
		expect(result.ok).toBe(true);
	});

	it('rejects SyntaxError (bad parameter list)', () => {
		// def f(: is a syntax error in CPython
		const result = compileCheck('/tmp/test.py', 'def f(:\n    pass\n');
		expect(result.ok).toBe(false);
		expect(result.tool).toBe('py_compile');
		expect(result.message).toMatch(/SyntaxError/);
	});

	it('rejects top-level return statement', () => {
		// top-level `return` is a SyntaxError in CPython (not valid outside a function)
		const result = compileCheck('/tmp/test.py', 'return 42\n');
		expect(result.ok).toBe(false);
		expect(result.message).toMatch(/SyntaxError/);
	});

	it('reports line number in error message', () => {
		const code = '# line 1\n# line 2\ndef f(\n    pass\n';
		const result = compileCheck('/tmp/test.py', code);
		expect(result.ok).toBe(false);
		expect(result.message).toMatch(/line \d+/);
	});

	it('accepts .pyi stub files', () => {
		const result = compileCheck('/tmp/stubs.pyi', 'def foo(x: int) -> str: ...\n');
		expect(result.ok).toBe(true);
		expect(result.tool).toBe('py_compile');
	});

	// NOTE: `python -m py_compile` does NOT check imports.
	// `import non_existent_module` compiles to bytecode without error.
	// This is intentional — import resolution needs the real venv, not a tmpfile.
});

// ── TypeScript (.ts / .tsx / .js / .jsx) ───────────────────────────────────────

describe('compileCheck — TypeScript / JavaScript', () => {
	it('accepts valid TypeScript', () => {
		const code = 'export interface User { id: number; name: string; }\nexport const u: User = { id: 1, name: "Alice" };\n';
		const result = compileCheck('/tmp/test.ts', code);
		expect(result.ok).toBe(true);
		expect(result.tool).toBe('tsc');
		expect(result.message).toBe('');
		expect(result.durationMs).toBeGreaterThanOrEqual(0);
	});

	it('rejects invalid TypeScript syntax', () => {
		const code = 'export function foo(x:) {\n    return x;\n}\n';
		const result = compileCheck('/tmp/test.ts', code);
		expect(result.ok).toBe(false);
		expect(result.tool).toBe('tsc');
		expect(result.message).toMatch(/TS\d+:/);
		expect(result.message).toMatch(/line \d+/);
	});

	it('accepts valid TSX component', () => {
		const code = 'export const App = () => <div>Hello World</div>;\n';
		const result = compileCheck('/tmp/test.tsx', code);
		expect(result.ok).toBe(true);
		expect(result.tool).toBe('tsc');
	});

	it('rejects invalid TSX syntax', () => {
		const code = 'export const App = () => <div unclosed;\n';
		const result = compileCheck('/tmp/test.tsx', code);
		expect(result.ok).toBe(false);
		expect(result.tool).toBe('tsc');
		expect(result.message).toMatch(/TS\d+:/);
	});

	it('accepts valid JavaScript', () => {
		const result = compileCheck('/tmp/test.js', 'const x = { a: 1, b: 2 };\n');
		expect(result.ok).toBe(true);
		expect(result.tool).toBe('tsc');
	});
});

// ── Unknown extensions ─────────────────────────────────────────────────────────

describe('compileCheck — unknown extension', () => {
	it('skips unknown extensions with ok=true', () => {
		const result = compileCheck('/tmp/test.rb', 'puts "hello"\n');
		expect(result.ok).toBe(true);
		expect(result.tool).toBe('skipped');
	});

	it('skips files with no extension', () => {
		const result = compileCheck('/tmp/Makefile', 'all:\n\techo hi\n');
		expect(result.ok).toBe(true);
		expect(result.tool).toBe('skipped');
	});
});
