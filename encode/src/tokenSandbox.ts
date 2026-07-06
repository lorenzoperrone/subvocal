/**
 * tokenSandbox.ts
 *
 * Epico 4.1.5 — Hardwired Token Sandbox.
 *
 * Scans project automation files (package.json, Makefile) at boot time
 * to build a whitelist of allowed shell commands. When the large model
 * requests a shell command, the sandbox verifies it against the whitelist
 * before execution.
 */

import { readFileSync } from 'fs';
import { resolve } from 'path';

// ── Types ──────────────────────────────────────────────────────────────────────

export interface SandboxRule {
	command: string;
	match: 'exact' | 'prefix';
	source: string;
	/** Internal flag to distinguish deny rules from allow rules. */
	_deny?: true;
}

export interface SandboxResult {
	allowed: boolean;
	matchedRule: SandboxRule | null;
	reason: string;
}

interface SubvocalSandboxConfig {
	allow: string[];
	deny: string[];
}

// ── Always-allowed commands (read-only, safe) ──────────────────────────────────

export const SAFE_COMMANDS: SandboxRule[] = [
	{ command: 'ls', match: 'prefix', source: 'builtin' },
	{ command: 'cat', match: 'prefix', source: 'builtin' },
	{ command: 'head', match: 'prefix', source: 'builtin' },
	{ command: 'tail', match: 'prefix', source: 'builtin' },
	{ command: 'grep', match: 'prefix', source: 'builtin' },
	{ command: 'wc', match: 'prefix', source: 'builtin' },
	{ command: 'find', match: 'prefix', source: 'builtin' },
	{ command: 'git status', match: 'exact', source: 'builtin' },
	{ command: 'git diff', match: 'prefix', source: 'builtin' },
	{ command: 'git log', match: 'prefix', source: 'builtin' },
	{ command: 'git branch', match: 'prefix', source: 'builtin' },
	{ command: 'echo', match: 'prefix', source: 'builtin' },
	{ command: 'node --version', match: 'exact', source: 'builtin' },
	{ command: 'npm --version', match: 'exact', source: 'builtin' },
	{ command: 'tsc --version', match: 'exact', source: 'builtin' },
	{ command: 'python --version', match: 'exact', source: 'builtin' },
];

// ── Scanner helpers ────────────────────────────────────────────────────────────

function parsePackageJson(filePath: string): SandboxRule[] {
	const rules: SandboxRule[] = [];
	try {
		const raw = readFileSync(filePath, 'utf-8');
		const pkg = JSON.parse(raw);
		if (pkg.scripts && typeof pkg.scripts === 'object') {
			for (const key of Object.keys(pkg.scripts)) {
				rules.push({ command: `npm run ${key}`, match: 'exact', source: 'package.json' });
			}
		}
	} catch {
		// package.json missing or malformed — skip
	}
	return rules;
}

const MAKEFILE_TARGET_RE = /^([a-zA-Z_-]+):/;

function parseMakefile(filePath: string): SandboxRule[] {
	const rules: SandboxRule[] = [];
	try {
		const raw = readFileSync(filePath, 'utf-8');
		for (const line of raw.split('\n')) {
			const m = line.match(MAKEFILE_TARGET_RE);
			if (m) {
				rules.push({ command: `make ${m[1]}`, match: 'exact', source: 'Makefile' });
			}
		}
	} catch {
		// Makefile missing — skip
	}
	return rules;
}

function parseSubvocalSandbox(filePath: string): { allow: SandboxRule[]; deny: SandboxRule[] } {
	const allow: SandboxRule[] = [];
	const deny: SandboxRule[] = [];
	try {
		const raw = readFileSync(filePath, 'utf-8');
		const config: SubvocalSandboxConfig = JSON.parse(raw);
		if (Array.isArray(config.allow)) {
			for (const cmd of config.allow) {
				allow.push({ command: cmd, match: 'exact', source: '.subvocal-sandbox.json' });
			}
		}
		if (Array.isArray(config.deny)) {
			for (const cmd of config.deny) {
				deny.push({ command: cmd, match: 'exact', source: '.subvocal-sandbox.json', _deny: true });
			}
		}
	} catch {
		// .subvocal-sandbox.json missing or malformed — skip
	}
	return { allow, deny };
}

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Scan a project directory for automation files and build a sandbox whitelist.
 *
 * Reads:
 *   - package.json → extracts "scripts" entries
 *   - Makefile      → extracts target names
 *   - .subvocal-sandbox.json → explicit allow/deny rules
 */
export function buildSandbox(projectDir: string): SandboxRule[] {
	const rules: SandboxRule[] = [];

	rules.push(...parsePackageJson(resolve(projectDir, 'package.json')));
	rules.push(...parseMakefile(resolve(projectDir, 'Makefile')));

	const { allow, deny } = parseSubvocalSandbox(resolve(projectDir, '.subvocal-sandbox.json'));
	rules.push(...deny);
	rules.push(...allow);

	rules.push(...SAFE_COMMANDS);

	return rules;
}

// ── Validator ──────────────────────────────────────────────────────────────────

function ruleMatches(command: string, rule: SandboxRule): boolean {
	const trimmed = command.trim();
	if (rule.match === 'exact') {
		return rule.command === trimmed;
	}
	return trimmed === rule.command || trimmed.startsWith(rule.command + ' ');
}

/**
 * Check if a shell command is allowed by the sandbox.
 *
 * Deny rules (from .subvocal-sandbox.json) take precedence — checked first.
 * Then custom allow rules and built-in SAFE_COMMANDS are checked in order.
 */
export function checkCommand(command: string, rules: SandboxRule[]): SandboxResult {
	const trimmed = command.trim();

	if (/[&|;<>$()\\]/.test(trimmed)) {
		return {
			allowed: false,
			matchedRule: null,
			reason: `Command contains forbidden shell metacharacters: ${trimmed}`,
		};
	}

	for (const rule of rules) {
		if (rule._deny && ruleMatches(command, rule)) {
			return {
				allowed: false,
				matchedRule: rule,
				reason: `Command denied by .subvocal-sandbox.json: ${trimmed}`,
			};
		}
	}

	for (const rule of rules) {
		if (!rule._deny && ruleMatches(command, rule)) {
			return { allowed: true, matchedRule: rule, reason: '' };
		}
	}

	return {
		allowed: false,
		matchedRule: null,
		reason: `Command not in sandbox whitelist: ${trimmed}`,
	};
}
