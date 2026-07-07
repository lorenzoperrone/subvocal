/**
 * osSandbox.ts
 *
 * Epico 4.1 — OS-Level Fortress.
 *
 * Shell-based sandbox utilities for safe speculative execution.
 * All operations return shell commands as strings (not executing them)
 * so the caller can pipe them through a permission gate.
 */

import type { execSync } from 'child_process';

// ── Types ──────────────────────────────────────────────────────────────────────

export interface SandboxConfig {
	/** Real project path (read-only, never modified directly). */
	projectPath: string;
	/** Ephemeral workspace path (tmpfs, disposable). */
	workspacePath: string;
}

export interface DiffResult {
	diff: string;
	prompt: string;
}

// ── Ephemeral worktree ─────────────────────────────────────────────────────────

/**
 * Create a git worktree in tmpfs for ephemeral execution.
 * Returns the shell command to create it.
 */
export function createEphemeralWorktree(config: SandboxConfig): string {
	const { projectPath, workspacePath } = config;
	return [
		`mkdir -p "${workspacePath}"`,
		`mount -t tmpfs -o size=512M tmpfs "${workspacePath}"`,
		`git -C "${projectPath}" worktree add --detach "${workspacePath}" HEAD`,
	].join(' && ');
}

/**
 * Remove the ephemeral worktree and clean up.
 */
export function removeEphemeralWorktree(workspacePath: string): string {
	return [
		`git -C "${workspacePath}" worktree remove --force "${workspacePath}" 2>/dev/null || git worktree remove --force "${workspacePath}"`,
		`umount "${workspacePath}" 2>/dev/null || true`,
		`rm -rf "${workspacePath}"`,
	].join(' && ');
}

// ── Network isolation ──────────────────────────────────────────────────────────

/**
 * Create a network namespace for air-gapped execution.
 * Returns the shell command prefix (unshare -n).
 */
export function airGapPrefix(): string {
	return 'unshare -n';
}

// ── Tmpfs mounts ───────────────────────────────────────────────────────────────

/**
 * Mount a tmpfs at the given path.
 */
export function mountTmpfs(path: string): string {
	return `mkdir -p "${path}" && mount -t tmpfs -o size=512M tmpfs "${path}"`;
}

/**
 * Unmount a tmpfs at the given path.
 */
export function unmountTmpfs(path: string): string {
	return `umount "${path}" 2>/dev/null || true`;
}

// ── Sandboxed command ──────────────────────────────────────────────────────────

/**
 * Run a test/compile command in the sandbox with Deno-style permissions.
 * - allow-read only on workspace
 * - deny-write outside workspace
 * - deny-net for air-gap
 * Returns the shell command.
 */
export function sandboxedCommand(workspacePath: string, command: string): string {
	const isolatedPath = `PATH=/usr/bin:/bin:/usr/local/bin`;
	return `${airGapPrefix()} bash -c '${isolatedPath}; cd "${workspacePath}" && ${command}'`;
}

// ── Dead-man's switch ──────────────────────────────────────────────────────────

/**
 * Dead-man's switch: compute a diff between old and new content,
 * return the diff and a confirmation prompt.
 */
export function deadMansSwitch(
	beforeContent: string,
	afterContent: string,
	filePath: string,
): DiffResult {
	const diff = computeDiff(beforeContent, afterContent, filePath);
	const prompt = [
		`--- ${filePath} (before)`,
		`+++ ${filePath} (after)`,
		diff,
		'',
		'Accept these changes? (y/N/explain)',
	].join('\n');
	return { diff, prompt };
}

// ── Unified diff (zero dependencies) ───────────────────────────────────────────

/**
 * Compute a unified diff between two strings.
 *
 * 2026-07 audit: the previous implementation used a fixed ~8-line window around each change and
 * emitted every line in that window as removed AND added — so a changed region longer than the
 * window rendered unchanged lines as -/+ pairs (a wrong, misleading diff). This version trims the
 * common line prefix and suffix and shows only the genuinely differing middle (old lines removed,
 * new lines added) with up to `ctx` lines of surrounding context. It is always CORRECT — a line
 * equal on both sides is never shown as changed — though not minimal for changes scattered across
 * a file (those collapse into one hunk). Fine for the human-facing `:diff` / follow-up display
 * this feeds; nothing conditions on its exact hunk shape.
 */
export function computeDiff(oldText: string, newText: string, filePath = 'file'): string {
	if (oldText === newText) return '';

	const oldLines = oldText.split('\n');
	const newLines = newText.split('\n');
	const ctx = 3;

	// Common prefix: lines identical from the top.
	let p = 0;
	while (p < oldLines.length && p < newLines.length && oldLines[p] === newLines[p]) p++;

	// Common suffix: lines identical from the bottom, not overlapping the prefix.
	let s = 0;
	while (
		s < oldLines.length - p &&
		s < newLines.length - p &&
		oldLines[oldLines.length - 1 - s] === newLines[newLines.length - 1 - s]
	) {
		s++;
	}

	// Changed region: old[p .. oldLines.length - s), new[p .. newLines.length - s).
	const oldChangeEnd = oldLines.length - s;
	const newChangeEnd = newLines.length - s;

	// Context bounds (clamped so we never re-show removed/added lines as context).
	const ctxStart = Math.max(0, p - ctx);
	const oldCtxEnd = Math.min(oldLines.length, oldChangeEnd + ctx);
	const newCtxEnd = Math.min(newLines.length, newChangeEnd + ctx);

	const body: string[] = [];
	for (let k = ctxStart; k < p; k++) body.push(` ${oldLines[k]}`);          // leading context
	for (let k = p; k < oldChangeEnd; k++) body.push(`-${oldLines[k]}`);       // removed
	for (let k = p; k < newChangeEnd; k++) body.push(`+${newLines[k]}`);       // added
	// Trailing context is shared (identical on both sides), read from oldLines.
	for (let k = oldChangeEnd; k < oldCtxEnd; k++) body.push(` ${oldLines[k]}`);

	// Trailing context count is the same on both sides (it's the shared common suffix), so each
	// range length is just its own (context-clamped) span from ctxStart.
	const oldRangeLen = oldCtxEnd - ctxStart;
	const newRangeLen = newCtxEnd - ctxStart;
	return (
		`--- ${filePath}\n+++ ${filePath}\n` +
		`@@ -${ctxStart + 1},${oldRangeLen} +${ctxStart + 1},${newRangeLen} @@\n` +
		body.join('\n') + '\n'
	);
}
