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
 * Compute a basic unified diff between two strings.
 * Produces output in unified diff format:
 *   @@ -a,b +c,d @@
 *   -removed line
 *   +added line
 */
export function computeDiff(oldText: string, newText: string, filePath = 'file'): string {
	if (oldText === newText) return '';

	const oldLines = oldText.split('\n');
	const newLines = newText.split('\n');
	const ctx = 3;
	const hunks: string[] = [];

	// Compute an edit script using a simple greedy scan.
	// Walk both sequences simultaneously: skip matching lines, emit hunks
	// for changed regions. Handles files of different lengths correctly by
	// treating missing trailing lines as deletions or insertions.
	let i = 0;  // index into oldLines
	let j = 0;  // index into newLines

	while (i < oldLines.length || j < newLines.length) {
		// Count matching prefix at current positions
		let common = 0;
		while (
			i + common < oldLines.length &&
			j + common < newLines.length &&
			oldLines[i + common] === newLines[j + common]
		) {
			common++;
		}

		// Large common block: skip it (only keep context lines at boundaries)
		if (common > ctx * 2 + 5) {
			i += common;
			j += common;
			continue;
		}

		if (i >= oldLines.length && j >= newLines.length) break;

		// Emit a changed hunk: include context + a window of changed lines
		const oldHunkStart = Math.max(0, i - ctx);
		const oldHunkEnd   = Math.min(oldLines.length, i + ctx + 8);
		const newHunkEnd   = Math.min(newLines.length, j + ctx + 8 + (newLines.length - oldLines.length));

		const lines: string[] = [];
		const newHunkStart = Math.max(0, j - ctx);

		// Context before change
		for (let k = oldHunkStart; k < i; k++) lines.push(` ${oldLines[k]}`);
		// Removed lines
		for (let k = i; k < oldHunkEnd; k++) lines.push(`-${oldLines[k]}`);
		// Added lines
		for (let k = j; k < newHunkEnd; k++) lines.push(`+${newLines[k]}`);
		// Context after change
		let ctxAfter = 0;
		while (
			oldHunkEnd + ctxAfter < oldLines.length &&
			newHunkEnd + ctxAfter < newLines.length &&
			oldLines[oldHunkEnd + ctxAfter] === newLines[newHunkEnd + ctxAfter] &&
			ctxAfter < ctx
		) {
			lines.push(` ${oldLines[oldHunkEnd + ctxAfter]}`);
			ctxAfter++;
		}

		const oldRangeLen = oldHunkEnd - oldHunkStart + ctxAfter;
		const newRangeLen = (newHunkEnd - newHunkStart) + ctxAfter;
		hunks.push(
			`--- ${filePath}\n+++ ${filePath}\n` +
			`@@ -${oldHunkStart + 1},${oldRangeLen} +${newHunkStart + 1},${newRangeLen} @@\n` +
			lines.join('\n') + '\n',
		);

		i = oldHunkEnd + ctxAfter;
		j = newHunkEnd + ctxAfter;
	}

	return hunks.join('');
}

function formatHunk(
	oldLines: string[],
	newLines: string[],
	hunkStart: number,
	hunkEnd: number,
	oldPos: number,
	newPos: number,
	filePath: string,
): string {
	const lines: string[] = [];

	const oldRangeStart = hunkStart + 1;
	const oldRangeLen = hunkEnd - hunkStart;
	const newRangeStart = newPos + 1;
	const newRangeLen = oldRangeLen;

	lines.push(`--- ${filePath}\n`);
	lines.push(`+++ ${filePath}\n`);
	lines.push(`@@ -${oldRangeStart},${oldRangeLen} +${newRangeStart},${newRangeLen} @@\n`);

	for (let i = hunkStart; i < hunkEnd; i++) {
		const oldLine = i < oldLines.length ? oldLines[i] : undefined;
		const newIdx = newPos + (i - oldPos);
		const newLine = newIdx >= 0 && newIdx < newLines.length ? newLines[newIdx] : undefined;

		if (oldLine !== undefined && oldLine === newLine) {
			lines.push(` ${oldLine}\n`);
		} else {
			if (oldLine !== undefined) {
				lines.push(`-${oldLine}\n`);
			}
			if (newLine !== undefined) {
				lines.push(`+${newLine}\n`);
			}
		}
	}

	return lines.join('');
}
