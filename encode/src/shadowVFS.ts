/**
 * shadowVFS.ts
 *
 * Epico 4.1 Fase C — Shadow Virtual File System.
 *
 * An in-memory file store that mirrors the real filesystem during speculative
 * editing. The model's edits are applied to the VFS first; the real disk is only
 * touched when the user confirms the changes (Dead-Man's Switch).
 *
 * Snapshot/rollback semantics:
 *   snapshot()  → pushes current VFS state onto a stack
 *   rollback()  → restores the last snapshot, discarding all writes since
 *   commit()    → pops the snapshot without restoring (action succeeded)
 *
 * Zero dependencies. Pure Map<string, string> with a stack of clones.
 */

export interface VFSFile {
	path: string;
	content: string;
}

// ── Implementation ─────────────────────────────────────────────────────────────

export class ShadowVFS {
	private files = new Map<string, string>();
	private snapshots: Array<Map<string, string>> = [];

	/** Write (or overwrite) a file in the VFS. */
	write(path: string, content: string): void {
		this.files.set(path, content);
	}

	/** Read a file from the VFS. Returns undefined if not present. */
	read(path: string): string | undefined {
		return this.files.get(path);
	}

	/** Delete a file from the VFS. */
	delete(path: string): boolean {
		return this.files.delete(path);
	}

	/** True if the file exists in the VFS. */
	has(path: string): boolean {
		return this.files.has(path);
	}

	/** All file paths currently in the VFS. */
	paths(): IterableIterator<string> {
		return this.files.keys();
	}

	/** Snapshot the current VFS state. Push onto the stack. */
	snapshot(): void {
		this.snapshots.push(new Map(this.files));
	}

	/**
	 * Rollback to the last snapshot. Restores the VFS to the state
	 * it was in when snapshot() was called. Returns true if a snapshot existed.
	 */
	rollback(): boolean {
		const prev = this.snapshots.pop();
		if (!prev) return false;
		this.files = prev;
		return true;
	}

	/**
	 * Discard the last snapshot without restoring. Use when the action succeeded
	 * and the VFS changes should be kept. Returns true if a snapshot existed.
	 */
	commit(): boolean {
		return this.snapshots.pop() !== undefined;
	}

	/** How many snapshots are on the stack. */
	get depth(): number {
		return this.snapshots.length;
	}

	/** Remove all files and snapshots. */
	reset(): void {
		this.files.clear();
		this.snapshots = [];
	}

	/** Export the VFS as an array of { path, content } entries. */
	export(): VFSFile[] {
		return Array.from(this.files.entries()).map(([path, content]) => ({ path, content }));
	}
}
