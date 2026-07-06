/**
 * shadowFileSystem.ts
 *
 * Substory 2.3: Shadow File System daemon.
 *
 * Background watcher (inotify via Node.js fs.watch) that pre-tokenizes source
 * files on every disk change. Keeps an in-RAM cache of Int32Array token arrays
 * so subsequent preprocess() calls can skip the tokenize() step entirely.
 *
 * Design contract:
 *   - Call watch() AFTER initSmallModel() so tokenize() is available.
 *   - Call scanDir() for the initial bulk pass over an existing codebase.
 *   - getCachedTokens(absolutePath) returns tokens or undefined (cold miss).
 *   - stop() releases the OS watcher handle; cache survives (read-only after stop).
 */

import * as fs from 'fs';
import * as path from 'path';
import { tokenize } from './tensorPayload.js';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ShadowFSOptions {
  /** Debounce window in ms to absorb editor atomic-save renames. Default: 50. */
  debounceMs?: number;
  /** File extensions to track. Default: .ts .tsx .js .jsx .py */
  extensions?: readonly string[];
  /** Paths matching any of these patterns are ignored. */
  ignore?: readonly RegExp[];
}

// ── Defaults ──────────────────────────────────────────────────────────────────

const DEFAULT_EXTENSIONS: readonly string[] = ['.ts', '.tsx', '.js', '.jsx', '.py'];
const DEFAULT_IGNORE: readonly RegExp[] = [
  /node_modules/,
  /[/\\]\.git[/\\]/,
  /build-/,
  /[/\\]dist[/\\]/,
  /\.node$/,
];

// ── Implementation ────────────────────────────────────────────────────────────

export class ShadowFileSystem {
  private tokenCache = new Map<string, Int32Array>();
  private watcher: fs.FSWatcher | null = null;
  private debounceMap = new Map<string, ReturnType<typeof setTimeout>>();
  private watchedDir = '';

  private readonly debounceMs: number;
  private readonly extensions: ReadonlySet<string>;
  private readonly ignore: readonly RegExp[];

  constructor(opts?: ShadowFSOptions) {
    this.debounceMs = opts?.debounceMs ?? 50;
    this.extensions = new Set(opts?.extensions ?? DEFAULT_EXTENSIONS);
    this.ignore = opts?.ignore ?? DEFAULT_IGNORE;
  }

  /**
   * Start watching `dir` recursively. File changes trigger a debounced
   * re-tokenization. Must be called after initSmallModel().
   */
  watch(dir: string): void {
    this.stop();
    this.watchedDir = path.resolve(dir);
    this.watcher = fs.watch(
      this.watchedDir,
      { recursive: true },
      (_event: string, filename: string | null) => {
        if (!filename) return;
        const fullPath = path.join(this.watchedDir, filename);
        if (!this.shouldProcess(fullPath)) return;
        this.scheduleProcess(fullPath);
      },
    );
    this.watcher.on('error', () => { /* ignore transient errors (unmount, etc.) */ });
  }

  /** Stop the OS watcher. The token cache remains readable after this call. */
  stop(): void {
    this.watcher?.close();
    this.watcher = null;
    for (const t of this.debounceMap.values()) clearTimeout(t);
    this.debounceMap.clear();
  }

  /**
   * Walk `dir` synchronously and tokenize every matching file.
   * Call once at startup (after initSmallModel) for an existing codebase.
   */
  scanDir(dir: string): void {
    const abs = path.resolve(dir);
    this.walkSync(abs);
  }

  /** Return cached token array for an absolute path, or undefined on cold miss. */
  getCachedTokens(filePath: string): Int32Array | undefined {
    return this.tokenCache.get(path.resolve(filePath));
  }

  /** Absolute paths of every cached file. */
  getAllPaths(): IterableIterator<string> {
    return this.tokenCache.keys();
  }

  /** Number of files currently in the cache. */
  get size(): number {
    return this.tokenCache.size;
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  private scheduleProcess(fullPath: string): void {
    const prev = this.debounceMap.get(fullPath);
    if (prev) clearTimeout(prev);
    this.debounceMap.set(
      fullPath,
      setTimeout(() => {
        this.debounceMap.delete(fullPath);
        this.processFile(fullPath);
      }, this.debounceMs),
    );
  }

  private processFile(filePath: string): void {
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      const tokens = tokenize(content);
      this.tokenCache.set(filePath, tokens);
    } catch {
      // File deleted, model not yet initialized, or binary file — skip silently.
      this.tokenCache.delete(filePath);
    }
  }

  private shouldProcess(filePath: string): boolean {
    if (!this.extensions.has(path.extname(filePath))) return false;
    if (this.ignore.some((re) => re.test(filePath))) return false;
    return true;
  }

  private walkSync(dir: string): void {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (!this.ignore.some((re) => re.test(full))) this.walkSync(full);
      } else if (entry.isFile() && this.shouldProcess(full)) {
        this.processFile(full);
      }
    }
  }
}
