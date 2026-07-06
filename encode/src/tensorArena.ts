/**
 * tensorArena.ts
 *
 * Substory 2.3: RAM Arena of latent tensors.
 *
 * Indexes all functions/classes in a codebase as normalized hidden-state
 * Float32Arrays using a dedicated ModelCPU with `embeddings: true`.
 * Keeps the embedding model separate from the shared smallModel singleton
 * so intent routing and tokenization latency are unaffected.
 *
 * Gemma 4 E2B hidden dim: 1536 floats per entry (6 KiB per function).
 * A codebase with 5000 functions ≈ 30 MiB of float data in RAM.
 *
 * Usage:
 *   const arena = new TensorArena();
 *   arena.init(modelPath);               // once at boot
 *   arena.indexFile(filePath, content);  // called by ShadowFileSystem on change
 *   const embedding = arena.getQueryEmbedding(promptText);
 *   const hits = arena.query(embedding, 5);
 */

import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import * as path from 'path';
import { type BaseModel, ModelCPU, normalizeL2, cosineSim } from '@subvocal/synapse';

const require = createRequire(import.meta.url);
// biome-ignore lint/suspicious/noExplicitAny: native addon untyped
const Parser = require('tree-sitter') as any;
// biome-ignore lint/suspicious/noExplicitAny: native addon untyped
const TSTypeScript = (require('tree-sitter-typescript') as any).typescript;
// Pass the whole module to setLanguage(), NOT `.language` -- see the comment in
// astTagger.ts for why: `.language` ships frozen and without its own
// nodeTypeInfo, which crashes tree.rootNode later inside tree-sitter's own
// unmarshalNode. The whole module has both as sibling properties and isn't frozen.
// biome-ignore lint/suspicious/noExplicitAny: native addon untyped
const TSPython = require('tree-sitter-python') as any;

// biome-ignore lint/suspicious/noExplicitAny: tree-sitter SyntaxNode is untyped
type SyntaxNode = any;

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ArenaEntry {
  /** Absolute path of the source file. */
  filePath: string;
  /** Name of the function/class (e.g. "calculateTotal"). */
  funcName: string;
  /** AST node type (e.g. "function_definition", "class_declaration"). */
  nodeType: string;
  /** Source text of the function (capped at MAX_SNIPPET_CHARS). */
  snippet: string;
  /** Tokens of `snippet` as produced by the embedding model. */
  tokens: Int32Array;
  /** L2-normalized hidden state of the last token after forward(tokens). */
  hiddenState: Float32Array;
}

export interface ArenaQueryHit {
  key: string;
  score: number;
  entry: ArenaEntry;
}

// ── Constants ─────────────────────────────────────────────────────────────────

/** Max UTF-16 chars taken from a function body for embedding. */
const MAX_SNIPPET_CHARS = 512;

/** Max tokens fed to the embedding model per function (guards KV overflow). */
const MAX_TOKENS = 256;

const EMBEDDED_NODE_TYPES = new Set([
  'function_declaration',
  'function_expression',
  'arrow_function',
  'method_definition',
  'class_declaration',
  'function_definition',
  'class_definition',
  'decorated_definition',
]);

// ── Implementation ────────────────────────────────────────────────────────────

export class TensorArena {
  private arena = new Map<string, ArenaEntry>();
  private model: BaseModel | null = null;

  /**
   * Load the embedding model. Defaults to ModelCPU (ik_llama.cpp) with the same GGUF as the
   * smallModel singleton, kept as a separate context so intent routing/tokenization latency
   * are unaffected — this is the Linux convention (CPU is idle while the GPU runs the large
   * model). Pass an already-constructed BaseModel (e.g. `new ModelGPU(embeddingGgufPath, {
   * embeddings: true })`) instead when there's no separate cheap CPU backend to dedicate to
   * this — e.g. on Mac, where the small-model/ik_llama.cpp path is off by default (see
   * doc/substories/M2.1-dual-brain-decision.md) and a dedicated embedding model
   * (embeddinggemma-300m) runs through the same Metal backend as everything else instead.
   */
  init(
    modelOrPath: BaseModel | string,
    opts?: { contextSize?: number; threads?: number },
  ): void {
    this.model?.free();
    this.model = typeof modelOrPath === 'string'
      ? new ModelCPU(modelOrPath, {
          contextSize: opts?.contextSize ?? 2048,
          threads: opts?.threads ?? 4,
          embeddings: true,
        })
      : modelOrPath;
  }

  /** Release the embedding model. Arena entries remain readable. */
  free(): void {
    this.model?.free();
    this.model = null;
  }

  /**
   * Parse `content` with tree-sitter, extract named functions/classes, compute
   * their hidden states, and store in the arena. Re-indexes the entire file on
   * every call (removes old entries for this path first).
   */
  indexFile(filePath: string, content: string): void {
    if (!this.model) throw new Error('TensorArena: call init() before indexFile()');

    // Remove stale entries from a previous index of this file.
    const prefix = `${path.resolve(filePath)}:`;
    for (const key of this.arena.keys()) {
      if (key.startsWith(prefix)) this.arena.delete(key);
    }

    const lang = path.extname(filePath) === '.py' ? 'python' : 'typescript';
    const spans = extractFunctionSpans(content, lang);

    for (const span of spans) {
      const snippet = content.slice(span.startIndex, span.endIndex).slice(0, MAX_SNIPPET_CHARS);
      try {
        const tokensRaw = this.model.tokenize(snippet, false, false);
        const tokens = tokensRaw.length > MAX_TOKENS
          ? tokensRaw.slice(0, MAX_TOKENS)
          : tokensRaw;
        if (tokens.length === 0) continue;

        this.model.forward(tokens);
        const hidden = this.model.getHiddenState();
        const hiddenState = normalizeL2(new Float32Array(hidden));

        const key = `${path.resolve(filePath)}:${span.name}`;
        this.arena.set(key, {
          filePath: path.resolve(filePath),
          funcName: span.name,
          nodeType: span.nodeType,
          snippet,
          tokens,
          hiddenState,
        });
      } catch {
        // Skip unemeddable functions (model error, empty body, etc.)
      }
    }
  }

  /**
   * Embed an arbitrary text string using the same model/layer as indexFile.
   * Use the returned Float32Array as the query vector for arena.query().
   */
  getQueryEmbedding(text: string): Float32Array {
    if (!this.model) throw new Error('TensorArena: call init() before getQueryEmbedding()');
    const tokensRaw = this.model.tokenize(text, false, false);
    const tokens = tokensRaw.length > MAX_TOKENS
      ? tokensRaw.slice(0, MAX_TOKENS)
      : tokensRaw;
    if (tokens.length === 0) return new Float32Array(0);
    this.model.forward(tokens);
    const hidden = this.model.getHiddenState();
    return normalizeL2(new Float32Array(hidden));
  }

  /**
   * Retrieve the top-N arena entries by cosine similarity to `queryState`.
   * Returns entries sorted descending by score.
   */
  query(queryState: Float32Array, topN: number): ArenaQueryHit[] {
    if (queryState.length === 0) return [];
    const hits: ArenaQueryHit[] = [];
    for (const [key, entry] of this.arena) {
      if (entry.hiddenState.length !== queryState.length) continue;
      const score = cosineSim(queryState, entry.hiddenState);
      hits.push({ key, score, entry });
    }
    hits.sort((a, b) => b.score - a.score);
    return hits.slice(0, topN);
  }

  /** Number of indexed function entries across all files. */
  get size(): number {
    return this.arena.size;
  }

  /** True if the embedding model is loaded. */
  get isReady(): boolean {
    return this.model !== null;
  }
}

// ── Internal: AST span extraction ─────────────────────────────────────────────

interface FunctionSpan {
  name: string;
  nodeType: string;
  startIndex: number;
  endIndex: number;
}

function extractFunctionSpans(
  source: string,
  lang: 'typescript' | 'python',
): FunctionSpan[] {
  const parser = new Parser();
  parser.setLanguage(lang === 'python' ? TSPython : TSTypeScript);
  const tree = parser.parse(source);

  const spans: FunctionSpan[] = [];
  traverseAST(tree.rootNode, (node: SyntaxNode) => {
    if (!EMBEDDED_NODE_TYPES.has(node.type)) return;

    // Resolve the canonical name from the first named identifier/name child.
    let name = '';
    for (let i = 0; i < node.childCount; i++) {
      const c = node.child(i);
      if (c && (c.type === 'identifier' || c.type === 'name' || c.type === 'property_identifier')) {
        name = c.text;
        break;
      }
    }
    if (!name) return; // skip anonymous functions

    spans.push({
      name,
      nodeType: node.type,
      startIndex: node.startIndex,
      endIndex: node.endIndex,
    });
  });

  return spans;
}

function traverseAST(node: SyntaxNode, visitor: (n: SyntaxNode) => void): void {
  visitor(node);
  for (let i = 0; i < node.childCount; i++) {
    traverseAST(node.child(i), visitor);
  }
}
