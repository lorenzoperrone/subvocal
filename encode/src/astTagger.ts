/**
 * astTagger.ts
 *
 * Substory 2.2: AST-driven tag injection.
 *
 * Takes raw source code (TypeScript or Python), parses its AST via tree-sitter,
 * and injects single-token ideogram tags from TagRegistry.json immediately before
 * each relevant syntactic block (functions, classes, loops).
 *
 * The output is:
 *   - taggedCode: the original source with ideogram markers prepended to each block
 *   - tagMap: a Map<tokenId, nodeLabel> so that when subvocal-large references a token
 *     ID in its output, we can resolve it back to the exact code block instantly.
 */

import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import * as path from 'path';
import * as fs from 'fs';
import { activeProfile } from './modelProfile.js';
import { murmurHash3 } from './lineLevelCRC.js';
import { nextFreeAnchor } from './ideogramAllocator.js';

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ── Dynamic import of native CJS tree-sitter modules ─────────────────────────
// tree-sitter and its language bindings are native addons (CJS), so we load
// them via require() rather than ESM import.
// biome-ignore lint/suspicious/noExplicitAny: native addon untyped
const Parser = require('tree-sitter') as any;
// tree-sitter-typescript exposes languages under .typescript and .tsx
// biome-ignore lint/suspicious/noExplicitAny: native addon untyped
const TSTypeScript = (require('tree-sitter-typescript') as any).typescript;
// Pass the whole tree-sitter-python module object to setLanguage(), NOT
// `.language` -- this grammar's `.language` sub-object ships frozen and
// without its own `nodeTypeInfo` (the wrapper's initializeLanguageNodeClasses
// needs to read+write directly on what it's given), which made
// tree.rootNode crash later with "Cannot read properties of undefined
// (reading '<nodeTypeId>')" in tree-sitter's unmarshalNode. The whole module
// has `.language` (for the native pointer) AND `.nodeTypeInfo` as sibling
// properties and isn't frozen, so it works -- same pattern already used in
// dependencyGraph.ts/astEditor.ts.
// biome-ignore lint/suspicious/noExplicitAny: native addon untyped
const TSPython = require('tree-sitter-python') as any;

// ── TagRegistry ───────────────────────────────────────────────────────────────
interface TagEntry {
  char: string;
  tokenId: number;
}

// Load per-profile registry (token IDs differ between model families).
const tagRegistry: TagEntry[] = JSON.parse(fs.readFileSync(activeProfile.tagRegistryPath, 'utf-8'));

// ── M15.2: content-addressed block anchors ────────────────────────────────────

// Statement/member granularity — finer than function-level tags, coarser than lines
// (identical lines like `}` would collide; blocks are almost always unique).
const ANCHORED_BLOCK_TYPES = new Set([
  // TypeScript / JavaScript statements & members
  'lexical_declaration', 'variable_declaration', 'expression_statement', 'return_statement',
  'if_statement', 'for_statement', 'for_in_statement', 'while_statement', 'switch_statement',
  'try_statement', 'throw_statement', 'public_field_definition', 'import_statement',
  'type_alias_declaration', 'interface_declaration',
  // NOT 'export_statement': it would wrap whole exported classes/functions, so any inner
  // edit would also flip the wrapper's content hash (containment noise in diffs). It is a
  // PARENT instead — the exported declaration inside gets the anchor.
  // Python statements
  'expression_statement', 'return_statement', 'if_statement', 'for_statement',
  'while_statement', 'try_statement', 'raise_statement', 'assert_statement',
  'import_statement', 'import_from_statement', 'assignment',
]);

// Anchor only DIRECT children of these containers — top-level statements, class members and
// function bodies — not expressions nested arbitrarily deep.
const ANCHOR_PARENT_TYPES = new Set([
  'program', 'module', 'class_body', 'statement_block', 'block', 'export_statement',
]);

/** Blocks smaller than this aren't worth an anchor token (single `break;` etc.). */
const MIN_ANCHOR_BLOCK_BYTES = 12;

// Node types that deserve a spatial tag.
// Extend this list to add more AST node types as needed.
const TAGGED_NODE_TYPES = new Set([
  // TypeScript / JavaScript
  'function_declaration',
  'function_expression',
  'arrow_function',
  'method_definition',
  'class_declaration',
  'for_statement',
  'for_in_statement',
  'for_of_statement',
  'while_statement',
  'do_statement',
  // Python
  'function_definition',
  'class_definition',
  'for_statement', // Python shares the name
  'while_statement',
  'decorated_definition',
]);

// ── Language detection ────────────────────────────────────────────────────────
export type SupportedLanguage = 'typescript' | 'python';

export function detectLanguage(filePath: string): SupportedLanguage {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === '.py') return 'python';
  return 'typescript'; // covers .ts, .tsx, .js, .jsx
}

// ── Core AST traversal ────────────────────────────────────────────────────────
// biome-ignore lint/suspicious/noExplicitAny: tree-sitter SyntaxNode is untyped
type SyntaxNode = any;

function traverseAST(node: SyntaxNode, visitor: (n: SyntaxNode) => void): void {
  visitor(node);
  for (let i = 0; i < node.childCount; i++) {
    traverseAST(node.child(i), visitor);
  }
}

// ── Exported result type ──────────────────────────────────────────────────────

/** Positional info for one tagged AST node. */
export interface TagInjection {
	/** Byte offset of the node in the ORIGINAL source file. */
	startIndex: number;
	/** The single-character ideogram injected before this node. */
	char: string;
	/** Model token ID for this ideogram. */
	tokenId: number;
	/** Human-readable label (e.g. "function_definition:calculateTotal"). */
	label: string;
}

export interface ASTTagResult {
	/** Source code with ideogram tags injected before each tagged node. */
	taggedCode: string;
	/**
	 * Maps each injected TokenID to a human-readable node label.
	 * When subvocal-large responds with [TOKEN_ID, ...], look up this map
	 * to find exactly which function/class it is referencing.
	 */
	tagMap: Map<number, string>;
	/** Number of tags injected. */
	tagCount: number;
	/** Ordered list of injections with positional data for AST editing. */
	injections: TagInjection[];
}

// ── Main export ───────────────────────────────────────────────────────────────
/**
 * Parse fileContent with tree-sitter, inject ideogram tags from TagRegistry
 * before each relevant AST node, and return the tagged code + resolution map.
 *
 * @param fileContent  Raw source code string.
 * @param language     'typescript' | 'python'
 * @param maxTags      Safety cap: max number of tags to inject (default: 64).
 */
export function injectASTTags(
  fileContent: string,
  language: SupportedLanguage = 'typescript',
  maxTags = 64,
): ASTTagResult {
  // Pick the correct tree-sitter grammar
  const parser = new Parser();
  parser.setLanguage(language === 'python' ? TSPython : TSTypeScript);

  const tree = parser.parse(fileContent);
  const tagMap = new Map<number, string>();

  // Collect (startIndex, tag, label) tuples sorted by startIndex ascending
  const injections: TagInjection[] = [];
  let tagIndex = 0;

  traverseAST(tree.rootNode, (node: SyntaxNode) => {
    if (!TAGGED_NODE_TYPES.has(node.type)) return;
    if (tagIndex >= maxTags || tagIndex >= tagRegistry.length) return;

    // Build a readable label: prefer the node's first named child (e.g. function name)
    let label = node.type;
    if (node.firstNamedChild?.type === 'identifier') {
      label = `${node.type}:${node.firstNamedChild.text}`;
    } else if (node.firstNamedChild?.type === 'name') {
      label = `${node.type}:${node.firstNamedChild.text}`;
    }

    const entry = tagRegistry[tagIndex];
    injections.push({
      startIndex: node.startIndex,
      char: entry.char,
      tokenId: entry.tokenId,
      label,
    });
    tagMap.set(entry.tokenId, label);
    tagIndex++;
  });

  // M15.2: content-addressed block anchors (opt-in via SUBVOCAL_BLOCK_ANCHORS=1).
  // Statement-granularity anchors whose ideogram derives from the block's murmur3 hash:
  // unchanged content re-renders with the SAME token (KV/trie stability, stale-edit
  // rejection, diff-only re-feeds — see doc/substories/M15.2-crc-block-anchors.md).
  // Same TagInjection shape → steering, astEditor and detag all work unchanged.
  if (process.env.SUBVOCAL_BLOCK_ANCHORS === '1') {
    const taggedStarts = new Set(injections.map((i) => i.startIndex));
    const usedAnchorIdx = new Set<number>();
    traverseAST(tree.rootNode, (node: SyntaxNode) => {
      if (!ANCHORED_BLOCK_TYPES.has(node.type)) return;
      if (taggedStarts.has(node.startIndex)) return; // node tag already marks this spot
      if (node.endIndex - node.startIndex < MIN_ANCHOR_BLOCK_BYTES) return;
      // Only direct statements/members — not expressions nested inside other statements.
      const parentType = node.parent?.type ?? '';
      if (!ANCHOR_PARENT_TYPES.has(parentType)) return;

      const hash = murmurHash3(node.text, 0);
      const { entry, index } = nextFreeAnchor(hash, usedAnchorIdx);
      usedAnchorIdx.add(index);
      // Label MUST start with the raw node type — astEditor's parseLabel() takes the text
      // before ':' as the tree-sitter type for node resolution. The '⌗<row>' suffix marks
      // this injection as a block anchor for consumers (and is Strategy-B-inert).
      const label = `${node.type}:⌗${node.startPosition.row + 1}`;
      injections.push({ startIndex: node.startIndex, char: entry.char, tokenId: entry.tokenId, label });
      tagMap.set(entry.tokenId, label);
      taggedStarts.add(node.startIndex);
    });
  }

  // Sort by position (tree-sitter traversal is DFS so usually already ordered,
  // but sort for safety to ensure correct reconstruction)
  injections.sort((a, b) => a.startIndex - b.startIndex);

  // Rebuild the source string with injected tags.
  // We walk the injections in order and splice each tag character + space.
  let taggedCode = '';
  let cursor = 0;
  for (const inj of injections) {
    taggedCode += fileContent.slice(cursor, inj.startIndex);
    taggedCode += `${inj.char} `; // 1 token + 1 space separator
    cursor = inj.startIndex;
  }
  taggedCode += fileContent.slice(cursor);

  return { taggedCode, tagMap, tagCount: injections.length, injections };
}

/**
 * M15.2: the content-addressed anchor set of a source — `char → block text` — WITHOUT
 * rendering. Used by callers that diff two versions of a file: blocks of the new content
 * whose anchor char is absent from the old set are exactly the changed/added blocks
 * (content-addressing makes set membership the diff). Independent of the
 * SUBVOCAL_BLOCK_ANCHORS flag — callers decide.
 */
export function computeBlockAnchors(
  fileContent: string,
  language: SupportedLanguage = 'typescript',
): Map<string, string> {
  const parser = new Parser();
  parser.setLanguage(language === 'python' ? TSPython : TSTypeScript);
  const tree = parser.parse(fileContent);
  const out = new Map<string, string>();
  const used = new Set<number>();
  traverseAST(tree.rootNode, (node: SyntaxNode) => {
    if (!ANCHORED_BLOCK_TYPES.has(node.type)) return;
    if (node.endIndex - node.startIndex < MIN_ANCHOR_BLOCK_BYTES) return;
    const parentType = node.parent?.type ?? '';
    if (!ANCHOR_PARENT_TYPES.has(parentType)) return;
    const { entry, index } = nextFreeAnchor(murmurHash3(node.text, 0), used);
    used.add(index);
    out.set(entry.char, node.text);
  });
  return out;
}
