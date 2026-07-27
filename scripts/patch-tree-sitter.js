import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

/**
 * Recursively find all `node_modules/tree-sitter/index.js` files.
 */
function findTreeSitterIndexFiles(dir, files = []) {
  if (!fs.existsSync(dir)) return files;
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === 'tree-sitter' && path.basename(dir) === 'node_modules') {
        const indexPath = path.join(fullPath, 'index.js');
        if (fs.existsSync(indexPath)) {
          files.push(indexPath);
        }
      } else {
        if (!['.git', 'dist', 'build', '.cache'].includes(entry.name)) {
          findTreeSitterIndexFiles(fullPath, files);
        }
      }
    }
  }

  return files;
}

function patchTreeSitterFile(filePath) {
  let content = fs.readFileSync(filePath, 'utf8');

  if (content.includes('_languageSubclasses')) {
    console.log(`[patch-tree-sitter] Already patched: ${filePath}`);
    return;
  }

  console.log(`[patch-tree-sitter] Patching Node 24 compatibility in: ${filePath}`);

  const targetBindingMarker = "const {Query, Parser, NodeMethods, Tree, TreeCursor, LookaheadIterator} = binding;";
  if (!content.includes(targetBindingMarker)) {
    console.warn(`[patch-tree-sitter] Warning: binding marker not found in ${filePath}`);
    return;
  }

  const patchWeakMap = `${targetBindingMarker}\n\n// SUBVOCAL: Node.js 24 compatibility — language objects from N-API are non-extensible\nconst _languageSubclasses = new WeakMap();`;
  content = content.replace(targetBindingMarker, patchWeakMap);

  const targetSetLang = "Parser.prototype.setLanguage = function(language) {";
  if (content.includes(targetSetLang) && !content.includes("_languageSubclasses.has")) {
    const replacementSetLang = `Parser.prototype.setLanguage = function(language) {
  if (this instanceof Parser && setLanguage) {
    setLanguage.call(this, language);
  }
  this[languageSymbol] = language;
  if (!_languageSubclasses.has(language)) {
    initializeLanguageNodeClasses(language)
  }
  return this;
};`;
    content = content.replace(/Parser\.prototype\.setLanguage = function\(language\) \{[\s\S]*?return this;\n\};/, replacementSetLang);
  }

  content = content.replace(
    /:\s*\(tree\.language\.nodeSubclasses \|\| \[\]\)\[nodeTypeId\];/,
    ": (_languageSubclasses.get(tree.language) || [])[nodeTypeId];"
  );

  content = content.replace(
    /language\.nodeSubclasses = nodeSubclasses;/,
    "try { language.nodeSubclasses = nodeSubclasses } catch (_) {}\n  _languageSubclasses.set(language, nodeSubclasses)"
  );

  fs.writeFileSync(filePath, content, 'utf8');
  console.log(`[patch-tree-sitter] Successfully patched: ${filePath}`);
}

function main() {
  console.log(`[patch-tree-sitter] Searching for tree-sitter packages under ${rootDir}...`);
  const targetFiles = findTreeSitterIndexFiles(rootDir);
  console.log(`[patch-tree-sitter] Found ${targetFiles.length} tree-sitter index.js file(s).`);

  for (const file of targetFiles) {
    patchTreeSitterFile(file);
  }
}

main();
