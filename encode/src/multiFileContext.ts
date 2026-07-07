import { readFileSync, existsSync } from 'fs';
import { resolve, dirname, relative, isAbsolute } from 'path';
import { injectASTTags, detectLanguage } from './astTagger.js';

export interface MultiFileBlock {
	filePath: string;
	content: string;
	folded: boolean;
}

export function resolveMultiFileContext(
	targetFile: string,
	projectDir: string,
): MultiFileBlock[] {
	const blocks: MultiFileBlock[] = [];

	let content: string;
	try {
		content = readFileSync(targetFile, 'utf-8');
	} catch {
		return blocks;
	}

	// 2026-07 audit: `projectDir` was previously ignored (`_projectDir`) — a relative import
	// chain (`../../../../etc/hosts`) resolved outside the project would be read and pasted into
	// the model's prompt. Contain resolution to the project root. `root` normalizes the boundary
	// once; a resolved dep must stay under it.
	const root = resolve(projectDir);
	const withinRoot = (p: string): boolean => {
		const rel = relative(root, p);
		return rel === '' || (!rel.startsWith('..') && !isAbsolute(rel));
	};

	const importRE = /(?:import|from)\s+['"]([^'"]+)['"]/g;
	const seen = new Set<string>();
	seen.add(targetFile);

	let match;
	while ((match = importRE.exec(content)) !== null) {
		const importPath = match[1];
		if (!importPath.startsWith('.')) continue;

		const resolved = resolve(dirname(targetFile), importPath);
		if (!withinRoot(resolved)) continue; // import escapes the project root — skip
		const extensions = ['.ts', '.tsx', '.py', '.js'];
		for (const ext of extensions) {
			const withExt = resolved + ext;
			if (!existsSync(withExt) || seen.has(withExt)) continue;
			seen.add(withExt);
			try {
				const depContent = readFileSync(withExt, 'utf-8');
				const lang = detectLanguage(withExt);
				const { taggedCode } = injectASTTags(depContent, lang);
				blocks.push({
					filePath: withExt,
					content: taggedCode,
					folded: false,
				});
			} catch {
				// best-effort: skip unreadable files
			}
			break;
		}
	}

	return blocks;
}
