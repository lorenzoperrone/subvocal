import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { injectASTTags, detectLanguage } from './astTagger.js';

export interface MultiFileBlock {
	filePath: string;
	content: string;
	folded: boolean;
}

export function resolveMultiFileContext(
	targetFile: string,
	_projectDir: string,
): MultiFileBlock[] {
	const blocks: MultiFileBlock[] = [];

	let content: string;
	try {
		content = readFileSync(targetFile, 'utf-8');
	} catch {
		return blocks;
	}

	const importRE = /(?:import|from)\s+['"]([^'"]+)['"]/g;
	const seen = new Set<string>();
	seen.add(targetFile);

	let match;
	while ((match = importRE.exec(content)) !== null) {
		const importPath = match[1];
		if (!importPath.startsWith('.')) continue;

		const resolved = resolve(dirname(targetFile), importPath);
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
