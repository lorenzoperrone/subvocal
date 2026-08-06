/**
 * envLoader.ts
 *
 * Lightweight, dependency-free environment variable loader for Subvocal.
 * Loads key=value pairs from .env and .env.local files across:
 *  - Current working directory (`process.cwd()/.env.local`)
 *  - Project root (`subvocal/.env.local`)
 *  - User config directory (`~/.config/subvocal/.env.local`)
 *
 * Existing process.env variables take precedence (never overwritten).
 */

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

function parseEnvFile(filePath: string): void {
	if (!existsSync(filePath)) return;
	try {
		const content = readFileSync(filePath, 'utf-8');
		for (const line of content.split('\n')) {
			const trimmed = line.trim();
			if (!trimmed || trimmed.startsWith('#')) continue;
			const eqIdx = trimmed.indexOf('=');
			if (eqIdx <= 0) continue;
			const key = trimmed.slice(0, eqIdx).trim();
			let val = trimmed.slice(eqIdx + 1).trim();
			if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
				val = val.slice(1, -1);
			}
			if (!(key in process.env)) {
				process.env[key] = val;
			}
		}
	} catch {
		/* Best-effort environment load */
	}
}

/** Automatically loads local environment files into process.env */
export function loadSubvocalEnv(): void {
	const userConfigDir = join(homedir(), '.config', 'subvocal');
	parseEnvFile(join(userConfigDir, '.env.local'));
	parseEnvFile(join(userConfigDir, '.env'));
	parseEnvFile(join(process.cwd(), '.env.local'));
	parseEnvFile(join(process.cwd(), '.env'));
}

// Load automatically on module import
loadSubvocalEnv();
