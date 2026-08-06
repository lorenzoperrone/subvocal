#!/usr/bin/env node
/**
 * Cross-platform automatic build dispatcher for Subvocal.
 * Detects host OS and available hardware (macOS Metal, Linux CUDA, Linux/Windows CPU)
 * and invokes the appropriate cmake-js build target for @subvocal/synapse.
 */

import { execSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { platform } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');

function hasNvidiaGpu() {
	if (platform() === 'win32' || platform() === 'linux') {
		try {
			execSync('nvidia-smi', { stdio: 'ignore' });
			return true;
		} catch {
			return false;
		}
	}
	return false;
}

function hasEngineFork() {
	return (
		existsSync(join(rootDir, 'llama.cpp', 'include', 'llama.h')) ||
		existsSync(join(rootDir, 'ik_llama.cpp', 'include', 'llama.h'))
	);
}

function getBuildCommand() {
	const os = platform();

	if (os === 'darwin') {
		console.log('[build-platform] Detected macOS (Apple Silicon Metal)');
		return 'npm run build:metal -w synapse';
	}

	if (os === 'linux') {
		if (hasNvidiaGpu()) {
			console.log('[build-platform] Detected Linux with NVIDIA CUDA GPU');
			return 'npm run build:gpu -w synapse';
		}
		console.log('[build-platform] Detected Linux (CPU AVX2/AVX512/OpenMP)');
		return 'npm run build:cpu -w synapse';
	}

	if (os === 'win32') {
		if (hasNvidiaGpu()) {
			console.log('[build-platform] Detected Windows with NVIDIA CUDA GPU');
			return 'npm run build:gpu -w synapse';
		}
		console.log('[build-platform] Detected Windows (CPU fallback)');
		return 'npm run build:cpu -w synapse';
	}

	console.log(`[build-platform] Unknown OS (${os}), falling back to CPU build`);
	return 'npm run build:cpu -w synapse';
}

try {
	if (hasEngineFork()) {
		const cmd = getBuildCommand();
		console.log(`[build-platform] Executing: ${cmd}`);
		execSync(cmd, { stdio: 'inherit' });
	} else {
		console.log('[build-platform] ℹ️ Upstream C++ engine forks (llama.cpp / ik_llama.cpp) not present in workspace; skipping C++ compilation.');
	}

	console.log('[build-platform] Building TypeScript types (@subvocal/synapse)...');
	execSync('npm run build:ts -w synapse', { stdio: 'inherit' });

	console.log('[build-platform] Building TypeScript orchestration (@subvocal/encode)...');
	execSync('npm run build:encode', { stdio: 'inherit' });

	console.log('[build-platform] ✅ Cross-platform build completed successfully!');
} catch (err) {
	console.error('[build-platform] ❌ Build failed:', err.message);
	process.exit(1);
}
