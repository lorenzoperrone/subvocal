// Substory 1.1 — TagRegistry generator.
// Scans a curated candidate set of single-character symbols and keeps only
// those that the model's tokenizer encodes as EXACTLY ONE token. Result is a
// JSON map char → tokenId, usable for AST tag injection (l_out spatial pointers).
//
// Usage:
//   npx tsx scripts/generate-tag-registry.ts /path/to/model.gguf
//
// Output: written to <repo-root>/models/tag-registries/<model-stem>.tag-registry.json so the
// registry is co-located with the model it's specific to (vocab IDs are model-bound).

import { writeFileSync, mkdirSync } from "node:fs";
import { basename, resolve } from "node:path";
import { ModelCPU } from "../src/index.js";

const MODEL = process.argv[2];
if (!MODEL) {
	console.error("Usage: npx tsx scripts/generate-tag-registry.ts /path/to/model.gguf");
	process.exit(1);
}

// Curated candidate set — visually distinctive single chars that compilers and
// programmers don't normally use. Grouped by category for readability.
const CANDIDATES = [
	// Greek (uppercase + lowercase rare ones)
	...["Α","Β","Γ","Δ","Ε","Ζ","Η","Θ","Ι","Κ","Λ","Μ","Ν","Ξ","Ο","Π","Ρ","Σ","Τ","Υ","Φ","Χ","Ψ","Ω"],
	...["α","β","γ","δ","ε","ζ","η","θ","ι","κ","λ","μ","ν","ξ","ο","π","ρ","σ","τ","υ","φ","χ","ψ","ω"],
	// Math operators / symbols
	...["∆","∇","∂","∏","∑","√","∞","∫","∮","∴","∵","≈","≠","≤","≥","≡","⊕","⊗","⊥","∥","∧","∨","¬","∀","∃"],
	// Geometric shapes
	...["▲","△","▼","▽","◆","◇","■","□","●","○","◐","◑","◒","◓","◢","◣","◤","◥"],
	// Stars / asterisms
	...["★","☆","✦","✧","✩","✪","✫","✬","✭","✮","✯","✰"],
	// Arrows
	...["←","→","↑","↓","↔","↕","↖","↗","↘","↙","⇐","⇒","⇑","⇓","⇔","⇕","⤴","⤵","⟵","⟶","⟷"],
	// Box drawing / blocks (used by TUI)
	...["┌","┐","└","┘","├","┤","┬","┴","┼","─","│","╭","╮","╯","╰","║","═","╔","╗","╚","╝","█","▌","▐","▀","▄","░","▒","▓"],
	// Misc dingbats / typographic
	...["§","¶","‡","†","‖","¦","℘","℧","℡","℅","℆","№","℗","™","℠","℘","℞","☉","☽","☾","☿","♀","♁","♂","♃","♄","♅","♆"],
	// Card / chess
	...["♠","♣","♥","♦","♔","♕","♖","♗","♘","♙","♚","♛","♜","♝","♞","♟"],
	// Music
	...["♩","♪","♫","♬","♭","♮","♯"],
	// CJK ideographs (very rare in code, distinct visually)
	...["一","二","三","四","五","六","七","八","九","十","百","千","万"],
	...["日","月","火","水","木","金","土","山","川","田","人","名","小","大","中"],
	// Misc CJK punctuation / fullwidth
	...["「","」","『","』","【","】","〈","〉","《","》","。","、","〜","・"],
];

const m = new ModelCPU(MODEL, { contextSize: 512, threads: 4 });

interface TagEntry { char: string; tokenId: number; category?: string; }
const accepted: TagEntry[] = [];
const rejected: { char: string; nTokens: number }[] = [];
const seenTokenIds = new Set<number>();

for (const char of CANDIDATES) {
	const tokens = m.tokenize(char, /* addSpecial */ false, /* parseSpecial */ false);
	if (tokens.length === 1) {
		const tokenId = tokens[0];
		if (seenTokenIds.has(tokenId)) continue; // dedupe (multiple chars may map to same token, e.g. byte fallback)
		seenTokenIds.add(tokenId);
		accepted.push({ char, tokenId });
	} else {
		rejected.push({ char, nTokens: tokens.length });
	}
}

console.log(`Tested ${CANDIDATES.length} candidates`);
console.log(`  Accepted (single-token, unique): ${accepted.length}`);
console.log(`  Rejected (multi-token or duplicate): ${CANDIDATES.length - accepted.length}`);

// Output dir: <repo-root>/models/tag-registries/ (co-located with the code, not the model file)
const modelStem = basename(MODEL, ".gguf").toLowerCase().replace(/[^a-z0-9.-]+/g, "_");
const projectOutDir = resolve(import.meta.dirname, "..", "..", "models", "tag-registries");
mkdirSync(projectOutDir, { recursive: true });
const outFile = `${projectOutDir}/${modelStem}.tag-registry.json`;

writeFileSync(outFile, JSON.stringify({
	model: MODEL,
	modelStem,
	generatedAt: new Date().toISOString(),
	totalCandidates: CANDIDATES.length,
	accepted: accepted.length,
	rejected: rejected.length,
	notes: "Single-token chars usable as AST tag pointers (Substory 1.1). vocab is model-specific.",
	entries: accepted,
	rejectedSample: rejected.slice(0, 20),
}, null, 2));

console.log(`Wrote: ${outFile}`);

// Quick sanity: show the first 15 accepted entries
console.log("\nSample accepted entries:");
for (const e of accepted.slice(0, 15)) {
	console.log(`  '${e.char}' → ${e.tokenId}`);
}

m.free();
