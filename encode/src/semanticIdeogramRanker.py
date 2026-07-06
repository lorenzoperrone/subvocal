"""
semanticIdeogramRanker.py

Finds the ideograms from TagRegistry.gemma4.json whose input embedding vector
is closest (cosine similarity) to a given set of concept words in Gemma 4's
own vector space.

Steps:
  1. Load token_embd.weight from the GGUF, dequantize Q4_0 → float32
  2. L2-normalize all rows (1 row = 1 token embedding)
  3. For each concept: compute avg embedding of its synonym tokens
  4. Cosine-rank all ideogram tokens against each concept
  5. Print top-N ideograms per concept

Usage:
  python3 semanticIdeogramRanker.py
"""

import json
import struct
import sys
import numpy as np
from pathlib import Path
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize

# ── Config ────────────────────────────────────────────────────────────────────

GGUF_PATH = Path(
    "/mnt/dati_cachy/LLM/lmstudio-community/"
    "unsloth-gemma-4-E2B-it-qat-GGUF/"
    "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"
)
REGISTRY_PATH = Path(__file__).parent / "TagRegistry.gemma4.json"
EMBED_CACHE   = Path(__file__).parent / "tok_embd_gemma4.npy"

TOP_N = 8  # ideograms to show per concept

# ── Concepts to map ───────────────────────────────────────────────────────────
# Each key = final label; value = list of Gemma 4 vocab words whose embeddings
# we average to build the concept centroid.
# Use common English words — they're guaranteed to tokenize as single tokens.

CONCEPTS: dict[str, list[str]] = {
    # ── Intent classes ────────────────────────────────────────────────────────
    "BUGFIX":       ["bug", "error", "fix", "patch", "crash", "broken", "Bug", "Error", "Fix"],
    "REFACTOR":     ["Ref", "ref", "cleanup", "simplify", "restructure", "refactoring", "Rewrite"],
    "EXPLAIN":      ["explain", "Explain", "describe", "understand", "document", "clarify", "Explanation"],
    "ADD_FEATURE":  ["add", "Add", "implement", "create", "feature", "new", "build", "ADD"],
    "WRITE_TEST":   ["test", "Test", "verify", "assert", "check", "validate", "testing", "spec"],
    "UNKNOWN":      ["unknown", "Unknown", "other", "misc", "unclear", "general", "UNK"],

    # ── Agentic tool tokens (EPIC 4.1) ────────────────────────────────────────
    "EDIT_AST":      ["edit", "Edit", "modify", "change", "update", "mutate", "rewrite", "write"],
    "CMD_READ":      ["read", "Read", "fetch", "get", "retrieve", "load", "open", "cat"],
    "CMD_EXEC":      ["execute", "run", "Run", "launch", "invoke", "call", "spawn", "exec"],
    "TASK_COMPLETE": ["done", "Done", "complete", "finish", "end", "success", "ready", "DONE"],
    "PAYLOAD_START": ["start", "Start", "begin", "open", "header", "block", "BEGIN"],
    "PAYLOAD_END":   ["end", "End", "close", "finish", "footer", "stop", "STOP"],
    "AFFECTED":      ["affected", "impact", "dependency", "related", "target", "Target"],
    "ERROR_SIGNAL":  ["error", "Error", "fail", "invalid", "undefined", "broken", "wrong", "ERR"],

    # ── AST node types ────────────────────────────────────────────────────────
    "FUNCTION":  ["function", "Function", "method", "procedure", "routine", "def", "func"],
    "CLASS":     ["class", "Class", "struct", "type", "object", "interface", "model"],
    "LOOP":      ["loop", "Loop", "iterate", "repeat", "cycle", "for", "while"],
    "CONDITION": ["if", "condition", "branch", "check", "guard", "predicate", "conditional"],
    "IMPORT":    ["import", "Import", "require", "include", "module", "dependency"],
    "VARIABLE":  ["variable", "Variable", "value", "data", "store", "assign", "binding", "var"],
}

# ── Tokenizer (fast, from GGUF metadata via gguf reader) ─────────────────────

def get_vocab(reader: GGUFReader) -> dict[str, int]:
    """Extract text->token_id mapping from GGUF metadata."""
    f = reader.get_field("tokenizer.ggml.tokens")
    if f is None:
        sys.exit("Could not find tokenizer.ggml.tokens in GGUF")
    vocab = {}
    for i in range(len(f.data)):
        part_idx = f.data[i]
        text = bytes(f.parts[part_idx]).decode("utf-8", errors="replace")
        vocab[text] = i
    return vocab

# ── Embedding extraction ──────────────────────────────────────────────────────

def load_embeddings(reader: GGUFReader) -> np.ndarray:
    """Load and dequantize token_embd.weight → float32 [vocab, embed_dim]."""
    if EMBED_CACHE.exists():
        print(f"Loading cached embeddings from {EMBED_CACHE} …", flush=True)
        emb = np.load(str(EMBED_CACHE))
        print(f"  shape={emb.shape} dtype={emb.dtype}", flush=True)
        return emb

    print("Extracting token_embd.weight from GGUF (Q4_0 → float32) …", flush=True)
    tensor = next(t for t in reader.tensors if t.name == "token_embd.weight")
    # tensor.data is the raw quantized bytes as a numpy uint8 array
    raw = tensor.data  # shape: (n_bytes,)

    print(f"  raw shape={raw.shape}, ggml_type={tensor.tensor_type.name}", flush=True)

    # dequantize: raw uint8 → float32 [vocab * embed_dim] (flattened)
    flat = dequantize(raw, GGMLQuantizationType.Q4_0)
    print(f"  dequantized flat shape={flat.shape}", flush=True)

    # token_embd.weight in GGUF is stored transposed: [embed_dim, vocab_size]
    embed_dim, vocab_size = tensor.shape
    emb = flat.reshape(int(embed_dim), int(vocab_size)).T.astype(np.float32)
    # → shape: [vocab_size, embed_dim]
    print(f"  final shape={emb.shape}", flush=True)

    np.save(str(EMBED_CACHE), emb)
    print(f"  cached to {EMBED_CACHE}", flush=True)
    return emb

# ── Cosine similarity ─────────────────────────────────────────────────────────

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*64}")
    print("  Semantic Ideogram Ranker — Gemma 4 E2B")
    print(f"{'='*64}\n")

    # Load registry
    registry = json.loads(REGISTRY_PATH.read_text())
    ideogram_ids  = np.array([e["tokenId"] for e in registry], dtype=np.int32)
    ideogram_chars = [e["char"] for e in registry]
    print(f"Registry: {len(registry)} single-token ideograms\n")

    # Load GGUF
    reader = GGUFReader(str(GGUF_PATH), "r")
    vocab  = get_vocab(reader)
    emb    = load_embeddings(reader)  # [vocab_size, embed_dim]

    # L2-normalize all token embeddings once
    emb_norm = l2_normalize(emb)

    # Extract ideogram embedding matrix (subset)
    ideogram_emb = emb_norm[ideogram_ids]  # [n_ideograms, embed_dim]

    results = {}

    for concept, synonyms in CONCEPTS.items():
        # Build concept centroid from synonym tokens
        vecs = []
        found = []
        missing = []
        for word in synonyms:
            # Try bare word and with leading space (SentencePiece convention)
            for candidate in [word, f"▁{word}", word.capitalize(), f"▁{word.capitalize()}"]:
                if candidate in vocab:
                    vecs.append(emb[vocab[candidate]])
                    found.append(word)
                    break
            else:
                missing.append(word)

        if not vecs:
            print(f"  {concept}: no vocab hits, skipping")
            continue

        centroid = l2_normalize(np.array(vecs).mean(axis=0, keepdims=True))[0]

        # Cosine similarity against all ideograms
        sims = ideogram_emb @ centroid  # [n_ideograms]
        top_idx = np.argsort(sims)[::-1][:TOP_N]

        results[concept] = [
            {"char": ideogram_chars[i], "tokenId": int(ideogram_ids[i]), "sim": float(sims[i])}
            for i in top_idx
        ]

        if missing:
            print(f"  [{concept}] missing from vocab: {missing}")

    # Print results
    print(f"\n{'='*64}")
    print("  RESULTS")
    print(f"{'='*64}\n")

    for concept, hits in results.items():
        bar = "  ".join(f"{h['char']}({h['sim']:.3f})" for h in hits)
        print(f"  {concept:16s} → {bar}")

    # Save JSON
    out_path = Path(__file__).parent / "ideogramMapping.gemma4.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n→ Saved to {out_path}")

if __name__ == "__main__":
    main()
