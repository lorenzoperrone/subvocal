# Subvocal patches (Mac / Metal port)

Version-controlled record of the local modifications to the vendored `../llama.cpp/` checkout
that the Mac port depends on. See [INDEX.md](INDEX.md) for what's applied.

## Why this exists
The `llama.cpp/` checkout has its own git and the Subvocal changes are kept as **uncommitted
working-tree edits** there (marked with `SUBVOCAL-PATCH-NNN-BEGIN/END`). That is fragile on its
own: a `git checkout .`, a fresh clone, or an upstream re-sync wipes them. This directory is the
durable copy — each patch has:
- `NNN-<slug>.md` — what/why/compatibility/validation.
- `NNN-<slug>.patch` — a `git diff` export from the real Mac checkout, re-appliable with `git apply`.

## Adding a new patch
1. Pick the next free ID from INDEX.md ("Next available IDs"). Never reuse an ID.
2. Edit the source in `llama.cpp/` wrapped in `SUBVOCAL-PATCH-NNN-BEGIN/END` markers.
3. Export: `git -C llama.cpp diff <files> > subvocal-patches/llama-cpp-upstream/NNN-<slug>.patch`.
4. Write `NNN-<slug>.md` and add a row to INDEX.md; bump "Next available IDs".
5. Rebuild `llama.cpp/build` + `synapse/build-metal`.
