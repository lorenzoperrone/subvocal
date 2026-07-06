# Third-party notices

Subvocal's own code is licensed under the MIT License (see [LICENSE](LICENSE)).

## utter/

`utter/` is a vendored, modified fork of the [pi agent harness](https://github.com/earendil-works/pi),
Copyright (c) 2025 Mario Zechner, licensed under the MIT License. The original license text is
preserved at [utter/LICENSE](utter/LICENSE) per its terms. Substantial parts of `utter/` have
been rewritten to integrate directly with Subvocal's `encode`/`synapse` inference layer in place
of the original provider abstraction — see [utter/README.md](utter/README.md).

## llama.cpp

Subvocal builds against [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT License) as a
separate, unmodified-history checkout with a small set of tracked patches applied on top — see
[subvocal-patches/](subvocal-patches/README.md). llama.cpp itself is not vendored in this
repository; it is cloned separately as a build step (see [README.md](README.md#building)).

## synapse/vendor/

`synapse/vendor/suffix-tree/{suffix-tree,log}.{cpp,h}` are vendored verbatim from
[ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)'s `common/` directory (MIT
License); `synapse/vendor/nlohmann/` is [nlohmann/json](https://github.com/nlohmann/json) (MIT
License). See [synapse/vendor/suffix-tree/ATTRIBUTION.md](synapse/vendor/suffix-tree/ATTRIBUTION.md)
for the full license text. Subvocal does not build or run ik_llama.cpp itself on Mac — only
these four files are used, for the suffix-tree speculative-decoding drafter.

## ds4 (DwarfStar 4)

Subvocal's design for SSD-backed KV caching was inspired by and partially adapts architectural patterns from [ds4](https://github.com/antirez/ds4) by Salvatore Sanfilippo (antirez), licensed under the MIT License.
