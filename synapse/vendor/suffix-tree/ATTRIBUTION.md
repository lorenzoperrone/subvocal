# Attribution

`suffix-tree.cpp`, `suffix-tree.h`, `log.cpp`, `log.h` are vendored verbatim (unmodified,
verified byte-identical to `main` as of 2026-07-06) from
[ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)'s `common/` directory,
MIT-licensed:

```
MIT License

Copyright (c) 2023-2024 The ggml authors (https://github.com/ggml-org/ggml/blob/master/AUTHORS)
Copyright (c) 2023-2024 The llama.cpp authors (https://github.com/ggml-org/llama.cpp/blob/master/AUTHORS)
Copyright (c) 2024-2025 The ik_llama.cpp authors (https://github.com/ikawrakow/ik_llama.cpp/blob/main/AUTHORS)
```

Vendored here (rather than requiring a full `ik_llama.cpp` clone at build time) because
Subvocal's Metal binding only ever needs these 4 files plus `../nlohmann/json.hpp` — not
ik_llama.cpp as a built engine, which Subvocal does not use on Mac.

`../nlohmann/{json.hpp,json_fwd.hpp}` are [nlohmann/json](https://github.com/nlohmann/json)
(MIT), vendored the same way inside the upstream `ik_llama.cpp/vendor/` tree originally.
