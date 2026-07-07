cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0042 NEW)

project(subvocal_ffi)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Strip the local checkout's absolute path from __FILE__/assert/debug-info strings baked into
# the compiled addon — without this, every user's own home-directory path ends up embedded in
# the shipped binary. Applies to synapse's own sources; llama.cpp's static libs get the matching
# flag from their own build invocation (see install.sh).
get_filename_component(SUBVOCAL_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
add_compile_options(-ffile-prefix-map=${SUBVOCAL_REPO_ROOT}=/subvocal)

# Backend selection — pass via -DSUBVOCAL_BACKEND=cpu|gpu|metal
if(NOT DEFINED SUBVOCAL_BACKEND)
  set(SUBVOCAL_BACKEND "cpu" CACHE STRING "Backend: cpu (ik_llama static), gpu (llama.cpp static + CUDA), or metal (llama.cpp static + Metal, Apple Silicon)")
endif()

# arm64 (Apple Silicon) has no AVX2/x86 SIMD — anything gated on this must not assume x86.
if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  set(SUBVOCAL_IS_ARM64 TRUE)
else()
  set(SUBVOCAL_IS_ARM64 FALSE)
endif()

# Common Node + N-API
include_directories(${CMAKE_JS_INC})
execute_process(
  COMMAND node -p "require('node-addon-api').include_dir"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE NODE_ADDON_API_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
include_directories(${NODE_ADDON_API_DIR})

# ---- Backend dispatch ----
if(SUBVOCAL_BACKEND STREQUAL "cpu")
  set(BACKEND_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ik_llama.cpp" CACHE PATH "ik_llama.cpp root")
  set(BACKEND_BUILD "${BACKEND_DIR}/build-static")
  set(SOURCE_FILES "src/native/binding.cpp" "src/native/logits_processor.cpp" "src/native/kv_cache.cpp" "src/native/suffix_tree_wrapper.cpp" "src/native/text_scanner.cpp")
  set(ADDON_NAME "subvocal_ffi_cpu")
  include_directories(${BACKEND_DIR}/include ${BACKEND_DIR}/ggml/include)
  set(BACKEND_LIBS
    ${BACKEND_BUILD}/src/libllama.a
    ${BACKEND_BUILD}/ggml/src/libggml.a
  )

elseif(SUBVOCAL_BACKEND STREQUAL "gpu")
  set(BACKEND_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp" CACHE PATH "llama.cpp upstream root")
  set(BACKEND_BUILD "${BACKEND_DIR}/build")
  add_definitions(-DSUBVOCAL_UPSTREAM_LLAMA)
  set(SOURCE_FILES "src/native/binding_gpu.cpp" "src/native/logits_processor.cpp" "src/native/kv_cache.cpp" "src/native/cuda_kernels.cu" "src/native/suffix_tree_wrapper.cpp" "src/native/text_scanner.cpp")
  set(ADDON_NAME "subvocal_ffi_gpu")
  # src/ (not just include/) for llama-ext.h — the "staging API" header (ctx_other/nextn
  # embeddings, used for MTP speculative decoding) isn't in the public include/ dir.
  include_directories(${BACKEND_DIR}/include ${BACKEND_DIR}/src ${BACKEND_DIR}/ggml/include)
  # Explicit core libs only — NOT the tools/server/cli/bench impls
  # Order matters: dependent libs first (linker resolves right-to-left)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  # ggml-cuda and ggml-cpu register themselves via static initializers — without
  # --whole-archive the linker drops their object files (no direct symbol refs),
  # causing "undefined symbol: ggml_backend_cuda_reg" at runtime.
  set(BACKEND_LIBS
    ${BACKEND_BUILD}/src/libllama.a
    "-Wl,--whole-archive"
    ${BACKEND_BUILD}/ggml/src/ggml-cuda/libggml-cuda.a
    ${BACKEND_BUILD}/ggml/src/libggml-cpu.a
    "-Wl,--no-whole-archive"
    ${BACKEND_BUILD}/ggml/src/libggml.a
    ${BACKEND_BUILD}/ggml/src/libggml-base.a
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    CUDA::cuda_driver
  )

elseif(SUBVOCAL_BACKEND STREQUAL "metal")
  set(BACKEND_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp" CACHE PATH "llama.cpp checkout root, built with -DGGML_METAL=ON")
  set(BACKEND_BUILD "${BACKEND_DIR}/build")
  add_definitions(-DSUBVOCAL_UPSTREAM_LLAMA)
  # Same binding source as gpu — llama.cpp's ggml-backend abstraction means the binding
  # itself is backend-agnostic (see doc/substories/M1.2-binding-metal.md for the verification
  # of that assumption). cuda_kernels.cu is CUDA-only and excluded here.
  set(SOURCE_FILES "src/native/binding_gpu.cpp" "src/native/logits_processor.cpp" "src/native/kv_cache.cpp" "src/native/suffix_tree_wrapper.cpp" "src/native/text_scanner.cpp")
  set(ADDON_NAME "subvocal_ffi_metal")
  # src/ (not just include/) for llama-ext.h — see the matching comment in the gpu branch above.
  include_directories(${BACKEND_DIR}/include ${BACKEND_DIR}/src ${BACKEND_DIR}/ggml/include)
  # ggml-cuda needs --whole-archive on the gpu branch because its self-registering static
  # initializer has no direct symbol references, so a classic-ld64-style linker drops the
  # object file entirely. Tested empirically on this toolchain (macOS 26, default linker,
  # 2026-06-28): ggml-metal's equivalent registration survives WITHOUT any force-load flag —
  # the default linker here doesn't dead-strip it. Confirmed by running (outside of Node, via
  # llama.cpp's own llama-cli binary) the exact same libggml-metal.a/libllama.a successfully.
  # If a future toolchain regresses this (symptom: "undefined symbol: ggml_backend_metal_reg"
  # at runtime, not at link time), the ld-prime equivalent to investigate is `-needed_library`
  # or re-deriving the right `-all_load`-style flag for whatever linker is in use then.
  # NOTE: libggml-blas.a is required even though nothing here calls BLAS directly.
  # ggml_backend_registry's constructor unconditionally calls register_backend(ggml_backend_blas_reg())
  # whenever GGML_USE_BLAS was defined at ggml's build time (true here: macOS + Accelerate
  # framework auto-enables it). Omitting libggml-blas.a doesn't fail at link time -- the
  # -undefined dynamic_lookup flag (needed for Node's N-API symbols) silently defers the
  # missing symbol to runtime, where it resolves to a null function pointer. Calling through
  # it crashes with EXC_BAD_ACCESS/SIGSEGV at address 0x0, surfacing inside
  # ggml_backend_registry::ggml_backend_registry() on the very first require() of the addon.
  # Found by linking the exact same .a set without -undefined dynamic_lookup in a standalone
  # test program: that build failed at proper link time with "undefined symbols:
  # _ggml_backend_blas_reg", which is what should have happened here too.
  set(BACKEND_LIBS
    ${BACKEND_BUILD}/src/libllama.a
    ${BACKEND_BUILD}/ggml/src/ggml-metal/libggml-metal.a
    ${BACKEND_BUILD}/ggml/src/ggml-blas/libggml-blas.a
    ${BACKEND_BUILD}/ggml/src/libggml-cpu.a
    ${BACKEND_BUILD}/ggml/src/libggml.a
    ${BACKEND_BUILD}/ggml/src/libggml-base.a
    "-framework Metal"
    "-framework Foundation"
    "-framework Accelerate"
    "-framework QuartzCore"
  )

else()
  message(FATAL_ERROR "Unknown SUBVOCAL_BACKEND='${SUBVOCAL_BACKEND}' (expected 'cpu', 'gpu', or 'metal')")
endif()

# Suffix-tree (V6.3 self-speculative drafting) + its logging, shared by ALL binders; nlohmann/json
# (header-only) alongside it. suffix-tree.{cpp,h} and log.{cpp,h} are unmodified upstream
# ik_llama.cpp/common files (MIT, see vendor/suffix-tree/ATTRIBUTION.md) vendored directly here so
# building any backend, including Mac's "metal", never requires cloning the full ik_llama.cpp repo
# just for these 4 files + json.hpp.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vendor/suffix-tree ${CMAKE_CURRENT_SOURCE_DIR}/vendor)
list(APPEND SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/vendor/suffix-tree/suffix-tree.cpp ${CMAKE_CURRENT_SOURCE_DIR}/vendor/suffix-tree/log.cpp)

message(STATUS "Subvocal backend: ${SUBVOCAL_BACKEND}")
message(STATUS "Addon name:        ${ADDON_NAME}")
message(STATUS "Backend dir:       ${BACKEND_DIR}")
message(STATUS "Backend libs:      ${BACKEND_LIBS}")

# ---- Build the .node ----
add_library(${ADDON_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})
set_target_properties(${ADDON_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

# src/index.ts's GPU_PATH loads build-metal/subvocal_ffi_metal.node directly, with no
# per-config subdirectory. Some CMake versions/generators nest single-config output under a
# Release/ subfolder by default even for "Unix Makefiles" — force it flat here so the addon
# always lands where the loader expects it, independent of the local CMake version.
if(SUBVOCAL_BACKEND STREQUAL "metal")
  set_target_properties(${ADDON_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}"
  )
endif()

target_link_libraries(${ADDON_NAME} PRIVATE ${BACKEND_LIBS} ${CMAKE_JS_LIB})

# Threads + OpenMP. cpu/gpu static llama builds link against libgomp and need it; llama.cpp's
# Metal build typically uses its own thread pool and doesn't require OpenMP (no libomp ships
# with macOS by default — would need `brew install libomp`), so make it optional there.
find_package(Threads REQUIRED)
if(SUBVOCAL_BACKEND STREQUAL "metal")
  find_package(OpenMP QUIET)
else()
  find_package(OpenMP REQUIRED)
endif()
# ---- libgit2 (optional — skip if not found) ----
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(LIBGIT2 libgit2)
endif()
if(LIBGIT2_FOUND)
    target_compile_definitions(${ADDON_NAME} PRIVATE SUBVOCAL_HAS_LIBGIT2)
    target_sources(${ADDON_NAME} PRIVATE src/native/git_binding.cpp)
    target_include_directories(${ADDON_NAME} PRIVATE ${LIBGIT2_INCLUDE_DIRS})
    target_link_libraries(${ADDON_NAME} PRIVATE ${LIBGIT2_LIBRARIES})
    message(STATUS "libgit2:           found — Git ops enabled")
else()
    message(STATUS "libgit2:           not found — Git ops disabled")
endif()

target_link_libraries(${ADDON_NAME} PRIVATE Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(${ADDON_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

add_definitions(-DNAPI_VERSION=8 -DNODE_ADDON_API_DISABLE_DEPRECATED)

# O3 always. AVX2 + -ffast-math for logits_processor.cpp SIMD paths and general throughput —
# Zen 3 (Ryzen 5800X) and CUDA host code both support AVX2 natively, but arm64 (Apple Silicon)
# has no AVX2: the SIMD path there goes through NEON instead (logits_processor.cpp's own
# portable fallback, or a NEON-specific path if/when one is added — see
# doc/substories/M1.2-binding-metal.md for the binding-level audit this affects).
if(SUBVOCAL_IS_ARM64)
  target_compile_options(${ADDON_NAME} PRIVATE -O3 -ffast-math)
else()
  target_compile_options(${ADDON_NAME} PRIVATE -O3 -mavx2 -ffast-math)
endif()

if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
  execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()
