# Linux Setup & Build Guide (CPU & CUDA GPU)

[Home](Home) | [Architecture](Architecture) | [macOS Installation](Installation-and-Setup) | **Linux Setup** | [Dual-Brain](Dual-Brain-Orchestration) | [KV Tiering](KV-Cache-Tiering) | [Native Protocol](Native-Tool-Calling-Protocol) | [Discarded Explorations](Discarded-Explorations)

---

## 🐧 Overview

Subvocal natively supports **Linux x86-64** environments via two high-performance C++ N-API backends:

1. **CUDA GPU Backend (`SUBVOCAL_BACKEND=gpu`)**: Leveraging discrete NVIDIA GPUs (e.g., RTX 4070 Ti, RTX 4090, A100) via CUDA Toolkit (cuBLAS, cuBLASLt) and custom CUDA kernels (`cuda_kernels.cu`).
2. **CPU Parallel Backend (`SUBVOCAL_BACKEND=cpu`)**: Leveraging multi-core CPUs via `ik_llama.cpp` with OpenMP multi-threading and AVX2/AVX512 SIMD vectorization.

---

## 🛠️ System Prerequisites

Ensure your Linux environment has the following software installed:

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build libopenblas-dev libgomp1 nodejs npm

# Fedora / RHEL
sudo dnf install -y gcc-c++ cmake ninja-build openblas-devel libgomp nodejs npm

# Arch Linux
sudo pacman -S --needed base-devel cmake ninja openblas nodejs npm
```

### For CUDA GPU Acceleration (NVIDIA):
- **NVIDIA Driver**: Version >= 535.xx
- **NVIDIA CUDA Toolkit**: Version >= 12.0 (`nvcc --version`)

---

## ⚙️ Building the Native Engine

Subvocal includes a cross-platform build dispatcher (`scripts/build-platform.js`). Running `npm run build` will automatically detect your OS and hardware capabilities (e.g. checking for `nvidia-smi`) and compile the appropriate native backend:

```bash
# Automatic build dispatcher (detects Linux CUDA vs CPU vs Mac Metal)
npm run build
```

Alternatively, you can compile explicit target backends manually via `cmake-js`:

### Option A: Build CUDA GPU Backend (NVIDIA)

```bash
# 1. Compile CUDA C++ N-API binding
npm run build:gpu -w synapse

# 2. Compile TypeScript orchestration package
npm run build:encode

# 3. Launch Subvocal with Linux CUDA model profile
SUBVOCAL_MODEL_PROFILE=gemma4 subvocal
```

### Option B: Build High-Performance CPU Backend

```bash
# 1. Compile CPU C++ N-API binding (OpenMP + AVX2/AVX512)
npm run build:cpu -w synapse

# 2. Compile TypeScript orchestration package
npm run build:encode

# 3. Launch Subvocal
subvocal
```

---

## 🌐 Environment & Model Paths (`.env.local`)

Subvocal automatically loads environment variables from `.env.local` or `~/.config/subvocal/.env.local`. Use `SUBVOCAL_MODEL_DIR` to specify where your GGUF checkpoints reside across different machines without editing codebase paths:

```bash
# Example ~/.config/subvocal/.env.local or .env.local in project root
SUBVOCAL_MODEL_DIR=/mnt/dati_cachy/LLM/lmstudio-community
SUBVOCAL_MODEL_PROFILE=gemma4
SUBVOCAL_CONTEXT_SIZE=65536
```

---

## 💾 Memory Layout on Linux Discrete GPUs

Unlike macOS Unified Memory, Linux desktop PCs with discrete NVIDIA GPUs utilize **3 distinct physical memory tiers**:

```
┌─────────────────────────────────────────────────────────────┐
│ HOT TIER (VRAM): Active SWA KV Window (~288 MiB)            │
├─────────────────────────────────────────────────────────────┤
│ WARM TIER (System DDR5 RAM): Global Attention KV (~5.1 GiB) │
├─────────────────────────────────────────────────────────────┤
│ COLD TIER (NVMe SSD): zstd Prefill Checkpoint Store (20 GiB)│
└─────────────────────────────────────────────────────────────┘
```

- **VRAM Headroom**: By placing the global KV in system RAM (`noKvOffload: false`), a 64k-token context fits easily within a 16 GB VRAM GPU without running out of VRAM.
- **NVMe Cold Store**: Checkpoints are stored under `/mnt/cache-llm/subvocal-kv-cold` or `~/.cache/subvocal/kv-cold`.
