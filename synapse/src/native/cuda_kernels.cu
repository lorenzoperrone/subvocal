#include "cuda_kernels.h"
#include <cstdio>
#include <limits>

__global__ void bias_kernel(float* logits,
                            const int* tokens,
                            const float* vals,
                            int n_biases,
                            int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_biases) {
        int token = tokens[idx];
        if (token >= 0 && token < vocab_size) {
            atomicAdd(&logits[token], vals[idx]);
        }
    }
}

__global__ void steering_kernel(float* logits,
                                const float* steering,
                                float strength,
                                int vocab_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vocab_size) {
        logits[i] += steering[i] * strength;
    }
}

__global__ void temperature_kernel(float* logits,
                                   float temperature,
                                   int vocab_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vocab_size) {
        logits[i] /= temperature;
    }
}

__global__ void token_mask_kernel(float* logits,
                                  const bool* token_mask,
                                  int vocab_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vocab_size) {
        if (!token_mask[i]) {
            logits[i] = -INFINITY;
        }
    }
}

void launch_apply_biases_and_steering(
    float* d_logits,
    const int* d_bias_tokens, const float* d_bias_vals, int n_biases,
    const float* d_steering, float steer_strength,
    float temperature, int vocab_size,
    cudaStream_t stream) {

    const int block = 256;

    if (n_biases > 0) {
        int grid = (n_biases + block - 1) / block;
        bias_kernel<<<grid, block, 0, stream>>>(d_logits,
                                                d_bias_tokens,
                                                d_bias_vals,
                                                n_biases,
                                                vocab_size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "bias kernel launch error: %s\n", cudaGetErrorString(err));
            return;
        }
    }

    if (steer_strength != 0.0f && d_steering != nullptr) {
        int grid = (vocab_size + block - 1) / block;
        steering_kernel<<<grid, block, 0, stream>>>(d_logits,
                                                    d_steering,
                                                    steer_strength,
                                                    vocab_size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "steering kernel launch error: %s\n", cudaGetErrorString(err));
            return;
        }
    }

    if (temperature != 1.0f) {
        int grid = (vocab_size + block - 1) / block;
        temperature_kernel<<<grid, block, 0, stream>>>(d_logits,
                                                       temperature,
                                                       vocab_size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "temperature kernel launch error: %s\n", cudaGetErrorString(err));
        }
    }
}

void launch_apply_token_mask(float* d_logits, const bool* d_token_mask, int vocab_size, cudaStream_t stream) {
    const int block = 256;
    int grid = (vocab_size + block - 1) / block;
    token_mask_kernel<<<grid, block, 0, stream>>>(d_logits,
                                                   d_token_mask,
                                                   vocab_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "token mask kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
