#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>

void launch_apply_biases_and_steering(
    float* d_logits,
    const int* d_bias_tokens, const float* d_bias_vals, int n_biases,
    const float* d_steering, float steer_strength,
    float temperature, int vocab_size,
    cudaStream_t stream);

void launch_apply_token_mask(float* d_logits, const bool* d_token_mask, int vocab_size, cudaStream_t stream);

#endif