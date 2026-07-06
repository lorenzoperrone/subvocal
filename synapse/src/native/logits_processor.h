#ifndef LOGITS_PROCESSOR_H
#define LOGITS_PROCESSOR_H
 
#include <vector>
#include <cstdint>
#include "llama.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

struct BiasEntry {
    llama_token token_id;
    float bias;
};

struct TokenMaskEntry {
    llama_token token_id;
    bool allowed;
};

class LogitProcessor {
public:
    LogitProcessor(int32_t vocab_size);
    ~LogitProcessor() = default;

    void SetPersistentBiases(const BiasEntry* biases, size_t count);
    void SetPendingBiases(const BiasEntry* biases, size_t count);
    void ClearAllBiases();
    void ClearPendingBiases();

    void SetSteeringVector(const float* vector, size_t size, float strength);
    void ClearSteeringVector();
    void Apply(float* logits);
    bool HasBias() const;
    bool HasSteering() const;

    void SetTokenMask(const std::vector<llama_token>& allowed);
    void ClearTokenMask();
    void ApplyTokenMask(float* logits);

    // V6.3: AST-aware logit masking
    void SetASTTokenMask(const std::vector<llama_token>& allowed);
    void ClearASTMask();
    void ApplyASTMask(float* logits);
    bool HasASTMask() const;

private:
    int32_t vocab_size_;
    std::vector<BiasEntry> persistent_biases_;
    std::vector<BiasEntry> pending_biases_;
    std::vector<float> steering_vector_;
    float steering_strength_ = 0.0f;
    bool has_steering_ = false;

    // uint8_t instead of bool so individual bytes are addressable by SIMD loads
    // (std::vector<bool> packs bits and makes &token_mask_[i] undefined behaviour).
    std::vector<uint8_t> token_mask_;
    bool has_token_mask_ = false;

    // Byte-aligned mask for SIMD-accelerated AST constraint masking.
    // uint8_t (not bool) so we can load 8 bytes at once with _mm_loadl_epi64.
    std::vector<uint8_t> ast_token_mask_;
    bool has_ast_mask_ = false;
};

#endif
