#include "logits_processor.h"
#include <algorithm>
#include <limits>

#ifdef __AVX2__
#include <immintrin.h>
#endif

LogitProcessor::LogitProcessor(int32_t vocab_size)
    : vocab_size_(vocab_size) {}

void LogitProcessor::SetPersistentBiases(const BiasEntry* biases, size_t count) {
    persistent_biases_.assign(biases, biases + count);
}

void LogitProcessor::SetPendingBiases(const BiasEntry* biases, size_t count) {
    pending_biases_.assign(biases, biases + count);
}

void LogitProcessor::ClearAllBiases() {
    persistent_biases_.clear();
    pending_biases_.clear();
}

void LogitProcessor::ClearPendingBiases() {
    pending_biases_.clear();
}

void LogitProcessor::SetSteeringVector(const float* vector, size_t size, float strength) {
    steering_vector_.assign(vector, vector + size);
    steering_strength_ = strength;
    has_steering_ = true;
}

void LogitProcessor::ClearSteeringVector() {
    steering_vector_.clear();
    steering_strength_ = 0.0f;
    has_steering_ = false;
}

bool LogitProcessor::HasBias() const {
    return !persistent_biases_.empty() || !pending_biases_.empty();
}

bool LogitProcessor::HasSteering() const {
    return has_steering_;
}

void LogitProcessor::SetTokenMask(const std::vector<llama_token>& allowed) {
    token_mask_.assign(vocab_size_, 0);
    for (size_t i = 0; i < allowed.size(); ++i) {
        if (allowed[i] >= 0 && static_cast<size_t>(allowed[i]) < static_cast<size_t>(vocab_size_)) {
            token_mask_[allowed[i]] = 1;
        }
    }
    has_token_mask_ = true;
}

void LogitProcessor::ClearTokenMask() {
    if (!token_mask_.empty()) {
        std::fill(token_mask_.begin(), token_mask_.end(), 0);
    }
    has_token_mask_ = false;
}

void LogitProcessor::ApplyTokenMask(float* logits) {
    if (!has_token_mask_ || logits == nullptr) return;

#ifdef __AVX2__
    // Load 8 uint8_t mask bytes, zero-extend to int32, compare with 0 →
    // all-ones blend mask for disallowed positions, then blend with -inf.
    // Same pattern as ApplyASTMask; token_mask_ is uint8_t so bytes are
    // individually addressable (std::vector<bool> would be UB here).
    const __m256 neg_inf = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    const __m256i zero_i  = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 8 <= static_cast<size_t>(vocab_size_); i += 8) {
        __m128i mask8  = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&token_mask_[i]));
        __m256i mask32 = _mm256_cvtepu8_epi32(mask8);
        __m256  zero_ps = _mm256_castsi256_ps(_mm256_cmpeq_epi32(mask32, zero_i));
        __m256  val    = _mm256_loadu_ps(&logits[i]);
        val = _mm256_blendv_ps(val, neg_inf, zero_ps);
        _mm256_storeu_ps(&logits[i], val);
    }
    for (; i < static_cast<size_t>(vocab_size_); ++i) {
        if (__builtin_expect(!token_mask_[i], 1))
            logits[i] = -std::numeric_limits<float>::infinity();
    }
#else
    for (size_t i = 0; i < static_cast<size_t>(vocab_size_); ++i) {
        if (__builtin_expect(!token_mask_[i], 1))
            logits[i] = -std::numeric_limits<float>::infinity();
    }
#endif
}

void LogitProcessor::SetASTTokenMask(const std::vector<llama_token>& allowed) {
    ast_token_mask_.assign(vocab_size_, 0);
    for (size_t i = 0; i < allowed.size(); ++i) {
        const llama_token t = allowed[i];
        if (t >= 0 && static_cast<size_t>(t) < static_cast<size_t>(vocab_size_)) {
            ast_token_mask_[t] = 1;
        }
    }
    has_ast_mask_ = !ast_token_mask_.empty();
}

void LogitProcessor::ClearASTMask() {
    if (!ast_token_mask_.empty()) {
        std::fill(ast_token_mask_.begin(), ast_token_mask_.end(), 0);
    }
    has_ast_mask_ = false;
}

bool LogitProcessor::HasASTMask() const {
    return has_ast_mask_;
}

void LogitProcessor::ApplyASTMask(float* logits) {
    if (!has_ast_mask_ || logits == nullptr) {
        return;
    }

#ifdef __AVX2__
    const __m256 neg_inf = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    const __m256i zero_i = _mm256_setzero_si256();
    size_t i = 0;

    // Process 8 floats per iteration with SIMD
    for (; i + 8 <= static_cast<size_t>(vocab_size_); i += 8) {
        // Load 8 byte mask entries and zero-extend to 32-bit integers
        __m128i mask8 = _mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(&ast_token_mask_[i]));
        __m256i mask32 = _mm256_cvtepu8_epi32(mask8);

        // Compare mask bytes with 0 → all-ones where disallowed
        __m256 zero_ps = _mm256_castsi256_ps(
            _mm256_cmpeq_epi32(mask32, zero_i));

        // Blend: keep original logit where allowed, -inf where disallowed
        __m256 val = _mm256_loadu_ps(&logits[i]);
        val = _mm256_blendv_ps(val, neg_inf, zero_ps);
        _mm256_storeu_ps(&logits[i], val);
    }

    // Scalar remainder for trailing entries (vocab_size_ % 8)
    for (; i < static_cast<size_t>(vocab_size_); ++i) {
        if (__builtin_expect(!ast_token_mask_[i], 1)) {
            logits[i] = -std::numeric_limits<float>::infinity();
        }
    }
#else
    // No AVX2: pure scalar with branch prediction hint
    for (size_t i = 0; i < static_cast<size_t>(vocab_size_); ++i) {
        if (__builtin_expect(!ast_token_mask_[i], 1)) {
            logits[i] = -std::numeric_limits<float>::infinity();
        }
    }
#endif
}

void LogitProcessor::Apply(float* logits) {
    for (const auto& b : persistent_biases_) {
        if (b.token_id >= 0 && b.token_id < vocab_size_) {
            logits[b.token_id] += b.bias;
        }
    }
    for (const auto& b : pending_biases_) {
        if (b.token_id >= 0 && b.token_id < vocab_size_) {
            logits[b.token_id] += b.bias;
        }
    }
    pending_biases_.clear();

    if (has_steering_) {
        const size_t sz = std::min<size_t>(steering_vector_.size(), static_cast<size_t>(vocab_size_));
#ifdef __AVX2__
        const __m256 strength256 = _mm256_set1_ps(steering_strength_);
        size_t i = 0;
        for (; i + 8 <= sz; i += 8) {
            __m256 v = _mm256_loadu_ps(&steering_vector_[i]);
            __m256 l = _mm256_loadu_ps(&logits[i]);
            __m256 scaled = _mm256_mul_ps(v, strength256);
            l = _mm256_add_ps(l, scaled);
            _mm256_storeu_ps(&logits[i], l);
        }
        for (; i < sz; ++i) {
            logits[i] += steering_vector_[i] * steering_strength_;
        }
#else
        for (size_t i = 0; i < sz; ++i) {
            logits[i] += steering_vector_[i] * steering_strength_;
        }
#endif
    }
}
