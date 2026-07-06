#include "text_scanner.h"
#include <cstring>

#ifdef __AVX2__
#include <immintrin.h>
#endif

std::vector<int32_t> scan_bytes(
    const uint8_t* haystack, size_t hay_len,
    const uint8_t* needle,   size_t needle_len)
{
    std::vector<int32_t> results;
    if (needle_len == 0 || hay_len < needle_len) return results;

#ifdef __AVX2__
    const __m256i first_vec = _mm256_set1_epi8(static_cast<char>(needle[0]));
    size_t i = 0;

    for (; i + 32 <= hay_len; i += 32) {
        __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(haystack + i));
        __m256i eq    = _mm256_cmpeq_epi8(chunk, first_vec);
        uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(eq));

        while (mask) {
            int bit    = __builtin_ctz(mask);
            size_t pos = i + static_cast<size_t>(bit);
            if (pos + needle_len <= hay_len) {
                // Verify the rest of the needle (tail bytes, scalar)
                if (needle_len == 1 ||
                    std::memcmp(haystack + pos + 1, needle + 1, needle_len - 1) == 0)
                {
                    results.push_back(static_cast<int32_t>(pos));
                }
            }
            mask &= mask - 1; // clear lowest set bit
        }
    }

    // Scalar tail for remaining bytes < 32
    for (; i + needle_len <= hay_len; ++i) {
        if (haystack[i] == needle[0] &&
            (needle_len == 1 ||
             std::memcmp(haystack + i + 1, needle + 1, needle_len - 1) == 0))
        {
            results.push_back(static_cast<int32_t>(i));
        }
    }

#else   // no AVX2 — scalar fallback
    for (size_t i = 0; i + needle_len <= hay_len; ++i) {
        if (haystack[i] == needle[0] &&
            std::memcmp(haystack + i, needle, needle_len) == 0)
        {
            results.push_back(static_cast<int32_t>(i));
        }
    }
#endif

    return results;
}
