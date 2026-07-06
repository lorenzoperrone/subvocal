#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

/**
 * Substory 1.4: AVX2-accelerated full-text scan.
 *
 * Searches for all byte-exact occurrences of `needle` in `haystack` and
 * returns the byte offsets of each match.  Used for full-text search across
 * the in-RAM codebase representation (prerequisite for Shadow FS, substory 2.3).
 *
 * Algorithm:
 *   - AVX2: broadcast needle[0] into 256-bit lane, scan haystack in 32-byte
 *     strides with _mm256_cmpeq_epi8 + movemask, verify tail bytes scalar.
 *   - Scalar fallback (no AVX2): straightforward byte-by-byte search.
 *
 * Complexity: O(n/32) typical (rare first-byte), O(n*m/32) worst case.
 */
std::vector<int32_t> scan_bytes(
    const uint8_t* haystack, size_t hay_len,
    const uint8_t* needle,   size_t needle_len);
