#pragma once
// VW_JOIN_BLOOM: the probe-side filter test, 4 hashes at a time with AVX2,
// for machines without AVX-512 (Zen 3: roquefort). The AVX-512 path in
// Operators.cpp tests 8 at a time; without either, the scalar loop tests
// one probe per iteration with a branch on the result, which costs 16-25%
// more instructions per tuple on roquefort's Q3/Q5/Q9.
//
// Same filter layout as bloomBits/bloomInsert (Operators.cpp): a key's 4 bits
// lie in one 64-bit word, word = hash bits 40.. & mask, bits = hash bits
// 0..23, 6 per bit. Survivors are written branch-free (every lane is stored,
// the output advances by its pass bit), so the result is the scalar loop's:
// the same survivors, in the same order.
//
// VW_JOIN_BLOOM_GATHER_HW: fetch the 4 filter words with vpgatherqq;
// otherwise with 4 scalar loads assembled in a register.
#include <cstdint>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace vectorwise {
namespace bloom {

inline uint64_t bits(uint64_t h) {
   return (uint64_t(1) << (h & 63)) | (uint64_t(1) << ((h >> 6) & 63)) |
          (uint64_t(1) << ((h >> 12) & 63)) | (uint64_t(1) << ((h >> 18) & 63));
}

inline bool test(const uint64_t* bloom, uint64_t mask, uint64_t h) {
   const uint64_t b = bits(h);
   return (bloom[(h >> 40) & mask] & b) == b;
}

#if defined(__AVX2__)
/// pass bits (lane l = bit l) for the 4 hashes in h
inline __attribute__((always_inline)) unsigned test4(const uint64_t* bloom,
                                                     uint64_t mask, __m256i h) {
   const __m256i idx = _mm256_and_si256(_mm256_srli_epi64(h, 40),
                                        _mm256_set1_epi64x((long long)mask));
#ifdef VW_JOIN_BLOOM_GATHER_HW
   const __m256i words =
       _mm256_i64gather_epi64(reinterpret_cast<const long long*>(bloom), idx, 8);
#else
   const __m256i words = _mm256_set_epi64x(
       (long long)bloom[_mm256_extract_epi64(idx, 3)],
       (long long)bloom[_mm256_extract_epi64(idx, 2)],
       (long long)bloom[_mm256_extract_epi64(idx, 1)],
       (long long)bloom[_mm256_extract_epi64(idx, 0)]);
#endif
   const __m256i b6 = _mm256_set1_epi64x(63);
   const __m256i one = _mm256_set1_epi64x(1);
   __m256i b = _mm256_sllv_epi64(one, _mm256_and_si256(h, b6));
   b = _mm256_or_si256(b, _mm256_sllv_epi64(one, _mm256_and_si256(_mm256_srli_epi64(h, 6), b6)));
   b = _mm256_or_si256(b, _mm256_sllv_epi64(one, _mm256_and_si256(_mm256_srli_epi64(h, 12), b6)));
   b = _mm256_or_si256(b, _mm256_sllv_epi64(one, _mm256_and_si256(_mm256_srli_epi64(h, 18), b6)));
   const __m256i pass = _mm256_cmpeq_epi64(_mm256_and_si256(words, b), b);
   return unsigned(_mm256_movemask_pd(_mm256_castsi256_pd(pass)));
}

/// filter hs[0..n) (hashes of probes base..base+n): survivors' probe
/// positions to sel, their hashes to out; returns the count. sel and out need
/// room for n + 1 entries (every lane is stored before the count advances).
template <typename Pos>
inline size_t filter4(const uint64_t* bloom, uint64_t mask, const uint64_t* hs,
                      size_t n, Pos* sel, uint64_t* out) {
   size_t m = 0, i = 0;
   for (; i + 4 <= n; i += 4) {
      const unsigned pass = test4(
          bloom, mask, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(hs + i)));
      for (unsigned l = 0; l < 4; ++l) {
         sel[m] = Pos(i + l);
         out[m] = hs[i + l];
         m += (pass >> l) & 1;
      }
   }
   for (; i < n; ++i) {
      sel[m] = Pos(i);
      out[m] = hs[i];
      m += test(bloom, mask, hs[i]);
   }
   return m;
}
#endif

} // namespace bloom
} // namespace vectorwise
