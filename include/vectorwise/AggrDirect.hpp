#pragma once
//===----------------------------------------------------------------------===//
// VW_AGGR_DIRECT kernels: grouped SUM for a vector that touches at most
// kMaxGroups groups. Each row carries its group's local id (ids[r]; ids
// >= G, e.g. kNone, belong to no group); every group's sum is kept in its own
// register accumulator, so there is no per-row store and no chain through
// memory. One compare + masked add per group and lane block.
//
// ISA, chosen at build time (CMake VW_AGGR_DIRECT_ISA):
//   avx512  8 int64 lanes, compare into a mask register, masked add
//   avx2    4 int64 lanes, compare into a lane mask, and + add
//   scalar  one row at a time, and with the 0/-1 compare result + add
// auto picks avx512 if the target has AVX-512F, else avx2 with AVX2, else
// scalar (ARM, Zen 3 without the option, ...).
//===----------------------------------------------------------------------===//
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(VW_AGGR_DIRECT_ISA_SCALAR)
#define VW_AGGR_DIRECT_SCALAR 1
#elif defined(VW_AGGR_DIRECT_ISA_AVX2)
#if !defined(__AVX2__)
#error "VW_AGGR_DIRECT_ISA=avx2 needs a target with AVX2"
#endif
#define VW_AGGR_DIRECT_AVX2 1
#elif defined(VW_AGGR_DIRECT_ISA_AVX512)
#if !defined(__AVX512F__)
#error "VW_AGGR_DIRECT_ISA=avx512 needs a target with AVX-512F"
#endif
#define VW_AGGR_DIRECT_AVX512 1
#elif defined(__x86_64__) && defined(__AVX512F__)
#define VW_AGGR_DIRECT_AVX512 1
#elif defined(__x86_64__) && defined(__AVX2__)
#define VW_AGGR_DIRECT_AVX2 1
#else
#define VW_AGGR_DIRECT_SCALAR 1
#endif

#if defined(VW_AGGR_DIRECT_AVX512) || defined(VW_AGGR_DIRECT_AVX2)
#include <immintrin.h>
#endif

namespace vectorwise {
namespace primitives {
namespace aggr_direct {

/// id of a row that belongs to no group
constexpr uint16_t kNone = 0xFFFF;
/// most groups a vector may touch for the direct path (the kernels are
/// instantiated up to 8; -DVW_AGGR_DIRECT_MAX_GROUPS=<n> lowers it, for
/// testing the fallback to the bucket path)
#ifdef VW_AGGR_DIRECT_MAX_GROUPS
constexpr unsigned kMaxGroups = VW_AGGR_DIRECT_MAX_GROUPS;
static_assert(kMaxGroups >= 1 && kMaxGroups <= 8, "1..8 groups");
#else
constexpr unsigned kMaxGroups = 8;
#endif

#if defined(VW_AGGR_DIRECT_AVX512)
constexpr const char* kIsa = "avx512";
#elif defined(VW_AGGR_DIRECT_AVX2)
constexpr const char* kIsa = "avx2";
#else
constexpr const char* kIsa = "scalar";
#endif

/// acc[g] += sum of col[r] over rows r < n with ids[r] == g; with Count also
/// cnt[g] += number of those rows, from the same compares
template <unsigned G, bool Count>
inline void sumImpl(const uint16_t* __restrict__ ids, size_t n,
                    const int64_t* __restrict__ col, int64_t* __restrict__ acc,
                    int64_t* __restrict__ cnt) {
   static_assert(G >= 1 && G <= 8, "direct path is for few groups");
   int64_t c[G] = {};
   size_t r = 0;
#if defined(VW_AGGR_DIRECT_AVX512)
   __m512i a[G];
   for (unsigned g = 0; g < G; ++g) a[g] = _mm512_setzero_si512();
   for (; r + 8 <= n; r += 8) {
      const __m512i v = _mm512_loadu_si512(col + r);
      const __m512i id = _mm512_cvtepu16_epi64(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(ids + r)));
      for (unsigned g = 0; g < G; ++g) {
         const __mmask8 m = _mm512_cmpeq_epi64_mask(id, _mm512_set1_epi64(g));
         a[g] = _mm512_mask_add_epi64(a[g], m, a[g], v);
         if constexpr (Count) c[g] += __builtin_popcount(m);
      }
   }
   for (unsigned g = 0; g < G; ++g) acc[g] += _mm512_reduce_add_epi64(a[g]);
#elif defined(VW_AGGR_DIRECT_AVX2)
   __m256i a[G];
   for (unsigned g = 0; g < G; ++g) a[g] = _mm256_setzero_si256();
   for (; r + 4 <= n; r += 4) {
      const __m256i v =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col + r));
      const __m256i id = _mm256_cvtepu16_epi64(
          _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ids + r)));
      for (unsigned g = 0; g < G; ++g) {
         const __m256i m = _mm256_cmpeq_epi64(id, _mm256_set1_epi64x(g));
         a[g] = _mm256_add_epi64(a[g], _mm256_and_si256(m, v));
         if constexpr (Count)
            c[g] += __builtin_popcount(
                _mm256_movemask_pd(_mm256_castsi256_pd(m)));
      }
   }
   for (unsigned g = 0; g < G; ++g) {
      alignas(32) int64_t l[4];
      _mm256_store_si256(reinterpret_cast<__m256i*>(l), a[g]);
      acc[g] += l[0] + l[1] + l[2] + l[3];
   }
#endif
   // scalar rows (all of them for the scalar ISA, else the tail)
   int64_t s[G] = {};
   for (; r < n; ++r) {
      const uint16_t id = ids[r];
      const int64_t v = col[r];
      for (unsigned g = 0; g < G; ++g) {
         s[g] += v & -int64_t(id == g);
         if constexpr (Count) c[g] += id == g;
      }
   }
   for (unsigned g = 0; g < G; ++g) {
      acc[g] += s[g];
      if constexpr (Count) cnt[g] += c[g];
   }
}
template <unsigned G>
inline void sum(const uint16_t* ids, size_t n, const int64_t* col,
                int64_t* acc) {
   sumImpl<G, false>(ids, n, col, acc, nullptr);
}
/// sum, plus the number of rows per group (COUNT(*) with no extra pass)
template <unsigned G>
inline void sumCount(const uint16_t* ids, size_t n, const int64_t* col,
                     int64_t* acc, int64_t* cnt) {
   sumImpl<G, true>(ids, n, col, acc, cnt);
}

/// cnt[g] += number of rows r < n with ids[r] == g
template <unsigned G>
inline void count(const uint16_t* __restrict__ ids, size_t n,
                  int64_t* __restrict__ cnt) {
   static_assert(G >= 1 && G <= 8, "direct path is for few groups");
   int64_t c[G] = {};
   size_t r = 0;
#if defined(VW_AGGR_DIRECT_AVX512)
   for (; r + 8 <= n; r += 8) {
      const __m512i id = _mm512_cvtepu16_epi64(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(ids + r)));
      for (unsigned g = 0; g < G; ++g)
         c[g] += __builtin_popcount(
             _mm512_cmpeq_epi64_mask(id, _mm512_set1_epi64(g)));
   }
#elif defined(VW_AGGR_DIRECT_AVX2)
   for (; r + 4 <= n; r += 4) {
      const __m256i id = _mm256_cvtepu16_epi64(
          _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ids + r)));
      for (unsigned g = 0; g < G; ++g)
         c[g] += __builtin_popcount(_mm256_movemask_pd(_mm256_castsi256_pd(
             _mm256_cmpeq_epi64(id, _mm256_set1_epi64x(g)))));
   }
#endif
   for (; r < n; ++r) {
      const uint16_t id = ids[r];
      for (unsigned g = 0; g < G; ++g) c[g] += id == g;
   }
   for (unsigned g = 0; g < G; ++g) cnt[g] += c[g];
}

/// sum / count dispatched on a run-time group count (1..kMaxGroups)
inline void sumN(unsigned groups, const uint16_t* ids, size_t n,
                 const int64_t* col, int64_t* acc) {
   switch (groups) {
   case 1: sum<1>(ids, n, col, acc); break;
   case 2: sum<2>(ids, n, col, acc); break;
   case 3: sum<3>(ids, n, col, acc); break;
   case 4: sum<4>(ids, n, col, acc); break;
   case 5: sum<5>(ids, n, col, acc); break;
   case 6: sum<6>(ids, n, col, acc); break;
   case 7: sum<7>(ids, n, col, acc); break;
   default: sum<8>(ids, n, col, acc); break;
   }
}
inline void sumCountN(unsigned groups, const uint16_t* ids, size_t n,
                      const int64_t* col, int64_t* acc, int64_t* cnt) {
   switch (groups) {
   case 1: sumCount<1>(ids, n, col, acc, cnt); break;
   case 2: sumCount<2>(ids, n, col, acc, cnt); break;
   case 3: sumCount<3>(ids, n, col, acc, cnt); break;
   case 4: sumCount<4>(ids, n, col, acc, cnt); break;
   case 5: sumCount<5>(ids, n, col, acc, cnt); break;
   case 6: sumCount<6>(ids, n, col, acc, cnt); break;
   case 7: sumCount<7>(ids, n, col, acc, cnt); break;
   default: sumCount<8>(ids, n, col, acc, cnt); break;
   }
}
inline void countN(unsigned groups, const uint16_t* ids, size_t n,
                   int64_t* cnt) {
   switch (groups) {
   case 1: count<1>(ids, n, cnt); break;
   case 2: count<2>(ids, n, cnt); break;
   case 3: count<3>(ids, n, cnt); break;
   case 4: count<4>(ids, n, cnt); break;
   case 5: count<5>(ids, n, cnt); break;
   case 6: count<6>(ids, n, cnt); break;
   case 7: count<7>(ids, n, cnt); break;
   default: count<8>(ids, n, cnt); break;
   }
}

} // namespace aggr_direct
} // namespace primitives
} // namespace vectorwise
