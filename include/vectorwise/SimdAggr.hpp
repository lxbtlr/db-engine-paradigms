#pragma once
//===----------------------------------------------------------------------===//
// Grouped SUM kernels for aggr_sel_col (Primitives.hpp), for benchmarking and
// later wiring into HashGroup behind CMake options.
//
// aggr_sel_col under VW_GROUP_AGGR walks, per group, the positions Lookup_T
// collected for it and keeps one scalar accumulator:
//   VW_GROUP_AGGR_SEL : value += param1[group->sel[i]]              (sum_sel)
//   otherwise         : value += param1[selParam1[group->pos[i]]]   (sum_pos)
// Without VW_GROUP_AGGR it updates one aggregate per selected row through
// entries[i] (the scatter path).
//
// Candidates:
//   B  gather: the same per-group walk with AVX-512 gathers, 8 rows per
//      vpgatherdq, two vector accumulators, one reduction per group
//      (sum_sel_gather / sum_pos_gather)
//   D  direct: no per-group lists. Given a group id per row of the vector
//      (kNoGroup for rows the selection dropped), accumulate each group from
//      the contiguous column: one load per 8 rows, one compare + masked add
//      per group (sum_direct_simd<G>), or a scalar acc[gid[r]] += col[r]
//      (sum_direct_scalar). Meant for small group counts (Q1: 4).
//
// All kernels compute the same sums as the scalar references; they add
// int64 with wraparound, which matches the references for every input whose
// true sum fits (the references would be UB on overflow).
//===----------------------------------------------------------------------===//
#include <cstddef>
#include <cstdint>

#include "vectorwise/defs.hpp"

#if defined(__x86_64__) && defined(__AVX512F__) && defined(__AVX512VL__) &&   \
    defined(__AVX512BW__)
#include <immintrin.h>
#define VW_HAVE_SIMD_AGGR 1
#endif

namespace vectorwise {
namespace primitives {
namespace simd_aggr {

/// group id of a row the selection dropped (direct aggregation)
constexpr uint8_t kNoGroup = 0xFF;

//--- scalar references (the loops aggr_sel_col runs today) ------------------

/// VW_GROUP_AGGR_SEL: sel holds row indices of one group's rows
inline int64_t sum_sel(const pos_t* sel, size_t size, const int64_t* col,
                       int64_t acc) {
   for (size_t i = 0; i < size; ++i) acc += col[sel[i]];
   return acc;
}

/// VW_GROUP_AGGR without _SEL: pos indexes the selection vector selParam1
inline int64_t sum_pos(const pos_t* pos, size_t size, const pos_t* selParam1,
                       const int64_t* col, int64_t acc) {
   for (size_t i = 0; i < size; ++i) acc += col[selParam1[pos[i]]];
   return acc;
}

/// D, scalar: acc[g] += col[r] for every row r of the vector with gid[r] == g
inline void sum_direct_scalar(const uint8_t* gid, size_t n, const int64_t* col,
                              int64_t* acc) {
   for (size_t r = 0; r < n; ++r)
      if (gid[r] != kNoGroup) acc[gid[r]] += col[r];
}

#ifdef VW_HAVE_SIMD_AGGR

/// 8 positions (the first `valid`, rest 0) widened to 32-bit lanes
inline __m256i load_idx8(const pos_t* p, __mmask8 valid) {
#ifdef VW_POS_16
   return _mm256_cvtepu16_epi32(_mm_maskz_loadu_epi16(valid, p));
#else
   return _mm256_maskz_loadu_epi32(valid, p);
#endif
}

inline __mmask8 tail_mask(size_t r) { return (__mmask8)((1u << r) - 1); }

/// B, VW_GROUP_AGGR_SEL list: gather col[sel[i]]
inline int64_t sum_sel_gather(const pos_t* sel, size_t size, const int64_t* col,
                              int64_t acc) {
   const auto* base = reinterpret_cast<const long long*>(col);
   __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
   size_t i = 0;
   for (; i + 16 <= size; i += 16) {
      a0 = _mm512_add_epi64(
          a0, _mm512_i32gather_epi64(load_idx8(sel + i, 0xFF), base, 8));
      a1 = _mm512_add_epi64(
          a1, _mm512_i32gather_epi64(load_idx8(sel + i + 8, 0xFF), base, 8));
   }
   for (; i < size; i += 8) {
      const __mmask8 m = size - i >= 8 ? (__mmask8)0xFF : tail_mask(size - i);
      a0 = _mm512_add_epi64(
          a0, _mm512_mask_i32gather_epi64(_mm512_setzero_si512(), m,
                                          load_idx8(sel + i, m), base, 8));
   }
   return acc + _mm512_reduce_add_epi64(_mm512_add_epi64(a0, a1));
}

/// row indices selParam1[pos[i..i+8)] as 32-bit lanes
inline __m256i load_pos_idx8(const pos_t* pos, const pos_t* selParam1,
                             __mmask8 valid) {
#ifdef VW_POS_16
   // no 16-bit gather: a 32-bit gather at scale 2 would read 2 bytes past the
   // last selection entry, so compose the 8 indices in scalar code
   alignas(32) uint32_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
   for (unsigned j = 0; j < 8; ++j)
      if (valid >> j & 1) idx[j] = selParam1[pos[j]];
   return _mm256_load_si256(reinterpret_cast<const __m256i*>(idx));
#else
   return _mm256_mmask_i32gather_epi32(_mm256_setzero_si256(), valid,
                                       load_idx8(pos, valid), selParam1, 4);
#endif
}

/// B, VW_GROUP_AGGR list without _SEL: gather col[selParam1[pos[i]]]
inline int64_t sum_pos_gather(const pos_t* pos, size_t size,
                              const pos_t* selParam1, const int64_t* col,
                              int64_t acc) {
   const auto* base = reinterpret_cast<const long long*>(col);
   __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
   size_t i = 0;
   for (; i + 16 <= size; i += 16) {
      a0 = _mm512_add_epi64(
          a0, _mm512_i32gather_epi64(load_pos_idx8(pos + i, selParam1, 0xFF),
                                     base, 8));
      a1 = _mm512_add_epi64(
          a1, _mm512_i32gather_epi64(
                  load_pos_idx8(pos + i + 8, selParam1, 0xFF), base, 8));
   }
   for (; i < size; i += 8) {
      const __mmask8 m = size - i >= 8 ? (__mmask8)0xFF : tail_mask(size - i);
      a0 = _mm512_add_epi64(
          a0, _mm512_mask_i32gather_epi64(_mm512_setzero_si512(), m,
                                          load_pos_idx8(pos + i, selParam1, m),
                                          base, 8));
   }
   return acc + _mm512_reduce_add_epi64(_mm512_add_epi64(a0, a1));
}

/// D, AVX-512: G groups (G <= 8), one contiguous load per 8 rows, one
/// compare + masked add per group. acc[g] += sum of col[r] with gid[r] == g.
template <unsigned G>
inline void sum_direct_simd(const uint8_t* gid, size_t n, const int64_t* col,
                            int64_t* acc) {
   static_assert(G >= 1 && G <= 8, "direct aggregation is for few groups");
   __m512i a[G];
   for (unsigned g = 0; g < G; ++g) a[g] = _mm512_setzero_si512();
   size_t r = 0;
   for (; r + 8 <= n; r += 8) {
      const __m512i v = _mm512_loadu_si512(col + r);
      const __m512i ids = _mm512_cvtepu8_epi64(
          _mm_loadl_epi64(reinterpret_cast<const __m128i*>(gid + r)));
      for (unsigned g = 0; g < G; ++g) {
         const __mmask8 m = _mm512_cmpeq_epi64_mask(ids, _mm512_set1_epi64(g));
         a[g] = _mm512_mask_add_epi64(a[g], m, a[g], v);
      }
   }
   if (r < n) {
      const __mmask8 t = tail_mask(n - r);
      const __m512i v = _mm512_maskz_loadu_epi64(t, col + r);
      // tail rows past n read as group 0 would be wrong; mask them with t
      const __m512i ids =
          _mm512_cvtepu8_epi64(_mm_maskz_loadu_epi8(t, gid + r));
      for (unsigned g = 0; g < G; ++g) {
         const __mmask8 m =
             _mm512_mask_cmpeq_epi64_mask(t, ids, _mm512_set1_epi64(g));
         a[g] = _mm512_mask_add_epi64(a[g], m, a[g], v);
      }
   }
   for (unsigned g = 0; g < G; ++g) acc[g] += _mm512_reduce_add_epi64(a[g]);
}

/// D dispatch on a runtime group count; false = too many groups for D
inline bool sum_direct_simd_n(unsigned groups, const uint8_t* gid, size_t n,
                              const int64_t* col, int64_t* acc) {
   switch (groups) {
   case 1: sum_direct_simd<1>(gid, n, col, acc); return true;
   case 2: sum_direct_simd<2>(gid, n, col, acc); return true;
   case 3: sum_direct_simd<3>(gid, n, col, acc); return true;
   case 4: sum_direct_simd<4>(gid, n, col, acc); return true;
   case 5: sum_direct_simd<5>(gid, n, col, acc); return true;
   case 6: sum_direct_simd<6>(gid, n, col, acc); return true;
   case 7: sum_direct_simd<7>(gid, n, col, acc); return true;
   case 8: sum_direct_simd<8>(gid, n, col, acc); return true;
   default: return false;
   }
}

#endif // VW_HAVE_SIMD_AGGR

} // namespace simd_aggr
} // namespace primitives
} // namespace vectorwise
