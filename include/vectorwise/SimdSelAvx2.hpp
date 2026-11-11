#pragma once
//===----------------------------------------------------------------------===//
// AVX2 selection kernels: the VW_SIMD_SEL path for x86 targets without
// AVX-512F (roquefort, Zen 3), or anywhere with VW_SIMD_SEL_ISA=avx2.
//
// AVX2 has no compress and no mask registers. Per block of 8 rows:
//   compare     vpcmpgtd / vpcmpeqd (int32, Date), two vpcmpgtq / vpcmpeqq
//               (int64); <=, >= negate the 8-bit mask
//   mask        vmovmskps (int32) or two vmovmskpd (int64): 8 bits
//   compress    kCompress8[mask] holds the positions of the set bits;
//               vpmovzxbd widens them, vpermd moves the selected row
//               positions to the front, all 8 lanes are stored and the
//               output advances by popcnt(mask)
// This is the mask pass + compress pass of the autovec addendum's
// VW_SEL_SPLIT (writeups/2026-11-03-compiler-autovec-coverage, item A4)
// fused into one pass, with no mask buffer.
//
// Store-overwrite contract, as the NEON kernels and the _bf templates: a
// block at row i stores result[found .. found+7] with found <= i, so
// nothing at or past n is written; entries between the last match and n
// may be. result may be inSel: every block loads its 8 inSel entries
// before it stores.
//
// Gathered input (selsel_*) follows VW_SIMD_SEL_GATHER: scalarload builds
// the value vector from 8 scalar loads, hwgather uses vpgatherdd /
// vpgatherqq.
//===----------------------------------------------------------------------===//
#include "common/runtime/Types.hpp"
#include "vectorwise/defs.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>

#if defined(__x86_64__) && defined(__AVX2__) && !defined(VW_POS_16)
#include <immintrin.h>
#define VW_HAVE_SIMD_SEL_AVX2 1

namespace vectorwise {
namespace primitives {
namespace simd {
namespace avx2 {

static_assert(sizeof(pos_t) == 4, "the AVX2 kernels permute 32-bit positions");

/// supported element types: int32_t and Date (32-bit lanes), int64_t
template <typename T>
constexpr bool kType32 =
    std::is_same<T, int32_t>::value || std::is_same<T, types::Date>::value;
template <typename T>
constexpr bool kType64 = std::is_same<T, int64_t>::value;

/// comparator -> (compare, negate): op(a, c) = cmp(a, c) xor negate
enum class Cmp { Eq, Gt, Lt };
template <template <typename> class Op> struct CmpOf {
   static constexpr bool ok = false;
};
template <> struct CmpOf<std::equal_to> {
   static constexpr bool ok = true;
   static constexpr Cmp cmp = Cmp::Eq;
   static constexpr bool neg = false;
};
template <> struct CmpOf<std::greater> {
   static constexpr bool ok = true;
   static constexpr Cmp cmp = Cmp::Gt;
   static constexpr bool neg = false;
};
template <> struct CmpOf<std::less> {
   static constexpr bool ok = true;
   static constexpr Cmp cmp = Cmp::Lt;
   static constexpr bool neg = false;
};
template <> struct CmpOf<std::less_equal> { // !(a > c)
   static constexpr bool ok = true;
   static constexpr Cmp cmp = Cmp::Gt;
   static constexpr bool neg = true;
};
template <> struct CmpOf<std::greater_equal> { // !(a < c)
   static constexpr bool ok = true;
   static constexpr Cmp cmp = Cmp::Lt;
   static constexpr bool neg = true;
};

template <typename T, template <typename> class Op>
constexpr bool kernel_ok = (kType32<T> || kType64<T>) && CmpOf<Op>::ok;

/// kCompress8.pos[m][j]: position of the j-th set bit of m (0 past the last)
struct Compress8Lut {
   alignas(64) uint8_t pos[256][8];
   constexpr Compress8Lut() : pos() {
      for (unsigned m = 0; m < 256; m++) {
         unsigned k = 0;
         for (unsigned b = 0; b < 8; b++)
            if (m & (1u << b)) pos[m][k++] = uint8_t(b);
      }
   }
};
inline constexpr Compress8Lut kCompress8{};

/// raw value of a supported element
template <typename T> inline auto raw(const T& v) {
   if constexpr (std::is_same<T, types::Date>::value)
      return int32_t(v.value);
   else
      return v;
}

//--- 8 values in registers: one ymm (32-bit) or two (64-bit) -------------
template <typename T> struct Vals { __m256i lo, hi; };

template <typename T> inline __m256i broadcast(const T& c) {
   if constexpr (kType64<T>)
      return _mm256_set1_epi64x(c);
   else
      return _mm256_set1_epi32(raw(c));
}

template <typename T> inline Vals<T> load8(const T* p) {
   if constexpr (kType64<T>)
      return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)),
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 4))};
   else
      return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)),
              _mm256_setzero_si256()};
}

/// p[s[0..7]] from 8 scalar loads
template <typename T> inline Vals<T> gather8_scalar(const T* p, const pos_t* s) {
   if constexpr (kType64<T>)
      return {_mm256_setr_epi64x(p[s[0]], p[s[1]], p[s[2]], p[s[3]]),
              _mm256_setr_epi64x(p[s[4]], p[s[5]], p[s[6]], p[s[7]])};
   else
      return {_mm256_setr_epi32(raw(p[s[0]]), raw(p[s[1]]), raw(p[s[2]]),
                                raw(p[s[3]]), raw(p[s[4]]), raw(p[s[5]]),
                                raw(p[s[6]]), raw(p[s[7]])),
              _mm256_setzero_si256()};
}

/// p[idx lanes] with vpgatherdd / vpgatherqq (positions are < 2^31)
template <typename T> inline Vals<T> gather8_hw(const T* p, __m256i idx) {
   if constexpr (kType64<T>) {
      const long long* b = reinterpret_cast<const long long*>(p);
      return {_mm256_i32gather_epi64(b, _mm256_castsi256_si128(idx), 8),
              _mm256_i32gather_epi64(b, _mm256_extracti128_si256(idx, 1), 8)};
   } else {
      return {_mm256_i32gather_epi32(reinterpret_cast<const int*>(p), idx, 4),
              _mm256_setzero_si256()};
   }
}

/// 8-bit mask of op(v[j], c)
template <typename T, template <typename> class Op>
inline unsigned mask8(const Vals<T>& v, __m256i c) {
   constexpr Cmp k = CmpOf<Op>::cmp;
   auto cmp = [](__m256i a, __m256i b) {
      if constexpr (kType64<T>) {
         if constexpr (k == Cmp::Eq) return _mm256_cmpeq_epi64(a, b);
         else if constexpr (k == Cmp::Gt) return _mm256_cmpgt_epi64(a, b);
         else return _mm256_cmpgt_epi64(b, a);
      } else {
         if constexpr (k == Cmp::Eq) return _mm256_cmpeq_epi32(a, b);
         else if constexpr (k == Cmp::Gt) return _mm256_cmpgt_epi32(a, b);
         else return _mm256_cmpgt_epi32(b, a);
      }
   };
   unsigned m;
   if constexpr (kType64<T>)
      m = unsigned(_mm256_movemask_pd(_mm256_castsi256_pd(cmp(v.lo, c)))) |
          unsigned(_mm256_movemask_pd(_mm256_castsi256_pd(cmp(v.hi, c))))
              << 4;
   else
      m = unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(cmp(v.lo, c))));
   if constexpr (CmpOf<Op>::neg) m ^= 0xFFu;
   return m;
}

/// store the positions of pos selected by m at out; returns popcnt(m)
inline unsigned emit(pos_t* out, unsigned m, __m256i pos) {
   const __m256i perm = _mm256_cvtepu8_epi32(
       _mm_loadl_epi64(reinterpret_cast<const __m128i*>(kCompress8.pos[m])));
   _mm256_storeu_si256(reinterpret_cast<__m256i*>(out),
                       _mm256_permutevar8x32_epi32(pos, perm));
   return unsigned(__builtin_popcount(m));
}

inline __m256i load_pos(const pos_t* s) {
   return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s));
}

//--- kernels (same signatures as the scalar templates) ----------------------

/// result = positions i with Op(param1[i], *param2)
template <typename T, template <typename> class Op>
pos_t sel_col_val(pos_t n, pos_t* RES result, T* RES param1, T* RES param2) {
   static_assert(kernel_ok<T, Op>, "unsupported type/comparator");
   const T con = *param2;
   const __m256i c = broadcast(con);
   const __m256i step = _mm256_set1_epi32(8);
   __m256i pos = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
   size_t found = 0, i = 0;
   for (; i + 8 <= n; i += 8) {
      found += emit(result + found, mask8<T, Op>(load8(param1 + i), c), pos);
      pos = _mm256_add_epi32(pos, step);
   }
   for (; i < n; ++i) {
      result[found] = pos_t(i);
      found += Op<T>()(param1[i], con);
   }
   return pos_t(found);
}

/// result = entries idx of inSel with Op(param1[idx], *param2)
template <typename T, template <typename> class Op, bool Hw>
pos_t selsel_col_val_g(pos_t n, pos_t* RES inSel, pos_t* RES result,
                       T* RES param1, T* RES param2) {
   static_assert(kernel_ok<T, Op>, "unsupported type/comparator");
   const T con = *param2;
   const __m256i c = broadcast(con);
   size_t found = 0, i = 0;
   for (; i + 8 <= n; i += 8) {
      const __m256i idx = load_pos(inSel + i);
      const Vals<T> v = Hw ? gather8_hw(param1, idx)
                           : gather8_scalar(param1, inSel + i);
      found += emit(result + found, mask8<T, Op>(v, c), idx);
   }
   for (; i < n; ++i) {
      const pos_t idx = inSel[i];
      result[found] = idx;
      found += Op<T>()(param1[idx], con);
   }
   return pos_t(found);
}

/// result = positions i (or entries of inSel) with Lo(x, *lo) && Hi(x, *hi);
/// out may be inSel
template <typename T, template <typename> class Lo, template <typename> class Hi,
          bool SelIn, bool Hw>
pos_t between_g(pos_t n, pos_t* inSel, pos_t* out, T* RES param1, T* RES lo,
                T* RES hi) {
   static_assert(kernel_ok<T, Lo> && kernel_ok<T, Hi>,
                 "unsupported type/comparator");
   const T l = *lo, h = *hi;
   const __m256i cl = broadcast(l), ch = broadcast(h);
   const __m256i step = _mm256_set1_epi32(8);
   __m256i pos = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
   size_t found = 0, i = 0;
   for (; i + 8 <= n; i += 8) {
      Vals<T> v;
      if constexpr (SelIn) {
         pos = load_pos(inSel + i);
         v = Hw ? gather8_hw(param1, pos) : gather8_scalar(param1, inSel + i);
      } else {
         v = load8(param1 + i);
      }
      const unsigned m = mask8<T, Lo>(v, cl) & mask8<T, Hi>(v, ch);
      found += emit(out + found, m, pos);
      if constexpr (!SelIn) pos = _mm256_add_epi32(pos, step);
   }
   for (; i < n; ++i) {
      const pos_t idx = SelIn ? inSel[i] : pos_t(i);
      const T x = param1[idx];
      out[found] = idx;
      found += Lo<T>()(x, l) & Hi<T>()(x, h);
   }
   return pos_t(found);
}

template <typename T, template <typename> class Lo, template <typename> class Hi>
pos_t sel_between(pos_t n, pos_t* result, T* RES param1, T* RES lo,
                  T* RES hi) {
   return between_g<T, Lo, Hi, false, false>(n, nullptr, result, param1, lo,
                                             hi);
}

/// Compile-time choice between the two gathered-input implementations.
template <typename T, template <typename> class Op>
pos_t selsel_col_val(pos_t n, pos_t* RES inSel, pos_t* RES result,
                     T* RES param1, T* RES param2) {
#ifdef VW_SIMD_SEL_GATHER_HWGATHER
   return selsel_col_val_g<T, Op, true>(n, inSel, result, param1, param2);
#else
   return selsel_col_val_g<T, Op, false>(n, inSel, result, param1, param2);
#endif
}

template <typename T, template <typename> class Lo, template <typename> class Hi>
pos_t selsel_between(pos_t n, pos_t* inSel, pos_t* result, T* RES param1,
                     T* RES lo, T* RES hi) {
#ifdef VW_SIMD_SEL_GATHER_HWGATHER
   return between_g<T, Lo, Hi, true, true>(n, inSel, result, param1, lo, hi);
#else
   return between_g<T, Lo, Hi, true, false>(n, inSel, result, param1, lo, hi);
#endif
}

} // namespace avx2
} // namespace simd
} // namespace primitives
} // namespace vectorwise

#endif // __AVX2__ && !VW_POS_16
