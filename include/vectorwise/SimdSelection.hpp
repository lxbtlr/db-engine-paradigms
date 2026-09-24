#pragma once
//===----------------------------------------------------------------------===//
// AVX-512 selection kernels (CMake options VW_SIMD_SEL, VW_SIMD_SEL_CHAR).
//
// Selection is the only primitive family with compressed (variable-length)
// output, and neither gcc nor clang autovectorizes it (see
// VW_AUTOVEC_STUDY.md). These kernels do compare -> compress -> masked store,
// 16 elements per iteration, with masked tails (no scalar epilogue).
//
// Every kernel has the same signature as the scalar template it replaces in
// Primitives.hpp, so the F3/F4 casts in Selection.cpp stay valid. The pick_*
// helpers at the bottom return the SIMD kernel when the option and ISA allow
// it, and the scalar pointer otherwise.
//
// ISA requirements:
//   int32 / int64 / Date kernels : x86-64 + AVX512F
//   Char<N> == constant kernel   : additionally AVX512BW (byte masks)
// Without them nothing below is defined and pick_* return the scalar pointer.
//===----------------------------------------------------------------------===//
#include "common/runtime/Types.hpp"
#include "vectorwise/defs.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <type_traits>

#if defined(__x86_64__) && defined(__AVX512F__)
#include <immintrin.h>
#define VW_HAVE_SIMD_SEL 1
#if defined(__AVX512BW__)
#define VW_HAVE_SIMD_SEL_CHAR 1
#endif
#endif

namespace vectorwise {
namespace primitives {
namespace simd {

#ifdef VW_HAVE_SIMD_SEL

static_assert(sizeof(types::Date) == 4 &&
                  std::is_standard_layout<types::Date>::value,
              "Date must be a plain int32 to use the int32 kernels");

/// Lane width in bits for the supported element types (0 = unsupported).
template <typename T> struct Width { static constexpr int bits = 0; };
template <> struct Width<int32_t> { static constexpr int bits = 32; };
template <> struct Width<int64_t> { static constexpr int bits = 64; };
template <> struct Width<types::Date> { static constexpr int bits = 32; };

/// std:: comparator -> AVX-512 integer compare predicate (-1 = unsupported).
/// All compares are signed, which matches int32_t, int64_t and Date.
template <template <typename> class Op> struct CmpImm {
   static constexpr int imm = -1;
};
template <> struct CmpImm<std::equal_to> {
   static constexpr int imm = _MM_CMPINT_EQ;
};
template <> struct CmpImm<std::less> {
   static constexpr int imm = _MM_CMPINT_LT;
};
template <> struct CmpImm<std::less_equal> {
   static constexpr int imm = _MM_CMPINT_LE;
};
template <> struct CmpImm<std::greater_equal> {
   static constexpr int imm = _MM_CMPINT_NLT;
};
template <> struct CmpImm<std::greater> {
   static constexpr int imm = _MM_CMPINT_NLE;
};

template <typename T, template <typename> class Op>
constexpr bool kernel_ok = Width<T>::bits != 0 && CmpImm<Op>::imm >= 0;

inline int32_t raw(int32_t v) { return v; }
inline int64_t raw(int64_t v) { return v; }
inline int32_t raw(const types::Date& d) { return d.value; }

/// Mask with the low r (1..16) bits set.
inline __mmask16 tail_mask(size_t r) { return (__mmask16)((1u << r) - 1); }

inline __m512i lane_ids() {
   return _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                            15);
}

/// Compress the lanes of `ids` selected by `m` and store them at `out`.
/// Uses a register compress plus a masked store: it never writes past
/// out[popcount(m)-1], and it avoids the memory-form vpcompressd, which is
/// microcoded on AMD Zen 4. Returns the number of positions written.
inline size_t emit(pos_t* RES out, __mmask16 m, __m512i ids) {
   const unsigned cnt = __builtin_popcount((unsigned)m);
   const __m512i packed = _mm512_maskz_compress_epi32(m, ids);
   const __mmask16 st = (__mmask16)((1u << cnt) - 1);
#ifdef VW_POS_16
   _mm512_mask_cvtepi32_storeu_epi16(out, st, packed);
#else
   _mm512_mask_storeu_epi32(out, st, packed);
#endif
   return cnt;
}

/// Load 16 selection-vector entries as 32-bit lanes (r valid entries, 1..16).
inline __m512i load_idx(const pos_t* p, size_t r) {
#ifdef VW_POS_16
   if (r == 16)
      return _mm512_cvtepu16_epi32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
   alignas(32) uint16_t tmp[16] = {0};
   std::memcpy(tmp, p, r * sizeof(pos_t));
   return _mm512_cvtepu16_epi32(
       _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp)));
#else
   if (r == 16) return _mm512_loadu_si512(p);
   return _mm512_maskz_loadu_epi32(tail_mask(r), p);
#endif
}

/// Compare 16 contiguous elements starting at element i (k = valid lanes).
template <typename T, template <typename> class Op>
inline __mmask16 cmp_contig(const T* a, size_t i, __mmask16 k, __m512i c) {
   constexpr int IMM = CmpImm<Op>::imm;
   if constexpr (Width<T>::bits == 32) {
      const __m512i v = _mm512_maskz_loadu_epi32(k, a + i);
      return _mm512_mask_cmp_epi32_mask(k, v, c, IMM);
   } else {
      const __mmask8 klo = (__mmask8)k, khi = (__mmask8)(k >> 8);
      const __m512i lo = _mm512_maskz_loadu_epi64(klo, a + i);
      const __m512i hi = _mm512_maskz_loadu_epi64(khi, a + i + 8);
      const __mmask8 mlo = _mm512_mask_cmp_epi64_mask(klo, lo, c, IMM);
      const __mmask8 mhi = _mm512_mask_cmp_epi64_mask(khi, hi, c, IMM);
      return (__mmask16)(mlo | ((unsigned)mhi << 8));
   }
}

/// Same, full 16 lanes, unmasked loads (hot loop).
template <typename T, template <typename> class Op>
inline __mmask16 cmp_contig_full(const T* a, size_t i, __m512i c) {
   constexpr int IMM = CmpImm<Op>::imm;
   if constexpr (Width<T>::bits == 32) {
      const __m512i v = _mm512_loadu_si512(a + i);
      return _mm512_cmp_epi32_mask(v, c, IMM);
   } else {
      const __m512i lo = _mm512_loadu_si512(a + i);
      const __m512i hi = _mm512_loadu_si512(a + i + 8);
      const __mmask8 mlo = _mm512_cmp_epi64_mask(lo, c, IMM);
      const __mmask8 mhi = _mm512_cmp_epi64_mask(hi, c, IMM);
      return (__mmask16)(mlo | ((unsigned)mhi << 8));
   }
}

/// Gather 16 elements a[idx[j]] (k = valid lanes) as one (32-bit) or two
/// (64-bit) registers.
template <typename T>
inline void gather(const T* a, __m512i idx, __mmask16 k, __m512i& lo,
                   __m512i& hi) {
   const __m512i z = _mm512_setzero_si512();
   if constexpr (Width<T>::bits == 32) {
      lo = _mm512_mask_i32gather_epi32(z, k, idx, a, 4);
   } else {
      lo = _mm512_mask_i32gather_epi64(z, (__mmask8)k,
                                       _mm512_castsi512_si256(idx), a, 8);
      hi = _mm512_mask_i32gather_epi64(z, (__mmask8)(k >> 8),
                                       _mm512_extracti64x4_epi64(idx, 1), a, 8);
   }
}

template <typename T, template <typename> class Op>
inline __mmask16 cmp_regs(__mmask16 k, __m512i alo, __m512i ahi, __m512i blo,
                          __m512i bhi) {
   constexpr int IMM = CmpImm<Op>::imm;
   if constexpr (Width<T>::bits == 32) {
      (void)ahi;
      (void)bhi;
      return _mm512_mask_cmp_epi32_mask(k, alo, blo, IMM);
   } else {
      const __mmask8 mlo =
          _mm512_mask_cmp_epi64_mask((__mmask8)k, alo, blo, IMM);
      const __mmask8 mhi =
          _mm512_mask_cmp_epi64_mask((__mmask8)(k >> 8), ahi, bhi, IMM);
      return (__mmask16)(mlo | ((unsigned)mhi << 8));
   }
}

template <typename T> inline __m512i broadcast(const T& v) {
   if constexpr (Width<T>::bits == 32)
      return _mm512_set1_epi32(raw(v));
   else
      return _mm512_set1_epi64(raw(v));
}

//--- continuous -> compressed ------------------------------------------------

/// result = positions i with Op(param1[i], *param2)
template <typename T, template <typename> class Op>
pos_t sel_col_val(pos_t n, pos_t* RES result, T* RES param1, T* RES param2) {
   static_assert(kernel_ok<T, Op>, "unsupported type/comparator");
   const __m512i c = broadcast(*param2);
   const __m512i step = _mm512_set1_epi32(16);
   __m512i ids = lane_ids();
   size_t found = 0, i = 0;
   for (; i + 16 <= n; i += 16) {
      found += emit(result + found, cmp_contig_full<T, Op>(param1, i, c), ids);
      ids = _mm512_add_epi32(ids, step);
   }
   if (i < n) {
      const __mmask16 k = tail_mask(n - i);
      found += emit(result + found, cmp_contig<T, Op>(param1, i, k, c), ids);
   }
   return found;
}

/// result = positions i with Op(param1[i], param2[i])
template <typename T, template <typename> class Op>
pos_t sel_col_col(pos_t n, pos_t* RES result, T* RES param1, T* RES param2) {
   static_assert(kernel_ok<T, Op>, "unsupported type/comparator");
   const __m512i step = _mm512_set1_epi32(16);
   __m512i ids = lane_ids();
   size_t found = 0;
   for (size_t i = 0; i < n; i += 16) {
      const size_t r = (n - i < 16) ? n - i : 16;
      const __mmask16 k = tail_mask(r);
      __m512i alo, ahi, blo, bhi;
      if constexpr (Width<T>::bits == 32) {
         alo = _mm512_maskz_loadu_epi32(k, param1 + i);
         blo = _mm512_maskz_loadu_epi32(k, param2 + i);
         ahi = bhi = alo;
      } else {
         alo = _mm512_maskz_loadu_epi64((__mmask8)k, param1 + i);
         ahi = _mm512_maskz_loadu_epi64((__mmask8)(k >> 8), param1 + i + 8);
         blo = _mm512_maskz_loadu_epi64((__mmask8)k, param2 + i);
         bhi = _mm512_maskz_loadu_epi64((__mmask8)(k >> 8), param2 + i + 8);
      }
      found += emit(result + found, cmp_regs<T, Op>(k, alo, ahi, blo, bhi), ids);
      ids = _mm512_add_epi32(ids, step);
   }
   return found;
}

//--- gather -> compressed ----------------------------------------------------

/// result = entries idx of inSel with Op(param1[idx], *param2)
template <typename T, template <typename> class Op>
pos_t selsel_col_val(pos_t n, pos_t* RES inSel, pos_t* RES result,
                     T* RES param1, T* RES param2) {
   static_assert(kernel_ok<T, Op>, "unsupported type/comparator");
   const __m512i c = broadcast(*param2);
   size_t found = 0;
   for (size_t i = 0; i < n; i += 16) {
      const size_t r = (n - i < 16) ? n - i : 16;
      const __mmask16 k = tail_mask(r);
      const __m512i idx = load_idx(inSel + i, r);
      __m512i lo, hi = c;
      gather(param1, idx, k, lo, hi);
      found += emit(result + found, cmp_regs<T, Op>(k, lo, hi, c, c), idx);
   }
   return found;
}

/// result = entries idx of inSel with Op(param1[idx], param2[idx])
template <typename T, template <typename> class Op>
pos_t selsel_col_col(pos_t n, pos_t* RES inSel, pos_t* RES result,
                     T* RES param1, T* RES param2) {
   static_assert(kernel_ok<T, Op>, "unsupported type/comparator");
   size_t found = 0;
   for (size_t i = 0; i < n; i += 16) {
      const size_t r = (n - i < 16) ? n - i : 16;
      const __mmask16 k = tail_mask(r);
      const __m512i idx = load_idx(inSel + i, r);
      __m512i alo, ahi, blo, bhi;
      ahi = bhi = _mm512_setzero_si512();
      gather(param1, idx, k, alo, ahi);
      gather(param2, idx, k, blo, bhi);
      found += emit(result + found, cmp_regs<T, Op>(k, alo, ahi, blo, bhi), idx);
   }
   return found;
}

//--- Char<N> == constant -----------------------------------------------------
#ifdef VW_HAVE_SIMD_SEL_CHAR

template <typename T> struct CharLen { static constexpr unsigned N = 0; };
template <unsigned L> struct CharLen<types::Char<L>> {
   static constexpr unsigned N = L;
};

/// Char<1> is a specialization without a length byte; it stays scalar.
template <typename T, template <typename> class Op>
constexpr bool char_kernel_ok =
    std::is_same<Op<T>, std::equal_to<T>>::value && CharLen<T>::N > 1 &&
    CharLen<T>::N + 1 <= 64;

/// result = positions i with param1[i] == *param2, for Char<N> laid out as
/// {uint8_t len; char value[N]} (stride N+1). One masked byte load per
/// element, compared against {con.len, con.value[0..con.len)} under a mask of
/// 1+con.len bytes; this is exactly `len == len && memcmp(len) == 0`, so the
/// padding bytes beyond len never matter.
template <unsigned N>
pos_t sel_char_eq_col_val(pos_t n, pos_t* RES result,
                          types::Char<N>* RES param1,
                          types::Char<N>* RES param2) {
   using C = types::Char<N>;
   static_assert(N > 1 && N + 1 <= 64, "Char<N> must fit in one zmm");
   static_assert(sizeof(C) == N + 1, "Char<N> must be {uint8 len; char[N]}");
   static_assert(offsetof(C, value) == 1, "Char<N> length byte must be first");
   constexpr __mmask64 load = (N + 1 == 64) ? ~0ull : ((1ull << (N + 1)) - 1);

   const C& con = *param2;
   const unsigned clen = con.len; // <= N
   alignas(64) uint8_t pat[64] = {0};
   pat[0] = (uint8_t)clen;
   std::memcpy(pat + 1, con.value, clen);
   const __m512i p = _mm512_load_si512(pat);
   const __mmask64 cmpm =
       (clen + 1 >= 64) ? ~0ull : ((1ull << (clen + 1)) - 1);

   const char* base = reinterpret_cast<const char*>(param1);
   const __m512i step = _mm512_set1_epi32(16);
   __m512i ids = lane_ids();
   size_t found = 0, i = 0;
   for (; i + 16 <= n; i += 16) {
      unsigned m = 0;
      for (unsigned j = 0; j < 16; ++j) {
         const __m512i v =
             _mm512_maskz_loadu_epi8(load, base + (i + j) * (N + 1));
         m |= (unsigned)(_mm512_mask_cmpneq_epi8_mask(cmpm, v, p) == 0) << j;
      }
      found += emit(result + found, (__mmask16)m, ids);
      ids = _mm512_add_epi32(ids, step);
   }
   if (i < n) {
      unsigned m = 0;
      for (unsigned j = 0; i + j < n; ++j) {
         const __m512i v =
             _mm512_maskz_loadu_epi8(load, base + (i + j) * (N + 1));
         m |= (unsigned)(_mm512_mask_cmpneq_epi8_mask(cmpm, v, p) == 0) << j;
      }
      found += emit(result + found, (__mmask16)m, ids);
   }
   return found;
}
#endif // VW_HAVE_SIMD_SEL_CHAR

#endif // VW_HAVE_SIMD_SEL

//--- pickers used by Selection.cpp -------------------------------------------
// Each returns the SIMD kernel when its CMake option is on, the ISA is
// available and the (type, comparator) is supported; otherwise `scalar`.
// The return type is the scalar pointer type, which the SIMD kernels share.

template <typename T, template <typename> class Op> constexpr bool use_simd() {
#if defined(VW_SIMD_SEL) && defined(VW_HAVE_SIMD_SEL)
   return kernel_ok<T, Op>;
#else
   return false;
#endif
}

template <typename T, template <typename> class Op> constexpr bool use_char() {
#if defined(VW_SIMD_SEL_CHAR) && defined(VW_HAVE_SIMD_SEL_CHAR)
   return char_kernel_ok<T, Op>;
#else
   return false;
#endif
}

template <typename T, template <typename> class Op, typename Fn>
constexpr Fn pick_sel_col_val(Fn scalar) {
#if defined(VW_HAVE_SIMD_SEL)
   if constexpr (use_simd<T, Op>()) return &sel_col_val<T, Op>;
#endif
#if defined(VW_HAVE_SIMD_SEL_CHAR)
   if constexpr (use_char<T, Op>())
      return &sel_char_eq_col_val<CharLen<T>::N>;
#endif
   return scalar;
}

template <typename T, template <typename> class Op, typename Fn>
constexpr Fn pick_sel_col_col(Fn scalar) {
#if defined(VW_HAVE_SIMD_SEL)
   if constexpr (use_simd<T, Op>()) return &sel_col_col<T, Op>;
#endif
   return scalar;
}

template <typename T, template <typename> class Op, typename Fn>
constexpr Fn pick_selsel_col_val(Fn scalar) {
#if defined(VW_HAVE_SIMD_SEL)
   if constexpr (use_simd<T, Op>()) return &selsel_col_val<T, Op>;
#endif
   return scalar;
}

template <typename T, template <typename> class Op, typename Fn>
constexpr Fn pick_selsel_col_col(Fn scalar) {
#if defined(VW_HAVE_SIMD_SEL)
   if constexpr (use_simd<T, Op>()) return &selsel_col_col<T, Op>;
#endif
   return scalar;
}

} // namespace simd
} // namespace primitives
} // namespace vectorwise
