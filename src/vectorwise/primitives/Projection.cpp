
#include "common/runtime/Hash.hpp"
#include "vectorwise/Operations.hpp"
#include "vectorwise/Primitives.hpp"
#include <functional>

using namespace types;
using namespace std;

namespace vectorwise {
namespace primitives {

#define MK_PROJ_COLCOL(type, op)                                               \
   F3 proj_##op##_##type##_col_##type##_col = (F3)&proj_col_col<type, op>;

#define MK_PROJ_COLVAL(type, op)                                               \
   F3 proj_##op##_##type##_col_##type##_val = (F3)&proj_col_val<type, op>;

#define MK_PROJ_SEL_BOTH_COLCOL(type, op)                                      \
   F4 proj_sel_both_##op##_##type##_col_##type##_col =                         \
       (F4)&proj_sel_both_col_col<type, op>;

#define MK_PROJ_SEL_COLCOL(type, op)                                           \
   F4 proj_##op##_sel_##type##_col_##type##_col =                              \
       (F4)&proj_sel_col_col<type, op>;
#define MK_PROJ_COL_SEL_COL(type, op)                                          \
   F4 proj_##op##_##type##_col_sel_##type##_col =                              \
       (F4)&proj_col_sel_col<type, op>;
#define MK_PROJ_SEL_COL_SEL_COL(type, op)                                      \
   F5 proj_##op##_sel_##type##_col_sel_##type##_col =                          \
       (F5)&proj_sel_col_sel_col<type, op>;

#define MK_PROJ_SEL_COLVAL(type, op)                                           \
   F4 proj_sel_##op##_##type##_col_##type##_val =                              \
       (F4)&proj_sel_col_val<type, op>;

#define MK_PROJ_VALCOL(type, op)                                               \
   F3 proj_##op##_##type##_val_##type##_col = (F3)&proj_val_col<type, op>;

#define MK_PROJ_SEL_VALCOL(type, op)                                           \
   F4 proj_sel_##op##_##type##_val_##type##_col =                              \
       (F4)&proj_sel_val_col<type, op>;

pos_t lookup_sel_(pos_t n, pos_t* RES target, pos_t* RES sel,
                  pos_t* RES source) {
   for (size_t i = 0; i < n; ++i) target[i] = source[sel[i]];
   return n;
}
F3 lookup_sel = (F3)&lookup_sel_;

template <typename int64_t> struct ExtractYear {
   Integer operator()(int64_t& d) { return types::extractYear(d); }
};
F2 apply_extract_year_col = (F2)&apply_col<Date, Integer, ExtractYear>;
F3 apply_extract_year_sel_col = (F3)&apply_sel_col<Date, Integer, ExtractYear>;

EACH_ARITH(EACH_TYPE_FULL, MK_PROJ_COLCOL)
EACH_ARITH(EACH_TYPE_FULL, MK_PROJ_COLVAL) // with second arg const
EACH_ARITH(EACH_TYPE_FULL,
           MK_PROJ_SEL_BOTH_COLCOL) // with input selection vector

EACH_ARITH(EACH_TYPE_FULL, MK_PROJ_SEL_COLCOL)
EACH_ARITH(EACH_TYPE_FULL, MK_PROJ_COL_SEL_COL)
EACH_ARITH(EACH_TYPE_FULL, MK_PROJ_SEL_COL_SEL_COL)

EACH_ARITH(EACH_TYPE_FULL,
           MK_PROJ_SEL_COLVAL) // with above and second arg const
EACH_ARITH_NON_COMM(EACH_TYPE_FULL, MK_PROJ_VALCOL)
EACH_ARITH_NON_COMM(EACH_TYPE_FULL, MK_PROJ_SEL_VALCOL)


#ifdef __AVX512F__

pos_t proj_sel8_minus_int64_t_val_int64_t_col_impl(pos_t n, pos_t* RES inSel, int64_t* RES result, int64_t* RES param1,
                                              int64_t* RES param2){
  size_t rest = n % 8;
  const auto constant = *param1;
  Vec8u consts = _mm512_set1_epi64(constant);
  for (uint64_t i = 0; i < n - rest; i += 8){
    auto idxs = _mm256_loadu_si256((const __m256i*)(inSel + i));
    Vec8u in = _mm512_i32gather_epi64(idxs, (const long long int*)param2, 8);
    auto res = consts - in;
    _mm512_store_epi64(result + i, res);
  }
  for (uint64_t i = n-rest; i < n; ++i) {
    const auto idx = inSel[i];
    result[i] = constant - param2[idx];
  }
  return n;
}
pos_t proj_sel8_plus_int64_t_col_int64_t_val_impl(pos_t n, pos_t* RES inSel, int64_t* RES result, int64_t* RES param1,
                                        int64_t* RES param2){
  size_t rest = n % 8;
  const auto constant = *param2;
  Vec8u consts = _mm512_set1_epi64(constant);
  for (uint64_t i = 0; i < n - rest; i += 8){
    auto idxs = _mm256_loadu_si256((const __m256i*)(inSel + i));
    Vec8u in = _mm512_i32gather_epi64(idxs, (const long long int*)param1, 8);
    auto res = consts + in;
    _mm512_store_epi64(result + i, res);
  }
  for (uint64_t i = n-rest; i < n; ++i) {
    const auto idx = inSel[i];
    result[i] = constant + param1[idx];
  }
  return n;
}

F4 proj_sel8_minus_int64_t_val_int64_t_col = (F4)&proj_sel8_minus_int64_t_val_int64_t_col_impl;
F4 proj_sel8_plus_int64_t_col_int64_t_val = (F4)&proj_sel8_plus_int64_t_col_int64_t_val_impl;
#endif


#ifdef __AVX512DQ__
pos_t proj8_multiplies_int64_t_col_int64_t_col_impl(pos_t n, int64_t* RES result,
                                              int64_t* RES param1, int64_t* RES param2){
  size_t rest = n % 8;
  for (uint64_t i = 0; i < n - rest; i += 8){
    Vec8u in1(param1 + i);
    Vec8u in2(param2 + i);
    auto res = in1 * in2;
    _mm512_store_epi64(result + i, res);
  }
  for (uint64_t i = n-rest; i < n; ++i) result[i] = param1[i] * param2[i];
  return n;
};
pos_t proj8_multiplies_sel_int64_t_col_int64_t_col_impl(pos_t n, pos_t* RES inSel, int64_t* RES result, int64_t* RES param1,
                                                    int64_t* RES param2){
  size_t rest = n % 8;
  for (uint64_t i = 0; i < n - rest; i += 8){
    auto idxs = _mm256_loadu_si256((const __m256i*)(inSel + i));
    Vec8u in1 = _mm512_i32gather_epi64(idxs, (const long long int*)param1, 8);
    Vec8u in2(param2 + i);
    auto res = in1 * in2;
    _mm512_store_epi64(result + i, res);
  }
  for (uint64_t i = n-rest; i < n; ++i) {
    const auto idx = inSel[i];
    result[i] = param1[idx] * param2[i];
  }
  return n;
}

F3 proj8_multiplies_int64_t_col_int64_t_col = (F3)&proj8_multiplies_int64_t_col_int64_t_col_impl;
F4 proj8_multiplies_sel_int64_t_col_int64_t_col = (F4)&proj8_multiplies_sel_int64_t_col_int64_t_col_impl;
#endif

#ifdef VW_PROJ_DENSE
// VW_PROJ_DENSE: compute every position up to the last selected one with
// contiguous loads instead of gathering through the selection. Wasted work on
// unselected rows is bounded by one vector; plus, minus and multiplies cannot
// trap, so computing them on rows the selection dropped is safe.
// Paths, widest first, each finishing what the previous left:
//   zmm  AVX-512 (vpaddq / vpsubq, vpmullq with DQ); skipped when
//        VW_PROJ_DENSE_WIDTH is 256 (AVX-512 frequency drop on Cascade Lake)
//   ymm  AVX2 (vpaddq / vpsubq); multiply with AVX-512DQ/VL vpmullq, else
//        emulated from three vpmuludq (AVX2 has no 64-bit multiply; Zen 3)
//   scalar tail, and the whole span on targets without AVX2 (ARM for now)
#if defined(__AVX512F__) && !defined(VW_PROJ_DENSE_WIDTH_256)
#define VW_PROJ_DENSE_ZMM 1
#endif
#ifdef __AVX2__
#define VW_PROJ_DENSE_YMM 1
#endif
namespace {
inline pos_t denseSpan(pos_t n, const pos_t* RES inSel) {
   return n ? inSel[n - 1] + 1 : 0;
}
#ifdef VW_PROJ_DENSE_YMM
/// low 64 bits of a * b per lane (same for signed and unsigned)
inline __m256i mullo64x4(__m256i a, __m256i b) {
#if defined(__AVX512DQ__) && defined(__AVX512VL__)
   return _mm256_mullo_epi64(a, b);
#else
   // a*b mod 2^64 = alo*blo + ((ahi*blo + alo*bhi) << 32)
   const __m256i lo = _mm256_mul_epu32(a, b);
   const __m256i cross =
       _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(a, 32), b),
                        _mm256_mul_epu32(a, _mm256_srli_epi64(b, 32)));
   return _mm256_add_epi64(lo, _mm256_slli_epi64(cross, 32));
#endif
}
#endif
} // namespace

pos_t proj_dense_minus_int64_t_val_int64_t_col_impl(pos_t n, pos_t* RES inSel,
                                                    int64_t* RES result,
                                                    int64_t* RES param1,
                                                    int64_t* RES param2) {
   const pos_t m = denseSpan(n, inSel);
   const int64_t c = *param1;
   pos_t i = 0;
#ifdef VW_PROJ_DENSE_ZMM
   const __m512i vc = _mm512_set1_epi64(c);
   for (; i + 8 <= m; i += 8)
      _mm512_storeu_si512(result + i,
                          _mm512_sub_epi64(vc, _mm512_loadu_si512(param2 + i)));
#endif
#ifdef VW_PROJ_DENSE_YMM
   const __m256i yc = _mm256_set1_epi64x(c);
   for (; i + 4 <= m; i += 4)
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(result + i),
          _mm256_sub_epi64(yc, _mm256_loadu_si256(
                                   reinterpret_cast<const __m256i*>(param2 + i))));
#endif
   for (; i < m; ++i) result[i] = c - param2[i];
   return n;
}

pos_t proj_dense_plus_int64_t_col_int64_t_val_impl(pos_t n, pos_t* RES inSel,
                                                   int64_t* RES result,
                                                   int64_t* RES param1,
                                                   int64_t* RES param2) {
   const pos_t m = denseSpan(n, inSel);
   const int64_t c = *param2;
   pos_t i = 0;
#ifdef VW_PROJ_DENSE_ZMM
   const __m512i vc = _mm512_set1_epi64(c);
   for (; i + 8 <= m; i += 8)
      _mm512_storeu_si512(result + i,
                          _mm512_add_epi64(_mm512_loadu_si512(param1 + i), vc));
#endif
#ifdef VW_PROJ_DENSE_YMM
   const __m256i yc = _mm256_set1_epi64x(c);
   for (; i + 4 <= m; i += 4)
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(result + i),
          _mm256_add_epi64(_mm256_loadu_si256(
                               reinterpret_cast<const __m256i*>(param1 + i)),
                           yc));
#endif
   for (; i < m; ++i) result[i] = param1[i] + c;
   return n;
}

pos_t proj_dense_multiplies_int64_t_col_int64_t_col_impl(
    pos_t n, pos_t* RES inSel, int64_t* RES result, int64_t* RES param1,
    int64_t* RES param2) {
   const pos_t m = denseSpan(n, inSel);
   pos_t i = 0;
#if defined(VW_PROJ_DENSE_ZMM) && defined(__AVX512DQ__)
   for (; i + 8 <= m; i += 8)
      _mm512_storeu_si512(result + i,
                          _mm512_mullo_epi64(_mm512_loadu_si512(param1 + i),
                                             _mm512_loadu_si512(param2 + i)));
#endif
#ifdef VW_PROJ_DENSE_YMM
   for (; i + 4 <= m; i += 4)
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(result + i),
          mullo64x4(
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(param1 + i)),
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(param2 + i))));
#endif
   for (; i < m; ++i) result[i] = param1[i] * param2[i];
   return n;
}

F4 proj_dense_minus_int64_t_val_int64_t_col =
    (F4)&proj_dense_minus_int64_t_val_int64_t_col_impl;
F4 proj_dense_plus_int64_t_col_int64_t_val =
    (F4)&proj_dense_plus_int64_t_col_int64_t_val_impl;
F4 proj_dense_multiplies_int64_t_col_int64_t_col =
    (F4)&proj_dense_multiplies_int64_t_col_int64_t_col_impl;
#endif
}
}
