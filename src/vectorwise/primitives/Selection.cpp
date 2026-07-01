#include "common/runtime/Hash.hpp"
#include "common/runtime/SIMD.hpp"
#include "vectorwise/Operations.hpp"
#include "vectorwise/Primitives.hpp"
#include <functional>


#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  #include <x86intrin.h>
#else
  #define SIMDE_ENABLE_NATIVE_ALIASES
  #include <simde/x86/avx512.h>
#endif

// Native ARM SVE backend for the selection kernels. SVE provides svcompact
// (a true cross-lane compress, the analogue of AVX-512 VPCOMPRESS/COMPRESSSTORE)
// and predicated gathers, which NEON lacks -- so on SVE-capable hardware we
// compile native kernels instead of emulating AVX-512 through SIMDe.
#if defined(__ARM_FEATURE_SVE)
  #include <arm_sve.h>
#endif


#include <string.h>

using namespace types;
using namespace std;

namespace vectorwise {
namespace primitives {

#define MK_SEL_COLCOL(type, op)                                                \
   F3 sel_##op##_##type##_col_##type##_col = (F3)&sel_col_col<type, op>;

#define MK_SEL_COLVAL(type, op)                                                \
   F3 sel_##op##_##type##_col_##type##_val = (F3)&sel_col_val<type, op>;

#define MK_SEL_COLVALORVAL(type, op)                                           \
   F4 sel_##op##_##type##_col_##type##_val_or_##type##_val =                   \
       (F4)&sel_col_val_or_val<type, op>;

#define MK_SELSEL_COLCOL(type, op)                                             \
   F4 selsel_##op##_##type##_col_##type##_col = (F4)&selsel_col_col<type, op>;

#define MK_SELSEL_COLVAL(type, op)                                             \
   F4 selsel_##op##_##type##_col_##type##_val = (F4)&selsel_col_val<type, op>;

#define MK_SEL_COLCOL_BF(type, op)                                             \
   F3 sel_##op##_##type##_col_##type##_col_bf = (F3)&sel_col_col_bf<type, op>;

#define MK_SEL_COLVAL_BF(type, op)                                             \
   F3 sel_##op##_##type##_col_##type##_val_bf = (F3)&sel_col_val_bf<type, op>;

#define MK_SELSEL_COLCOL_BF(type, op)                                          \
   F4 selsel_##op##_##type##_col_##type##_col_bf =                             \
       (F4)&selsel_col_col_bf<type, op>;

#define MK_SELSEL_COLVAL_BF(type, op)                                          \
   F4 selsel_##op##_##type##_col_##type##_val_bf =                             \
       (F4)&selsel_col_val_bf<type, op>;

// NOTE: to support KNL we shadow masked compress store ---
#if defined(__AVX512F__) && !defined(__AVX512VL__)
    // Intercept the 256-bit call on KNL
    #ifndef _mm256_mask_compressstoreu_epi32
    #define _mm256_mask_compressstoreu_epi32(mem_addr, mask, a) \
        _mm512_mask_compressstoreu_epi32((mem_addr), (__mmask16)(mask), _mm512_castsi256_si512(a))
    #endif

    // If you also use the 64-bit version in Selection.cpp:
    #ifndef _mm256_mask_compressstoreu_epi64
    #define _mm256_mask_compressstoreu_epi64(mem_addr, mask, a) \
        _mm512_mask_compressstoreu_epi64((mem_addr), (__mmask8)(mask), _mm512_castsi256_si512(a))
    #endif
#endif

// instantiate selection primitives for each type and for each comparator
EACH_COMP(EACH_TYPE, MK_SEL_COLCOL)
EACH_COMP(EACH_TYPE, MK_SEL_COLVAL)      // with second arg const
EACH_COMP(EACH_TYPE, MK_SEL_COLVALORVAL) // with second and third arg const
EACH_COMP(EACH_TYPE, MK_SELSEL_COLCOL)   // with input selection vector
EACH_COMP(EACH_TYPE, MK_SELSEL_COLVAL)   // with above and second arg const
EACH_COMP(EACH_TYPE, MK_SEL_COLCOL_BF)
EACH_COMP(EACH_TYPE, MK_SEL_COLVAL_BF)    // with second arg const
EACH_COMP(EACH_TYPE, MK_SELSEL_COLCOL_BF) // with input selection vector
EACH_COMP(EACH_TYPE, MK_SELSEL_COLVAL_BF) // with above and second arg const

template <typename T> struct Contains {
   bool operator()(const T& haystack, const T& needle) {
      return memmem(haystack.value, haystack.len, needle.value, needle.len) !=
             nullptr;
   }
};
F3 sel_contains_Varchar_55_col_Varchar_55_val =
    (F3)&sel_col_val<Varchar_55, Contains>;


// ---------------------------------------------------------------------------
// Backend selection for the data-parallel (simdsel) selection kernels.
//
// The same public function-pointer symbols (the historical *_avx512 names that
// Primitives.hpp / the experiment harness expect) are bound to whichever native
// kernel matches the build target. Priority:
//
//   1. ARM SVE              -> native SVE kernels (svcompact + gather)
//   2. x86 AVX-512 / SIMDe  -> the original AVX-512 kernels
//                              (native on x86, SIMDe-emulated NEON otherwise)
//
// SVE takes priority on ARM so the experiment measures a genuine ARM kernel
// rather than AVX-512 semantics emulated on top of NEON.
// ---------------------------------------------------------------------------
#if defined(__ARM_FEATURE_SVE)
  #define SIMDSEL_BACKEND_SVE 1
#elif defined(__AVX512F__) || defined(SIMDE_X86_AVX512F_NATIVE) || defined(SIMDE_ENABLE_NATIVE_ALIASES)
  #define SIMDSEL_BACKEND_AVX512 1
#endif


#if defined(SIMDSEL_BACKEND_AVX512)
// #define PREFETCH(E) __builtin_prefetch(E);
#define PREFETCH(E)

pos_t sel_less_int32_t_col_int32_t_val_avx512_impl(pos_t n, pos_t* RES result,
                                                   int32_t* RES param1,
                                                   int32_t* RES param2) {
   static_assert(sizeof(pos_t) == 4,
                 "This implementation only supports sizeof(pos_t) == 4");
   uint64_t found = 0;
   size_t rest = n % 16;
   auto ids =
       _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
   auto con = *param2;
   auto consts = _mm512_set1_epi32(con);
   for (uint64_t i = 0; i < n - rest; i += 16) {
      Vec8u in(param1 + i);
      mask16_t less = _mm512_cmplt_epi32_mask(in, consts);
      _mm512_mask_compressstoreu_epi32(result + found, less, ids);
      found += __builtin_popcount(less);
      ids = _mm512_add_epi32(ids, _mm512_set1_epi32(16));
   }
   for (uint64_t i = n - rest; i < n; ++i)
      if (param1[i] < con) result[found++] = i;
   return found;
}

const size_t lead = 16;

pos_t selsel_greater_equal_int32_t_col_int32_t_val_avx512_impl(
    pos_t n, pos_t* RES inSel, pos_t* RES result, int32_t* RES param1,
    int32_t* RES param2) {
   static_assert(sizeof(pos_t) == 4,
                 "This implementation only supports sizeof(pos_t) == 4");

   uint64_t found = 0;
   size_t rest = n % 16;
   auto con = *param2;
   auto consts = _mm512_set1_epi32(con);
   for (uint64_t i = 0; i < n - rest; i += 16) {
      Vec8u idxs(inSel + i);
      auto in = _mm512_i32gather_epi32(idxs, param1, 4);
      PREFETCH(&param1[inSel[i + lead]]);
      mask16_t ge = _mm512_cmpge_epi32_mask(in, consts);
      _mm512_mask_compressstoreu_epi32(result + found, ge, idxs);
      found += __builtin_popcount(ge);
   }
   for (uint64_t i = n - rest; i < n; ++i) {
      const auto idx = inSel[i];
      if (param1[idx] >= con) result[found++] = idx;
   }
   return found;
}

#if defined(__AVX512L__) || defined(SIMDE_X86_AVX512L_NATIVE) || defined(SIMDE_ENABLE_NATIVE_ALIASES)
pos_t selsel_less_int64_t_col_int64_t_val_avx512_impl(pos_t n, pos_t* RES inSel,
                                                      pos_t* RES result,
                                                      int64_t* RES param1,
                                                      int64_t* RES param2) {
   static_assert(sizeof(pos_t) == 4,
                 "This implementation only supports sizeof(pos_t) == 4");

   uint64_t found = 0;
   size_t rest = n % 8;
   auto con = *param2;
   auto consts = _mm512_set1_epi64(con);
   for (uint64_t i = 0; i < n - rest; i += 8) {
      auto idxs = _mm256_loadu_si256((const __m256i*)(inSel + i));
      auto in = _mm512_i32gather_epi64(idxs, (const long long int*)param1, 8);
      PREFETCH(&param1[inSel[i + lead]]);
      mask8_t less = _mm512_cmplt_epi64_mask(in, consts);
      _mm256_mask_compressstoreu_epi32(result + found, less, idxs);
      found += __builtin_popcount(less);
   }
   for (uint64_t i = n - rest; i < n; ++i) {
      const auto idx = inSel[i];
      if (param1[idx] < con) result[found++] = idx;
   }
   return found;
}

pos_t selsel_greater_equal_int64_t_col_int64_t_val_avx512_impl(
    pos_t n, pos_t* RES inSel, pos_t* RES result, int64_t* RES param1,
    int64_t* RES param2) {
   static_assert(sizeof(pos_t) == 4,
                 "This implementation only supports sizeof(pos_t) == 4");

   uint64_t found = 0;
   size_t rest = n % 8;
   auto con = *param2;
   auto consts = _mm512_set1_epi64(con);
   for (uint64_t i = 0; i < n - rest; i += 8) {
      auto idxs = _mm256_loadu_si256((const __m256i*)(inSel + i));
      auto in = _mm512_i32gather_epi64(idxs, (const long long int*)param1, 8);
      PREFETCH(&param1[inSel[i + lead]]);
      mask8_t less = _mm512_cmpge_epi64_mask(in, consts);
      _mm256_mask_compressstoreu_epi32(result + found, less, idxs);
      found += __builtin_popcount(less);
   }
   for (uint64_t i = n - rest; i < n; ++i) {
      const auto idx = inSel[i];
      if (param1[idx] >= con) result[found++] = idx;
   }
   return found;
}

pos_t selsel_less_equal_int64_t_col_int64_t_val_avx512_impl(
    pos_t n, pos_t* RES inSel, pos_t* RES result, int64_t* RES param1,
    int64_t* RES param2) {
   static_assert(sizeof(pos_t) == 4,
                 "This implementation only supports sizeof(pos_t) == 4");

   uint64_t found = 0;
   size_t rest = n % 8;
   auto con = *param2;
   auto consts = _mm512_set1_epi64(con);
   for (uint64_t i = 0; i < n - rest; i += 8) {
      auto idxs = _mm256_loadu_si256((const __m256i*)(inSel + i));
      auto in = _mm512_i32gather_epi64(idxs, (const long long int*)param1, 8);
      PREFETCH(&param1[inSel[i + lead]]);
      mask8_t less = _mm512_cmple_epi64_mask(in, consts);
      _mm256_mask_compressstoreu_epi32(result + found, less, idxs);
      found += __builtin_popcount(less);
   }
   for (uint64_t i = n - rest; i < n; ++i) {
      const auto idx = inSel[i];
      if (param1[idx] <= con) result[found++] = idx;
   }
   return found;
}

#else

pos_t selsel_less_int64_t_col_int64_t_val_avx512_impl(pos_t n, pos_t* RES inSel,
                                                      pos_t* RES result,
                                                      int64_t* RES param1,
                                                      int64_t* RES param2) {
   static_assert(sizeof(pos_t) == 4,
                 "This implementation only supports sizeof(pos_t) == 4");

   uint64_t found = 0;
   size_t rest = n % 16;
   auto con = *param2;
   auto consts = _mm512_set1_epi64(con);
   for (uint64_t i = 0; i < n - rest; i += 16) {
      mask16_t l = 0;
      {
         auto idxs = _mm256_loadu_si256((const __m256i*)(inSel + i));
         auto in =
             _mm512_i32gather_epi64(idxs, (const long long int*)param1, 8);
         mask8_t less = _mm512_cmplt_epi64_mask(in, consts);
         l = less;
      }
      {
         auto idxs = _mm256_loadu_si256((const __m256i*)(inSel + i + 8));
         auto in =
             _mm512_i32gather_epi64(idxs, (const long long int*)param1, 8);
         mask8_t less = _mm512_cmplt_epi64_mask(in, consts);
         l |= less << 8;
      }

      _mm512_mask_compressstoreu_epi32(result + found, l,
                                       _mm512_loadu_si512(inSel + i));
      found += __builtin_popcount(l);
   }
   for (uint64_t i = n - rest; i < n; ++i) {
      const auto idx = inSel[i];
      if (param1[idx] < con) result[found++] = idx;
   }
   return found;
}

pos_t selsel_greater_equal_int64_t_col_int64_t_val_avx512_impl(
    pos_t n, pos_t* RES inSel, pos_t* RES result, int64_t* RES param1,
    int64_t* RES param2) {
   static_assert(sizeof(pos_t) == 4,
                 "This implementation only supports sizeof(pos_t) == 4");

   uint64_t found = 0;
   size_t rest = n % 16;
   auto con = *param2;
   auto consts = _mm512_set1_epi64(con);
   for (uint64_t i = 0; i < n - rest; i += 16) {
      mask16_t l;
      {
         auto idxs = _mm256_loadu_si256((const __m256i*)(inSel + i));
         auto in =
             _mm512_i32gather_epi64(idxs, (const long long int*)param1, 8);
         mask8_t less = _mm512_cmpge_epi64_mask(in, consts);
         l = less;
      }
      {
         auto idxs = _mm256_loadu_si256((const __m256i*)(inSel + i + 8));
         auto in =
             _mm512_i32gather_epi64(idxs, (const long long int*)param1, 8);
         mask16_t less = _mm512_cmpge_epi64_mask(in, consts);
         l |= less << 8;
      }
      _mm512_mask_compressstoreu_epi32(result + found, l,
                                       _mm512_loadu_si512(inSel + i));
      found += __builtin_popcount(l);
   }
   for (uint64_t i = n - rest; i < n; ++i) {
      const auto idx = inSel[i];
      if (param1[idx] >= con) result[found++] = idx;
   }
   return found;
}

pos_t selsel_less_equal_int64_t_col_int64_t_val_avx512_impl(
    pos_t n, pos_t* RES inSel, pos_t* RES result, int64_t* RES param1,
    int64_t* RES param2) {
   static_assert(sizeof(pos_t) == 4,
                 "This implementation only supports sizeof(pos_t) == 4");

   uint64_t found = 0;
   size_t rest = n % 16;
   auto con = *param2;
   auto consts = _mm512_set1_epi64(con);
   for (uint64_t i = 0; i < n - rest; i += 16) {
      mask16_t l = 0;

      {
         auto idxs = _mm256_loadu_si256((const __m256i*)(inSel + i));
         auto in =
             _mm512_i32gather_epi64(idxs, (const long long int*)param1, 8);
         mask8_t less = _mm512_cmple_epi64_mask(in, consts);
         l = less;
      }

      {
         auto idxs = _mm256_loadu_si256((const __m256i*)(inSel + i + 8));
         auto in =
             _mm512_i32gather_epi64(idxs, (const long long int*)param1, 8);
         mask16_t less = _mm512_cmple_epi64_mask(in, consts);
         l |= less << 8;
      }

      _mm512_mask_compressstoreu_epi32(result + found, l,
                                       _mm512_loadu_si512(inSel + i));
      found += __builtin_popcount(l);
   }
   for (uint64_t i = n - rest; i < n; ++i) {
      const auto idx = inSel[i];
      if (param1[idx] <= con) result[found++] = idx;
   }
   return found;
}

#endif

F3 sel_less_int32_t_col_int32_t_val_avx512 =
    (F3)&sel_less_int32_t_col_int32_t_val_avx512_impl;
F4 selsel_greater_equal_int32_t_col_int32_t_val_avx512 =
    (F4)&selsel_greater_equal_int32_t_col_int32_t_val_avx512_impl;
F4 selsel_less_int64_t_col_int64_t_val_avx512 =
    (F4)&selsel_less_int64_t_col_int64_t_val_avx512_impl;
F4 selsel_greater_equal_int64_t_col_int64_t_val_avx512 =
    (F4)&selsel_greater_equal_int64_t_col_int64_t_val_avx512_impl;
F4 selsel_less_equal_int64_t_col_int64_t_val_avx512 =
    (F4)&selsel_less_equal_int64_t_col_int64_t_val_avx512_impl;


#elif defined(SIMDSEL_BACKEND_SVE)
// ---------------------------------------------------------------------------
// ARM SVE selection kernels.
//
// Mapping from the paper's AVX-512 COMPRESSSTORE design onto SVE:
//   svcmp*_s{32,64}        -> predicate           (analogue of an AVX-512 mask)
//   svcompact_{u32,u64}    -> pack active lanes    (analogue of VPCOMPRESS)
//   svld1_gather_*         -> gather               (selection-vector inputs)
//   svcntp_b{32,64}        -> popcount of the predicate
//
// SVE is vector-length agnostic, so the loop is driven by svwhilelt, which
// yields a partial predicate on the final iteration. That removes the scalar
// remainder loop the AVX-512 kernels carry. After svcompact the matching
// elements occupy the low lanes; we store the first popcount of them.
// ---------------------------------------------------------------------------

pos_t sel_less_int32_t_col_int32_t_val_sve_impl(pos_t n, pos_t* RES result,
                                                int32_t* RES param1,
                                                int32_t* RES param2) {
   static_assert(sizeof(pos_t) == 4,
                 "This implementation only supports sizeof(pos_t) == 4");
   uint64_t found = 0;
   const int32_t con = *param2;
   const svint32_t consts = svdup_s32(con);
   const uint64_t vl = svcntw();
   for (uint64_t i = 0; i < n; i += vl) {
      svbool_t pg = svwhilelt_b32_u64(i, (uint64_t)n);
      svint32_t in = svld1_s32(pg, param1 + i);
      svbool_t lt = svcmplt_s32(pg, in, consts);
      svuint32_t ids = svindex_u32((uint32_t)i, 1);
      svuint32_t packed = svcompact_u32(lt, ids);
      uint64_t cnt = svcntp_b32(pg, lt);
      svbool_t store_pg = svwhilelt_b32_u64((uint64_t)0, cnt);
      svst1_u32(store_pg, result + found, packed);
      found += cnt;
   }
   return found;
}

pos_t selsel_greater_equal_int32_t_col_int32_t_val_sve_impl(
    pos_t n, pos_t* RES inSel, pos_t* RES result, int32_t* RES param1,
    int32_t* RES param2) {
   static_assert(sizeof(pos_t) == 4,
                 "This implementation only supports sizeof(pos_t) == 4");
   uint64_t found = 0;
   const int32_t con = *param2;
   const svint32_t consts = svdup_s32(con);
   const uint64_t vl = svcntw();
   for (uint64_t i = 0; i < n; i += vl) {
      svbool_t pg = svwhilelt_b32_u64(i, (uint64_t)n);
      svuint32_t idxs = svld1_u32(pg, inSel + i);
      svint32_t in = svld1_gather_u32index_s32(pg, param1, idxs);
      svbool_t ge = svcmpge_s32(pg, in, consts);
      svuint32_t packed = svcompact_u32(ge, idxs);
      uint64_t cnt = svcntp_b32(pg, ge);
      svbool_t store_pg = svwhilelt_b32_u64((uint64_t)0, cnt);
      svst1_u32(store_pg, result + found, packed);
      found += cnt;
   }
   return found;
}

// The three int64 selection-vector kernels differ only in the comparison.
// Indices (pos_t) stay 32-bit: we widen them to 64-bit lanes for the gather
// (svld1uw_u64), compact in 64-bit lanes, then truncate back to 32-bit on the
// store (svst1w_u64).
#define MK_SELSEL_INT64_SVE(NAME, SVCMP)                                       \
   pos_t NAME(pos_t n, pos_t* RES inSel, pos_t* RES result,                    \
              int64_t* RES param1, int64_t* RES param2) {                      \
      static_assert(sizeof(pos_t) == 4,                                        \
                    "This implementation only supports sizeof(pos_t) == 4");   \
      uint64_t found = 0;                                                      \
      const int64_t con = *param2;                                            \
      const svint64_t consts = svdup_s64(con);                                \
      const uint64_t vl = svcntd();                                           \
      for (uint64_t i = 0; i < n; i += vl) {                                  \
         svbool_t pg = svwhilelt_b64_u64(i, (uint64_t)n);                     \
         svuint64_t idxs = svld1uw_u64(pg, inSel + i);                        \
         svint64_t in = svld1_gather_u64index_s64(pg, param1, idxs);          \
         svbool_t m = SVCMP(pg, in, consts);                                  \
         svuint64_t packed = svcompact_u64(m, idxs);                          \
         uint64_t cnt = svcntp_b64(pg, m);                                    \
         svbool_t store_pg = svwhilelt_b64_u64((uint64_t)0, cnt);             \
         svst1w_u64(store_pg, result + found, packed);                        \
         found += cnt;                                                        \
      }                                                                       \
      return found;                                                           \
   }

MK_SELSEL_INT64_SVE(selsel_less_int64_t_col_int64_t_val_sve_impl, svcmplt_s64)
MK_SELSEL_INT64_SVE(selsel_greater_equal_int64_t_col_int64_t_val_sve_impl,
                    svcmpge_s64)
MK_SELSEL_INT64_SVE(selsel_less_equal_int64_t_col_int64_t_val_sve_impl,
                    svcmple_s64)

#undef MK_SELSEL_INT64_SVE

// Public symbols keep their historical *_avx512 names so Primitives.hpp and the
// experiment harness need no changes; they are bound to the SVE kernels here.
F3 sel_less_int32_t_col_int32_t_val_avx512 =
    (F3)&sel_less_int32_t_col_int32_t_val_sve_impl;
F4 selsel_greater_equal_int32_t_col_int32_t_val_avx512 =
    (F4)&selsel_greater_equal_int32_t_col_int32_t_val_sve_impl;
F4 selsel_less_int64_t_col_int64_t_val_avx512 =
    (F4)&selsel_less_int64_t_col_int64_t_val_sve_impl;
F4 selsel_greater_equal_int64_t_col_int64_t_val_avx512 =
    (F4)&selsel_greater_equal_int64_t_col_int64_t_val_sve_impl;
F4 selsel_less_equal_int64_t_col_int64_t_val_avx512 =
    (F4)&selsel_less_equal_int64_t_col_int64_t_val_sve_impl;

#endif
}
} // namespace vectorwise
