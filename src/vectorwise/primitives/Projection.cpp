#include "common/runtime/Hash.hpp"
#include "vectorwise/Operations.hpp"
#include "vectorwise/Primitives.hpp"
#include <functional>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  #include <x86intrin.h>
#else
  #define SIMDE_ENABLE_NATIVE_ALIASES
  #include <simde/x86/avx512.h>
  // On AArch64 keep SIMDe available for the rest of the TU, but route the
  // projection kernels below to native ARM SIMD instead of emulated AVX-512.
  // Priority: SVE2 (hardware gather + native 64-bit multiply) > NEON.
  #if defined(__ARM_FEATURE_SVE2)
    #include <arm_sve.h>
    #define VW_TARGET_SVE2 1
  #elif defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define VW_TARGET_NEON 1
  #endif
#endif

// Any native ARM SIMD path owns the projection symbols, so the x86/SIMDe
// kernels must be excluded on those targets.
#if defined(VW_TARGET_SVE2) || defined(VW_TARGET_NEON)
  #define VW_TARGET_ARM 1
#endif

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

pos_t lookup_sel_(pos_t n, pos_t* target, pos_t* sel, pos_t* source) {
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

// ===========================================================================
// Hand-vectorized projection kernels.
//
// Architecture dispatch (macro expansion):
//   * x86 with AVX-512 (or SIMDe emulating it on a non-ARM host) -> AVX-512.
//   * AArch64 with SVE2                                          -> native SVE2.
//   * AArch64 / ARM NEON (no SVE2)                               -> native NEON.
// The exported F3/F4 symbols are identical on every target, so the rest of the
// engine links unchanged; only the implementation behind them is swapped.
// ===========================================================================

// ---------------------------------------------------------------------------
// x86 AVX-512 path (also taken by SIMDe on non-ARM hosts).
// `&& !defined(VW_TARGET_ARM)` ensures AArch64 does NOT compile these, so the
// native SVE2/NEON definitions below own the symbols there.
// ---------------------------------------------------------------------------
#if (defined(__AVX512F__) || defined(SIMDE_X86_AVX512F_NATIVE) ||              \
     defined(SIMDE_ENABLE_NATIVE_ALIASES)) &&                                  \
    !defined(VW_TARGET_ARM)

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


#if (defined(__AVX512Q__) || defined(SIMDE_X86_AVX512Q_NATIVE) ||             \
     defined(SIMDE_ENABLE_NATIVE_ALIASES)) &&                                 \
    !defined(VW_TARGET_ARM)
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


// ---------------------------------------------------------------------------
// Native ARM NEON path.
//
// NEON registers are 128-bit -> 2x int64 per vector (vs. 8x in AVX-512). Two
// hardware gaps shape these kernels and are themselves part of what the
// experiment measures:
//   (1) No gather. Lanes touched through a selection vector are filled with
//       scalar loads, then the arithmetic is vectorized.
//   (2) No 64-bit SIMD integer multiply (the AVX-512 analogue, vpmullq, needs
//       the DQ extension). The low 64 bits of the product are emulated with
//       three 32x32->64 widening multiplies via vmull_u32.
// ---------------------------------------------------------------------------
#if defined(VW_TARGET_NEON)

// low 64 bits of a*b, lane-wise, for int64x2_t.
// a*b mod 2^64 = al*bl + ((ah*bl + al*bh) << 32) mod 2^64
static inline int64x2_t vw_neon_mullo_s64(int64x2_t a, int64x2_t b) {
  const uint64x2_t ua = vreinterpretq_u64_s64(a);
  const uint64x2_t ub = vreinterpretq_u64_s64(b);
  const uint32x2_t al = vmovn_u64(ua);        // low 32 of each lane
  const uint32x2_t bl = vmovn_u64(ub);
  const uint32x2_t ah = vshrn_n_u64(ua, 32);  // high 32 of each lane
  const uint32x2_t bh = vshrn_n_u64(ub, 32);
  const uint64x2_t lo  = vmull_u32(al, bl);
  const uint64x2_t mid = vaddq_u64(vmull_u32(ah, bl), vmull_u32(al, bh));
  const uint64x2_t res = vaddq_u64(lo, vshlq_n_u64(mid, 32));
  return vreinterpretq_s64_u64(res);
}

pos_t proj_sel8_minus_int64_t_val_int64_t_col_impl(pos_t n, pos_t* RES inSel, int64_t* RES result, int64_t* RES param1,
                                              int64_t* RES param2){
  size_t rest = n % 2;
  const int64_t constant = *param1;
  const int64x2_t consts = vdupq_n_s64(constant);
  for (uint64_t i = 0; i < n - rest; i += 2){
    // scalar gather -> NEON lanes (no native gather on NEON)
    const int64x2_t in = {param2[inSel[i]], param2[inSel[i + 1]]};
    vst1q_s64(result + i, vsubq_s64(consts, in));
  }
  for (uint64_t i = n - rest; i < n; ++i) {
    const auto idx = inSel[i];
    result[i] = constant - param2[idx];
  }
  return n;
}
pos_t proj_sel8_plus_int64_t_col_int64_t_val_impl(pos_t n, pos_t* RES inSel, int64_t* RES result, int64_t* RES param1,
                                        int64_t* RES param2){
  size_t rest = n % 2;
  const int64_t constant = *param2;
  const int64x2_t consts = vdupq_n_s64(constant);
  for (uint64_t i = 0; i < n - rest; i += 2){
    const int64x2_t in = {param1[inSel[i]], param1[inSel[i + 1]]};
    vst1q_s64(result + i, vaddq_s64(consts, in));
  }
  for (uint64_t i = n - rest; i < n; ++i) {
    const auto idx = inSel[i];
    result[i] = constant + param1[idx];
  }
  return n;
}

pos_t proj8_multiplies_int64_t_col_int64_t_col_impl(pos_t n, int64_t* RES result,
                                              int64_t* RES param1, int64_t* RES param2){
  size_t rest = n % 2;
  for (uint64_t i = 0; i < n - rest; i += 2){
    const int64x2_t in1 = vld1q_s64(param1 + i);
    const int64x2_t in2 = vld1q_s64(param2 + i);
    vst1q_s64(result + i, vw_neon_mullo_s64(in1, in2));
  }
  for (uint64_t i = n - rest; i < n; ++i) result[i] = param1[i] * param2[i];
  return n;
}
pos_t proj8_multiplies_sel_int64_t_col_int64_t_col_impl(pos_t n, pos_t* RES inSel, int64_t* RES result, int64_t* RES param1,
                                                    int64_t* RES param2){
  size_t rest = n % 2;
  for (uint64_t i = 0; i < n - rest; i += 2){
    const int64x2_t in1 = {param1[inSel[i]], param1[inSel[i + 1]]}; // scalar gather
    const int64x2_t in2 = vld1q_s64(param2 + i);
    vst1q_s64(result + i, vw_neon_mullo_s64(in1, in2));
  }
  for (uint64_t i = n - rest; i < n; ++i) {
    const auto idx = inSel[i];
    result[i] = param1[idx] * param2[i];
  }
  return n;
}

F4 proj_sel8_minus_int64_t_val_int64_t_col = (F4)&proj_sel8_minus_int64_t_val_int64_t_col_impl;
F4 proj_sel8_plus_int64_t_col_int64_t_val = (F4)&proj_sel8_plus_int64_t_col_int64_t_val_impl;
F3 proj8_multiplies_int64_t_col_int64_t_col = (F3)&proj8_multiplies_int64_t_col_int64_t_col_impl;
F4 proj8_multiplies_sel_int64_t_col_int64_t_col = (F4)&proj8_multiplies_sel_int64_t_col_int64_t_col_impl;
#endif


// ---------------------------------------------------------------------------
// Native ARM SVE2 path (priority over NEON when the target supports it).
//
// SVE closes both NEON gaps for these kernels:
//   * Hardware gather (svld1_gather_*index) -> the real analogue of AVX-512's
//     i32gather, replacing NEON's scalar lane fills.
//   * Native 64-bit lane multiply (svmul_s64_x) -> no 32x32 emulation needed.
//
// Vector-length agnostic: svcntd() lanes per step and a svwhilelt_b64
// predicate folds the tail in, so there is no scalar remainder loop.
//
// The selection vector holds 32-bit positions (matching the AVX-512 i32gather
// path); svld1uw_u64 loads and zero-extends them to 64-bit gather offsets.
// ---------------------------------------------------------------------------
#if defined(VW_TARGET_SVE2)

pos_t proj_sel8_minus_int64_t_val_int64_t_col_impl(pos_t n, pos_t* RES inSel, int64_t* RES result, int64_t* RES param1,
                                              int64_t* RES param2){
  const svint64_t consts = svdup_n_s64(*param1);
  const uint64_t vl = svcntd();
  for (uint64_t i = 0; i < n; i += vl) {
    const svbool_t pg = svwhilelt_b64_u64(i, (uint64_t)n);
    const svuint64_t idx = svld1uw_u64(pg, (const uint32_t*)(inSel + i));
    const svint64_t in = svld1_gather_u64index_s64(pg, param2, idx);
    svst1_s64(pg, result + i, svsub_s64_x(pg, consts, in));
  }
  return n;
}
pos_t proj_sel8_plus_int64_t_col_int64_t_val_impl(pos_t n, pos_t* RES inSel, int64_t* RES result, int64_t* RES param1,
                                        int64_t* RES param2){
  const svint64_t consts = svdup_n_s64(*param2);
  const uint64_t vl = svcntd();
  for (uint64_t i = 0; i < n; i += vl) {
    const svbool_t pg = svwhilelt_b64_u64(i, (uint64_t)n);
    const svuint64_t idx = svld1uw_u64(pg, (const uint32_t*)(inSel + i));
    const svint64_t in = svld1_gather_u64index_s64(pg, param1, idx);
    svst1_s64(pg, result + i, svadd_s64_x(pg, consts, in));
  }
  return n;
}

pos_t proj8_multiplies_int64_t_col_int64_t_col_impl(pos_t n, int64_t* RES result,
                                              int64_t* RES param1, int64_t* RES param2){
  const uint64_t vl = svcntd();
  for (uint64_t i = 0; i < n; i += vl) {
    const svbool_t pg = svwhilelt_b64_u64(i, (uint64_t)n);
    const svint64_t a = svld1_s64(pg, param1 + i);
    const svint64_t b = svld1_s64(pg, param2 + i);
    svst1_s64(pg, result + i, svmul_s64_x(pg, a, b));
  }
  return n;
}
pos_t proj8_multiplies_sel_int64_t_col_int64_t_col_impl(pos_t n, pos_t* RES inSel, int64_t* RES result, int64_t* RES param1,
                                                    int64_t* RES param2){
  const uint64_t vl = svcntd();
  for (uint64_t i = 0; i < n; i += vl) {
    const svbool_t pg = svwhilelt_b64_u64(i, (uint64_t)n);
    const svuint64_t idx = svld1uw_u64(pg, (const uint32_t*)(inSel + i));
    const svint64_t a = svld1_gather_u64index_s64(pg, param1, idx);
    const svint64_t b = svld1_s64(pg, param2 + i);
    svst1_s64(pg, result + i, svmul_s64_x(pg, a, b));
  }
  return n;
}

F4 proj_sel8_minus_int64_t_val_int64_t_col = (F4)&proj_sel8_minus_int64_t_val_int64_t_col_impl;
F4 proj_sel8_plus_int64_t_col_int64_t_val = (F4)&proj_sel8_plus_int64_t_col_int64_t_val_impl;
F3 proj8_multiplies_int64_t_col_int64_t_col = (F3)&proj8_multiplies_int64_t_col_int64_t_col_impl;
F4 proj8_multiplies_sel_int64_t_col_int64_t_col = (F4)&proj8_multiplies_sel_int64_t_col_int64_t_col_impl;
#endif
}
}
