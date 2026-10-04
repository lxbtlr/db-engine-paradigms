#pragma once
//===----------------------------------------------------------------------===//
// NEON (and SVE) kernels for the aarch64 port of the AVX-512 VectorWise
// kernels. Measured by run_neonbench (src/benchmarks/primitives/neonbench.cpp)
// against the scalar engine templates and the AVX-512 kernels compiled
// through SIMDe; see NEON_PORT_STUDY.md for the per-kernel decision.
//
// NOT WIRED INTO THE ENGINE YET. Every AVX-512 kernel in the engine is
// guarded by __AVX512F__ (and friends), so an aarch64 build runs the scalar
// templates everywhere. The "engine site" line above each kernel names the
// guard that would get a VW_HAVE_NEON branch.
//
// What AVX-512 has and NEON does not, and the substitute used here:
//   mask registers   compares give all-ones lanes; a lane bitmask costs
//                    AND with {1,2,4,8} + ADDV (+ FMOV to a GPR). mask16()
//                    folds four compares into one ADDV.
//   vpcompressd/q    16-entry TBL (vqtbl1q_u8) table moves the selected
//                    lanes to the front; the whole register is stored and
//                    the output pointer advances by the match count.
//                    Store-overwrite contract: up to 3 (32-bit) or 1
//                    (64-bit) lanes past the last match are written. All
//                    kernels here write at found <= i while i + 4 <= n, so
//                    an n-entry output buffer is enough (as for the
//                    engine's _bf templates, which also write result[found]
//                    unconditionally).
//   vpgather         none: scalar loads (LDR + INS into lanes) or scalar
//                    code. Index arithmetic is done in scalar registers from
//                    memory, not extracted from vector lanes (UMOV is a
//                    cross-domain move).
//   vpmullq          no 64-bit lane multiply: 3 UMULL/UMLAL per 2 lanes
//                    (mullo64). SVE has MUL (vectors) .D.
//   512-bit vectors  128-bit: 4 x int32 / 2 x int64 per instruction.
// SVE (VW_HAVE_SVE, compile time only: neoverse-v1/v2 profiles) has
// COMPACT, gathers, predicates and 64-bit MUL, so the AVX-512 kernels map
// almost 1:1; Neoverse N1 (burrata) and Cortex-A76 (rpi5) have no SVE.
//===----------------------------------------------------------------------===//
#include "common/runtime/Types.hpp"
#include "vectorwise/defs.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <type_traits>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define VW_HAVE_NEON 1
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#define VW_HAVE_SVE 1
#endif
#endif

namespace vectorwise {
namespace primitives {
namespace neon {

//--- tables (constexpr, any target) ------------------------------------------

/// TBL byte indices that move the 32-bit lanes selected by a 4-bit mask to
/// the front (unused tail lanes repeat lane 0; their values do not matter)
struct Compress32Lut {
   uint8_t idx[16][16];
};
constexpr Compress32Lut makeCompress32Lut() {
   Compress32Lut l{};
   for (unsigned m = 0; m < 16; ++m) {
      unsigned o = 0;
      for (unsigned lane = 0; lane < 4; ++lane)
         if (m >> lane & 1) {
            for (unsigned b = 0; b < 4; ++b) l.idx[m][o * 4 + b] = uint8_t(lane * 4 + b);
            ++o;
         }
      for (; o < 4; ++o)
         for (unsigned b = 0; b < 4; ++b) l.idx[m][o * 4 + b] = uint8_t(b);
   }
   return l;
}
alignas(16) inline constexpr Compress32Lut kCompress32 = makeCompress32Lut();

/// Same for 2 x 64-bit lanes (mask 0..3): only mask 2 moves anything
alignas(16) inline constexpr uint8_t kCompress64[4][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};

/// popcount of a 4-bit mask (a scalar CNT needs FMOV + CNT + ADDV on cores
/// without FEAT_CSSC)
inline constexpr uint8_t kPop4[16] = {0, 1, 1, 2, 1, 2, 2, 3,
                                      1, 2, 2, 3, 2, 3, 3, 4};

#ifdef VW_HAVE_NEON
static_assert(sizeof(pos_t) == 4,
              "the NEON kernels write 32-bit positions (VW_POS_16 is ON)");

#define VW_NEON_INLINE inline __attribute__((always_inline))

//--- building blocks ---------------------------------------------------------

/// 4-bit lane mask of a 4 x 32 compare result (all-ones / zero lanes)
VW_NEON_INLINE unsigned mask4(uint32x4_t m) {
   const uint32x4_t w = {1, 2, 4, 8};
   return vaddvq_u32(vandq_u32(m, w));
}
/// two 2 x 64 compare results as one 4 x 32 compare result
VW_NEON_INLINE uint32x4_t narrow2x64(uint64x2_t lo, uint64x2_t hi) {
   return vcombine_u32(vmovn_u64(lo), vmovn_u64(hi));
}
/// 16-bit lane mask of four 4 x 32 compare results with a single ADDV
VW_NEON_INLINE unsigned mask16(uint32x4_t a, uint32x4_t b, uint32x4_t c,
                               uint32x4_t d) {
   const uint32x4_t w0 = {1, 2, 4, 8};
   const uint32x4_t w1 = {16, 32, 64, 128};
   const uint32x4_t w2 = {256, 512, 1024, 2048};
   const uint32x4_t w3 = {4096, 8192, 16384, 32768};
   const uint32x4_t s = vorrq_u32(vorrq_u32(vandq_u32(a, w0), vandq_u32(b, w1)),
                                  vorrq_u32(vandq_u32(c, w2), vandq_u32(d, w3)));
   return vaddvq_u32(s);
}
/// 8-bit lane mask of an 8 x 16 compare result
VW_NEON_INLINE unsigned mask8(uint16x8_t m) {
   const uint16x8_t w = {1, 2, 4, 8, 16, 32, 64, 128};
   return vaddvq_u16(vandq_u16(m, w));
}

/// the 32-bit lanes of v selected by m (0..15) moved to the front
VW_NEON_INLINE uint32x4_t compress4(uint32x4_t v, unsigned m) {
   return vreinterpretq_u32_u8(
       vqtbl1q_u8(vreinterpretq_u8_u32(v), vld1q_u8(kCompress32.idx[m])));
}
/// the 64-bit lanes of v selected by m (0..3) moved to the front
VW_NEON_INLINE uint64x2_t compress2(uint64x2_t v, unsigned m) {
   return vreinterpretq_u64_u8(
       vqtbl1q_u8(vreinterpretq_u8_u64(v), vld1q_u8(kCompress64[m])));
}
/// store the selected lanes of ids at out (store-overwrite, see top);
/// returns the number of positions written
VW_NEON_INLINE size_t emit4(pos_t* out, uint32x4_t ids, unsigned m) {
   vst1q_u32(out, compress4(ids, m));
   return kPop4[m];
}

/// low 64 bits of a * b per lane: a*b mod 2^64 = al*bl + ((ah*bl + al*bh) << 32)
VW_NEON_INLINE uint64x2_t mullo64(uint64x2_t a, uint32x2_t bl, uint32x2_t bh) {
   const uint32x2_t al = vmovn_u64(a), ah = vshrn_n_u64(a, 32);
   const uint64x2_t cross = vmlal_u32(vmull_u32(ah, bl), al, bh);
   return vaddq_u64(vmull_u32(al, bl), vshlq_n_u64(cross, 32));
}
VW_NEON_INLINE uint64x2_t mullo64(uint64x2_t a, uint64x2_t b) {
   return mullo64(a, vmovn_u64(b), vshrn_n_u64(b, 32));
}

/// MurmurHash64A of 2 keys; bit-identical to runtime::MurMurHash::hashKey
/// (engine: SimdHash.hpp murmur(), the zmm version)
VW_NEON_INLINE uint64x2_t murmur2(uint64x2_t k, uint64x2_t seed) {
   constexpr uint64_t m = 0xc6a4a7935bd1e995ull;
   const uint32x2_t ml = vdup_n_u32(uint32_t(m)), mh = vdup_n_u32(uint32_t(m >> 32));
   uint64x2_t h = veorq_u64(seed, vdupq_n_u64(0x8445d61a4e774912ull ^ (8 * m)));
   k = mullo64(k, ml, mh);
   k = veorq_u64(k, vshrq_n_u64(k, 47));
   k = mullo64(k, ml, mh);
   h = veorq_u64(h, k);
   h = mullo64(h, ml, mh);
   h = veorq_u64(h, vshrq_n_u64(h, 47));
   h = mullo64(h, ml, mh);
   h = veorq_u64(h, vshrq_n_u64(h, 47));
   return h;
}

//--- comparators -------------------------------------------------------------

/// std:: comparator -> NEON signed compare (all-ones lanes where true)
template <template <typename> class Op> struct Cmp;
template <> struct Cmp<std::equal_to> {
   static uint32x4_t f(int32x4_t a, int32x4_t b) { return vceqq_s32(a, b); }
   static uint64x2_t f(int64x2_t a, int64x2_t b) { return vceqq_s64(a, b); }
};
template <> struct Cmp<std::less> {
   static uint32x4_t f(int32x4_t a, int32x4_t b) { return vcltq_s32(a, b); }
   static uint64x2_t f(int64x2_t a, int64x2_t b) { return vcltq_s64(a, b); }
};
template <> struct Cmp<std::less_equal> {
   static uint32x4_t f(int32x4_t a, int32x4_t b) { return vcleq_s32(a, b); }
   static uint64x2_t f(int64x2_t a, int64x2_t b) { return vcleq_s64(a, b); }
};
template <> struct Cmp<std::greater> {
   static uint32x4_t f(int32x4_t a, int32x4_t b) { return vcgtq_s32(a, b); }
   static uint64x2_t f(int64x2_t a, int64x2_t b) { return vcgtq_s64(a, b); }
};
template <> struct Cmp<std::greater_equal> {
   static uint32x4_t f(int32x4_t a, int32x4_t b) { return vcgeq_s32(a, b); }
   static uint64x2_t f(int64x2_t a, int64x2_t b) { return vcgeq_s64(a, b); }
};

/// compare 4 contiguous int32 / int64 elements against c (broadcast)
template <typename R, template <typename> class Op>
VW_NEON_INLINE uint32x4_t cmp4(const R* p, R c) {
   if constexpr (sizeof(R) == 4) {
      return Cmp<Op>::f(vld1q_s32(p), vdupq_n_s32(c));
   } else {
      const int64x2_t vc = vdupq_n_s64(c);
      return narrow2x64(Cmp<Op>::f(vld1q_s64(p), vc),
                        Cmp<Op>::f(vld1q_s64(p + 2), vc));
   }
}
/// compare a[idx[0..3]] (scalar loads into lanes) against c
template <typename R, template <typename> class Op>
VW_NEON_INLINE uint32x4_t cmp4_at(const R* a, const pos_t* idx, R c) {
   if constexpr (sizeof(R) == 4) {
      int32x4_t v = vdupq_n_s32(a[idx[0]]);
      v = vld1q_lane_s32(a + idx[1], v, 1);
      v = vld1q_lane_s32(a + idx[2], v, 2);
      v = vld1q_lane_s32(a + idx[3], v, 3);
      return Cmp<Op>::f(v, vdupq_n_s32(c));
   } else {
      const int64x2_t vc = vdupq_n_s64(c);
      int64x2_t lo = vdupq_n_s64(a[idx[0]]), hi = vdupq_n_s64(a[idx[2]]);
      lo = vld1q_lane_s64(a + idx[1], lo, 1);
      hi = vld1q_lane_s64(a + idx[3], hi, 1);
      return narrow2x64(Cmp<Op>::f(lo, vc), Cmp<Op>::f(hi, vc));
   }
}

/// int32_t / int64_t / types::Date as the integer the kernels compare
template <typename T> using raw_t = typename std::conditional<
    sizeof(T) == 4, int32_t, int64_t>::type;
template <typename T> VW_NEON_INLINE raw_t<T> rawValue(const T& v) {
   raw_t<T> r;
   std::memcpy(&r, &v, sizeof(r));
   return r;
}

//--- selection (engine: SimdSelection.hpp, VW_SIMD_SEL) ----------------------

/// sel_col_val, 4 elements per step.
/// engine site: SimdSelection.hpp:235 sel_col_val_v, picked by
/// pick_sel_col_val under `__x86_64__ && __AVX512F__` (SimdSelection.hpp:28)
template <typename T, template <typename> class Op>
pos_t sel_col_val(pos_t n, pos_t* RES result, T* RES param1, T* RES param2) {
   using R = raw_t<T>;
   static_assert(sizeof(T) == sizeof(R), "int32 / int64 / Date only");
   const R* a = reinterpret_cast<const R*>(param1);
   const R c = rawValue(*param2);
   uint32x4_t ids = {0, 1, 2, 3};
   const uint32x4_t four = vdupq_n_u32(4);
   size_t found = 0, i = 0;
   for (; i + 4 <= n; i += 4) {
      found += emit4(result + found, ids, mask4(cmp4<R, Op>(a + i, c)));
      ids = vaddq_u32(ids, four);
   }
   for (; i < n; ++i) {
      result[found] = pos_t(i);
      found += Op<R>()(a[i], c);
   }
   return pos_t(found);
}

/// sel_col_val, 16 elements per step: one ADDV for four compares (the mask
/// move to a GPR is the long-latency part of mask4)
template <typename T, template <typename> class Op>
pos_t sel_col_val_x16(pos_t n, pos_t* RES result, T* RES param1,
                      T* RES param2) {
   using R = raw_t<T>;
   static_assert(sizeof(T) == sizeof(R), "int32 / int64 / Date only");
   const R* a = reinterpret_cast<const R*>(param1);
   const R c = rawValue(*param2);
   uint32x4_t ids = {0, 1, 2, 3};
   const uint32x4_t four = vdupq_n_u32(4), sixteen = vdupq_n_u32(16);
   size_t found = 0, i = 0;
   for (; i + 16 <= n; i += 16) {
      const unsigned m =
          mask16(cmp4<R, Op>(a + i, c), cmp4<R, Op>(a + i + 4, c),
                 cmp4<R, Op>(a + i + 8, c), cmp4<R, Op>(a + i + 12, c));
      found += emit4(result + found, ids, m & 15);
      found += emit4(result + found, vaddq_u32(ids, four), m >> 4 & 15);
      found += emit4(result + found, vaddq_u32(ids, vdupq_n_u32(8)), m >> 8 & 15);
      found += emit4(result + found, vaddq_u32(ids, vdupq_n_u32(12)), m >> 12);
      ids = vaddq_u32(ids, sixteen);
   }
   for (; i + 4 <= n; i += 4) {
      found += emit4(result + found, ids, mask4(cmp4<R, Op>(a + i, c)));
      ids = vaddq_u32(ids, four);
   }
   for (; i < n; ++i) {
      result[found] = pos_t(i);
      found += Op<R>()(a[i], c);
   }
   return pos_t(found);
}

/// selsel_col_val: scalar loads through inSel into lanes, NEON compare and
/// compress of the selection entries.
/// engine site: SimdSelection.hpp selsel_col_val (VW_SIMD_SEL_GATHER picks
/// scalar / scalarload / hwgather), same guard as sel_col_val
template <typename T, template <typename> class Op>
pos_t selsel_col_val(pos_t n, pos_t* RES inSel, pos_t* RES result,
                     T* RES param1, T* RES param2) {
   using R = raw_t<T>;
   const R* a = reinterpret_cast<const R*>(param1);
   const R c = rawValue(*param2);
   size_t found = 0, i = 0;
   for (; i + 4 <= n; i += 4)
      found += emit4(result + found, vld1q_u32(inSel + i),
                     mask4(cmp4_at<R, Op>(a, inSel + i, c)));
   for (; i < n; ++i) {
      const pos_t idx = inSel[i];
      result[found] = idx;
      found += Op<R>()(a[idx], c);
   }
   return pos_t(found);
}

//--- hashing (engine: SimdHash.hpp, VW_SIMD_HASH) ----------------------------

/// hash<int32_t, MurMurHash>: keys sign-extended to 64 bits, seed `seed`,
/// 4 keys per step (two independent 2-lane chains).
/// engine site: SimdHash.hpp hash() (VW_HAVE_SIMD_HASH needs AVX512F+DQ)
inline void murmur_hash_i32(size_t n, uint64_t* RES out, const int32_t* RES in,
                            uint64_t seed) {
   const uint64x2_t s = vdupq_n_u64(seed);
   size_t i = 0;
   for (; i + 4 <= n; i += 4) {
      const int32x4_t k = vld1q_s32(in + i);
      const uint64x2_t k0 = vreinterpretq_u64_s64(vmovl_s32(vget_low_s32(k)));
      const uint64x2_t k1 = vreinterpretq_u64_s64(vmovl_high_s32(k));
      vst1q_u64(out + i, murmur2(k0, s));
      vst1q_u64(out + i + 2, murmur2(k1, s));
   }
   for (; i < n; ++i) {
      const uint64x2_t k = vdupq_n_u64(uint64_t(int64_t(in[i])));
      out[i] = vgetq_lane_u64(murmur2(k, s), 0);
   }
}
/// hash_sel<int32_t, MurMurHash>: keys in[sel[i]] loaded into lanes
inline void murmur_hash_sel_i32(size_t n, uint64_t* RES out,
                                const int32_t* RES in, const pos_t* RES sel,
                                uint64_t seed) {
   const uint64x2_t s = vdupq_n_u64(seed);
   size_t i = 0;
   for (; i + 4 <= n; i += 4) {
      int32x4_t k = vdupq_n_s32(in[sel[i]]);
      k = vld1q_lane_s32(in + sel[i + 1], k, 1);
      k = vld1q_lane_s32(in + sel[i + 2], k, 2);
      k = vld1q_lane_s32(in + sel[i + 3], k, 3);
      vst1q_u64(out + i, murmur2(vreinterpretq_u64_s64(vmovl_s32(vget_low_s32(k))), s));
      vst1q_u64(out + i + 2, murmur2(vreinterpretq_u64_s64(vmovl_high_s32(k)), s));
   }
   for (; i < n; ++i) {
      const uint64x2_t k = vdupq_n_u64(uint64_t(int64_t(in[sel[i]])));
      out[i] = vgetq_lane_u64(murmur2(k, s), 0);
   }
}

//--- dense projections (the removed VW_PROJ_DENSE; kept as measured kernels) -
// Same contract as the engine's proj_dense_*: every position in
// [0, inSel[n-1]] is computed, results are position-indexed.

inline pos_t denseSpan(pos_t n, const pos_t* RES inSel) {
   return n ? inSel[n - 1] + 1 : 0;
}

// 8 elements (4 registers) per iteration with a 64-bit index: N1 is
// dispatch-bound on these loops, so loop overhead and the address arithmetic
// of a 32-bit (pos_t) index are a large share of the instructions.

/// result[i] = *param1 - param2[i]  (Q1: 1 - l_discount)
inline pos_t proj_dense_minus_val_col(pos_t n, pos_t* RES inSel,
                                      int64_t* RES result, int64_t* RES param1,
                                      int64_t* RES param2) {
   const size_t m = denseSpan(n, inSel);
   const int64x2_t c = vdupq_n_s64(*param1);
   size_t i = 0;
   for (; i + 8 <= m; i += 8) {
      int64x2x4_t v = vld1q_s64_x4(param2 + i);
      for (auto& r : v.val) r = vsubq_s64(c, r);
      vst1q_s64_x4(result + i, v);
   }
   for (; i < m; ++i) result[i] = *param1 - param2[i];
   return n;
}
/// result[i] = param1[i] + *param2  (Q1: 1 + l_tax)
inline pos_t proj_dense_plus_col_val(pos_t n, pos_t* RES inSel,
                                     int64_t* RES result, int64_t* RES param1,
                                     int64_t* RES param2) {
   const size_t m = denseSpan(n, inSel);
   const int64x2_t c = vdupq_n_s64(*param2);
   size_t i = 0;
   for (; i + 8 <= m; i += 8) {
      int64x2x4_t v = vld1q_s64_x4(param1 + i);
      for (auto& r : v.val) r = vaddq_s64(r, c);
      vst1q_s64_x4(result + i, v);
   }
   for (; i < m; ++i) result[i] = param1[i] + *param2;
   return n;
}
/// result[i] = param1[i] * param2[i], 64-bit multiply emulated (3 UMULL
/// per 2 lanes; Q1: extendedprice * (1 - discount))
inline pos_t proj_dense_multiplies_col_col(pos_t n, pos_t* RES inSel,
                                           int64_t* RES result,
                                           int64_t* RES param1,
                                           int64_t* RES param2) {
   const size_t m = denseSpan(n, inSel);
   size_t i = 0;
   for (; i + 4 <= m; i += 4) {
      const uint64x2_t a0 = vreinterpretq_u64_s64(vld1q_s64(param1 + i));
      const uint64x2_t a1 = vreinterpretq_u64_s64(vld1q_s64(param1 + i + 2));
      const uint64x2_t b0 = vreinterpretq_u64_s64(vld1q_s64(param2 + i));
      const uint64x2_t b1 = vreinterpretq_u64_s64(vld1q_s64(param2 + i + 2));
      vst1q_s64(result + i, vreinterpretq_s64_u64(mullo64(a0, b0)));
      vst1q_s64(result + i + 2, vreinterpretq_s64_u64(mullo64(a1, b1)));
   }
   for (; i < m; ++i) result[i] = param1[i] * param2[i];
   return n;
}

//--- HashGroup run heads, pass 1 (engine: Operators.cpp:2027-2054, 2115-2140,
// VW_GROUP_RUN_HEADS; guard __AVX512BW__ && __AVX512VL__) --------------------
// From row `i` (>= 1) on: heads[cnt++] = r for every row r whose key differs
// from row r-1's, flags[r] = 1 for those rows and 0 otherwise. Returns the new
// cnt. Keys are 2 bytes (Q1's packed l_returnflag, l_linestatus) or 4 bytes
// (Q18's l_orderkey).

template <typename K>
pos_t run_heads(pos_t n, const K* RES keys, pos_t* RES heads,
                uint8_t* RES flags, pos_t cnt0, pos_t i0) {
   size_t cnt = cnt0, i = i0;
   static_assert(sizeof(K) == 2 || sizeof(K) == 4, "2- or 4-byte keys");
   if constexpr (sizeof(K) == 4) {
      const uint32_t* k = reinterpret_cast<const uint32_t*>(keys);
      const uint32x4_t lane = {0, 1, 2, 3};
      for (; i + 4 <= n; i += 4) {
         const uint32x4_t ne =
             vmvnq_u32(vceqq_u32(vld1q_u32(k + i), vld1q_u32(k + i - 1)));
         cnt += emit4(heads + cnt, vaddq_u32(vdupq_n_u32(uint32_t(i)), lane), mask4(ne));
         // 0/1 per row: narrow the all-ones lanes to bytes and keep bit 0
         const uint8x8_t f = vand_u8(
             vmovn_u16(vcombine_u16(vmovn_u32(ne), vdup_n_u16(0))), vdup_n_u8(1));
         vst1_lane_u32(reinterpret_cast<uint32_t*>(flags + i),
                       vreinterpret_u32_u8(f), 0);
      }
   } else {
      const uint16_t* k = reinterpret_cast<const uint16_t*>(keys);
      const uint32x4_t lane = {0, 1, 2, 3};
      for (; i + 8 <= n; i += 8) {
         const uint16x8_t ne =
             vmvnq_u16(vceqq_u16(vld1q_u16(k + i), vld1q_u16(k + i - 1)));
         const unsigned m = mask8(ne);
         const uint32x4_t idx = vaddq_u32(vdupq_n_u32(uint32_t(i)), lane);
         cnt += emit4(heads + cnt, idx, m & 15);
         cnt += emit4(heads + cnt, vaddq_u32(idx, vdupq_n_u32(4)), m >> 4);
         vst1_u8(flags + i, vand_u8(vmovn_u16(ne), vdup_n_u8(1)));
      }
   }
   for (; i < n; ++i) {
      const uint8_t f = keys[i] != keys[i - 1];
      flags[i] = f;
      heads[cnt] = pos_t(i);
      cnt += f;
   }
   return pos_t(cnt);
}

//--- Hashjoin Bloom filter (engine: Operators.cpp:1008 bloomFilter,
// VW_JOIN_BLOOM; guard __AVX512F__ && !VW_POS_16) ----------------------------
// Survivors' probe index -> sel, hash -> hs (store-overwrite); returns the
// survivor count. The 4 word loads are scalar, indexed from the hashes in
// memory; the bit test and compress are NEON.

VW_NEON_INLINE uint64x2_t bloomBits2(uint64x2_t h) {
   const uint64x2_t b6 = vdupq_n_u64(63), one = vdupq_n_u64(1);
   auto bit = [&](uint64x2_t s) {
      return vshlq_u64(one, vreinterpretq_s64_u64(vandq_u64(s, b6)));
   };
   return vorrq_u64(vorrq_u64(bit(h), bit(vshrq_n_u64(h, 6))),
                    vorrq_u64(bit(vshrq_n_u64(h, 12)), bit(vshrq_n_u64(h, 18))));
}

inline size_t bloom_filter(size_t n, const uint64_t* RES hashes,
                           const uint64_t* RES bloom, uint64_t mask,
                           pos_t* RES sel, uint64_t* RES hs) {
   size_t m = 0, i = 0;
   const uint32x4_t lane = {0, 1, 2, 3};
   for (; i + 4 <= n; i += 4) {
      const uint64x2_t h0 = vld1q_u64(hashes + i), h1 = vld1q_u64(hashes + i + 2);
      uint64x2_t w0 = vdupq_n_u64(bloom[(hashes[i] >> 40) & mask]);
      uint64x2_t w1 = vdupq_n_u64(bloom[(hashes[i + 2] >> 40) & mask]);
      w0 = vld1q_lane_u64(bloom + ((hashes[i + 1] >> 40) & mask), w0, 1);
      w1 = vld1q_lane_u64(bloom + ((hashes[i + 3] >> 40) & mask), w1, 1);
      const uint64x2_t b0 = bloomBits2(h0), b1 = bloomBits2(h1);
      const unsigned p = mask4(narrow2x64(vceqq_u64(vandq_u64(w0, b0), b0),
                                          vceqq_u64(vandq_u64(w1, b1), b1)));
      vst1q_u32(sel + m, compress4(vaddq_u32(vdupq_n_u32(uint32_t(i)), lane), p));
      vst1q_u64(hs + m, compress2(h0, p & 3));
      vst1q_u64(hs + m + kPop4[p & 3], compress2(h1, p >> 2));
      m += kPop4[p];
   }
   for (; i < n; ++i) {
      const uint64_t h = hashes[i];
      const uint64_t b = (uint64_t(1) << (h & 63)) | (uint64_t(1) << ((h >> 6) & 63)) |
                         (uint64_t(1) << ((h >> 12) & 63)) |
                         (uint64_t(1) << ((h >> 18) & 63));
      sel[m] = pos_t(i);
      hs[m] = h;
      m += (bloom[(h >> 40) & mask] & b) == b;
   }
   return m;
}

//--- Hashjoin semi bitmap probe (engine: Operators.cpp:918 semiProbe,
// VW_JOIN_SEMI; guard __AVX512F__ && !VW_POS_16) -----------------------------
// Dense probe keys: probe positions i with keys[i] - lo in [0, span] and its
// bit set -> out (store-overwrite). Word loads are scalar from a clamped
// offset, the range check, bit test and compress are NEON.

inline size_t semi_probe(size_t n, const int32_t* RES keys,
                         const uint32_t* RES bits, int32_t lo, uint32_t span,
                         pos_t* RES out) {
   const int32x4_t vlo = vdupq_n_s32(lo);
   const uint32x4_t vspan = vdupq_n_u32(span), b31 = vdupq_n_u32(31),
                    one = vdupq_n_u32(1), lane = {0, 1, 2, 3};
   size_t found = 0, i = 0;
   for (; i + 4 <= n; i += 4) {
      const uint32x4_t d =
          vreinterpretq_u32_s32(vsubq_s32(vld1q_s32(keys + i), vlo));
      const uint32x4_t in = vcleq_u32(d, vspan);
      // clamped word offsets in scalar registers (no lane extraction)
      auto w = [&](size_t j) {
         const uint32_t dj = uint32_t(keys[i + j]) - uint32_t(lo);
         return bits[(dj <= span ? dj : span) >> 5];
      };
      uint32x4_t words = vdupq_n_u32(w(0));
      words = vsetq_lane_u32(w(1), words, 1);
      words = vsetq_lane_u32(w(2), words, 2);
      words = vsetq_lane_u32(w(3), words, 3);
      const uint32x4_t bit =
          vshlq_u32(one, vreinterpretq_s32_u32(vandq_u32(d, b31)));
      const uint32x4_t hit = vandq_u32(in, vtstq_u32(words, bit));
      found += emit4(out + found, vaddq_u32(vdupq_n_u32(uint32_t(i)), lane),
                     mask4(hit));
   }
   for (; i < n; ++i) {
      const uint64_t d = uint64_t(int64_t(keys[i]) - lo);
      out[found] = pos_t(i);
      found += d <= span && (bits[d >> 5] >> (d & 31) & 1);
   }
   return found;
}

//--- SVE ---------------------------------------------------------------------
#ifdef VW_HAVE_SVE
/// sel_col_val<int32_t, Op> with COMPACT: the direct analogue of
/// sel_col_val_v (compare -> compress -> store of cnt lanes), VL-agnostic
template <template <typename> class Op>
pos_t sve_sel_col_val_i32(pos_t n, pos_t* RES result, const int32_t* RES a,
                          int32_t c) {
   size_t found = 0;
   const svint32_t vc = svdup_s32(c);
   for (size_t i = 0; i < n; i += svcntw()) {
      const svbool_t pg = svwhilelt_b32_u64(i, n);
      const svint32_t v = svld1_s32(pg, a + i);
      svbool_t m;
      if constexpr (std::is_same<Op<int>, std::less<int>>::value) m = svcmplt_s32(pg, v, vc);
      else if constexpr (std::is_same<Op<int>, std::less_equal<int>>::value) m = svcmple_s32(pg, v, vc);
      else if constexpr (std::is_same<Op<int>, std::greater<int>>::value) m = svcmpgt_s32(pg, v, vc);
      else if constexpr (std::is_same<Op<int>, std::greater_equal<int>>::value) m = svcmpge_s32(pg, v, vc);
      else m = svcmpeq_s32(pg, v, vc);
      const svuint32_t ids = svindex_u32(uint32_t(i), 1);
      const uint64_t cnt = svcntp_b32(pg, m);
      svst1_u32(svwhilelt_b32_u64(0, cnt), result + found, svcompact_u32(m, ids));
      found += cnt;
   }
   return pos_t(found);
}
/// selsel_col_val<int32_t, Op> with a 32-bit index gather
template <template <typename> class Op>
pos_t sve_selsel_col_val_i32(pos_t n, const pos_t* RES inSel, pos_t* RES result,
                             const int32_t* RES a, int32_t c) {
   size_t found = 0;
   const svint32_t vc = svdup_s32(c);
   for (size_t i = 0; i < n; i += svcntw()) {
      const svbool_t pg = svwhilelt_b32_u64(i, n);
      const svuint32_t idx = svld1_u32(pg, inSel + i);
      const svint32_t v = svld1_gather_u32index_s32(pg, a, idx);
      static_assert(std::is_same<Op<int>, std::less<int>>::value, "less only");
      const svbool_t m = svcmplt_s32(pg, v, vc);
      const uint64_t cnt = svcntp_b32(pg, m);
      svst1_u32(svwhilelt_b32_u64(0, cnt), result + found, svcompact_u32(m, idx));
      found += cnt;
   }
   return pos_t(found);
}
/// MurmurHash64A of int32 keys with 64-bit vector MUL
inline void sve_murmur_hash_i32(size_t n, uint64_t* RES out,
                                const int32_t* RES in, uint64_t seed) {
   constexpr uint64_t mc = 0xc6a4a7935bd1e995ull;
   const svuint64_t m = svdup_u64(mc);
   const svuint64_t h0 = svdup_u64(seed ^ 0x8445d61a4e774912ull ^ (8 * mc));
   for (size_t i = 0; i < n; i += svcntd()) {
      const svbool_t pg = svwhilelt_b64_u64(i, n);
      svuint64_t k = svreinterpret_u64_s64(svld1sw_s64(pg, in + i));
      k = svmul_u64_x(pg, k, m);
      k = sveor_u64_x(pg, k, svlsr_n_u64_x(pg, k, 47));
      k = svmul_u64_x(pg, k, m);
      svuint64_t h = sveor_u64_x(pg, h0, k);
      h = svmul_u64_x(pg, h, m);
      h = sveor_u64_x(pg, h, svlsr_n_u64_x(pg, h, 47));
      h = svmul_u64_x(pg, h, m);
      h = sveor_u64_x(pg, h, svlsr_n_u64_x(pg, h, 47));
      svst1_u64(pg, out + i, h);
   }
}
/// VW_PROJ_DENSE multiplies with 64-bit vector MUL
inline pos_t sve_proj_dense_multiplies(pos_t n, pos_t* RES inSel,
                                       int64_t* RES result, int64_t* RES p1,
                                       int64_t* RES p2) {
   const pos_t m = denseSpan(n, inSel);
   for (size_t i = 0; i < m; i += svcntd()) {
      const svbool_t pg = svwhilelt_b64_u64(i, m);
      svst1_s64(pg, result + i,
                svmul_s64_x(pg, svld1_s64(pg, p1 + i), svld1_s64(pg, p2 + i)));
   }
   return n;
}
#endif // VW_HAVE_SVE

#endif // VW_HAVE_NEON

//=============================================================================
// Engine wiring (SKELETON: nothing below is called yet). One entry per
// AVX-512 site, with the decision from NEON_PORT_STUDY.md section 4:
//   PORT     the NEON kernel above replaces the scalar path under the same
//            CMake option
//   SVE      only worth it with SVE (V1/V2); N1 keeps scalar
//   REWORK   the AVX-512 algorithm depends on gathers / masked lanes; the
//            aarch64 path is a different scalar loop
//   SCALAR   keep the scalar path; no NEON kernel pays
//
// site (guard today)                         option               decision
// SimdSelection.hpp:28 pick_sel_col_val      VW_SIMD_SEL          PORT  sel_col_val_x16
//   (__x86_64__ && __AVX512F__)                                         (int32/int64/Date)
// SimdSelection.hpp pick_sel_col_col         VW_SIMD_SEL          PORT  same shape, two loads
// SimdSelection.hpp pick_selsel_*            VW_SIMD_SEL_GATHER   SCALAR on N1, SVE on V1
// SimdSelection.hpp sel_char_eq (AVX512BW)   VW_SIMD_SEL_CHAR     not measured (byte compare
//                                                                 + vmaxvq: likely PORT)
// SimdHash.hpp:31 (AVX512F+DQ)               VW_SIMD_HASH         SCALAR (use VW_USE_CRC32)
// SimdCrc.hpp:31 (VPCLMULQDQ)                VW_CRC32_VPCLMUL     SCALAR (PMULL 2.6x slower
//                                                                 than crc32cx on burrata)
// (removed) dense projections               VW_PROJ_DENSE        REMOVED: choosing dense needs
//                                                                 the selection density (data);
//                                                                 plan facts only
// Operators.cpp:2027 runHeadMask16 (BW+VL)   VW_GROUP_RUN_HEADS   PORT  run_heads<K>
// Operators.cpp:1008 bloomFilter (AVX512F)   VW_JOIN_BLOOM        REWORK scalar_bf + prefetch
// Operators.cpp:1114 fusedHashFilter         VW_JOIN_FUSED_PROBE  REWORK scalar fused loop
//                                                                 (crc32cx + filter, no hash
//                                                                 vector written)
// Operators.cpp:918 semiProbe (AVX512F)      VW_JOIN_SEMI         SCALAR (+ prefetch for
//                                                                 bitmaps > L2)
// Operators.cpp:1259 joinNewFirstPass        VW_NEW_JOIN          REWORK block-8 scalar loop +
//                                                                 group prefetch
// Operators.cpp:346,658 joinAllSIMD/SelSIMD  VW_JOIN_SIMD         SCALAR (use joinAllParallel /
//                                                                 VW_JOIN_TWOPHASE)
// SimdAggr.hpp (bench only)                  -                    not wired on x86 either
//=============================================================================
#ifdef VW_HAVE_NEON
/// pick_sel_col_val for aarch64: the 16-per-step NEON kernel for int32,
/// int64 and Date, the scalar pointer otherwise. Would be called from
/// SimdSelection.hpp's pick_sel_col_val in a VW_HAVE_NEON branch.
template <typename T, template <typename> class Op, typename Fn>
constexpr Fn pick_sel_col_val(Fn scalar) {
#ifdef VW_SIMD_SEL
   if constexpr (std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value ||
                 std::is_same<T, types::Date>::value)
      return &sel_col_val_x16<T, Op>;
#endif
   return scalar;
}
#endif

} // namespace neon
} // namespace primitives
} // namespace vectorwise
