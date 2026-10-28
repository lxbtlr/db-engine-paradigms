#pragma once
//===----------------------------------------------------------------------===//
// VPCLMULQDQ CRC32Hash kernels (CMake option VW_CRC32_VPCLMUL, with
// VW_USE_CRC32). Bit-identical to runtime::CRC32Hash::hashKey(T k, seed).
//
// crc32q(seed, k) is (k ^ seed32) * x^32 mod P over GF(2) (bit-reflected,
// P = CRC-32C). Per 64-bit lane, with three carry-less multiplies:
//   v   = k ^ lo32(seed)
//   y   = clmul(lo32(v), x^63 mod P) ^ (v >> 32)    fold the first 32 bits
//   t   = clmul(lo32(clmul(lo32(y), mu)), P)        Barrett: q * P
//   crc = (y ^ t) >> 32
// with reflected constants x^63 mod P = 0xdd45aab8, mu = floor(x^64 / P) =
// 0xdea713f1, P = 0x105ec76f1. VPCLMULQDQ on a zmm does four 64x64
// products (one per 128-bit lane, low or high qword by immediate), so 8 keys
// take two instructions per multiply. hashKey's second crc32 (seed
// 0x04c11db7) is the first one XOR crc32(seed ^ 0x04c11db7, 0), a constant
// for a fixed seed (CRC linearity).
//
// Covered: the hash / hash_sel primitives (constant seed) and HashGroup's
// packed keys <= 8 bytes (seed 0). rehash (a per-row seed would need two
// reductions per key) and wide packed keys (a serial chain) stay scalar.
// Measured on Sapphire Rapids: 0.45 ns/key against 0.84 scalar and 0.56
// with VW_CRC32_FAST (run_crcbench).
//
// Needs AVX-512F/DQ/VL/BW and VPCLMULQDQ (Ice Lake+, Sapphire Rapids,
// Zen 4) at compile time and 64-bit hashes; otherwise VW_HAVE_SIMD_CRC stays
// undefined and the pick_* helpers return the scalar pointer.
//===----------------------------------------------------------------------===//
#include "vectorwise/SimdHash.hpp"

#if defined(VW_HAVE_SIMD_HASH) && defined(__AVX512VL__) &&                    \
    defined(__AVX512BW__) && defined(__VPCLMULQDQ__)
#define VW_HAVE_SIMD_CRC 1
#endif

namespace vectorwise {
namespace primitives {
namespace simd_crc {

#ifdef VW_HAVE_SIMD_CRC
using simd_hash::gather8;
using simd_hash::gather_tail;
using simd_hash::supported;
using simd_hash::tail8;
using simd_hash::widen8;
using simd_hash::widen_tail;
using hash_t = defs::hash_t;

/// a * k per 64-bit lane, carry-less, for a < 2^32 and k < 2^33 (the
/// product fits 64 bits): even qwords from imm 0x00, odd from imm 0x01
VW_HASH_INLINE __m512i clmul8(__m512i a, __m512i k) {
   return _mm512_unpacklo_epi64(_mm512_clmulepi64_epi128(a, k, 0x00),
                                _mm512_clmulepi64_epi128(a, k, 0x01));
}

/// crc32q(0, v) per lane, in the low 32 bits
VW_HASH_INLINE __m512i crc8(__m512i v) {
   const __m512i lo32 = _mm512_set1_epi64(0xffffffffll);
   const __m512i kFold = _mm512_set1_epi64(0xdd45aab8ll);
   const __m512i kMu = _mm512_set1_epi64(0xdea713f1ll);
   const __m512i kP = _mm512_set1_epi64(0x105ec76f1ll);
   const __m512i y = _mm512_xor_si512(clmul8(_mm512_and_si512(v, lo32), kFold),
                                      _mm512_srli_epi64(v, 32));
   const __m512i t =
       clmul8(_mm512_and_si512(clmul8(_mm512_and_si512(y, lo32), kMu), lo32), kP);
   return _mm512_srli_epi64(_mm512_xor_si512(y, t), 32);
}

/// Per-seed constants: the seed's low 32 bits (XORed into each key) and
/// crc32q(seed ^ 0x04c11db7, 0) (turns crc32q(seed, k) into
/// crc32q(0x04c11db7, k)).
struct Seed {
   __m512i lo32;
   __m512i term;
   explicit Seed(hash_t seed)
       : lo32(_mm512_set1_epi64((long long)(uint32_t)seed)),
         term(_mm512_set1_epi64(
             (long long)_mm_crc32_u64((uint32_t)seed ^ 0x04c11db7u, 0))) {}
};

/// CRC32Hash::hashKey(k, seed) for 8 widened keys
VW_HASH_INLINE __m512i crc_hash8(__m512i k, const Seed& s) {
   const __m512i r1 = crc8(_mm512_xor_si512(k, s.lo32));
   const __m512i r2 = _mm512_xor_si512(r1, s.term);
   return _mm512_mullo_epi64(_mm512_or_si512(_mm512_slli_epi64(r2, 32), r1),
                             _mm512_set1_epi64(0x2545F4914F6CDD1Dll));
}

//--- primitive replacements (signatures as in Primitives.hpp) ---------------

template <typename T>
pos_t hash(pos_t n, hash_t* RES result, T* RES input) {
   static_assert(supported<T>, "unsupported key type");
   const Seed s(primitives::seed);
   size_t i = 0;
   for (; i + 8 <= n; i += 8)
      _mm512_storeu_si512(result + i, crc_hash8(widen8<T>(input + i), s));
   if (i < n) {
      const size_t r = n - i;
      _mm512_mask_storeu_epi64(result + i, tail8(r),
                               crc_hash8(widen_tail<T>(input + i, r), s));
   }
   return n;
}

template <typename T>
pos_t hash_sel(pos_t n, pos_t* RES inSel, hash_t* RES result, T* RES input) {
   static_assert(supported<T>, "unsupported key type");
   const Seed s(primitives::seed);
   size_t i = 0;
   for (; i + 8 <= n; i += 8)
      _mm512_storeu_si512(result + i, crc_hash8(gather8<T>(input, inSel + i), s));
   if (i < n) {
      const size_t r = n - i;
      _mm512_mask_storeu_epi64(result + i, tail8(r),
                               crc_hash8(gather_tail<T>(input, inSel + i, r), s));
   }
   return n;
}

/// HashGroup packed keys: out[i] = CRC32Hash::hashKey(uint64_t(key_i)),
/// keys are n consecutive unsigned T (zero-extended, seed 0)
template <typename T>
void hash_keys(size_t n, const char* RES keys, hash_t* RES out) {
   static_assert(std::is_unsigned<T>::value, "packed keys are unsigned");
   const Seed s(0);
   size_t i = 0;
   for (; i + 8 <= n; i += 8)
      _mm512_storeu_si512(out + i, crc_hash8(widen8<T>(keys + i * sizeof(T)), s));
   if (i < n) {
      const size_t r = n - i;
      _mm512_mask_storeu_epi64(out + i, tail8(r),
                               crc_hash8(widen_tail<T>(keys + i * sizeof(T), r), s));
   }
}
#endif // VW_HAVE_SIMD_CRC

//--- pickers used by Hash.cpp -------------------------------------------------
// The VPCLMULQDQ kernel when VW_CRC32_VPCLMUL is on, the ISA is available,
// the hash is CRC32Hash and T is supported; otherwise `scalar`.

template <typename T, typename Op> constexpr bool use_simd() {
#if defined(VW_CRC32_VPCLMUL) && defined(VW_HAVE_SIMD_CRC)
   return simd_hash::supported<T> && std::is_same<Op, runtime::CRC32Hash>::value;
#else
   return false;
#endif
}

template <typename T, typename Op, typename Fn>
constexpr Fn pick_hash(Fn scalar) {
#if defined(VW_HAVE_SIMD_CRC)
   if constexpr (use_simd<T, Op>()) return &hash<T>;
#endif
   return scalar;
}
template <typename T, typename Op, typename Fn>
constexpr Fn pick_hash_sel(Fn scalar) {
#if defined(VW_HAVE_SIMD_CRC)
   if constexpr (use_simd<T, Op>()) return &hash_sel<T>;
#endif
   return scalar;
}

} // namespace simd_crc
} // namespace primitives
} // namespace vectorwise
