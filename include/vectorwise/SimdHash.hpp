#pragma once
//===----------------------------------------------------------------------===//
// SIMD MurmurHash64A kernels (CMake option VW_SIMD_HASH), one tier per ISA:
//   avx512  8 keys per zmm, vpmullq (AVX512F + AVX512DQ)
//   avx2    4 keys per ymm; no 64-bit lane multiply, so each multiply by the
//           constant is 3 vpmuludq: lo*m_lo + ((hi*m_lo + lo*m_hi) << 32)
//   neon    2 keys per register, the same 3-multiply form with UMULL/UMLAL
//           (mullo64), 8 keys per step in 4 independent chains
// The widest tier the target has is used (VW_SIMD_HASH_TIER_*); with
// VW_SIMD_HASH_WIDTH_256 (CMake VW_SIMD_HASH_WIDTH=256) an AVX-512 target
// uses the avx2 tier. Every compiled tier stays reachable as
// simd_hash::<tier>:: for the tests and run_hashbench.
//
// Replacements for the hash / hash_sel / rehash / rehash_sel primitive
// templates (Primitives.hpp) and for HashGroup's packed-key hash. Results are
// bit-identical to runtime::MurMurHash, because
// build and probe sides of a join may hash through different paths:
//   - keys are widened to 64 bits exactly as the scalar overloads do it:
//     signed types (int8/16/32, Date) sign-extend, unsigned ones zero-extend
//     (the old hash4 SIMD kernel zero-extends int32, which differs for
//     negative keys);
//   - hash/hash_sel use primitives::seed, rehash* use result[i] as the seed,
//     packed group keys use seed 0 (MurMurHash::hashKey(uint64_t)).
// Gathered input (hash_sel, rehash_sel) is collected with scalar loads into a
// stack buffer instead of vpgather, which is slow under the GDS microcode
// mitigation on Skylake..Ice Lake parts (see SimdSelection.hpp).
//
// Needs 64-bit hashes and x86-64 AVX2 or aarch64 NEON; otherwise nothing
// below is defined and pick_* return the scalar pointer. VW_HAVE_SIMD_HASH
// keeps its meaning (the AVX-512 tier and its zmm helpers murmur / widen8 /
// gather8, which SimdCrc.hpp, Concat.cpp and Operators.cpp use directly);
// VW_HAVE_SIMD_HASH_ANY means some tier is compiled in.
//===----------------------------------------------------------------------===//
#include "common/runtime/Hash.hpp"
#include "common/runtime/Types.hpp"
#include "vectorwise/Primitives.hpp"
#include "vectorwise/defs.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if !(defined(HASH_SIZE) && HASH_SIZE == 32)
#if defined(__x86_64__) && defined(__AVX512F__) && defined(__AVX512DQ__)
#define VW_HAVE_SIMD_HASH 1
#endif
#if defined(__x86_64__) && defined(__AVX2__)
#define VW_HAVE_SIMD_HASH_AVX2 1
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
#define VW_HAVE_SIMD_HASH_NEON 1
#endif
#endif
#if defined(VW_HAVE_SIMD_HASH) || defined(VW_HAVE_SIMD_HASH_AVX2)
#include <immintrin.h>
#endif
#ifdef VW_HAVE_SIMD_HASH_NEON
#include <arm_neon.h>
#endif

// the tier behind the public names
#if defined(VW_HAVE_SIMD_HASH) && !defined(VW_SIMD_HASH_WIDTH_256)
#define VW_SIMD_HASH_TIER_AVX512 1
#elif defined(VW_HAVE_SIMD_HASH_AVX2)
#define VW_SIMD_HASH_TIER_AVX2 1
#elif defined(VW_HAVE_SIMD_HASH_NEON)
#define VW_SIMD_HASH_TIER_NEON 1
#endif
#if defined(VW_SIMD_HASH_TIER_AVX512) || defined(VW_SIMD_HASH_TIER_AVX2) ||   \
    defined(VW_SIMD_HASH_TIER_NEON)
#define VW_HAVE_SIMD_HASH_ANY 1
#endif

namespace vectorwise {
namespace primitives {
namespace simd_hash {

/// the tier behind the public names, for logs and benchmarks
#if defined(VW_SIMD_HASH_TIER_AVX512)
constexpr const char* kTier = "avx512";
#elif defined(VW_SIMD_HASH_TIER_AVX2)
constexpr const char* kTier = "avx2";
#elif defined(VW_SIMD_HASH_TIER_NEON)
constexpr const char* kTier = "neon";
#else
constexpr const char* kTier = "scalar";
#endif

#if defined(VW_HAVE_SIMD_HASH) || defined(VW_HAVE_SIMD_HASH_AVX2) ||          \
    defined(VW_HAVE_SIMD_HASH_NEON)

using hash_t = defs::hash_t;
static_assert(sizeof(hash_t) == 8, "SIMD MurmurHash64A needs 64-bit hashes");

#if defined(__GNUC__) || defined(__clang__)
#define VW_HASH_INLINE inline __attribute__((always_inline))
#else
#define VW_HASH_INLINE inline
#endif
// keep the scalar gather loops scalar (vectorizing them yields vpgather)
#if defined(__clang__)
#define VW_HASH_SCALAR_LOOP _Pragma("clang loop vectorize(disable) unroll(full)")
#elif defined(__GNUC__) && __GNUC__ >= 14
#define VW_HASH_SCALAR_LOOP _Pragma("GCC unroll 8") _Pragma("GCC novector")
#else
#define VW_HASH_SCALAR_LOOP _Pragma("GCC unroll 8")
#endif

/// Types the kernels accept: the element types of EACH_TYPE except Char<N>.
template <typename T>
constexpr bool supported =
    std::is_same<T, int8_t>::value || std::is_same<T, int16_t>::value ||
    std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value ||
    std::is_same<T, uint64_t>::value || std::is_same<T, types::Date>::value;

/// Widen to 64 bits like the scalar Hash<> overloads (see file comment).
template <typename T>
constexpr bool sign_extends =
    std::is_same<T, types::Date>::value || std::is_signed<T>::value;

/// MurmurHash64A constants (runtime::MurMurHash::hashKey(uint64_t, seed))
constexpr uint64_t kMurmurM = 0xc6a4a7935bd1e995ull;
constexpr uint64_t kMurmurInit = 0x8445d61a4e774912ull ^ (8 * kMurmurM);

/// key i of a supported column widened to 64 bits as the scalar overloads
/// do it (the gather of the avx2 / neon tiers)
template <typename T> VW_HASH_INLINE uint64_t key64(const T* in, pos_t i) {
   if constexpr (std::is_same<T, types::Date>::value)
      return uint64_t(int64_t(in[i].value));
   else if constexpr (std::is_signed<T>::value)
      return uint64_t(int64_t(in[i]));
   else
      return uint64_t(in[i]);
}

#endif // any tier

#ifdef VW_HAVE_SIMD_HASH

/// MurmurHash64A of one 64-bit key per lane; bit-identical to
/// runtime::MurMurHash::hashKey(uint64_t k, hash_t seed).
VW_HASH_INLINE __m512i murmur(__m512i k, __m512i seed) {
   constexpr uint64_t m = 0xc6a4a7935bd1e995ull;
   const __m512i vm = _mm512_set1_epi64((long long)m);
   __m512i h = _mm512_xor_si512(
       seed, _mm512_set1_epi64((long long)(0x8445d61a4e774912ull ^ (8 * m))));
   k = _mm512_mullo_epi64(k, vm);
   k = _mm512_xor_si512(k, _mm512_srli_epi64(k, 47));
   k = _mm512_mullo_epi64(k, vm);
   h = _mm512_xor_si512(h, k);
   h = _mm512_mullo_epi64(h, vm);
   h = _mm512_xor_si512(h, _mm512_srli_epi64(h, 47));
   h = _mm512_mullo_epi64(h, vm);
   h = _mm512_xor_si512(h, _mm512_srli_epi64(h, 47));
   return h;
}

/// Load 8 consecutive keys of type T (sizeof 1/2/4/8) and widen to 64 bits.
/// Reads exactly 8 * sizeof(T) bytes.
template <typename T> VW_HASH_INLINE __m512i widen8(const void* p) {
   constexpr bool S = sign_extends<T>;
   if constexpr (sizeof(T) == 1) {
      const __m128i v = _mm_loadl_epi64(static_cast<const __m128i*>(p));
      return S ? _mm512_cvtepi8_epi64(v) : _mm512_cvtepu8_epi64(v);
   } else if constexpr (sizeof(T) == 2) {
      const __m128i v = _mm_loadu_si128(static_cast<const __m128i*>(p));
      return S ? _mm512_cvtepi16_epi64(v) : _mm512_cvtepu16_epi64(v);
   } else if constexpr (sizeof(T) == 4) {
      const __m256i v = _mm256_loadu_si256(static_cast<const __m256i*>(p));
      return S ? _mm512_cvtepi32_epi64(v) : _mm512_cvtepu32_epi64(v);
   } else {
      static_assert(sizeof(T) == 8, "unsupported key size");
      return _mm512_loadu_si512(p);
   }
}

/// Same for the last r < 8 keys (copied to a stack buffer; no over-read).
template <typename T> VW_HASH_INLINE __m512i widen_tail(const void* p, size_t r) {
   alignas(64) unsigned char tmp[64] = {0};
   std::memcpy(tmp, p, r * sizeof(T));
   return widen8<T>(tmp);
}

/// Gather 8 keys in[sel[j]] with scalar loads and widen them.
#ifndef VW_SIMD_HASH_GATHER_STACK
/// The keys are assembled in registers (vmovd/vpinsrd, or vmovq/vpinsrq for
/// 8-byte keys): the stack-buffer form (VW_SIMD_HASH_GATHER=stack) stores 8
/// narrow keys and reloads them as one vector, which cannot be forwarded from
/// the stores and stalls (gcc; clang already emits inserts). run_crcbench
/// hash_sel, Zen 4, gcc: 1.27 -> 0.30 ns/key for MurmurHash, 1.73 -> 0.64 for
/// the VPCLMULQDQ CRC; manchego lost Q3 22% / 12% to the stall.
/// Keys of <= 4 bytes are all signed here (int8/16/32, Date): each is widened
/// to int32 by the scalar load and the vector sign-extends to 64 bits, as
/// widen8 does.
template <typename T> VW_HASH_INLINE int32_t key32(const T* in, pos_t i) {
   if constexpr (std::is_same<T, types::Date>::value)
      return in[i].value;
   else
      return int32_t(in[i]);
}
template <typename T>
VW_HASH_INLINE __m512i gather8(const T* in, const pos_t* sel) {
   if constexpr (sizeof(T) == 8) {
      __m128i a = _mm_cvtsi64_si128((long long)in[sel[0]]);
      __m128i b = _mm_cvtsi64_si128((long long)in[sel[2]]);
      __m128i c = _mm_cvtsi64_si128((long long)in[sel[4]]);
      __m128i d = _mm_cvtsi64_si128((long long)in[sel[6]]);
      a = _mm_insert_epi64(a, (long long)in[sel[1]], 1);
      b = _mm_insert_epi64(b, (long long)in[sel[3]], 1);
      c = _mm_insert_epi64(c, (long long)in[sel[5]], 1);
      d = _mm_insert_epi64(d, (long long)in[sel[7]], 1);
      const __m256i lo = _mm256_inserti128_si256(_mm256_castsi128_si256(a), b, 1);
      const __m256i hi = _mm256_inserti128_si256(_mm256_castsi128_si256(c), d, 1);
      return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
   } else {
      static_assert(sign_extends<T>, "keys of <= 4 bytes are signed");
      __m128i lo = _mm_cvtsi32_si128(key32(in, sel[0]));
      __m128i hi = _mm_cvtsi32_si128(key32(in, sel[4]));
      lo = _mm_insert_epi32(lo, key32(in, sel[1]), 1);
      hi = _mm_insert_epi32(hi, key32(in, sel[5]), 1);
      lo = _mm_insert_epi32(lo, key32(in, sel[2]), 2);
      hi = _mm_insert_epi32(hi, key32(in, sel[6]), 2);
      lo = _mm_insert_epi32(lo, key32(in, sel[3]), 3);
      hi = _mm_insert_epi32(hi, key32(in, sel[7]), 3);
      return _mm512_cvtepi32_epi64(
          _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1));
   }
}
#else
template <typename T>
VW_HASH_INLINE __m512i gather8(const T* in, const pos_t* sel) {
   alignas(64) T tmp[8];
   VW_HASH_SCALAR_LOOP
   for (unsigned j = 0; j < 8; ++j) tmp[j] = in[sel[j]];
   return widen8<T>(tmp);
}
#endif

template <typename T>
VW_HASH_INLINE __m512i gather_tail(const T* in, const pos_t* sel, size_t r) {
   alignas(64) T tmp[8];
   std::memset(static_cast<void*>(tmp), 0, sizeof(tmp));
   for (size_t j = 0; j < r; ++j) tmp[j] = in[sel[j]];
   return widen8<T>(tmp);
}

VW_HASH_INLINE __mmask8 tail8(size_t r) { return (__mmask8)((1u << r) - 1); }

//--- primitive replacements (signatures as in Primitives.hpp) ---------------

namespace avx512 {

/// result[i] = hash(input[i], seed)
template <typename T, typename Op>
pos_t hash(pos_t n, hash_t* RES result, T* RES input) {
   static_assert(supported<T>, "unsupported key type");
   const __m512i s = _mm512_set1_epi64((long long)primitives::seed);
   size_t i = 0;
   for (; i + 16 <= n; i += 16) {
      const __m512i h0 = murmur(widen8<T>(input + i), s);
      const __m512i h1 = murmur(widen8<T>(input + i + 8), s);
      _mm512_storeu_si512(result + i, h0);
      _mm512_storeu_si512(result + i + 8, h1);
   }
   for (; i < n; i += 8) {
      const size_t r = (n - i < 8) ? n - i : 8;
      const __m512i k = r == 8 ? widen8<T>(input + i) : widen_tail<T>(input + i, r);
      _mm512_mask_storeu_epi64(result + i, tail8(r), murmur(k, s));
   }
   return n;
}

/// result[i] = hash(input[inSel[i]], seed)
template <typename T, typename Op>
pos_t hash_sel(pos_t n, pos_t* RES inSel, hash_t* RES result, T* RES input) {
   static_assert(supported<T>, "unsupported key type");
   const __m512i s = _mm512_set1_epi64((long long)primitives::seed);
   size_t i = 0;
   for (; i + 16 <= n; i += 16) {
      const __m512i h0 = murmur(gather8<T>(input, inSel + i), s);
      const __m512i h1 = murmur(gather8<T>(input, inSel + i + 8), s);
      _mm512_storeu_si512(result + i, h0);
      _mm512_storeu_si512(result + i + 8, h1);
   }
   for (; i < n; i += 8) {
      const size_t r = (n - i < 8) ? n - i : 8;
      const __m512i k = r == 8 ? gather8<T>(input, inSel + i)
                               : gather_tail<T>(input, inSel + i, r);
      _mm512_mask_storeu_epi64(result + i, tail8(r), murmur(k, s));
   }
   return n;
}

/// result[i] = hash(input[i], result[i])
template <typename T, typename Op>
pos_t rehash(pos_t n, hash_t* RES result, T* RES input) {
   static_assert(supported<T>, "unsupported key type");
   size_t i = 0;
   for (; i + 16 <= n; i += 16) {
      const __m512i h0 =
          murmur(widen8<T>(input + i), _mm512_loadu_si512(result + i));
      const __m512i h1 =
          murmur(widen8<T>(input + i + 8), _mm512_loadu_si512(result + i + 8));
      _mm512_storeu_si512(result + i, h0);
      _mm512_storeu_si512(result + i + 8, h1);
   }
   for (; i < n; i += 8) {
      const size_t r = (n - i < 8) ? n - i : 8;
      const __mmask8 k8 = tail8(r);
      const __m512i k = r == 8 ? widen8<T>(input + i) : widen_tail<T>(input + i, r);
      const __m512i seed = _mm512_maskz_loadu_epi64(k8, result + i);
      _mm512_mask_storeu_epi64(result + i, k8, murmur(k, seed));
   }
   return n;
}

/// result[i] = hash(input[inSel[i]], result[i])
template <typename T, typename Op>
pos_t rehash_sel(pos_t n, pos_t* RES inSel, hash_t* RES result,
                 T* RES input) {
   static_assert(supported<T>, "unsupported key type");
   size_t i = 0;
   for (; i + 16 <= n; i += 16) {
      const __m512i h0 = murmur(gather8<T>(input, inSel + i),
                                _mm512_loadu_si512(result + i));
      const __m512i h1 = murmur(gather8<T>(input, inSel + i + 8),
                                _mm512_loadu_si512(result + i + 8));
      _mm512_storeu_si512(result + i, h0);
      _mm512_storeu_si512(result + i + 8, h1);
   }
   for (; i < n; i += 8) {
      const size_t r = (n - i < 8) ? n - i : 8;
      const __mmask8 k8 = tail8(r);
      const __m512i k = r == 8 ? gather8<T>(input, inSel + i)
                               : gather_tail<T>(input, inSel + i, r);
      const __m512i seed = _mm512_maskz_loadu_epi64(k8, result + i);
      _mm512_mask_storeu_epi64(result + i, k8, murmur(k, seed));
   }
   return n;
}

/// HashGroup packed keys: out[i] = MurMurHash::hashKey(uint64_t(key_i)),
/// keys are n consecutive unsigned T (zero-extended, seed 0).
template <typename T>
void hash_keys(size_t n, const char* RES keys, hash_t* RES out) {
   static_assert(std::is_unsigned<T>::value, "packed keys are unsigned");
   const __m512i s = _mm512_setzero_si512();
   size_t i = 0;
   for (; i + 16 <= n; i += 16) {
      const __m512i h0 = murmur(widen8<T>(keys + i * sizeof(T)), s);
      const __m512i h1 = murmur(widen8<T>(keys + (i + 8) * sizeof(T)), s);
      _mm512_storeu_si512(out + i, h0);
      _mm512_storeu_si512(out + i + 8, h1);
   }
   for (; i < n; i += 8) {
      const size_t r = (n - i < 8) ? n - i : 8;
      const char* p = keys + i * sizeof(T);
      const __m512i k = r == 8 ? widen8<T>(p) : widen_tail<T>(p, r);
      _mm512_mask_storeu_epi64(out + i, tail8(r), murmur(k, s));
   }
}

} // namespace avx512

#endif // VW_HAVE_SIMD_HASH

#ifdef VW_HAVE_SIMD_HASH_AVX2
//--- avx2 tier: 4 keys per ymm, 2 ymm per step ------------------------------
namespace avx2 {

/// MurmurHash64A of one 64-bit key per lane, as simd_hash::murmur; built
/// once per kernel call.
struct Murmur4 {
   __m256i ml, mh, init;
   VW_HASH_INLINE Murmur4()
       : ml(_mm256_set1_epi64x((long long)(kMurmurM & 0xffffffffull))),
         mh(_mm256_set1_epi64x((long long)(kMurmurM >> 32))),
         init(_mm256_set1_epi64x((long long)kMurmurInit)) {}
   /// low 64 bits of k * kMurmurM per lane, from 32 x 32 -> 64 multiplies.
   /// With AVX512DQ+VL clang recognizes the sum as one 64-bit multiply and
   /// emits vpmullq ymm; the empty asm on the cross term keeps the AVX2 code
   /// an AVX2-only machine runs (VW_SIMD_HASH_WIDTH=256 measures that code).
   VW_HASH_INLINE __m256i mul(__m256i k) const {
      __m256i cross = _mm256_add_epi64(
          _mm256_mul_epu32(_mm256_srli_epi64(k, 32), ml), _mm256_mul_epu32(k, mh));
#if defined(__clang__) && defined(__AVX512DQ__)
      asm("" : "+x"(cross));
#endif
      return _mm256_add_epi64(_mm256_mul_epu32(k, ml), _mm256_slli_epi64(cross, 32));
   }
   VW_HASH_INLINE __m256i operator()(__m256i k, __m256i seed) const {
      __m256i h = _mm256_xor_si256(seed, init);
      k = mul(k);
      k = _mm256_xor_si256(k, _mm256_srli_epi64(k, 47));
      k = mul(k);
      h = _mm256_xor_si256(h, k);
      h = mul(h);
      h = _mm256_xor_si256(h, _mm256_srli_epi64(h, 47));
      h = mul(h);
      h = _mm256_xor_si256(h, _mm256_srli_epi64(h, 47));
      return h;
   }
};

/// 4 consecutive keys of type T (sizeof 1/2/4/8) widened to 64 bits; reads
/// exactly 4 * sizeof(T) bytes
template <typename T> VW_HASH_INLINE __m256i widen4(const void* p) {
   constexpr bool S = sign_extends<T>;
   if constexpr (sizeof(T) == 1) {
      int32_t v;
      std::memcpy(&v, p, 4);
      const __m128i x = _mm_cvtsi32_si128(v);
      return S ? _mm256_cvtepi8_epi64(x) : _mm256_cvtepu8_epi64(x);
   } else if constexpr (sizeof(T) == 2) {
      long long v;
      std::memcpy(&v, p, 8);
      const __m128i x = _mm_cvtsi64_si128(v);
      return S ? _mm256_cvtepi16_epi64(x) : _mm256_cvtepu16_epi64(x);
   } else if constexpr (sizeof(T) == 4) {
      const __m128i x = _mm_loadu_si128(static_cast<const __m128i*>(p));
      return S ? _mm256_cvtepi32_epi64(x) : _mm256_cvtepu32_epi64(x);
   } else {
      static_assert(sizeof(T) == 8, "unsupported key size");
      return _mm256_loadu_si256(static_cast<const __m256i*>(p));
   }
}

/// keys in[sel[0..3]], scalar loads assembled in a register
template <typename T>
VW_HASH_INLINE __m256i gather4(const T* in, const pos_t* sel) {
   return _mm256_set_epi64x((long long)key64(in, sel[3]), (long long)key64(in, sel[2]),
                            (long long)key64(in, sel[1]), (long long)key64(in, sel[0]));
}

VW_HASH_INLINE void store4(hash_t* out, __m256i h) {
   _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), h);
}
VW_HASH_INLINE __m256i load4(const hash_t* in) {
   return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in));
}

/// result[i] = hash(input[i], seed)
template <typename T, typename Op>
pos_t hash(pos_t n, hash_t* RES result, T* RES input) {
   static_assert(supported<T>, "unsupported key type");
   const Murmur4 mm;
   const __m256i s = _mm256_set1_epi64x((long long)primitives::seed);
   size_t i = 0;
   for (; i + 8 <= n; i += 8) {
      const __m256i h0 = mm(widen4<T>(input + i), s);
      const __m256i h1 = mm(widen4<T>(input + i + 4), s);
      store4(result + i, h0);
      store4(result + i + 4, h1);
   }
   for (; i < n; ++i) result[i] = Op()(input[i], primitives::seed);
   return n;
}

/// result[i] = hash(input[inSel[i]], seed)
template <typename T, typename Op>
pos_t hash_sel(pos_t n, pos_t* RES inSel, hash_t* RES result, T* RES input) {
   static_assert(supported<T>, "unsupported key type");
   const Murmur4 mm;
   const __m256i s = _mm256_set1_epi64x((long long)primitives::seed);
   size_t i = 0;
   for (; i + 8 <= n; i += 8) {
      const __m256i h0 = mm(gather4<T>(input, inSel + i), s);
      const __m256i h1 = mm(gather4<T>(input, inSel + i + 4), s);
      store4(result + i, h0);
      store4(result + i + 4, h1);
   }
   for (; i < n; ++i) result[i] = Op()(input[inSel[i]], primitives::seed);
   return n;
}

/// result[i] = hash(input[i], result[i])
template <typename T, typename Op>
pos_t rehash(pos_t n, hash_t* RES result, T* RES input) {
   static_assert(supported<T>, "unsupported key type");
   const Murmur4 mm;
   size_t i = 0;
   for (; i + 8 <= n; i += 8) {
      const __m256i h0 = mm(widen4<T>(input + i), load4(result + i));
      const __m256i h1 = mm(widen4<T>(input + i + 4), load4(result + i + 4));
      store4(result + i, h0);
      store4(result + i + 4, h1);
   }
   for (; i < n; ++i) result[i] = Op()(input[i], result[i]);
   return n;
}

/// result[i] = hash(input[inSel[i]], result[i])
template <typename T, typename Op>
pos_t rehash_sel(pos_t n, pos_t* RES inSel, hash_t* RES result,
                 T* RES input) {
   static_assert(supported<T>, "unsupported key type");
   const Murmur4 mm;
   size_t i = 0;
   for (; i + 8 <= n; i += 8) {
      const __m256i h0 = mm(gather4<T>(input, inSel + i), load4(result + i));
      const __m256i h1 =
          mm(gather4<T>(input, inSel + i + 4), load4(result + i + 4));
      store4(result + i, h0);
      store4(result + i + 4, h1);
   }
   for (; i < n; ++i) result[i] = Op()(input[inSel[i]], result[i]);
   return n;
}

/// HashGroup packed keys, as simd_hash::avx512::hash_keys
template <typename T>
void hash_keys(size_t n, const char* RES keys, hash_t* RES out) {
   static_assert(std::is_unsigned<T>::value, "packed keys are unsigned");
   const Murmur4 mm;
   const __m256i s = _mm256_setzero_si256();
   size_t i = 0;
   for (; i + 8 <= n; i += 8) {
      const __m256i h0 = mm(widen4<T>(keys + i * sizeof(T)), s);
      const __m256i h1 = mm(widen4<T>(keys + (i + 4) * sizeof(T)), s);
      store4(out + i, h0);
      store4(out + i + 4, h1);
   }
   const runtime::MurMurHash h;
   for (; i < n; ++i) {
      T k;
      std::memcpy(&k, keys + i * sizeof(T), sizeof(T));
      out[i] = h.hashKey(uint64_t(k));
   }
}

} // namespace avx2
#endif // VW_HAVE_SIMD_HASH_AVX2

#ifdef VW_HAVE_SIMD_HASH_NEON
//--- neon tier: 2 keys per register, 4 registers per step -------------------
namespace neon {

/// low 64 bits of a * b per lane: a*b mod 2^64 = al*bl + ((ah*bl + al*bh) << 32)
VW_HASH_INLINE uint64x2_t mullo64(uint64x2_t a, uint32x2_t bl, uint32x2_t bh) {
   const uint32x2_t al = vmovn_u64(a), ah = vshrn_n_u64(a, 32);
   const uint64x2_t cross = vmlal_u32(vmull_u32(ah, bl), al, bh);
   return vaddq_u64(vmull_u32(al, bl), vshlq_n_u64(cross, 32));
}
VW_HASH_INLINE uint64x2_t mullo64(uint64x2_t a, uint64x2_t b) {
   return mullo64(a, vmovn_u64(b), vshrn_n_u64(b, 32));
}

/// MurmurHash64A of 2 keys, as simd_hash::murmur
VW_HASH_INLINE uint64x2_t murmur2(uint64x2_t k, uint64x2_t seed) {
   const uint32x2_t ml = vdup_n_u32(uint32_t(kMurmurM)),
                    mh = vdup_n_u32(uint32_t(kMurmurM >> 32));
   uint64x2_t h = veorq_u64(seed, vdupq_n_u64(kMurmurInit));
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

/// 4 int32 / uint32 lanes widened to 2 x 2 64-bit lanes
template <bool S> VW_HASH_INLINE void widen32(uint32x4_t x, uint64x2_t* k) {
   if constexpr (S) {
      const int32x4_t s = vreinterpretq_s32_u32(x);
      k[0] = vreinterpretq_u64_s64(vmovl_s32(vget_low_s32(s)));
      k[1] = vreinterpretq_u64_s64(vmovl_high_s32(s));
   } else {
      k[0] = vmovl_u32(vget_low_u32(x));
      k[1] = vmovl_high_u32(x);
   }
}
/// 8 int16 / uint16 lanes widened to 4 x 2 64-bit lanes
template <bool S> VW_HASH_INLINE void widen16(uint16x8_t x, uint64x2_t* k) {
   if constexpr (S) {
      const int16x8_t s = vreinterpretq_s16_u16(x);
      widen32<true>(vreinterpretq_u32_s32(vmovl_s16(vget_low_s16(s))), k);
      widen32<true>(vreinterpretq_u32_s32(vmovl_high_s16(s)), k + 2);
   } else {
      widen32<false>(vmovl_u16(vget_low_u16(x)), k);
      widen32<false>(vmovl_high_u16(x), k + 2);
   }
}

/// 8 consecutive keys of type T widened to 64 bits in k[0..3]; reads exactly
/// 8 * sizeof(T) bytes
template <typename T> VW_HASH_INLINE void widen8(const void* p, uint64x2_t* k) {
   constexpr bool S = sign_extends<T>;
   const uint8_t* b = static_cast<const uint8_t*>(p);
   if constexpr (sizeof(T) == 1) {
      const uint8x8_t x = vld1_u8(b);
      widen16<S>(S ? vreinterpretq_u16_s16(vmovl_s8(vreinterpret_s8_u8(x)))
                   : vmovl_u8(x),
                 k);
   } else if constexpr (sizeof(T) == 2) {
      widen16<S>(vreinterpretq_u16_u8(vld1q_u8(b)), k);
   } else if constexpr (sizeof(T) == 4) {
      widen32<S>(vreinterpretq_u32_u8(vld1q_u8(b)), k);
      widen32<S>(vreinterpretq_u32_u8(vld1q_u8(b + 16)), k + 2);
   } else {
      static_assert(sizeof(T) == 8, "unsupported key size");
      for (int j = 0; j < 4; ++j) k[j] = vreinterpretq_u64_u8(vld1q_u8(b + 16 * j));
   }
}

/// keys in[sel[0..7]] in k[0..3], scalar loads into lanes
template <typename T>
VW_HASH_INLINE void gather8(const T* in, const pos_t* sel, uint64x2_t* k) {
   for (int j = 0; j < 4; ++j)
      k[j] = vcombine_u64(vcreate_u64(key64(in, sel[2 * j])),
                          vcreate_u64(key64(in, sel[2 * j + 1])));
}

/// out[0..7] = murmur of k[0..3] with the seeds s[0..3]
VW_HASH_INLINE void hash8(hash_t* out, const uint64x2_t* k, const uint64x2_t* s) {
   uint64x2_t h[4];
   for (int j = 0; j < 4; ++j) h[j] = murmur2(k[j], s[j]);
   for (int j = 0; j < 4; ++j) vst1q_u64(out + 2 * j, h[j]);
}

/// result[i] = hash(input[i], seed)
template <typename T, typename Op>
pos_t hash(pos_t n, hash_t* RES result, T* RES input) {
   static_assert(supported<T>, "unsupported key type");
   const uint64x2_t s1 = vdupq_n_u64(primitives::seed);
   const uint64x2_t s[4] = {s1, s1, s1, s1};
   size_t i = 0;
   for (; i + 8 <= n; i += 8) {
      uint64x2_t k[4];
      widen8<T>(input + i, k);
      hash8(result + i, k, s);
   }
   for (; i < n; ++i) result[i] = Op()(input[i], primitives::seed);
   return n;
}

/// result[i] = hash(input[inSel[i]], seed)
template <typename T, typename Op>
pos_t hash_sel(pos_t n, pos_t* RES inSel, hash_t* RES result, T* RES input) {
   static_assert(supported<T>, "unsupported key type");
   const uint64x2_t s1 = vdupq_n_u64(primitives::seed);
   const uint64x2_t s[4] = {s1, s1, s1, s1};
   size_t i = 0;
   for (; i + 8 <= n; i += 8) {
      uint64x2_t k[4];
      gather8<T>(input, inSel + i, k);
      hash8(result + i, k, s);
   }
   for (; i < n; ++i) result[i] = Op()(input[inSel[i]], primitives::seed);
   return n;
}

/// result[i] = hash(input[i], result[i])
template <typename T, typename Op>
pos_t rehash(pos_t n, hash_t* RES result, T* RES input) {
   static_assert(supported<T>, "unsupported key type");
   size_t i = 0;
   for (; i + 8 <= n; i += 8) {
      uint64x2_t k[4], s[4];
      widen8<T>(input + i, k);
      for (int j = 0; j < 4; ++j) s[j] = vld1q_u64(result + i + 2 * j);
      hash8(result + i, k, s);
   }
   for (; i < n; ++i) result[i] = Op()(input[i], result[i]);
   return n;
}

/// result[i] = hash(input[inSel[i]], result[i])
template <typename T, typename Op>
pos_t rehash_sel(pos_t n, pos_t* RES inSel, hash_t* RES result,
                 T* RES input) {
   static_assert(supported<T>, "unsupported key type");
   size_t i = 0;
   for (; i + 8 <= n; i += 8) {
      uint64x2_t k[4], s[4];
      gather8<T>(input, inSel + i, k);
      for (int j = 0; j < 4; ++j) s[j] = vld1q_u64(result + i + 2 * j);
      hash8(result + i, k, s);
   }
   for (; i < n; ++i) result[i] = Op()(input[inSel[i]], result[i]);
   return n;
}

/// HashGroup packed keys, as simd_hash::avx512::hash_keys
template <typename T>
void hash_keys(size_t n, const char* RES keys, hash_t* RES out) {
   static_assert(std::is_unsigned<T>::value, "packed keys are unsigned");
   const uint64x2_t z = vdupq_n_u64(0);
   const uint64x2_t s[4] = {z, z, z, z};
   size_t i = 0;
   for (; i + 8 <= n; i += 8) {
      uint64x2_t k[4];
      widen8<T>(keys + i * sizeof(T), k);
      hash8(out + i, k, s);
   }
   const runtime::MurMurHash h;
   for (; i < n; ++i) {
      T k;
      std::memcpy(&k, keys + i * sizeof(T), sizeof(T));
      out[i] = h.hashKey(uint64_t(k));
   }
}

} // namespace neon
#endif // VW_HAVE_SIMD_HASH_NEON

//--- the public names: the tier VW_SIMD_HASH_TIER_* selects -----------------
#if defined(VW_SIMD_HASH_TIER_AVX512)
using avx512::hash;
using avx512::hash_keys;
using avx512::hash_sel;
using avx512::rehash;
using avx512::rehash_sel;
#elif defined(VW_SIMD_HASH_TIER_AVX2)
using avx2::hash;
using avx2::hash_keys;
using avx2::hash_sel;
using avx2::rehash;
using avx2::rehash_sel;
#elif defined(VW_SIMD_HASH_TIER_NEON)
using neon::hash;
using neon::hash_keys;
using neon::hash_sel;
using neon::rehash;
using neon::rehash_sel;
#endif

//--- pickers used by Hash.cpp -------------------------------------------------
// Return the SIMD kernel when VW_SIMD_HASH is on, some tier is available, the
// hash is MurMurHash and T is supported; otherwise `scalar`.

template <typename T, typename Op> constexpr bool use_simd() {
#if defined(VW_SIMD_HASH) && defined(VW_HAVE_SIMD_HASH_ANY)
   return supported<T> && std::is_same<Op, runtime::MurMurHash>::value;
#else
   return false;
#endif
}

template <typename T, typename Op, typename Fn>
constexpr Fn pick_hash(Fn scalar) {
#if defined(VW_HAVE_SIMD_HASH_ANY)
   if constexpr (use_simd<T, Op>()) return &hash<T, Op>;
#endif
   return scalar;
}
template <typename T, typename Op, typename Fn>
constexpr Fn pick_hash_sel(Fn scalar) {
#if defined(VW_HAVE_SIMD_HASH_ANY)
   if constexpr (use_simd<T, Op>()) return &hash_sel<T, Op>;
#endif
   return scalar;
}
template <typename T, typename Op, typename Fn>
constexpr Fn pick_rehash(Fn scalar) {
#if defined(VW_HAVE_SIMD_HASH_ANY)
   if constexpr (use_simd<T, Op>()) return &rehash<T, Op>;
#endif
   return scalar;
}
template <typename T, typename Op, typename Fn>
constexpr Fn pick_rehash_sel(Fn scalar) {
#if defined(VW_HAVE_SIMD_HASH_ANY)
   if constexpr (use_simd<T, Op>()) return &rehash_sel<T, Op>;
#endif
   return scalar;
}

} // namespace simd_hash
} // namespace primitives
} // namespace vectorwise
