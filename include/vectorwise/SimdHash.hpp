#pragma once
//===----------------------------------------------------------------------===//
// AVX-512 MurmurHash64A kernels (CMake option VW_SIMD_HASH).
//
// Replacements for the hash / hash_sel / rehash / rehash_sel primitive
// templates (Primitives.hpp) and for HashGroup's packed-key hash, computing 8
// hashes per zmm. Results are bit-identical to runtime::MurMurHash, because
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
// Needs x86-64 + AVX512F + AVX512DQ (vpmullq) and 64-bit hashes; otherwise
// nothing below is defined and pick_* return the scalar pointer.
//===----------------------------------------------------------------------===//
#include "common/runtime/Hash.hpp"
#include "common/runtime/Types.hpp"
#include "vectorwise/Primitives.hpp"
#include "vectorwise/defs.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__x86_64__) && defined(__AVX512F__) && defined(__AVX512DQ__) &&   \
    !(defined(HASH_SIZE) && HASH_SIZE == 32)
#include <immintrin.h>
#define VW_HAVE_SIMD_HASH 1
#endif

namespace vectorwise {
namespace primitives {
namespace simd_hash {

#ifdef VW_HAVE_SIMD_HASH

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
template <typename T>
VW_HASH_INLINE __m512i gather8(const T* in, const pos_t* sel) {
   alignas(64) T tmp[8];
   VW_HASH_SCALAR_LOOP
   for (unsigned j = 0; j < 8; ++j) tmp[j] = in[sel[j]];
   return widen8<T>(tmp);
}

template <typename T>
VW_HASH_INLINE __m512i gather_tail(const T* in, const pos_t* sel, size_t r) {
   alignas(64) T tmp[8];
   std::memset(static_cast<void*>(tmp), 0, sizeof(tmp));
   for (size_t j = 0; j < r; ++j) tmp[j] = in[sel[j]];
   return widen8<T>(tmp);
}

VW_HASH_INLINE __mmask8 tail8(size_t r) { return (__mmask8)((1u << r) - 1); }

//--- primitive replacements (signatures as in Primitives.hpp) ---------------

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

#endif // VW_HAVE_SIMD_HASH

//--- pickers used by Hash.cpp -------------------------------------------------
// Return the SIMD kernel when VW_SIMD_HASH is on, the ISA is available, the
// hash is MurMurHash and T is supported; otherwise `scalar`.

template <typename T, typename Op> constexpr bool use_simd() {
#if defined(VW_SIMD_HASH) && defined(VW_HAVE_SIMD_HASH)
   return supported<T> && std::is_same<Op, runtime::MurMurHash>::value;
#else
   return false;
#endif
}

template <typename T, typename Op, typename Fn>
constexpr Fn pick_hash(Fn scalar) {
#if defined(VW_HAVE_SIMD_HASH)
   if constexpr (use_simd<T, Op>()) return &hash<T, Op>;
#endif
   return scalar;
}
template <typename T, typename Op, typename Fn>
constexpr Fn pick_hash_sel(Fn scalar) {
#if defined(VW_HAVE_SIMD_HASH)
   if constexpr (use_simd<T, Op>()) return &hash_sel<T, Op>;
#endif
   return scalar;
}
template <typename T, typename Op, typename Fn>
constexpr Fn pick_rehash(Fn scalar) {
#if defined(VW_HAVE_SIMD_HASH)
   if constexpr (use_simd<T, Op>()) return &rehash<T, Op>;
#endif
   return scalar;
}
template <typename T, typename Op, typename Fn>
constexpr Fn pick_rehash_sel(Fn scalar) {
#if defined(VW_HAVE_SIMD_HASH)
   if constexpr (use_simd<T, Op>()) return &rehash_sel<T, Op>;
#endif
   return scalar;
}

} // namespace simd_hash
} // namespace primitives
} // namespace vectorwise
