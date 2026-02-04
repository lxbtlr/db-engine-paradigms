#pragma once

#ifndef SIMDE_ENABLE_NATIVE_ALIASES
#define SIMDE_ENABLE_NATIVE_ALIASES
#endif

// --- SIMDe Extension Headers ---
// Provides maskz_loadu variants
#if !defined(__x86_64__)
    #include <simde/x86/avx512/types.h>
    #include <simde/x86/avx512/mov.h>      // Essential for maskz_loadu visibility
    #include <simde/x86/avx512/setzero.h>
    #include <simde/x86/avx512/set.h>
    #include <simde/x86/avx512/movm.h>     // for movm_epi32

    #ifndef _mm256_maskz_loadu_epi32
    #if defined(SIMDE_X86_AVX512VL_NATIVE) && defined(SIMDE_X86_AVX512F_NATIVE)
        #define _mm256_maskz_loadu_epi32(k, mem_addr) _mm256_maskz_loadu_epi32(k, mem_addr)
    #else
        // Manual fallback using movm (vector mask)
        static inline simde__m256i _mm256_maskz_loadu_epi32(simde__mmask8 k, void const* mem_addr) {
            simde__m256i v = simde_mm256_loadu_si256((simde__m256i const*)mem_addr);
            simde__m256i m = simde_mm256_movm_epi32(k);
            return simde_mm256_and_si256(v, m);
        }
    #endif
    #endif

    // 2. Specific feature headers
    #include <simde/x86/avx512/loadu.h>    // Resolves _mm256_maskz_loadu_epi32
    #include <simde/x86/avx512/gather.h>   // Resolves i32gather
    #include <simde/x86/avx512/cvt.h>      // Resolves _mm512_cvtepu32_epi64

// Manual implementation of _mm512_cvtepi32_epi64 (Signed 32->64 ext)
static inline simde__m512i simde_mm512_cvtepi32_epi64(simde__m256i a) {
    simde__m128i lo = simde_mm256_castsi256_si128(a);
    simde__m128i hi = simde_mm256_extracti128_si256(a, 1);
    simde__m256i lo64 = simde_mm256_cvtepi32_epi64(lo);
    simde__m256i hi64 = simde_mm256_cvtepi32_epi64(hi);
    // Combine into 512
    simde__m512i res = simde_mm512_setzero_si512();
    res = simde_mm512_inserti64x4(res, lo64, 0);
    res = simde_mm512_inserti64x4(res, hi64, 1);
    return res;
}

// Manual implementation of _mm512_cvtepu32_epi64 (Unsigned 32->64 ext)
static inline simde__m512i simde_mm512_cvtepu32_epi64(simde__m256i a) {
    simde__m128i lo = simde_mm256_castsi256_si128(a);
    simde__m128i hi = simde_mm256_extracti128_si256(a, 1);
    simde__m256i lo64 = simde_mm256_cvtepu32_epi64(lo);
    simde__m256i hi64 = simde_mm256_cvtepu32_epi64(hi);
    simde__m512i res = simde_mm512_setzero_si512();
    res = simde_mm512_inserti64x4(res, lo64, 0);
    res = simde_mm512_inserti64x4(res, hi64, 1);
    return res;
}

#define _mm512_cvtepi32_epi64(a) simde_mm512_cvtepi32_epi64(a)
#define _mm512_cvtepu32_epi64(a) simde_mm512_cvtepu32_epi64(a)

    #ifndef _mm512_i32gather_epi64
    // Manual fallback: Upcast indices to 64-bit and use i64gather
    static inline simde__m512i _mm512_i32gather_epi64(simde__m256i idxs, void const* base_addr, int scale) {
        simde__m512i idxs64 = simde_mm512_cvtepi32_epi64(idxs);
        return simde_mm512_i64gather_epi64(idxs64, base_addr, scale);
    }
    #endif

#define _mm512_cvtepi32_epi64(a) simde_mm512_cvtepi32_epi64(a)
#define _mm512_cvtepu32_epi64(a) simde_mm512_cvtepu32_epi64(a)
#endif
#include <algorithm>
#include <iostream>
#include <ostream>
#include <vector>
#include <cstdint>

// --- Architecture Detection & SIMD Mapping ---
#if defined(__x86_64__) && defined(__AVX512F__)
    #include <immintrin.h>
    using vec_reg_t = __m512i;
    using mask8_t = __mmask8;
    using mask16_t = __mmask16;
    using mask32_t = __mmask32;
    using mask64_t = __mmask64;
#else
    // Raspberry Pi / ARM Path
    using vec_reg_t = simde__m512i;
    using mask8_t = simde__mmask8;
    using mask16_t = simde__mmask16;
    using mask32_t = simde__mmask32;
    using mask64_t = simde__mmask64;

    // --- The Crucial Fix: Explicit Declaration ---
    // This tells the compiler exactly what _mm512_cvtepu32_epi64 is 
    // before Primitives.hpp tries to use it.
    #if !defined(_mm512_cvtepu32_epi64)
        #define _mm512_cvtepu32_epi64(a) simde_mm512_cvtepu32_epi64(a)
    #endif
#endif

// --- Vec8u (64-bit elements x 8) ---
struct Vec8u {
   union {
      vec_reg_t reg;
      uint64_t entry[8];
   };

   explicit Vec8u(uint64_t x) { reg = _mm512_set1_epi64(x); }
   explicit Vec8u(void* p) { reg = _mm512_loadu_si512(p); }
   Vec8u(vec_reg_t x) { reg = x; }
   Vec8u(uint64_t x0, uint64_t x1, uint64_t x2, uint64_t x3, uint64_t x4, uint64_t x5, uint64_t x6, uint64_t x7) { 
       reg = _mm512_set_epi64(x0, x1, x2, x3, x4, x5, x6, x7); 
   }

   operator vec_reg_t() const { return reg; }

   friend std::ostream& operator<< (std::ostream& stream, const Vec8u& v) {
      for (auto& e : v.entry) stream << e << " ";
      return stream;
   }
};

struct Vec8uM {
   Vec8u vec;
   mask8_t mask;
};

// --- Vec8u Operators ---
inline Vec8u operator+ (const Vec8u& a, const Vec8u& b) { return _mm512_add_epi64(a.reg, b.reg); }
inline Vec8u operator- (const Vec8u& a, const Vec8u& b) { return _mm512_sub_epi64(a.reg, b.reg); }
inline Vec8u operator* (const Vec8u& a, const Vec8u& b) { return _mm512_mullo_epi64(a.reg, b.reg); }
inline Vec8u operator^ (const Vec8u& a, const Vec8u& b) { return _mm512_xor_epi64(a.reg, b.reg); }
inline Vec8u operator& (const Vec8u& a, const Vec8u& b) { return _mm512_and_epi64(a.reg, b.reg); }
inline Vec8u operator| (const Vec8u& a, const Vec8u& b) { return _mm512_or_epi64(a.reg, b.reg); }
inline Vec8u operator>> (const Vec8u& a, const unsigned int shift) { return _mm512_srli_epi64(a.reg, shift); }
inline Vec8u operator<< (const Vec8u& a, const unsigned int shift) { return _mm512_slli_epi64(a.reg, shift); }
inline Vec8u operator>> (const Vec8u& a, const Vec8u& b) { return _mm512_srlv_epi64(a.reg, b.reg); }
inline Vec8u operator<< (const Vec8u& a, const Vec8u& b) { return _mm512_sllv_epi64(a.reg, b.reg); }

// --- Vec8u Comparisons (0:==, 1:<, 2:<=, 4:!=, 5:>=, 6:>) ---
inline mask8_t operator== (const Vec8u& a, const Vec8u& b) { return _mm512_cmp_epi64_mask(a.reg, b.reg, 0); }
inline mask8_t operator<  (const Vec8u& a, const Vec8u& b) { return _mm512_cmp_epi64_mask(a.reg, b.reg, 1); }
inline mask8_t operator<= (const Vec8u& a, const Vec8u& b) { return _mm512_cmp_epi64_mask(a.reg, b.reg, 2); }
inline mask8_t operator!= (const Vec8u& a, const Vec8u& b) { return _mm512_cmp_epi64_mask(a.reg, b.reg, 4); }
inline mask8_t operator>= (const Vec8u& a, const Vec8u& b) { return _mm512_cmp_epi64_mask(a.reg, b.reg, 5); }
inline mask8_t operator>  (const Vec8u& a, const Vec8u& b) { return _mm512_cmp_epi64_mask(a.reg, b.reg, 6); }

// --- Vec16u (32-bit elements x 16) ---
struct Vec16u {
   union {
      vec_reg_t reg;
      uint32_t entry[16];
   };

   explicit Vec16u(uint32_t x) { reg = _mm512_set1_epi32(x); }
   explicit Vec16u(void* p) { reg = _mm512_loadu_si512(p); }
   Vec16u(vec_reg_t x) { reg = x; }
   Vec16u(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, uint32_t x4, uint32_t x5, uint32_t x6, uint32_t x7,
          uint32_t x8, uint32_t x9, uint32_t x10, uint32_t x11, uint32_t x12, uint32_t x13, uint32_t x14, uint32_t x15) {
       reg = _mm512_set_epi32(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);
   }

   operator vec_reg_t() const { return reg; }

   friend std::ostream& operator<< (std::ostream& stream, const Vec16u& v) {
      for (auto& e : v.entry) stream << e << " ";
      return stream;
   }
};

struct Vec16uM {
   Vec16u vec;
   mask16_t mask;
};

// --- Vec16u Arithmetic & Logic ---
inline Vec16u operator+ (const Vec16u& a, const Vec16u& b) { return _mm512_add_epi32(a.reg, b.reg); }
inline Vec16u operator- (const Vec16u& a, const Vec16u& b) { return _mm512_sub_epi32(a.reg, b.reg); }
inline Vec16u operator* (const Vec16u& a, const Vec16u& b) { return _mm512_mullo_epi32(a.reg, b.reg); }
inline Vec16u operator^ (const Vec16u& a, const Vec16u& b) { return _mm512_xor_epi32(a.reg, b.reg); }
inline Vec16u operator& (const Vec16u& a, const Vec16u& b) { return _mm512_and_epi32(a.reg, b.reg); }
inline Vec16u operator| (const Vec16u& a, const Vec16u& b) { return _mm512_or_epi32(a.reg, b.reg); }
inline Vec16u operator>> (const Vec16u& a, const unsigned int shift) { return _mm512_srli_epi32(a.reg, shift); }
inline Vec16u operator<< (const Vec16u& a, const unsigned int shift) { return _mm512_slli_epi32(a.reg, shift); }
inline Vec16u operator>> (const Vec16u& a, const Vec16u& b) { return _mm512_srlv_epi32(a.reg, b.reg); }
inline Vec16u operator<< (const Vec16u& a, const Vec16u& b) { return _mm512_sllv_epi32(a.reg, b.reg); }

// --- Vec16u Comparisons ---
inline mask16_t operator== (const Vec16u& a, const Vec16u& b) { return _mm512_cmp_epi32_mask(a.reg, b.reg, 0); }
inline mask16_t operator<  (const Vec16u& a, const Vec16u& b) { return _mm512_cmp_epi32_mask(a.reg, b.reg, 1); }
inline mask16_t operator<= (const Vec16u& a, const Vec16u& b) { return _mm512_cmp_epi32_mask(a.reg, b.reg, 2); }
inline mask16_t operator!= (const Vec16u& a, const Vec16u& b) { return _mm512_cmp_epi32_mask(a.reg, b.reg, 4); }
inline mask16_t operator>= (const Vec16u& a, const Vec16u& b) { return _mm512_cmp_epi32_mask(a.reg, b.reg, 5); }
inline mask16_t operator>  (const Vec16u& a, const Vec16u& b) { return _mm512_cmp_epi32_mask(a.reg, b.reg, 6); }
