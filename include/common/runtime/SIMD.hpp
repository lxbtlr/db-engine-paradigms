#pragma once

#ifndef SIMDE_ENABLE_NATIVE_ALIASES
#define SIMDE_ENABLE_NATIVE_ALIASES
#endif

#include <algorithm>
#include <iostream>
#include <ostream>
#include <vector>
#include <cstdint>

// --- Architecture Detection & SIMD Mapping ---
#if defined(__x86_64__) && defined(__AVX512F__)
    // Native AVX512 Path
    #include <immintrin.h>
    using vec_reg_t = __m512i;
    using mask8_t = __mmask8;
    using mask16_t = __mmask16;
    using mask32_t = __mmask32;
    using mask64_t = __mmask64;
#else
    // SIMDe Path (ARM or x86_64 without AVX512)
    #include <simde/x86/avx512.h>
    #include <simde/simde-common.h>
    // Specific feature headers for manual fixes/visibility
    #include <simde/x86/avx512/types.h>
    #include <simde/x86/avx512/mov.h>
    #include <simde/x86/avx512/setzero.h>
    #include <simde/x86/avx512/set.h>
    #include <simde/x86/avx512/movm.h>
    #include <simde/x86/avx512/loadu.h>
    #include <simde/x86/avx512/gather.h>
    #include <simde/x86/avx512/cvt.h>
    #include <simde/x86/avx512/insert.h>   // Resolves simde_mm512_inserti64x4
    #include <simde/x86/avx512/extract.h>  // Resolves simde_mm512_extracti64x4_epi64.h>

    using vec_reg_t = simde__m512i;
    using mask8_t = simde__mmask8;
    using mask16_t = simde__mmask16;
    using mask32_t = simde__mmask32;
    using mask64_t = simde__mmask64;

    // --- Manual Polyfills for missing intrinsics in older SIMDe ---

    // 1. _mm256_maskz_loadu_epi32
    #ifndef _mm256_maskz_loadu_epi32
    #if defined(SIMDE_X86_AVX512VL_NATIVE) && defined(SIMDE_X86_AVX512F_NATIVE)
        #define _mm256_maskz_loadu_epi32(k, mem_addr) _mm256_maskz_loadu_epi32(k, mem_addr)
    #else
        static inline simde__m256i _mm256_maskz_loadu_epi32(simde__mmask8 k, void const* mem_addr) {
            simde__m256i v = simde_mm256_loadu_si256((simde__m256i const*)mem_addr);
            simde__m256i m = simde_mm256_movm_epi32(k);
            return simde_mm256_and_si256(v, m);
        }
    #endif
    #endif

    // 2. Conversion Intrinsics (Signed/Unsigned 32->64)
    #if !defined(_mm512_cvtepi32_epi64)
    static inline simde__m512i simde_mm512_cvtepi32_epi64(simde__m256i a) {
        simde__m128i lo = simde_mm256_castsi256_si128(a);
        simde__m128i hi = simde_mm256_extracti128_si256(a, 1);
        simde__m256i lo64 = simde_mm256_cvtepi32_epi64(lo);
        simde__m256i hi64 = simde_mm256_cvtepi32_epi64(hi);
        simde__m512i res = simde_mm512_setzero_si512();
        res = simde_mm512_inserti64x4(res, lo64, 0);
        res = simde_mm512_inserti64x4(res, hi64, 1);
        return res;
    }
    #define _mm512_cvtepi32_epi64(a) simde_mm512_cvtepi32_epi64(a)
    #endif

    #if !defined(_mm512_cvtepu32_epi64)
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
    #define _mm512_cvtepu32_epi64(a) simde_mm512_cvtepu32_epi64(a)
    #endif

    // 4. Comparison Aliases (Missing in some SIMDe builds)
    #ifndef _mm512_cmplt_epi32_mask
    #define _mm512_cmplt_epi32_mask(a, b) simde_mm512_cmp_epi32_mask(a, b, 1)
    #endif
    #ifndef _mm512_cmplt_epi64_mask
    #define _mm512_cmplt_epi64_mask(a, b) simde_mm512_cmp_epi64_mask(a, b, 1)
    #endif
    #ifndef _mm512_cmpge_epi32_mask
    #define _mm512_cmpge_epi32_mask(a, b) simde_mm512_cmp_epi32_mask(a, b, 5)
    #endif

    // 5. Gather Fallback (i32 indices -> 32-bit values)
    #ifndef _mm512_i32gather_epi32
    // Manual Split Implementation: 512 -> 2x256
    static inline simde__m512i _mm512_i32gather_epi32(simde__m512i idxs, void const* base_addr, int scale) {
        simde__m256i idx_lo = simde_mm512_castsi512_si256(idxs);
        // Extract high 256 bits. 0x1 selects the high half.
        simde__m256i idx_hi = simde_mm512_extracti64x4_epi64(idxs, 1);
        
        simde__m256i res_lo, res_hi;

        switch (scale) {
            case 1:
                res_lo = simde_mm256_i32gather_epi32((int const*)base_addr, idx_lo, 1);
                res_hi = simde_mm256_i32gather_epi32((int const*)base_addr, idx_hi, 1);
                break;
            case 2:
                res_lo = simde_mm256_i32gather_epi32((int const*)base_addr, idx_lo, 2);
                res_hi = simde_mm256_i32gather_epi32((int const*)base_addr, idx_hi, 2);
                break;
            case 4:
                res_lo = simde_mm256_i32gather_epi32((int const*)base_addr, idx_lo, 4);
                res_hi = simde_mm256_i32gather_epi32((int const*)base_addr, idx_hi, 4);
                break;
            case 8:
                res_lo = simde_mm256_i32gather_epi32((int const*)base_addr, idx_lo, 8);
                res_hi = simde_mm256_i32gather_epi32((int const*)base_addr, idx_hi, 8);
                break;
            default:
                // Should not happen for valid gathers, but fallback or crash
                res_lo = simde_mm256_setzero_si256();
                res_hi = simde_mm256_setzero_si256();
                break;
        }
        
        simde__m512i res = simde_mm512_setzero_si512();
        res = simde_mm512_inserti64x4(res, res_lo, 0);
        res = simde_mm512_inserti64x4(res, res_hi, 1);
        return res;
    }
    #endif

    // 3. Gather Fallback (i32 indices -> 64-bit values)
    #ifndef _mm512_i32gather_epi64
    static inline simde__m512i _mm512_i32gather_epi64(simde__m256i idxs, void const* base_addr, int scale) {
        simde__m512i idxs64 = simde_mm512_cvtepi32_epi64(idxs);
        // Note: i64gather usage requires checking if headers define it properly
        switch (scale) {
            case 1: return simde_mm512_i64gather_epi64(idxs64, base_addr, 1);
            case 2: return simde_mm512_i64gather_epi64(idxs64, base_addr, 2);
            case 4: return simde_mm512_i64gather_epi64(idxs64, base_addr, 4);
            case 8: return simde_mm512_i64gather_epi64(idxs64, base_addr, 8);
            default: return simde_mm512_setzero_si512();
        }
    }
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
