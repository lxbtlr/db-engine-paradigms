// VW_GROUP_CONCAT_SIMD kernels, see SimdConcat.hpp. Compiled always (the
// microbenchmark and tests call them); HashGroup uses them only with the
// option.
#include "vectorwise/SimdConcat.hpp"
#include "common/runtime/Hash.hpp"
#include "vectorwise/SimdHash.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <utility>
#if defined(VW_CONCAT_HAVE_VBMI) || defined(VW_CONCAT_HAVE_AVX2)
#include <immintrin.h>
#endif
#ifdef VW_CONCAT_HAVE_NEON
#include <arm_neon.h>
#endif
// fused concat + hash on AVX-512 (simd_hash::murmur), except where the
// target downclocks under AVX-512 (VW_CONCAT_NO_AVX512_HASH, dubliner)
#if defined(VW_HAVE_SIMD_HASH) && !defined(VW_CONCAT_NO_AVX512_HASH)
#define VW_CONCAT_FUSED_SIMD 1
#endif

namespace vectorwise {
namespace primitives {
namespace simd_concat {

namespace {

/// one strided pass over rows [from, n) of a W-byte column (Concat_T)
template <size_t W>
void columnPass(pos_t from, pos_t n, const char* __restrict__ src,
                const pos_t* __restrict__ sel, char* __restrict__ dest,
                uint32_t keySize) {
   if (sel) {
      for (pos_t i = from; i < n; i++)
         std::memcpy(dest + size_t(i) * keySize, src + size_t(sel[i]) * W, W);
   } else {
      for (pos_t i = from; i < n; i++)
         std::memcpy(dest + size_t(i) * keySize, src + size_t(i) * W, W);
   }
}

/// the same for a column whose width is only known at run time
/// (Concat_T<char*>)
void columnPassWide(pos_t from, pos_t n, const char* __restrict__ src,
                    const pos_t* __restrict__ sel, char* __restrict__ dest,
                    uint32_t keySize, uint32_t w) {
   if (sel) {
      for (pos_t i = from; i < n; i++)
         std::memcpy(dest + size_t(i) * keySize, src + size_t(sel[i]) * w, w);
   } else {
      for (pos_t i = from; i < n; i++)
         std::memcpy(dest + size_t(i) * keySize, src + size_t(i) * w, w);
   }
}

void columnsFrom(size_t cols, const uint32_t* width, const uint32_t* offset,
                 uint32_t keySize, pos_t from, pos_t n,
                 const char* const* data, const pos_t* const* sel,
                 char* out) {
   for (size_t c = 0; c < cols; ++c) {
      char* dest = out + offset[c];
      switch (width[c]) {
      case 1: columnPass<1>(from, n, data[c], sel[c], dest, keySize); break;
      case 2: columnPass<2>(from, n, data[c], sel[c], dest, keySize); break;
      case 4: columnPass<4>(from, n, data[c], sel[c], dest, keySize); break;
      case 8: columnPass<8>(from, n, data[c], sel[c], dest, keySize); break;
      default:
         columnPassWide(from, n, data[c], sel[c], dest, keySize, width[c]);
      }
   }
}

/// rows [from, n) of the plan's columns, after the last full block
void tail(const Plan& p, pos_t from, pos_t n, const char* const* data,
          const pos_t* const* sel, char* out) {
   if (from >= n) return;
   uint32_t width[kMaxCols];
   for (size_t c = 0; c < p.cols; ++c) width[c] = p.width[c];
   const pos_t* s[kMaxCols];
   for (size_t c = 0; c < p.cols; ++c) s[c] = p.sel[c] ? sel[c] : nullptr;
   columnsFrom(p.cols, width, p.offset, p.keySize, from, n, data, s, out);
}

/// Stage: the m values of a column read through s, packed into st
template <size_t W>
inline void stageW(char* __restrict__ st, const char* __restrict__ src,
                   const pos_t* __restrict__ s, uint32_t m) {
   for (uint32_t j = 0; j < m; ++j)
      std::memcpy(st + j * W, src + size_t(s[j]) * W, W);
}
inline void stage(char* st, uint32_t w, const char* src, const pos_t* s,
                  uint32_t m) {
   switch (w) {
   case 1: stageW<1>(st, src, s, m); break;
   case 2: stageW<2>(st, src, s, m); break;
   case 4: stageW<4>(st, src, s, m); break;
   default: stageW<8>(st, src, s, m); break;
   }
}

#ifdef VW_CONCAT_HAVE_AVX2
inline __m256i idx8(const pos_t* s) {
#ifdef VW_POS_16
   return _mm256_cvtepu16_epi32(
       _mm_loadu_si128(reinterpret_cast<const __m128i*>(s)));
#else
   return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s));
#endif
}
inline __m128i idx4(const pos_t* s) {
#ifdef VW_POS_16
   return _mm_cvtepu16_epi32(
       _mm_loadl_epi64(reinterpret_cast<const __m128i*>(s)));
#else
   return _mm_loadu_si128(reinterpret_cast<const __m128i*>(s));
#endif
}
inline __m128i idx2(const pos_t* s) {
#ifdef VW_POS_16
   uint32_t v;
   std::memcpy(&v, s, sizeof(v));
   return _mm_cvtepu16_epi32(_mm_cvtsi32_si128(int(v)));
#else
   return _mm_loadl_epi64(reinterpret_cast<const __m128i*>(s));
#endif
}

/// vpgather: the W-byte values of rows s[0..] packed into the low bytes of
/// an xmm (8 rows for W = 1, 2; 4 for W = 4; 2 for W = 8). A 4-byte load at
/// a 1- or 2-byte value would read up to 3 bytes past the column's last
/// value: each lane loads the 4 bytes that end with the value instead
/// (o + W - 4), or for the column's first values the 4 at its base, and
/// shifts the value down.
template <size_t W> inline __m128i gatherLane(const char* src, const pos_t* s) {
   if constexpr (W == 1 || W == 2) {
      const __m256i o = _mm256_slli_epi32(idx8(s), W == 2 ? 1 : 0);
      const __m256i a = _mm256_max_epi32(
          _mm256_add_epi32(o, _mm256_set1_epi32(int(W) - 4)),
          _mm256_setzero_si256());
      __m256i g =
          _mm256_i32gather_epi32(reinterpret_cast<const int*>(src), a, 1);
      g = _mm256_srlv_epi32(g, _mm256_slli_epi32(_mm256_sub_epi32(o, a), 3));
      if constexpr (W == 1) {
         const __m256i pick = _mm256_setr_epi8(
             0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, //
             0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
         g = _mm256_shuffle_epi8(g, pick);
         g = _mm256_permutevar8x32_epi32(g,
                                         _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0));
      } else {
         const __m256i pick = _mm256_setr_epi8(
             0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, //
             0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
         g = _mm256_shuffle_epi8(g, pick);
         g = _mm256_permute4x64_epi64(g, _MM_SHUFFLE(0, 0, 2, 0));
      }
      return _mm256_castsi256_si128(g);
   } else if constexpr (W == 4) {
      return _mm_i32gather_epi32(reinterpret_cast<const int*>(src), idx4(s), 4);
   } else {
      return _mm_i32gather_epi64(reinterpret_cast<const long long*>(src),
                                 idx2(s), 8);
   }
}

template <size_t W>
inline __m256i gatherAvx2(const char* src, const pos_t* s, uint32_t r) {
   return _mm256_set_m128i(gatherLane<W>(src, s + r), gatherLane<W>(src, s));
}

/// blocks of two 16-byte lanes, R rows each
void runAvx2(const Plan& p, pos_t n, const char* const* data,
             const pos_t* const* sel, char* out) {
   const uint32_t K = p.keySize, R = p.rows, cols = p.cols;
   __m256i ctrl[kMaxCols];
   for (uint32_t c = 0; c < cols; ++c)
      ctrl[c] = _mm256_broadcastsi128_si256(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(p.ctrl[c])));
   alignas(32) char st[32];
   pos_t i = 0;
   for (; i + p.lookahead <= n; i += 2 * R) {
      __m256i acc = _mm256_setzero_si256();
      for (uint32_t c = 0; c < cols; ++c) {
         const uint32_t w = p.width[c];
         const char* src = data[c];
         __m256i v;
         if (!p.sel[c]) {
            v = _mm256_loadu2_m128i(
                reinterpret_cast<const __m128i*>(src + size_t(i + R) * w),
                reinterpret_cast<const __m128i*>(src + size_t(i) * w));
         } else if (p.selLoad == SelLoad::Stage) {
            stage(st, w, src, sel[c] + i, 2 * R);
            v = _mm256_loadu2_m128i(reinterpret_cast<const __m128i*>(st + R * w),
                                    reinterpret_cast<const __m128i*>(st));
         } else {
            const pos_t* s = sel[c] + i;
            switch (w) {
            case 1: v = gatherAvx2<1>(src, s, R); break;
            case 2: v = gatherAvx2<2>(src, s, R); break;
            case 4: v = gatherAvx2<4>(src, s, R); break;
            default: v = gatherAvx2<8>(src, s, R); break;
            }
         }
         acc = _mm256_or_si256(acc, _mm256_shuffle_epi8(v, ctrl[c]));
      }
      char* d = out + size_t(i) * K;
      if (R * K == 16) {
         _mm256_storeu_si256(reinterpret_cast<__m256i*>(d), acc);
      } else {
         // lane 1 after lane 0: it overwrites lane 0's bytes past R * K
         _mm_storeu_si128(reinterpret_cast<__m128i*>(d),
                          _mm256_castsi256_si128(acc));
         _mm_storeu_si128(reinterpret_cast<__m128i*>(d + R * K),
                          _mm256_extracti128_si256(acc, 1));
      }
   }
   tail(p, i, n, data, sel, out);
}
#endif

#ifdef VW_CONCAT_HAVE_VBMI
inline __mmask64 bytesMask(uint32_t x) {
   return x >= 64 ? ~__mmask64(0) : (__mmask64(1) << x) - 1;
}
inline __m512i idx16(const pos_t* s, __mmask16 m) {
#ifdef VW_POS_16
   return _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(m, s));
#else
   return _mm512_maskz_loadu_epi32(m, s);
#endif
}
inline __m256i idx8x(const pos_t* s, __mmask8 m) {
#ifdef VW_POS_16
   return _mm256_cvtepu16_epi32(_mm_maskz_loadu_epi16(m, s));
#else
   return _mm256_maskz_loadu_epi32(m, s);
#endif
}

/// 16 rows (from row 16 * k on) of a 1- or 2-byte column, as dwords (loads
/// as in gatherLane)
template <size_t W>
inline __m512i gather16(const char* src, const pos_t* s, uint32_t m,
                        uint32_t k) {
   const uint32_t mk = std::min<uint32_t>(m - 16 * k, 16);
   const __mmask16 mm = __mmask16((1u << mk) - 1);
   const __m512i o = _mm512_slli_epi32(idx16(s + 16 * k, mm), W == 2 ? 1 : 0);
   const __m512i a = _mm512_max_epi32(
       _mm512_add_epi32(o, _mm512_set1_epi32(int(W) - 4)),
       _mm512_setzero_si512());
   const __m512i g =
       _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), mm, a, src, 1);
   return _mm512_srlv_epi32(g, _mm512_slli_epi32(_mm512_sub_epi32(o, a), 3));
}

/// vpgather: the W-byte values of rows s[0, m) packed into a zmm
template <size_t W>
inline __m512i gatherVbmi(const char* src, const pos_t* s, uint32_t m) {
   __m512i v = _mm512_setzero_si512();
   if constexpr (W == 1) {
      v = _mm512_inserti32x4(v, _mm512_cvtepi32_epi8(gather16<1>(src, s, m, 0)), 0);
      if (m > 16)
         v = _mm512_inserti32x4(v, _mm512_cvtepi32_epi8(gather16<1>(src, s, m, 1)), 1);
      if (m > 32)
         v = _mm512_inserti32x4(v, _mm512_cvtepi32_epi8(gather16<1>(src, s, m, 2)), 2);
      if (m > 48)
         v = _mm512_inserti32x4(v, _mm512_cvtepi32_epi8(gather16<1>(src, s, m, 3)), 3);
   } else if constexpr (W == 2) {
      v = _mm512_inserti64x4(v, _mm512_cvtepi32_epi16(gather16<2>(src, s, m, 0)), 0);
      if (m > 16)
         v = _mm512_inserti64x4(v, _mm512_cvtepi32_epi16(gather16<2>(src, s, m, 1)), 1);
   } else if constexpr (W == 4) {
      const __mmask16 mm = __mmask16((1u << m) - 1);
      v = _mm512_mask_i32gather_epi32(v, mm, idx16(s, mm), src, 4);
   } else {
      const __mmask8 mm = __mmask8((1u << m) - 1);
      v = _mm512_mask_i32gather_epi64(v, mm, idx8x(s, mm), src, 8);
   }
   return v;
}

/// blocks of R = 64 / K rows; the last one masked
void runVbmi(const Plan& p, pos_t n, const char* const* data,
             const pos_t* const* sel, char* out) {
   const uint32_t K = p.keySize, R = p.rows, cols = p.cols;
   __m512i ctrl[kMaxCols];
   for (uint32_t c = 0; c < cols; ++c) ctrl[c] = _mm512_loadu_si512(p.ctrl[c]);
   alignas(64) char st[64];
   for (pos_t i = 0; i < n; i += R) {
      const uint32_t m = std::min<uint32_t>(R, n - i);
      __m512i acc = _mm512_setzero_si512();
      for (uint32_t c = 0; c < cols; ++c) {
         const uint32_t w = p.width[c];
         const char* src = data[c];
         __m512i v;
         if (!p.sel[c]) {
            v = _mm512_maskz_loadu_epi8(bytesMask(m * w), src + size_t(i) * w);
         } else if (p.selLoad == SelLoad::Stage) {
            stage(st, w, src, sel[c] + i, m);
            v = _mm512_maskz_loadu_epi8(bytesMask(m * w), st);
         } else {
            const pos_t* s = sel[c] + i;
            switch (w) {
            case 1: v = gatherVbmi<1>(src, s, m); break;
            case 2: v = gatherVbmi<2>(src, s, m); break;
            case 4: v = gatherVbmi<4>(src, s, m); break;
            default: v = gatherVbmi<8>(src, s, m); break;
            }
         }
         acc = _mm512_mask_permutexvar_epi8(acc, p.lanes[c], ctrl[c], v);
      }
      _mm512_mask_storeu_epi8(out + size_t(i) * K, bytesMask(m * K), acc);
   }
}
#endif

#ifdef VW_CONCAT_HAVE_NEON
template <size_t W, int J>
inline uint8x16_t setLane(uint8x16_t v, const char* p) {
   if constexpr (W == 1) {
      uint8_t x;
      std::memcpy(&x, p, W);
      return vsetq_lane_u8(x, v, J);
   } else if constexpr (W == 2) {
      uint16_t x;
      std::memcpy(&x, p, W);
      return vreinterpretq_u8_u16(vsetq_lane_u16(x, vreinterpretq_u16_u8(v), J));
   } else if constexpr (W == 4) {
      uint32_t x;
      std::memcpy(&x, p, W);
      return vreinterpretq_u8_u32(vsetq_lane_u32(x, vreinterpretq_u32_u8(v), J));
   } else {
      uint64_t x;
      std::memcpy(&x, p, W);
      return vreinterpretq_u8_u64(vsetq_lane_u64(x, vreinterpretq_u64_u8(v), J));
   }
}
template <size_t W, size_t... J>
inline uint8x16_t laneLoads(const char* src, const pos_t* s,
                            std::index_sequence<J...>) {
   uint8x16_t v = vdupq_n_u8(0);
   ((v = setLane<W, int(J)>(v, src + size_t(s[J]) * W)), ...);
   return v;
}
/// lane loads of the next power of two >= r rows (r <= 16 / W)
template <size_t W>
inline uint8x16_t gatherNeon(const char* src, const pos_t* s, uint32_t r) {
   if (r <= 1) return laneLoads<W>(src, s, std::make_index_sequence<1>());
   if constexpr (16 / W >= 2)
      if (r <= 2) return laneLoads<W>(src, s, std::make_index_sequence<2>());
   if constexpr (16 / W >= 4)
      if (r <= 4) return laneLoads<W>(src, s, std::make_index_sequence<4>());
   if constexpr (16 / W >= 8)
      if (r <= 8) return laneLoads<W>(src, s, std::make_index_sequence<8>());
   if constexpr (16 / W >= 16)
      return laneLoads<W>(src, s, std::make_index_sequence<16>());
   return vdupq_n_u8(0); // r > 16 / W: not planned
}

/// blocks of R = 16 / K rows
void runNeon(const Plan& p, pos_t n, const char* const* data,
             const pos_t* const* sel, char* out) {
   const uint32_t K = p.keySize, R = p.rows, cols = p.cols;
   uint8x16_t ctrl[kMaxCols];
   for (uint32_t c = 0; c < cols; ++c) ctrl[c] = vld1q_u8(p.ctrl[c]);
   alignas(16) char st[16];
   pos_t i = 0;
   for (; i + p.lookahead <= n; i += R) {
      uint8x16_t acc = vdupq_n_u8(0);
      for (uint32_t c = 0; c < cols; ++c) {
         const uint32_t w = p.width[c];
         const char* src = data[c];
         uint8x16_t v;
         if (!p.sel[c]) {
            v = vld1q_u8(reinterpret_cast<const uint8_t*>(src + size_t(i) * w));
         } else if (p.selLoad == SelLoad::Stage) {
            stage(st, w, src, sel[c] + i, R);
            v = vld1q_u8(reinterpret_cast<const uint8_t*>(st));
         } else {
            const pos_t* s = sel[c] + i;
            switch (w) {
            case 1: v = gatherNeon<1>(src, s, R); break;
            case 2: v = gatherNeon<2>(src, s, R); break;
            case 4: v = gatherNeon<4>(src, s, R); break;
            default: v = gatherNeon<8>(src, s, R); break;
            }
         }
         acc = vorrq_u8(acc, vqtbl1q_u8(v, ctrl[c]));
      }
      vst1q_u8(reinterpret_cast<uint8_t*>(out + size_t(i) * K), acc);
   }
   tail(p, i, n, data, sel, out);
}
#endif

// Row kernels: per row, each column's value is loaded at its exact width,
// shifted to its key offset and or-ed into one 64-bit register, which is
// stored once (8 bytes for keys of 3, 5, 6, 7 bytes; kSlack covers the
// last row). With the widths fixed at compile time the column loop unrolls
// to straight-line loads and immediate shifts; with them as data (a loop
// over the plan's columns) the per-column overhead outweighs the saved
// passes (run_concatbench row_gpr).
template <size_t W> inline uint64_t loadW(const char* p) {
   if constexpr (W == 1) {
      uint8_t v;
      std::memcpy(&v, p, W);
      return v;
   } else if constexpr (W == 2) {
      uint16_t v;
      std::memcpy(&v, p, W);
      return v;
   } else if constexpr (W == 4) {
      uint32_t v;
      std::memcpy(&v, p, W);
      return v;
   } else {
      uint64_t v;
      std::memcpy(&v, p, W);
      return v;
   }
}
template <bool Sel, size_t... W, size_t... C>
void rowImpl(pos_t n, const char* const* data, const pos_t* __restrict__ sel,
             char* __restrict__ out, std::index_sequence<C...>) {
   constexpr size_t K = (W + ...);
   static_assert(K <= 8, "row kernels hold the key in 64 bits");
   constexpr size_t ws[] = {W...};
   constexpr auto offs = [] {
      std::array<size_t, sizeof...(W)> o{};
      size_t acc = 0;
      for (size_t c = 0; c < sizeof...(W); ++c) o[c] = acc, acc += ws[c];
      return o;
   }();
   const char* d[] = {data[C]...};
   for (pos_t i = 0; i < n; ++i) {
      const size_t r = Sel ? sel[i] : i;
      const uint64_t key =
          ((loadW<ws[C]>(d[C] + r * ws[C]) << (8 * offs[C])) | ...);
      std::memcpy(out + size_t(i) * K, &key,
                  K == 1 || K == 2 || K == 4 || K == 8 ? K : 8);
   }
}
template <bool Sel, size_t... W>
void rowKernel(pos_t n, const char* const* data, const pos_t* sel,
               char* out) {
   rowImpl<Sel, W...>(n, data, sel, out,
                      std::make_index_sequence<sizeof...(W)>());
}

// Fused row kernels: HashGroup hashes a packed key of 1/2/4/8 bytes as
// MurMurHash::hashKey(uint64_t(key)), seed 0, which is the row kernel's
// register zero-extended, and other keys by their bytes
// (hashKey(key, K, 0)); so the key is hashed in the row loop instead of in
// a second pass over packedKeys. AVX-512 (VW_CONCAT_FUSED_SIMD, 1/2/4/8
// bytes): 8 keys move from their registers into one zmm, simd_hash::murmur
// hashes them, and they are stored by one narrowing store. The blocked
// kernel runs the row kernel over kHashBlock rows and then
// simd_hash::hash_keys over that block, still in L1.
constexpr bool kHash64 = sizeof(runtime::MurMurHash::hash_t) == 8;

template <bool Sel, bool Simd, size_t... W, size_t... C>
void rowHashImpl(pos_t n, const char* const* data,
                 const pos_t* __restrict__ sel, char* __restrict__ out,
                 uint64_t* __restrict__ hashes, std::index_sequence<C...>) {
   constexpr size_t K = (W + ...);
   static_assert(K <= 8, "row kernels hold the key in 64 bits");
   constexpr size_t ws[] = {W...};
   constexpr auto offs = [] {
      std::array<size_t, sizeof...(W)> o{};
      size_t acc = 0;
      for (size_t c = 0; c < sizeof...(W); ++c) o[c] = acc, acc += ws[c];
      return o;
   }();
   constexpr bool pow2 = K == 1 || K == 2 || K == 4 || K == 8;
   const char* d[] = {data[C]...};
   auto keyAt = [&](pos_t i) -> uint64_t {
      const size_t r = Sel ? sel[i] : i;
      return ((loadW<ws[C]>(d[C] + r * ws[C]) << (8 * offs[C])) | ...);
   };
   auto store = [&](pos_t i, uint64_t key) {
      std::memcpy(out + size_t(i) * K, &key, pow2 ? K : 8);
   };
   pos_t i = 0;
#ifdef VW_CONCAT_FUSED_SIMD
   if constexpr (Simd && pow2) {
      const __m512i seed = _mm512_setzero_si512();
      for (; i + 8 <= n; i += 8) {
         uint64_t k[8];
         for (int j = 0; j < 8; ++j) k[j] = keyAt(i + j);
         const __m512i v = _mm512_set_epi64(
             (long long)k[7], (long long)k[6], (long long)k[5],
             (long long)k[4], (long long)k[3], (long long)k[2],
             (long long)k[1], (long long)k[0]);
         char* dst = out + size_t(i) * K;
         if constexpr (K == 1) {
            _mm_storel_epi64(reinterpret_cast<__m128i*>(dst),
                             _mm512_cvtepi64_epi8(v));
         } else if constexpr (K == 2) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst),
                             _mm512_cvtepi64_epi16(v));
         } else if constexpr (K == 4) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst),
                                _mm512_cvtepi64_epi32(v));
         } else {
            _mm512_storeu_si512(dst, v);
         }
         _mm512_storeu_si512(hashes + i, primitives::simd_hash::murmur(v, seed));
      }
   }
#endif
   const runtime::MurMurHash h;
   for (; i < n; ++i) {
      const uint64_t key = keyAt(i);
      store(i, key);
      if constexpr (pow2)
         hashes[i] = h.hashKey(key);
      else
         hashes[i] = h.hashKey(out + size_t(i) * K, int(K), 0);
   }
}
template <bool Sel, bool Simd, size_t... W>
void rowHashKernel(pos_t n, const char* const* data, const pos_t* sel,
                   char* out, uint64_t* hashes) {
   rowHashImpl<Sel, Simd, W...>(n, data, sel, out, hashes,
                                std::make_index_sequence<sizeof...(W)>());
}
/// the hash pass of the blocked kernel over m packed keys: Hash_T's
template <size_t K>
void hashBlock(pos_t m, const char* keys, uint64_t* hashes) {
#ifdef VW_CONCAT_FUSED_SIMD
   using hash_t = primitives::simd_hash::hash_t;
   auto* out = reinterpret_cast<hash_t*>(hashes);
   if constexpr (K == 1) return primitives::simd_hash::hash_keys<uint8_t>(m, keys, out);
   if constexpr (K == 2) return primitives::simd_hash::hash_keys<uint16_t>(m, keys, out);
   if constexpr (K == 4) return primitives::simd_hash::hash_keys<uint32_t>(m, keys, out);
   if constexpr (K == 8) return primitives::simd_hash::hash_keys<uint64_t>(m, keys, out);
#endif
   const runtime::MurMurHash h;
   for (pos_t j = 0; j < m; ++j) {
      const char* key = keys + size_t(j) * K;
      if constexpr (K == 1 || K == 2 || K == 4 || K == 8) {
         uint64_t v = 0;
         std::memcpy(&v, key, K);
         hashes[j] = h.hashKey(v);
      } else {
         hashes[j] = h.hashKey(key, int(K), 0);
      }
   }
}
template <bool Sel, size_t... W, size_t... C>
void rowHashBlockedImpl(pos_t n, const char* const* data, const pos_t* sel,
                        char* out, uint64_t* hashes,
                        std::index_sequence<C...> cs) {
   constexpr size_t K = (W + ...);
   constexpr size_t ws[] = {W...};
   for (pos_t i = 0; i < n; i += kHashBlock) {
      const pos_t m = std::min<pos_t>(kHashBlock, pos_t(n - i));
      const char* d[] = {(data[C] + (Sel ? 0 : size_t(i) * ws[C]))...};
      rowImpl<Sel, W...>(m, d, Sel ? sel + i : nullptr, out + size_t(i) * K,
                         cs);
      hashBlock<K>(m, out + size_t(i) * K, hashes + i);
   }
}
template <bool Sel, size_t... W>
void rowHashBlockedKernel(pos_t n, const char* const* data, const pos_t* sel,
                          char* out, uint64_t* hashes) {
   rowHashBlockedImpl<Sel, W...>(n, data, sel, out, hashes,
                                 std::make_index_sequence<sizeof...(W)>());
}
template <bool Sel, size_t... W> constexpr RowHashFn rowHashBlocked() {
   return kHash64 ? rowHashBlockedKernel<Sel, W...> : nullptr;
}
template <bool Sel, size_t... W> constexpr RowHashFn rowHashScalar() {
   return kHash64 ? rowHashKernel<Sel, false, W...> : nullptr;
}
template <bool Sel, size_t... W> constexpr RowHashFn rowHashSimd() {
#ifdef VW_CONCAT_FUSED_SIMD
   return kHash64 ? rowHashKernel<Sel, true, W...> : nullptr;
#else
   return nullptr;
#endif
}

/// The instantiated key shapes: every column through the one selection
/// vector (sel) or every column dense. Only the compound group keys of the
/// TPC-H / SSB plans that fit 8 bytes of 1/2/4/8-byte columns are listed
/// (TPC-H Q1: l_returnflag, l_linestatus); the others group on wider keys.
/// All 127 width tuples x sel/dense could be generated: a kernel only runs
/// for its own shape, so the cost would be build and test time, not query
/// time; the table is kept to the shapes in use to keep those short.
struct RowEntry {
   bool sel;
   uint32_t cols;
   uint8_t width[kMaxCols];
   RowFn fn;
   RowHashFn hash, hashSimd, hashBlocked;
};
#define VW_ROW_SHAPE(SEL, COLS, ...)                                          \
   {SEL, COLS, {__VA_ARGS__}, rowKernel<SEL, __VA_ARGS__>,                     \
    rowHashScalar<SEL, __VA_ARGS__>(), rowHashSimd<SEL, __VA_ARGS__>(),      \
    rowHashBlocked<SEL, __VA_ARGS__>()}
const RowEntry kRowKernels[] = {
    VW_ROW_SHAPE(true, 2, 1, 1), // Q1
};
#undef VW_ROW_SHAPE

uint32_t pow2ceil(uint32_t x) {
   uint32_t p = 1;
   while (p < x) p <<= 1;
   return p;
}

} // namespace

Isa best() {
#if defined(VW_CONCAT_HAVE_VBMI)
   return Isa::Vbmi;
#elif defined(VW_CONCAT_HAVE_AVX2)
   return Isa::Avx2;
#elif defined(VW_CONCAT_HAVE_NEON)
   return Isa::Neon;
#else
   return Isa::Scalar;
#endif
}

bool available(Isa isa) {
   switch (isa) {
   case Isa::Scalar:
   case Isa::Row: return true;
#ifdef VW_CONCAT_HAVE_AVX2
   case Isa::Avx2: return true;
#endif
#ifdef VW_CONCAT_HAVE_VBMI
   case Isa::Vbmi: return true;
#endif
#ifdef VW_CONCAT_HAVE_NEON
   case Isa::Neon: return true;
#endif
   default: return false;
   }
}

const char* name(Isa isa) {
   switch (isa) {
   case Isa::Avx2: return "avx2";
   case Isa::Vbmi: return "vbmi";
   case Isa::Neon: return "neon";
   case Isa::Row: return "row";
   default: return "scalar";
   }
}

bool build(Plan& p, Isa isa, SelLoad selLoad, size_t cols,
           const uint32_t* width, const uint32_t* offset, const bool* sel,
           uint32_t keySize) {
   p = Plan();
   if (isa == Isa::Row) {
      if (cols == 0 || cols > kMaxCols) return false;
      for (const RowEntry& e : kRowKernels) {
         if (e.cols != cols) continue;
         bool match = true;
         for (size_t c = 0; c < cols && match; ++c)
            match = e.width[c] == width[c] && e.sel == sel[c] &&
                    offset[c] == (c ? offset[c - 1] + width[c - 1] : 0);
         if (!match) continue;
         p.isa = isa;
         p.keySize = keySize;
         p.cols = uint32_t(cols);
         for (size_t c = 0; c < cols; ++c) {
            p.width[c] = uint8_t(width[c]);
            p.offset[c] = offset[c];
            p.sel[c] = sel[c];
         }
         p.row = e.fn;
         p.rowHash = e.hash;
         p.rowHashSimd = e.hashSimd;
         p.rowHashBlocked = e.hashBlocked;
         p.ok = true;
         return true;
      }
      return false;
   }
   if (isa == Isa::Scalar || !available(isa)) return false;
   if (cols == 0 || cols > kMaxCols || keySize == 0) return false;
   const uint32_t lane = isa == Isa::Vbmi ? 64 : 16;
   if (keySize > lane) return false;
   const uint32_t R = lane / keySize;
   for (size_t c = 0; c < cols; ++c) {
      const uint32_t w = width[c];
      if (w != 1 && w != 2 && w != 4 && w != 8) return false;
      if (offset[c] + w > keySize) return false;
      // AVX2 gathers 8 rows of a 1-byte column per lane (R = 16: K = 1)
      if (isa == Isa::Avx2 && selLoad == SelLoad::Gather && sel[c] && R > 8)
         return false;
   }
   p.isa = isa;
   p.selLoad = selLoad;
   p.keySize = keySize;
   p.cols = uint32_t(cols);
   p.rows = R;
   const uint8_t none = isa == Isa::Neon ? 0xff : 0x80;
   for (size_t c = 0; c < cols; ++c) {
      const uint32_t w = width[c], off = offset[c];
      p.width[c] = uint8_t(w);
      p.offset[c] = off;
      p.sel[c] = sel[c];
      for (uint32_t b = 0; b < lane; ++b) {
         const uint32_t r = b / keySize, q = b % keySize;
         const bool mine = b < R * keySize && q >= off && q < off + w;
         p.ctrl[c][b] = mine ? uint8_t(r * w + q - off)
                             : (isa == Isa::Vbmi ? 0 : none);
         if (mine && isa == Isa::Vbmi) p.lanes[c] |= uint64_t(1) << b;
      }
   }
   // rows a block reads past its first row
   uint32_t la = isa == Isa::Avx2 ? 2 * R : R;
   for (size_t c = 0; c < cols; ++c) {
      const uint32_t w = width[c];
      uint32_t need = 0;
      if (isa == Isa::Avx2) {
         if (!sel[c])
            need = R + 16 / w;
         else if (selLoad == SelLoad::Stage)
            need = 2 * R;
         else
            need = R + (w <= 2 ? 8 : w == 4 ? 4 : 2);
      } else if (isa == Isa::Neon) {
         if (!sel[c])
            need = 16 / w;
         else if (selLoad == SelLoad::Stage)
            need = R;
         else
            need = pow2ceil(R);
      }
      la = std::max(la, need);
   }
   p.lookahead = la;
   p.ok = true;
   return true;
}

void run(const Plan& p, pos_t n, const char* const* data,
         const pos_t* const* sel, char* out) {
   switch (p.isa) {
#ifdef VW_CONCAT_HAVE_VBMI
   case Isa::Vbmi: runVbmi(p, n, data, sel, out); return;
#endif
#ifdef VW_CONCAT_HAVE_AVX2
   case Isa::Avx2: runAvx2(p, n, data, sel, out); return;
#endif
#ifdef VW_CONCAT_HAVE_NEON
   case Isa::Neon: runNeon(p, n, data, sel, out); return;
#endif
   case Isa::Row: {
      // the kernel reads one selection vector for every column
      const pos_t* s = nullptr;
      for (uint32_t c = 0; c < p.cols; ++c) {
         if (!p.sel[c]) continue;
         if (s && sel[c] != s) {
            tail(p, 0, n, data, sel, out);
            return;
         }
         s = sel[c];
      }
      p.row(n, data, s, out);
      return;
   }
   default: tail(p, 0, n, data, sel, out); return;
   }
}

bool runHash(const Plan& p, HashMode mode, pos_t n, const char* const* data,
             const pos_t* const* sel, char* out, uint64_t* hashes) {
   if (p.isa != Isa::Row || !p.rowHash) return false;
   const pos_t* s = nullptr;
   for (uint32_t c = 0; c < p.cols; ++c) {
      if (!p.sel[c]) continue;
      if (s && sel[c] != s) return false;
      s = sel[c];
   }
   RowHashFn fn = p.rowHash;
   if (mode == HashMode::Simd && p.rowHashSimd) fn = p.rowHashSimd;
   if (mode == HashMode::Blocked && p.rowHashBlocked) fn = p.rowHashBlocked;
   fn(n, data, s, out, hashes);
   return true;
}

bool simdHashAvailable() {
#ifdef VW_CONCAT_FUSED_SIMD
   return kHash64;
#else
   return false;
#endif
}

void runColumns(size_t cols, const uint32_t* width, const uint32_t* offset,
                uint32_t keySize, pos_t n, const char* const* data,
                const pos_t* const* sel, char* out) {
   columnsFrom(cols, width, offset, keySize, 0, n, data, sel, out);
}

} // namespace simd_concat
} // namespace primitives
} // namespace vectorwise
