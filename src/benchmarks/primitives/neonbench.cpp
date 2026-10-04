// run_neonbench: throughput of the engine's AVX-512 kernels against their
// aarch64 options, per kernel family:
//   scalar*       the engine's scalar template or loop (what an aarch64 build
//                 runs today: every AVX-512 path is compiled out there)
//   avx512_simde  the engine's AVX-512 kernel, spelled with simde_ names: on
//                 an AVX-512 x86 build these are the native instructions (the
//                 reference rows), on aarch64 SIMDe's emulation (the "just
//                 translate it" option)
//   neon*         hand NEON kernels from include/vectorwise/SimdNeon.hpp
//   sve           SVE kernels, only when the target has SVE (neoverse-v1/v2)
// Every impl must produce the same output as the family's first impl
// (count + checksum); exit status 1 on any MISMATCH.
//
// Families, the engine code each one stands for, and its TPC-H users:
//   sel       sel_col_val int32/int64 less       SimdSelection.hpp:235      Q6, Q3, Q1
//             (VW_SIMD_SEL)
//   selsel    selsel_col_val, input through a    SimdSelection.hpp:594      Q6, Q3
//             selection vector (VW_SIMD_SEL_GATHER)
//   hash      hash / hash_sel of int32 keys      SimdHash.hpp:74,184,204    Q3/Q5/Q9 probes
//             (VW_SIMD_HASH; the CRC32 lowerings are run_crcbench's)
//   proj      Q1's dense projections (VW_PROJ_DENSE, since removed)        Q1
//             and the gathered ones they replace  Primitives.hpp:204-253
//   runheads  HashGroup run-head pass 1          Operators.cpp:2027,2115    Q1, Q18
//             (VW_GROUP_RUN_HEADS)
//   bloom     Hashjoin::bloomFilter (VW_JOIN_BLOOM)  Operators.cpp:1008     Q3, Q5, Q9
//   semi      Hashjoin::semiProbe (VW_JOIN_SEMI)     Operators.cpp:918      Q3, Q9, Q18
//   dirprobe  Hashjoin::joinNewFirstPass         Operators.cpp:1221,1274,1340  Q3/Q5/Q9
//             (VW_NEW_JOIN): directory + tag + head hash, real/maybe lists
//   roof      1-thread read bandwidth and dependent-load latency per working
//             set: the ceilings the memory-bound families are judged against
//
// Modes:
//   l1     one vecSize vector, called reps times (cache resident; compute)
//   stream a column of -s MiB processed vector by vector (as a Scan does)
//   sweep  random-access families: probe stream against tables of growing
//          size (param = table bytes), the memory-stall question
//
// Usage: run_neonbench [-v vec] [-r reps] [-s streamMiB] [-p probes]
//                      [-t maxTableMiB] [-m l1|stream|sweep|all]
//                      [-f family,...] [-i impl,...] [-y type,...]
//                      [-q param,...] [-T trials]
//   -i/-y/-q keep only matching rows (the family's first impl still runs
//   once as the check reference); -T timed trials per row (best is kept,
//   default 3). NEONBENCH_TRACE=1 prints "elems <row> <elements per trial>"
//   on stderr, which study/neon_port/insns.sh uses for instructions per
//   element under qemu.
// CSV on stdout:
//   family,impl,type,mode,param,vec,ns_per_elem,gb_per_s,check,autovec
// gb_per_s = input bytes per element / ns (the column bytes the kernel
// reads, not counting random table accesses). autovec = 1 for the
// run_neonbench_autovec build (CMake drops -fno-tree-vectorize for it; SIMDe's
// generic fallbacks rely on the vectorizer).
#include "common/runtime/Hash.hpp"
#include "vectorwise/Primitives.hpp"
#include "vectorwise/SimdNeon.hpp"
#include <simde/x86/avx512.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#ifndef NEONBENCH_AUTOVEC
#define NEONBENCH_AUTOVEC 0
#endif

using namespace vectorwise;
using namespace vectorwise::primitives;
using hash_t = defs::hash_t;
static_assert(sizeof(pos_t) == 4, "run_neonbench needs 32-bit pos_t (VW_POS_16 is ON)");
static_assert(sizeof(hash_t) == 8, "run_neonbench needs 64-bit hashes");

namespace {

size_t vecSize = 1024;
size_t reps = 2000;
size_t streamBytes = 64ull << 20;
size_t probes = 1ull << 21;
size_t maxTable = 128ull << 20;
std::string mode = "all";
std::string families = "all";
std::string implFilter, typeFilter, paramFilter; // empty = all
int trials = 3;
bool trace = false; // NEONBENCH_TRACE: elements per timed trial on stderr
int fails = 0;
volatile uint64_t sink;

bool inList(const std::string& list, const std::string& v) {
   std::stringstream ss(list);
   std::string s;
   while (std::getline(ss, s, ','))
      if (s == v) return true;
   return false;
}
bool wantMode(const char* m) { return mode == "all" || mode == m; }
bool wantFam(const char* f) { return families == "all" || inList(families, f); }
bool wantRow(const char* impl, const char* type, const std::string& param) {
   return (implFilter.empty() || inList(implFilter, impl)) &&
          (typeFilter.empty() || inList(typeFilter, type)) &&
          (paramFilter.empty() || inList(paramFilter, param));
}

using Clock = std::chrono::steady_clock;
template <typename F> double bestNs(F&& f) {
   double best = 1e300;
   for (int t = 0; t < trials; ++t) {
      const auto t0 = Clock::now();
      sink = f();
      const auto t1 = Clock::now();
      best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
   }
   return best;
}

void row(const char* fam, const char* impl, const char* type, const char* m,
         const std::string& param, double ns, double bytesPerElem, bool ok,
         size_t elemsPerTrial = 0) {
   if (trace)
      std::fprintf(stderr, "elems %s %s %s %s %s %zu\n", fam, impl, type, m,
                   param.c_str(), elemsPerTrial);
   std::printf("%s,%s,%s,%s,%s,%zu,%.4f,%.3f,%s,%d\n", fam, impl, type, m,
               param.c_str(), vecSize, ns, bytesPerElem / ns,
               ok ? "ok" : "MISMATCH", NEONBENCH_AUTOVEC);
   std::fflush(stdout);
   fails += !ok;
}

/// One implementation of a family: fn(off, len, check) processes input
/// elements [off, off + len); with check it returns an order-sensitive
/// checksum of its output, without it something cheap (the timed path).
struct Impl {
   const char* name;
   std::function<uint64_t(size_t off, size_t len, bool check)> fn;
};

uint64_t mix(uint64_t h, uint64_t v) { return (h ^ v) * 0x9E3779B97F4A7C15ull; }

/// l1 and stream rows for vector kernels; total = input elements available
/// perCall = elements one call processes (default vecSize; selection-indexed
/// families pass their selection length, so ns_per_elem is per entry)
void runVector(const char* fam, const char* type, const std::string& param,
               const std::vector<Impl>& impls, size_t total, double bpe,
               size_t perCall = 0) {
   if (!perCall) perCall = vecSize;
   for (const char* m : {"l1", "stream"}) {
      if (!wantMode(m)) continue;
      const bool l1 = !std::strcmp(m, "l1");
      const size_t elems = l1 ? perCall : total / vecSize * perCall;
      auto pass = [&](const Impl& im, bool check) {
         uint64_t c = 0;
         if (l1) {
            if (check) return im.fn(0, vecSize, true);
            for (size_t r = 0; r < reps; ++r) c += im.fn(0, vecSize, false);
         } else {
            for (size_t off = 0; off + vecSize <= total; off += vecSize)
               c = check ? mix(c, im.fn(off, vecSize, true)) : c + im.fn(off, vecSize, false);
         }
         return c;
      };
      // impls[0] is the reference even when -i filters it out
      const uint64_t ref = pass(impls[0], true);
      for (size_t k = 0; k < impls.size(); ++k) {
         if (!wantRow(impls[k].name, type, param)) continue;
         const uint64_t c = k ? pass(impls[k], true) : ref;
         const size_t perTrial = l1 ? elems * reps : elems;
         const double ns = bestNs([&] { return pass(impls[k], false); }) / double(perTrial);
         row(fam, impls[k].name, type, m, param, ns, bpe, c == ref, perTrial);
      }
   }
}

/// sweep rows: the whole probe stream once per timing (param = table bytes)
void runSweep(const char* fam, const char* type, const std::string& param,
              const std::vector<Impl>& impls, size_t total, double bpe) {
   auto pass = [&](const Impl& im, bool check) {
      uint64_t c = 0;
      for (size_t off = 0; off + vecSize <= total; off += vecSize)
         c = check ? mix(c, im.fn(off, vecSize, true)) : c + im.fn(off, vecSize, false);
      return c;
   };
   const uint64_t ref = pass(impls[0], true);
   for (size_t k = 0; k < impls.size(); ++k) {
      if (!wantRow(impls[k].name, type, param)) continue;
      const uint64_t c = k ? pass(impls[k], true) : ref;
      const size_t perTrial = total / vecSize * vecSize;
      const double ns = bestNs([&] { return pass(impls[k], false); }) / double(perTrial);
      row(fam, impls[k].name, type, "sweep", param, ns, bpe, c == ref, perTrial);
   }
}

template <typename T> uint64_t sumSel(const T* out, size_t found, bool check) {
   if (!check) return found;
   uint64_t h = found;
   for (size_t j = 0; j < found; ++j) h = mix(h, uint64_t(out[j]));
   return h;
}

std::vector<size_t> tableSizes() {
   std::vector<size_t> s;
   for (size_t b : {32ull << 10, 256ull << 10, 2ull << 20, 16ull << 20, 128ull << 20})
      if (b <= maxTable) s.push_back(b);
   return s;
}
std::string bytesStr(size_t b) {
   char buf[32];
   if (b >= (1u << 20)) std::snprintf(buf, sizeof buf, "%zuMiB", b >> 20);
   else std::snprintf(buf, sizeof buf, "%zuKiB", b >> 10);
   return buf;
}

//=============================================================================
// AVX-512 kernels, spelled with simde_ names (native on AVX-512 x86, SIMDe's
// emulation on aarch64). Each copies the engine kernel named above it; tails
// are scalar here because the bench only runs whole vectors.
//=============================================================================
namespace avx {

// Intrinsics the engine uses that SIMDe (3rdparty/simde, 2026-02) does not
// provide: native where the target has them, otherwise the per-lane
// polyfills SIMD.hpp also carries for the engine.
#if defined(SIMDE_X86_AVX512F_NATIVE)
inline simde__m512i cvtepi32_epi64(simde__m256i a) { return _mm512_cvtepi32_epi64(a); }
inline simde__m512i i32gather_epi32(simde__m512i idx, const void* b) { return _mm512_i32gather_epi32(idx, b, 4); }
inline simde__m512i i32gather_epi64(simde__m256i idx, const void* b) { return _mm512_i32gather_epi64(idx, b, 8); }
inline simde__m512i mask_i32gather_epi32(simde__m512i src, simde__mmask16 k, simde__m512i idx, const void* b) {
   return _mm512_mask_i32gather_epi32(src, k, idx, b, 4);
}
inline simde__m512i mask_i64gather_epi64(simde__m512i src, simde__mmask8 k, simde__m512i idx, const void* b) {
   return _mm512_mask_i64gather_epi64(src, k, idx, b, 1);
}
#else
// simde_mm512_mask_i64gather_epi64 is an UNMASKED gather plus a blend: it
// loads masked-off lanes too, which faults where the engine relies on the
// mask (joinNewFirstPass gathers through null heads). Per-lane instead.
inline simde__m512i mask_i64gather_epi64(simde__m512i src, simde__mmask8 k, simde__m512i idx, const void* b) {
   alignas(64) int64_t i[8], r[8];
   simde_mm512_store_si512(i, idx);
   simde_mm512_store_si512(r, src);
   for (unsigned l = 0; l < 8; ++l)
      if (k >> l & 1) std::memcpy(&r[l], static_cast<const char*>(b) + i[l], 8);
   return simde_mm512_load_si512(r);
}
inline simde__m512i cvtepi32_epi64(simde__m256i a) {
   return simde_mm512_inserti64x4(
       simde_mm512_castsi256_si512(simde_mm256_cvtepi32_epi64(simde_mm256_castsi256_si128(a))),
       simde_mm256_cvtepi32_epi64(simde_mm256_extracti128_si256(a, 1)), 1);
}
inline simde__m512i i32gather_epi32(simde__m512i idx, const void* b) {
   alignas(64) int32_t i[16], r[16];
   simde_mm512_store_si512(i, idx);
   for (unsigned l = 0; l < 16; ++l) std::memcpy(&r[l], static_cast<const char*>(b) + int64_t(i[l]) * 4, 4);
   return simde_mm512_load_si512(r);
}
inline simde__m512i i32gather_epi64(simde__m256i idx, const void* b) {
   return simde_mm512_i64gather_epi64(cvtepi32_epi64(idx), b, 8);
}
inline simde__m512i mask_i32gather_epi32(simde__m512i src, simde__mmask16 k, simde__m512i idx, const void* b) {
   alignas(64) int32_t i[16], r[16];
   simde_mm512_store_si512(i, idx);
   simde_mm512_store_si512(r, src);
   for (unsigned l = 0; l < 16; ++l)
      if (k >> l & 1) std::memcpy(&r[l], static_cast<const char*>(b) + int64_t(i[l]) * 4, 4);
   return simde_mm512_load_si512(r);
}
#endif
#if defined(SIMDE_X86_AVX512BW_NATIVE) && defined(SIMDE_X86_AVX512VL_NATIVE)
inline simde__m128i maskz_set1_epi8(simde__mmask16 m, char v) { return _mm_maskz_set1_epi8(m, v); }
#else
inline simde__m128i maskz_set1_epi8(simde__mmask16 m, char v) {
   return simde_mm_and_si128(simde_mm_movm_epi8(m), simde_mm_set1_epi8(v));
}
#endif

inline size_t emit16(pos_t* out, simde__mmask16 m, simde__m512i ids) {
   const unsigned cnt = __builtin_popcount(unsigned(m));
   simde_mm512_mask_storeu_epi32(out, simde__mmask16((1u << cnt) - 1),
                                 simde_mm512_maskz_compress_epi32(m, ids));
   return cnt;
}
inline void compressStore32(void* dst, simde__mmask8 m, simde__m512i v) {
   simde_mm512_mask_storeu_epi32(dst, simde__mmask16((1u << __builtin_popcount(m)) - 1),
                                 simde_mm512_maskz_compress_epi32(simde__mmask16(m), v));
}
inline void compressStore64(void* dst, simde__mmask8 m, simde__m512i v) {
   simde_mm512_mask_storeu_epi64(dst, simde__mmask8((1u << __builtin_popcount(m)) - 1),
                                 simde_mm512_maskz_compress_epi64(m, v));
}
inline simde__m512i laneIds() {
   return simde_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}

// SimdSelection.hpp:235 sel_col_val_v<T, std::less, Emit::Reg, 1>
template <typename T>
pos_t sel_less(pos_t n, pos_t* result, const T* a, T c) {
   const simde__m512i vc = sizeof(T) == 4 ? simde_mm512_set1_epi32(int32_t(c))
                                          : simde_mm512_set1_epi64(int64_t(c));
   const simde__m512i step = simde_mm512_set1_epi32(16);
   simde__m512i ids = laneIds();
   size_t found = 0, i = 0;
   for (; i + 16 <= n; i += 16) {
      simde__mmask16 m;
      if constexpr (sizeof(T) == 4) {
         m = simde_mm512_cmp_epi32_mask(simde_mm512_loadu_si512(a + i), vc, SIMDE_MM_CMPINT_LT);
      } else {
         const simde__mmask8 lo = simde_mm512_cmp_epi64_mask(simde_mm512_loadu_si512(a + i), vc, SIMDE_MM_CMPINT_LT);
         const simde__mmask8 hi = simde_mm512_cmp_epi64_mask(simde_mm512_loadu_si512(a + i + 8), vc, SIMDE_MM_CMPINT_LT);
         m = simde__mmask16(lo | (unsigned(hi) << 8));
      }
      found += emit16(result + found, m, ids);
      ids = simde_mm512_add_epi32(ids, step);
   }
   for (; i < n; ++i) {
      result[found] = pos_t(i);
      found += a[i] < c;
   }
   return pos_t(found);
}

// SimdSelection.hpp:594 selsel_col_val_hwgather<T, std::less>
template <typename T>
pos_t selsel_less(pos_t n, const pos_t* inSel, pos_t* result, const T* a, T c) {
   size_t found = 0, i = 0;
   for (; i + 16 <= n; i += 16) {
      const simde__m512i idx = simde_mm512_loadu_si512(inSel + i);
      simde__mmask16 m;
      if constexpr (sizeof(T) == 4) {
         const simde__m512i v = i32gather_epi32(idx, a);
         m = simde_mm512_cmp_epi32_mask(v, simde_mm512_set1_epi32(int32_t(c)), SIMDE_MM_CMPINT_LT);
      } else {
         const simde__m512i vc = simde_mm512_set1_epi64(int64_t(c));
         const simde__m512i lo = i32gather_epi64(simde_mm512_castsi512_si256(idx), a);
         const simde__m512i hi = i32gather_epi64(simde_mm512_extracti64x4_epi64(idx, 1), a);
         m = simde__mmask16(simde_mm512_cmp_epi64_mask(lo, vc, SIMDE_MM_CMPINT_LT) |
                            (unsigned(simde_mm512_cmp_epi64_mask(hi, vc, SIMDE_MM_CMPINT_LT)) << 8));
      }
      found += emit16(result + found, m, idx);
   }
   for (; i < n; ++i) {
      const pos_t idx = inSel[i];
      result[found] = idx;
      found += a[idx] < c;
   }
   return pos_t(found);
}

// SimdHash.hpp:74 murmur
inline simde__m512i murmur(simde__m512i k, simde__m512i seed) {
   constexpr uint64_t m = 0xc6a4a7935bd1e995ull;
   const simde__m512i vm = simde_mm512_set1_epi64((long long)m);
   simde__m512i h = simde_mm512_xor_si512(
       seed, simde_mm512_set1_epi64((long long)(0x8445d61a4e774912ull ^ (8 * m))));
   k = simde_mm512_mullo_epi64(k, vm);
   k = simde_mm512_xor_si512(k, simde_mm512_srli_epi64(k, 47));
   k = simde_mm512_mullo_epi64(k, vm);
   h = simde_mm512_xor_si512(h, k);
   h = simde_mm512_mullo_epi64(h, vm);
   h = simde_mm512_xor_si512(h, simde_mm512_srli_epi64(h, 47));
   h = simde_mm512_mullo_epi64(h, vm);
   h = simde_mm512_xor_si512(h, simde_mm512_srli_epi64(h, 47));
   return h;
}
// SimdHash.hpp:184 hash<int32_t> (widen8: sign-extend 8 keys)
inline void murmur_hash_i32(size_t n, uint64_t* out, const int32_t* in, uint64_t seed) {
   const simde__m512i s = simde_mm512_set1_epi64((long long)seed);
   for (size_t i = 0; i + 8 <= n; i += 8)
      simde_mm512_storeu_si512(out + i, murmur(avx::cvtepi32_epi64(simde_mm256_loadu_si256(in + i)), s));
}
// SimdHash.hpp:204 hash_sel<int32_t> (gather8, stack form)
inline void murmur_hash_sel_i32(size_t n, uint64_t* out, const int32_t* in,
                                const pos_t* sel, uint64_t seed) {
   const simde__m512i s = simde_mm512_set1_epi64((long long)seed);
   for (size_t i = 0; i + 8 <= n; i += 8) {
      alignas(32) int32_t tmp[8];
      for (unsigned j = 0; j < 8; ++j) tmp[j] = in[sel[i + j]];
      simde_mm512_storeu_si512(out + i, murmur(avx::cvtepi32_epi64(simde_mm256_load_si256(reinterpret_cast<const simde__m256i*>(tmp))), s));
   }
}

// Projection.cpp:197-268 proj_dense_* (zmm loops)
inline void dense_minus(pos_t m, int64_t* r, int64_t c, const int64_t* p2) {
   const simde__m512i vc = simde_mm512_set1_epi64(c);
   for (pos_t i = 0; i + 8 <= m; i += 8)
      simde_mm512_storeu_si512(r + i, simde_mm512_sub_epi64(vc, simde_mm512_loadu_si512(p2 + i)));
   for (pos_t i = m / 8 * 8; i < m; ++i) r[i] = c - p2[i];
}
inline void dense_plus(pos_t m, int64_t* r, const int64_t* p1, int64_t c) {
   const simde__m512i vc = simde_mm512_set1_epi64(c);
   for (pos_t i = 0; i + 8 <= m; i += 8)
      simde_mm512_storeu_si512(r + i, simde_mm512_add_epi64(simde_mm512_loadu_si512(p1 + i), vc));
   for (pos_t i = m / 8 * 8; i < m; ++i) r[i] = p1[i] + c;
}
inline void dense_mul(pos_t m, int64_t* r, const int64_t* p1, const int64_t* p2) {
   for (pos_t i = 0; i + 8 <= m; i += 8)
      simde_mm512_storeu_si512(r + i, simde_mm512_mullo_epi64(simde_mm512_loadu_si512(p1 + i),
                                                              simde_mm512_loadu_si512(p2 + i)));
   for (pos_t i = m / 8 * 8; i < m; ++i) r[i] = p1[i] * p2[i];
}

// Operators.cpp:2031 runHeadMask16 + 2123-2137 (pass 1, AVX-512BW/VL)
template <typename K>
pos_t run_heads(pos_t n, const K* keys, pos_t* heads, uint8_t* flags, pos_t cnt, pos_t i) {
   const simde__m512i iota = laneIds();
   for (; i + 16 <= n; i += 16) {
      simde__mmask16 m;
      if constexpr (sizeof(K) == 2)
         m = simde_mm256_cmpneq_epi16_mask(simde_mm256_loadu_si256(keys + i),
                                           simde_mm256_loadu_si256(keys + i - 1));
      else
         m = simde_mm512_cmpneq_epi32_mask(simde_mm512_loadu_si512(keys + i),
                                           simde_mm512_loadu_si512(keys + i - 1));
      const simde__m512i idx = simde_mm512_add_epi32(simde_mm512_set1_epi32(int(i)), iota);
      const unsigned k = __builtin_popcount(unsigned(m));
      simde_mm512_mask_storeu_epi32(heads + cnt, simde__mmask16((1u << k) - 1),
                                    simde_mm512_maskz_compress_epi32(m, idx));
      simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(flags + i), maskz_set1_epi8(m, 1));
      cnt += k;
   }
   for (; i < n; ++i) {
      const bool head = keys[i] != keys[i - 1];
      flags[i] = head;
      heads[cnt] = i;
      cnt += head;
   }
   return cnt;
}

// Operators.cpp:1008 bloomFilter (AVX-512 loop)
inline size_t bloom(size_t n, const uint64_t* H, const uint64_t* bloom, uint64_t mask,
                    pos_t* sel, uint64_t* hs) {
   const simde__m512i vMask = simde_mm512_set1_epi64((long long)mask);
   const simde__m512i b6 = simde_mm512_set1_epi64(63);
   const simde__m512i one = simde_mm512_set1_epi64(1);
   const simde__m512i lane = simde_mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0);
   size_t m = 0;
   for (size_t i = 0; i + 8 <= n; i += 8) {
      const simde__m512i h = simde_mm512_loadu_si512(H + i);
      const simde__m512i words = simde_mm512_i64gather_epi64(
          simde_mm512_and_si512(simde_mm512_srli_epi64(h, 40), vMask), bloom, 8);
      simde__m512i bits = simde_mm512_sllv_epi64(one, simde_mm512_and_si512(h, b6));
      bits = simde_mm512_or_si512(bits, simde_mm512_sllv_epi64(one, simde_mm512_and_si512(simde_mm512_srli_epi64(h, 6), b6)));
      bits = simde_mm512_or_si512(bits, simde_mm512_sllv_epi64(one, simde_mm512_and_si512(simde_mm512_srli_epi64(h, 12), b6)));
      bits = simde_mm512_or_si512(bits, simde_mm512_sllv_epi64(one, simde_mm512_and_si512(simde_mm512_srli_epi64(h, 18), b6)));
      const simde__mmask8 pass = simde_mm512_cmpeq_epi64_mask(simde_mm512_and_si512(words, bits), bits);
      compressStore32(sel + m, pass, simde_mm512_add_epi32(simde_mm512_set1_epi32(int(i)), lane));
      compressStore64(hs + m, pass, h);
      m += __builtin_popcount(pass);
   }
   return m;
}

// Operators.cpp:918 semiProbe (AVX-512 loop, dense probe keys)
inline size_t semi(size_t n, const int32_t* keys, const uint32_t* bits, int32_t lo,
                   uint32_t span, pos_t* out) {
   const simde__m512i vlo = simde_mm512_set1_epi32(lo);
   const simde__m512i vspan = simde_mm512_set1_epi32(int32_t(span));
   const simde__m512i b31 = simde_mm512_set1_epi32(31);
   const simde__m512i one = simde_mm512_set1_epi32(1);
   const simde__m512i lane = laneIds();
   size_t found = 0;
   for (size_t i = 0; i + 16 <= n; i += 16) {
      const simde__m512i pos = simde_mm512_add_epi32(simde_mm512_set1_epi32(int32_t(i)), lane);
      const simde__m512i d = simde_mm512_sub_epi32(simde_mm512_loadu_si512(keys + i), vlo);
      const simde__mmask16 in = simde_mm512_cmple_epu32_mask(d, vspan);
      const simde__m512i w = mask_i32gather_epi32(
          simde_mm512_setzero_si512(), in, simde_mm512_srli_epi32(d, 5), bits);
      const simde__mmask16 hit = simde_mm512_mask_test_epi32_mask(
          in, w, simde_mm512_sllv_epi32(one, simde_mm512_and_si512(d, b31)));
      found += emit16(out + found, hit, pos);
   }
   return found;
}

} // namespace avx

//=============================================================================
// families
//=============================================================================

//--- sel ---------------------------------------------------------------------
template <typename T> void famSel(const char* tname) {
   const size_t total = std::max(vecSize, streamBytes / sizeof(T));
   std::vector<T> col(total + 16);
   std::mt19937_64 rng(1);
   for (auto& v : col) v = T(rng() % 100);
   std::vector<pos_t> out(vecSize + 16);
   for (int sel : {1, 10, 50, 90, 99}) {
      T c = T(sel);
      T* a = col.data();
      pos_t* o = out.data();
      std::vector<Impl> im = {
          {"scalar_bf", [&](size_t off, size_t len, bool ck) {
              return sumSel(o, sel_col_val_bf<T, std::less>(pos_t(len), o, a + off, &c), ck); }},
          {"scalar_branch", [&](size_t off, size_t len, bool ck) {
              return sumSel(o, sel_col_val<T, std::less>(pos_t(len), o, a + off, &c), ck); }},
          {"avx512_simde", [&](size_t off, size_t len, bool ck) {
              return sumSel(o, avx::sel_less<T>(pos_t(len), o, a + off, c), ck); }},
      };
#ifdef VW_HAVE_NEON
      im.push_back({"neon4", [&](size_t off, size_t len, bool ck) {
         return sumSel(o, neon::sel_col_val<T, std::less>(pos_t(len), o, a + off, &c), ck); }});
      im.push_back({"neon16", [&](size_t off, size_t len, bool ck) {
         return sumSel(o, neon::sel_col_val_x16<T, std::less>(pos_t(len), o, a + off, &c), ck); }});
#endif
#ifdef VW_HAVE_SVE
      if constexpr (sizeof(T) == 4)
         im.push_back({"sve", [&](size_t off, size_t len, bool ck) {
            return sumSel(o, neon::sve_sel_col_val_i32<std::less>(pos_t(len), o, a + off, c), ck); }});
#endif
      runVector("sel", tname, std::to_string(sel), im, total, sizeof(T));
   }
}

//--- selsel ------------------------------------------------------------------
template <typename T> void famSelsel(const char* tname) {
   const size_t total = std::max(vecSize, streamBytes / sizeof(T));
   std::vector<T> col(total + 16);
   std::mt19937_64 rng(2);
   for (auto& v : col) v = T(rng() % 100);
   // one input selection per vector: 50% of the positions, sorted (as a
   // previous selection leaves them); the timed unit is a selection entry
   std::vector<pos_t> inSel;
   for (size_t i = 0; i < vecSize; ++i)
      if (rng() % 2) inSel.push_back(pos_t(i));
   const size_t ns = inSel.size() / 16 * 16;
   std::vector<pos_t> out(vecSize + 16);
   for (int sel : {10, 50, 90}) {
      T c = T(sel);
      T* a = col.data();
      pos_t* o = out.data();
      pos_t* is = inSel.data();
      std::vector<Impl> im = {
          {"scalar_bf", [&](size_t off, size_t, bool ck) {
              return sumSel(o, selsel_col_val_bf<T, std::less>(pos_t(ns), is, o, a + off, &c), ck); }},
          {"avx512_simde", [&](size_t off, size_t, bool ck) {
              return sumSel(o, avx::selsel_less<T>(pos_t(ns), is, o, a + off, c), ck); }},
      };
#ifdef VW_HAVE_NEON
      im.push_back({"neon", [&](size_t off, size_t, bool ck) {
         return sumSel(o, neon::selsel_col_val<T, std::less>(pos_t(ns), is, o, a + off, &c), ck); }});
#endif
#ifdef VW_HAVE_SVE
      if constexpr (sizeof(T) == 4)
         im.push_back({"sve", [&](size_t off, size_t, bool ck) {
            return sumSel(o, neon::sve_selsel_col_val_i32<std::less>(pos_t(ns), is, o, a + off, c), ck); }});
#endif
      runVector("selsel", tname, std::to_string(sel), im, total, sizeof(T), ns);
   }
}

//--- hash --------------------------------------------------------------------
void famHash() {
   const size_t total = std::max(vecSize, streamBytes / sizeof(int32_t));
   std::vector<int32_t> keys(total + 16);
   std::mt19937_64 rng(3);
   for (auto& k : keys) k = int32_t(rng());
   keys[0] = -1, keys[1] = INT32_MIN; // sign extension matters
   std::vector<uint64_t> out(vecSize + 16);
   auto sum = [&](size_t n, bool ck) {
      uint64_t h = 0;
      if (ck) for (size_t j = 0; j < n; ++j) h = mix(h, out[j]);
      else h = out[n - 1];
      return h;
   };
   int32_t* k = keys.data();
   hash_t* o = out.data();
   {
      std::vector<Impl> im = {
          {"murmur_scalar", [&](size_t off, size_t len, bool ck) {
              primitives::hash<int32_t, runtime::MurMurHash>(pos_t(len), o, k + off); return sum(len, ck); }},
          {"murmur_avx512_simde", [&](size_t off, size_t len, bool ck) {
              avx::murmur_hash_i32(len, o, k + off, primitives::seed); return sum(len, ck); }},
      };
#ifdef VW_HAVE_NEON
      im.push_back({"murmur_neon", [&](size_t off, size_t len, bool ck) {
         neon::murmur_hash_i32(len, o, k + off, primitives::seed); return sum(len, ck); }});
#endif
#ifdef VW_HAVE_SVE
      im.push_back({"murmur_sve", [&](size_t off, size_t len, bool ck) {
         neon::sve_murmur_hash_i32(len, o, k + off, primitives::seed); return sum(len, ck); }});
#endif
      runVector("hash", "int32", "dense", im, total, 4);
      // CRC32Hash for scale (VW_USE_CRC32; its vector lowerings are in
      // run_crcbench): a different hash, so its own reference row
      std::vector<Impl> crc = {
          {"crc_scalar", [&](size_t off, size_t len, bool ck) {
              primitives::hash<int32_t, runtime::CRC32Hash>(pos_t(len), o, k + off); return sum(len, ck); }}};
      runVector("hash", "int32", "dense_crc", crc, total, 4);
   }
   {
      // hash_sel: keys through a 50% selection vector
      std::vector<pos_t> sel;
      for (size_t i = 0; i < vecSize; ++i)
         if (rng() % 2) sel.push_back(pos_t(i));
      const size_t ns = sel.size() / 8 * 8;
      pos_t* s = sel.data();
      std::vector<Impl> im = {
          {"murmur_scalar", [&](size_t off, size_t, bool ck) {
              hash_sel<int32_t, runtime::MurMurHash>(pos_t(ns), s, o, k + off); return sum(ns, ck); }},
          {"murmur_avx512_simde", [&](size_t off, size_t, bool ck) {
              avx::murmur_hash_sel_i32(ns, o, k + off, s, primitives::seed); return sum(ns, ck); }},
      };
#ifdef VW_HAVE_NEON
      im.push_back({"murmur_neon", [&](size_t off, size_t, bool ck) {
         neon::murmur_hash_sel_i32(ns, o, k + off, s, primitives::seed); return sum(ns, ck); }});
#endif
      runVector("hash", "int32", "sel50", im, total, 4, ns);
      std::vector<Impl> crc = {
          {"crc_scalar", [&](size_t off, size_t, bool ck) {
              hash_sel<int32_t, runtime::CRC32Hash>(pos_t(ns), s, o, k + off); return sum(ns, ck); }}};
      runVector("hash", "int32", "sel50_crc", crc, total, 4, ns);
   }
}

//--- proj (Q1) ---------------------------------------------------------------
// Q1's projections over the l_shipdate selection (~98%): gather path today
// (proj_sel_* templates, Q1 without VW_PROJ_DENSE) against VW_PROJ_DENSE.
// Checksums read every result through the selection, so position- and
// selection-indexed outputs compare equal. ns_per_elem is per selected row.
void famProj() {
   const size_t total = std::max(vecSize, streamBytes / (2 * sizeof(int64_t)));
   std::vector<int64_t> p1(total + 16), p2(total + 16);
   std::mt19937_64 rng(4);
   for (auto& v : p1) v = int64_t(rng() % 10000000);
   for (auto& v : p2) v = int64_t(rng() % 100);
   std::vector<int64_t> out(vecSize + 16);
   for (int pct : {98, 50}) {
      std::vector<pos_t> sel;
      for (size_t i = 0; i < vecSize; ++i)
         if (int(rng() % 100) < pct) sel.push_back(pos_t(i));
      const pos_t ns = pos_t(sel.size());
      const pos_t span = ns ? sel[ns - 1] + 1 : 0;
      pos_t* s = sel.data();
      int64_t* o = out.data();
      auto bySel = [&](bool dense, bool ck) -> uint64_t {
         if (!ck) return uint64_t(o[0]);
         uint64_t h = 0;
         for (pos_t j = 0; j < ns; ++j) h = mix(h, uint64_t(dense ? o[s[j]] : o[j]));
         return h;
      };
      int64_t one = 100;
      const std::string param = std::to_string(pct);
      // 1 - l_discount
      {
         std::vector<Impl> im = {
             {"gather_scalar", [&](size_t off, size_t, bool ck) {
                 proj_sel_val_col<int64_t, std::minus>(ns, s, o, &one, p2.data() + off); return bySel(false, ck); }},
             {"dense_scalar", [&](size_t off, size_t, bool ck) {
                 // Projection.cpp:217 scalar loop (the whole span on aarch64)
                 const int64_t* q = p2.data() + off;
                 for (pos_t i = 0; i < span; ++i) o[i] = one - q[i];
                 return bySel(true, ck); }},
             {"dense_avx512_simde", [&](size_t off, size_t, bool ck) {
                 avx::dense_minus(span, o, one, p2.data() + off); return bySel(true, ck); }},
         };
#ifdef VW_HAVE_NEON
         im.push_back({"dense_neon", [&](size_t off, size_t, bool ck) {
            neon::proj_dense_minus_val_col(ns, s, o, &one, p2.data() + off); return bySel(true, ck); }});
#endif
         runVector("proj", "int64", "minus_val_col/" + param, im, total, 8, ns);
      }
      // l_tax + 1
      {
         std::vector<Impl> im = {
             {"gather_scalar", [&](size_t off, size_t, bool ck) {
                 proj_sel_col_val<int64_t, std::plus>(ns, s, o, p2.data() + off, &one); return bySel(false, ck); }},
             {"dense_scalar", [&](size_t off, size_t, bool ck) {
                 const int64_t* q = p2.data() + off;
                 for (pos_t i = 0; i < span; ++i) o[i] = q[i] + one;
                 return bySel(true, ck); }},
             {"dense_avx512_simde", [&](size_t off, size_t, bool ck) {
                 avx::dense_plus(span, o, p2.data() + off, one); return bySel(true, ck); }},
         };
#ifdef VW_HAVE_NEON
         im.push_back({"dense_neon", [&](size_t off, size_t, bool ck) {
            neon::proj_dense_plus_col_val(ns, s, o, p2.data() + off, &one); return bySel(true, ck); }});
#endif
         runVector("proj", "int64", "plus_col_val/" + param, im, total, 8, ns);
      }
      // l_extendedprice * (1 - l_discount): param1 through the selection,
      // param2 the previous result (selection-indexed, or position-indexed
      // under VW_PROJ_DENSE; here a second column at the same index)
      {
         std::vector<int64_t> q2(vecSize + 16);
         for (auto& v : q2) v = int64_t(rng() % 100);
         // the previous result, selection-indexed as the gather path has it
         std::vector<int64_t> q2c(ns);
         for (pos_t j = 0; j < ns; ++j) q2c[j] = q2[s[j]];
         std::vector<Impl> im = {
             {"gather_scalar", [&](size_t off, size_t, bool ck) {
                 // Q1: proj_multiplies_sel_int64_t_col_int64_t_col
                 proj_sel_col_col<int64_t, std::multiplies>(ns, s, o, p1.data() + off, q2c.data());
                 return bySel(false, ck); }},
             {"dense_scalar", [&](size_t off, size_t, bool ck) {
                 const int64_t* q = p1.data() + off;
                 for (pos_t i = 0; i < span; ++i) o[i] = q[i] * q2[i];
                 return bySel(true, ck); }},
             {"dense_avx512_simde", [&](size_t off, size_t, bool ck) {
                 avx::dense_mul(span, o, p1.data() + off, q2.data()); return bySel(true, ck); }},
         };
#ifdef VW_HAVE_NEON
         im.push_back({"dense_neon", [&](size_t off, size_t, bool ck) {
            neon::proj_dense_multiplies_col_col(ns, s, o, p1.data() + off, q2.data()); return bySel(true, ck); }});
#endif
#ifdef VW_HAVE_SVE
         im.push_back({"dense_sve", [&](size_t off, size_t, bool ck) {
            neon::sve_proj_dense_multiplies(ns, s, o, p1.data() + off, q2.data()); return bySel(true, ck); }});
#endif
         runVector("proj", "int64", "mul_col_col/" + param, im, total, 16, ns);
      }
   }
}

//--- runheads ----------------------------------------------------------------
// Keys in runs: a row repeats its predecessor's key with probability rep
// (Q1's 2-byte group key: 64% repeats; Q18's l_orderkey: ~75%).
template <typename K> void famRunHeads(const char* tname, int repPct) {
   const size_t total = std::max(vecSize, streamBytes / sizeof(K));
   std::vector<K> keys(total + 16);
   std::mt19937_64 rng(5);
   K cur = 0;
   for (auto& k : keys) {
      if (int(rng() % 100) >= repPct) cur = K(cur + 1 + rng() % 3);
      k = cur;
   }
   std::vector<pos_t> heads(vecSize + 16);
   std::vector<uint8_t> flags(vecSize + 16);
   auto res = [&](pos_t cnt, size_t n, bool ck) -> uint64_t {
      if (!ck) return cnt;
      uint64_t h = sumSel(heads.data(), cnt, true);
      for (size_t i = 1; i < n; ++i) h = mix(h, flags[i]);
      return h;
   };
   pos_t* hd = heads.data();
   uint8_t* fl = flags.data();
   std::vector<Impl> im = {
       {"scalar_bf", [&](size_t off, size_t len, bool ck) {
           // Operators.cpp:2141-2146 (the non-AVX-512 pass 1; restrict as
           // keyStepT's base / heads / flags, else the byte stores to flags
           // force reloads)
           const K* __restrict__ k = keys.data() + off;
           pos_t* __restrict__ h = hd;
           uint8_t* __restrict__ f = fl;
           pos_t cnt = 1;
           h[0] = 0, f[0] = 1;
           for (pos_t i = 1; i < len; ++i) {
              const bool head = k[i] != k[i - 1];
              f[i] = head;
              h[cnt] = i;
              cnt += head;
           }
           return res(cnt, len, ck); }},
       {"avx512_simde", [&](size_t off, size_t len, bool ck) {
           hd[0] = 0, fl[0] = 1;
           return res(avx::run_heads<K>(pos_t(len), keys.data() + off, hd, fl, 1, 1), len, ck); }},
   };
#ifdef VW_HAVE_NEON
   im.push_back({"neon", [&](size_t off, size_t len, bool ck) {
      hd[0] = 0, fl[0] = 1;
      return res(neon::run_heads<K>(pos_t(len), keys.data() + off, hd, fl, 1, 1), len, ck); }});
#endif
   // keys + off - 1 is read for row 1 of every vector but the first
   runVector("runheads", tname, std::to_string(repPct), im, total, sizeof(K));
}

//--- bloom -------------------------------------------------------------------
inline uint64_t bloomBits(uint64_t h) {
   return (uint64_t(1) << (h & 63)) | (uint64_t(1) << ((h >> 6) & 63)) |
          (uint64_t(1) << ((h >> 12) & 63)) | (uint64_t(1) << ((h >> 18) & 63));
}
void famBloom() {
   std::mt19937_64 rng(6);
   std::vector<uint64_t> H(probes + 16);
   std::vector<pos_t> sel(vecSize + 16);
   std::vector<uint64_t> hs(vecSize + 16);
   for (size_t bytes : tableSizes()) {
      const size_t words = bytes / 8;
      const uint64_t mask = words - 1;
      std::vector<uint64_t> bloom(words, 0);
      // VW_JOIN_BLOOM_BITS 16 bits per build key
      const size_t keys = bytes * 8 / 16;
      std::vector<uint64_t> built(keys);
      for (auto& h : built) {
         h = rng();
         bloom[(h >> 40) & mask] |= bloomBits(h);
      }
      // probes: 5% build keys (Q3-like hit rate), the rest random
      for (auto& h : H) h = (rng() % 100 < 5) ? built[rng() % keys] : rng();
      const uint64_t* B = bloom.data();
      pos_t* S = sel.data();
      uint64_t* HS = hs.data();
      auto res = [&](size_t m, bool ck) -> uint64_t {
         if (!ck) return m;
         uint64_t h = sumSel(S, m, true);
         for (size_t j = 0; j < m; ++j) h = mix(h, HS[j]);
         return h;
      };
      std::vector<Impl> im = {
          {"scalar", [&](size_t off, size_t len, bool ck) {
              // Operators.cpp:1043-1047 (the non-AVX-512 loop)
              const uint64_t* p = H.data() + off;
              size_t m = 0;
              for (size_t i = 0; i < len; ++i) {
                 const auto h = p[i];
                 const uint64_t b = bloomBits(h);
                 if ((B[(h >> 40) & mask] & b) == b) S[m] = pos_t(i), HS[m++] = h;
              }
              return res(m, ck); }},
          {"scalar_bf", [&](size_t off, size_t len, bool ck) {
              const uint64_t* p = H.data() + off;
              size_t m = 0;
              for (size_t i = 0; i < len; ++i) {
                 const auto h = p[i];
                 const uint64_t b = bloomBits(h);
                 S[m] = pos_t(i), HS[m] = h;
                 m += (B[(h >> 40) & mask] & b) == b;
              }
              return res(m, ck); }},
          {"scalar_bf_pf", [&](size_t off, size_t len, bool ck) {
              // + software prefetch of the word 16 probes ahead
              const uint64_t* p = H.data() + off;
              size_t m = 0;
              for (size_t i = 0; i < len; ++i) {
                 __builtin_prefetch(&B[(p[i + 16] >> 40) & mask]);
                 const auto h = p[i];
                 const uint64_t b = bloomBits(h);
                 S[m] = pos_t(i), HS[m] = h;
                 m += (B[(h >> 40) & mask] & b) == b;
              }
              return res(m, ck); }},
          {"avx512_simde", [&](size_t off, size_t len, bool ck) {
              return res(avx::bloom(len, H.data() + off, B, mask, S, HS), ck); }},
      };
#ifdef VW_HAVE_NEON
      im.push_back({"neon", [&](size_t off, size_t len, bool ck) {
         return res(neon::bloom_filter(len, H.data() + off, B, mask, S, HS), ck); }});
#endif
      runSweep("bloom", "hash64", bytesStr(bytes), im, probes, 8);
   }
}

//--- semi --------------------------------------------------------------------
void famSemi() {
   std::mt19937_64 rng(7);
   std::vector<int32_t> keys(probes + 16);
   std::vector<pos_t> out(vecSize + 16);
   for (size_t bytes : tableSizes()) {
      const uint32_t span = uint32_t(bytes * 8 - 1); // last offset
      const int32_t lo = 1000;
      std::vector<uint32_t> bits(bytes / 4, 0);
      // 10% of the range are build keys
      for (size_t j = 0; j < (size_t(span) + 1) / 10; ++j) {
         const uint32_t d = uint32_t(rng() % (uint64_t(span) + 1));
         bits[d >> 5] |= 1u << (d & 31);
      }
      // probe keys: 90% inside [lo, lo + span], 10% below lo
      for (auto& k : keys)
         k = (rng() % 10) ? lo + int32_t(rng() % (uint64_t(span) + 1))
                          : lo - 1 - int32_t(rng() % 1000);
      const uint32_t* Bt = bits.data();
      pos_t* o = out.data();
      std::vector<Impl> im = {
          {"scalar", [&](size_t off, size_t len, bool ck) {
              // Operators.cpp:956-960 (the non-AVX-512 loop, dense keys)
              const int32_t* k = keys.data() + off;
              size_t found = 0;
              for (size_t i = 0; i < len; ++i) {
                 const uint64_t d = uint64_t(int64_t(k[i]) - lo);
                 if (d <= span && (Bt[d >> 5] >> (d & 31) & 1)) o[found++] = pos_t(i);
              }
              return sumSel(o, found, ck); }},
          {"scalar_bf", [&](size_t off, size_t len, bool ck) {
              const int32_t* k = keys.data() + off;
              size_t found = 0;
              for (size_t i = 0; i < len; ++i) {
                 const uint32_t d = uint32_t(k[i]) - uint32_t(lo);
                 const uint32_t dc = d <= span ? d : span;
                 o[found] = pos_t(i);
                 found += (d <= span) & (Bt[dc >> 5] >> (dc & 31));
              }
              return sumSel(o, found, ck); }},
          {"scalar_bf_pf", [&](size_t off, size_t len, bool ck) {
              // + software prefetch of the bitmap word 16 probes ahead
              const int32_t* k = keys.data() + off;
              size_t found = 0;
              for (size_t i = 0; i < len; ++i) {
                 const uint32_t dp = uint32_t(k[i + 16]) - uint32_t(lo);
                 __builtin_prefetch(&Bt[(dp <= span ? dp : span) >> 5]);
                 const uint32_t d = uint32_t(k[i]) - uint32_t(lo);
                 const uint32_t dc = d <= span ? d : span;
                 o[found] = pos_t(i);
                 found += (d <= span) & (Bt[dc >> 5] >> (dc & 31));
              }
              return sumSel(o, found, ck); }},
          {"avx512_simde", [&](size_t off, size_t len, bool ck) {
              return sumSel(o, avx::semi(len, keys.data() + off, Bt, lo, span, o), ck); }},
      };
#ifdef VW_HAVE_NEON
      im.push_back({"neon", [&](size_t off, size_t len, bool ck) {
         return sumSel(o, neon::semi_probe(len, keys.data() + off, Bt, lo, span, o), ck); }});
#endif
      runSweep("semi", "int32", bytesStr(bytes), im, probes, 4);
   }
}

//--- dirprobe ----------------------------------------------------------------
// The first pass of VW_NEW_JOIN on a tagged directory (runtime::Hashmap
// layout: 8-byte slots, 16 tag bits on top of a 48-bit pointer; tag bit
// 48 + top 4 hash bits). Per probe: slot load; full = tag bit set; match =
// head->hash == h; real list (head, id); successors of real and maybe
// (full && !match) heads. Load factor 0.5; 5% of probes are build hashes.
struct DEntry {
   DEntry* next;
   uint64_t hash;
   uint64_t payload[2];
};
constexpr uint64_t kPtrMask = (uint64_t(1) << 48) - 1;
inline uint64_t tagOf(uint64_t h) { return uint64_t(1) << (48 + (h >> 60)); }

struct DirOut {
   std::vector<DEntry*> real, rf, mf;
   std::vector<pos_t> realId, rfId, mfId;
   explicit DirOut(size_t n)
       : real(n + 8), rf(n + 8), mf(n + 8), realId(n + 8), rfId(n + 8), mfId(n + 8) {}
};

void famDirProbe() {
   std::mt19937_64 rng(8);
   std::vector<uint64_t> H(probes + 16);
   DirOut out(vecSize);
   for (size_t bytes : tableSizes()) {
      const size_t slots = bytes / 8;
      const uint64_t mask = slots - 1;
      std::vector<uint64_t> dir(slots, 0);
      std::vector<DEntry> ent(slots / 2 + 1);
      DEntry* dummy = &ent.back(); // branch-free variant's stand-in head
      dummy->next = nullptr, dummy->hash = ~uint64_t(0);
      for (size_t e = 0; e + 1 < ent.size(); ++e) {
         const uint64_t h = rng();
         ent[e].hash = h;
         uint64_t& d = dir[h & mask];
         ent[e].next = reinterpret_cast<DEntry*>(d & kPtrMask);
         d = reinterpret_cast<uint64_t>(&ent[e]) | (d & ~kPtrMask) | tagOf(h);
      }
      for (auto& h : H) h = (rng() % 100 < 5) ? ent[rng() % (ent.size() - 1)].hash : rng();
      const uint64_t* D = dir.data();
      auto res = [&](size_t f, size_t r, size_t m, bool ck) -> uint64_t {
         if (!ck) return f + r + m;
         uint64_t h = mix(mix(f, r), m);
         for (size_t j = 0; j < f; ++j) h = mix(mix(h, uint64_t(out.real[j]->hash)), out.realId[j]);
         for (size_t j = 0; j < r; ++j) h = mix(mix(h, out.rf[j]->hash), out.rfId[j]);
         for (size_t j = 0; j < m; ++j) h = mix(mix(h, out.mf[j]->hash), out.mfId[j]);
         return h;
      };
      // Operators.cpp:1340-1373: the 8-lane scalar block (non-AVX-512 build)
      auto branchy = [&](const uint64_t* p, size_t len, bool prefetched, bool ck) {
         (void)prefetched;
         size_t f = 0, r = 0, m = 0;
         for (size_t i = 0; i < len; i += 8) {
            DEntry* head[8];
            DEntry* next[8];
            bool match[8];
            for (size_t l = 0; l < 8; ++l) {
               const uint64_t h = p[i + l];
               const uint64_t dv = D[h & mask];
               head[l] = reinterpret_cast<DEntry*>(dv & kPtrMask);
               const bool full = (dv & tagOf(h)) != 0;
               match[l] = full && head[l]->hash == h;
               next[l] = full ? head[l]->next : nullptr;
            }
            for (size_t l = 0; l < 8; ++l)
               if (match[l]) out.real[f] = head[l], out.realId[f++] = pos_t(i + l);
            for (size_t l = 0; l < 8; ++l) {
               if (!next[l]) continue;
               if (match[l]) out.rf[r] = next[l], out.rfId[r++] = pos_t(i + l);
               else out.mf[m] = next[l], out.mfId[m++] = pos_t(i + l);
            }
         }
         return res(f, r, m, ck);
      };
      std::vector<Impl> im = {
          {"scalar_block8", [&](size_t off, size_t len, bool ck) {
              return branchy(H.data() + off, len, false, ck); }},
          {"scalar_2phase", [&](size_t off, size_t len, bool ck) {
              // VW_JOIN_TWOPHASE-style: prefetch every slot of the vector
              // first, then the same block loop
              const uint64_t* p = H.data() + off;
              for (size_t i = 0; i < len; ++i) __builtin_prefetch(&D[p[i] & mask]);
              return branchy(p, len, true, ck); }},
          {"scalar_bf", [&](size_t off, size_t len, bool ck) {
              // branch-free: a probe that is not full reads the dummy head,
              // every output slot is written and advanced by a flag
              const uint64_t* p = H.data() + off;
              size_t f = 0, r = 0, m = 0;
              for (size_t i = 0; i < len; ++i) {
                 const uint64_t h = p[i];
                 const uint64_t dv = D[h & mask];
                 const bool full = (dv & tagOf(h)) != 0;
                 DEntry* head = full ? reinterpret_cast<DEntry*>(dv & kPtrMask) : dummy;
                 const bool match = head->hash == h;
                 DEntry* next = head->next;
                 const bool has = next != nullptr;
                 out.real[f] = head, out.realId[f] = pos_t(i);
                 f += match;
                 out.rf[r] = next, out.rfId[r] = pos_t(i);
                 r += match & has;
                 out.mf[m] = next, out.mfId[m] = pos_t(i);
                 m += full & !match & has;
              }
              return res(f, r, m, ck); }},
          {"avx512_simde", [&](size_t off, size_t len, bool ck) {
              // Operators.cpp:1274-1337 (VW_NEW_JOIN_FULL_TAG, no Bloom);
              // the absolute-address gathers become base + offset gathers
              const uint64_t* p = H.data() + off;
              const char* base = reinterpret_cast<const char*>(ent.data());
              const simde__m512i vBase = simde_mm512_set1_epi64((long long)(uintptr_t)base);
              const simde__m512i vMask = simde_mm512_set1_epi64((long long)mask);
              const simde__m512i vMaskPtr = simde_mm512_set1_epi64((long long)kPtrMask);
              const simde__m512i zero = simde_mm512_setzero_si512();
              const simde__m512i hashOff = simde_mm512_set1_epi64((long long)offsetof(DEntry, hash));
              const simde__m512i lane = simde_mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0);
              const simde__m512i tagBase = simde_mm512_set1_epi64(48), one = simde_mm512_set1_epi64(1);
              size_t f = 0, r = 0, m = 0;
              for (size_t i = 0; i + 8 <= len; i += 8) {
                 const simde__m512i h = simde_mm512_loadu_si512(p + i);
                 const simde__m512i dv = simde_mm512_i64gather_epi64(simde_mm512_and_si512(h, vMask), D, 8);
                 const simde__m512i head = simde_mm512_and_si512(dv, vMaskPtr);
                 const simde__m512i tag = simde_mm512_sllv_epi64(
                     one, simde_mm512_add_epi64(simde_mm512_srli_epi64(h, 60), tagBase));
                 const simde__mmask8 full = simde_mm512_test_epi64_mask(dv, tag);
                 if (!full) continue;
                 const simde__m512i rel = simde_mm512_sub_epi64(head, vBase);
                 const simde__m512i headHash = avx::mask_i64gather_epi64(
                     zero, full, simde_mm512_add_epi64(rel, hashOff), base);
                 const simde__mmask8 match = simde_mm512_mask_cmpeq_epi64_mask(full, headHash, h);
                 const simde__mmask8 maybe = full & simde__mmask8(~match);
                 const simde__m512i next = avx::mask_i64gather_epi64(zero, full, rel, base);
                 const simde__mmask8 hasNext = simde_mm512_mask_cmpneq_epi64_mask(full, next, zero);
                 const simde__m512i idx = simde_mm512_add_epi32(simde_mm512_set1_epi32(int(i)), lane);
                 avx::compressStore64(out.real.data() + f, match, head);
                 avx::compressStore32(out.realId.data() + f, match, idx);
                 f += __builtin_popcount(match);
                 const simde__mmask8 rfm = match & hasNext, mfm = maybe & hasNext;
                 avx::compressStore64(out.rf.data() + r, rfm, next);
                 avx::compressStore32(out.rfId.data() + r, rfm, idx);
                 r += __builtin_popcount(rfm);
                 avx::compressStore64(out.mf.data() + m, mfm, next);
                 avx::compressStore32(out.mfId.data() + m, mfm, idx);
                 m += __builtin_popcount(mfm);
              }
              return res(f, r, m, ck); }},
      };
      runSweep("dirprobe", "hash64", bytesStr(bytes), im, probes, 8);
   }
}

//--- roof --------------------------------------------------------------------
void famRoof() {
   if (wantMode("stream")) {
      const size_t n = streamBytes / 8;
      std::vector<uint64_t> a(n);
      for (size_t i = 0; i < n; ++i) a[i] = i;
      const double ns = bestNs([&] {
         uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
         for (size_t i = 0; i + 4 <= n; i += 4)
            s0 += a[i], s1 += a[i + 1], s2 += a[i + 2], s3 += a[i + 3];
         return s0 + s1 + s2 + s3;
      }) / double(n);
      row("roof", "read_scalar", "uint64", "stream", bytesStr(streamBytes), ns, 8, true);
#ifdef VW_HAVE_NEON
      const double nsv = bestNs([&] {
         uint64x2_t s0 = vdupq_n_u64(0), s1 = s0, s2 = s0, s3 = s0;
         for (size_t i = 0; i + 8 <= n; i += 8) {
            s0 = vaddq_u64(s0, vld1q_u64(&a[i]));
            s1 = vaddq_u64(s1, vld1q_u64(&a[i + 2]));
            s2 = vaddq_u64(s2, vld1q_u64(&a[i + 4]));
            s3 = vaddq_u64(s3, vld1q_u64(&a[i + 6]));
         }
         return vaddvq_u64(vaddq_u64(vaddq_u64(s0, s1), vaddq_u64(s2, s3)));
      }) / double(n);
      row("roof", "read_neon", "uint64", "stream", bytesStr(streamBytes), nsv, 8, true);
#endif
   }
   if (wantMode("sweep")) {
      // dependent loads over a random cycle of cache lines: latency per level
      std::mt19937_64 rng(9);
      for (size_t bytes : tableSizes()) {
         const size_t lines = bytes / 64;
         std::vector<uint32_t> perm(lines);
         for (size_t i = 0; i < lines; ++i) perm[i] = uint32_t(i);
         std::shuffle(perm.begin(), perm.end(), rng);
         std::vector<uint32_t> next(lines * 16);
         for (size_t i = 0; i < lines; ++i)
            next[size_t(perm[i]) * 16] = perm[(i + 1) % lines] * 16;
         const size_t hops = std::max<size_t>(probes / 4, 1 << 16);
         const double ns = bestNs([&] {
            uint32_t p = 0;
            for (size_t i = 0; i < hops; ++i) p = next[p];
            return uint64_t(p);
         }) / double(hops);
         row("roof", "chase_latency", "line", "sweep", bytesStr(bytes), ns, 0, true);
         // independent loads at the same addresses: what MLP buys
         const double nsi = bestNs([&] {
            uint64_t s = 0;
            for (size_t i = 0; i < hops; ++i) s += next[size_t(perm[i & (lines - 1)]) * 16];
            return s;
         }) / double(hops);
         row("roof", "random_throughput", "line", "sweep", bytesStr(bytes), nsi, 0, true);
      }
   }
}

} // namespace

int main(int argc, char** argv) {
   int opt;
   while ((opt = getopt(argc, argv, "v:r:s:p:t:m:f:i:y:q:T:")) != -1) {
      switch (opt) {
      case 'v': vecSize = std::strtoull(optarg, nullptr, 10); break;
      case 'r': reps = std::strtoull(optarg, nullptr, 10); break;
      case 's': streamBytes = std::strtoull(optarg, nullptr, 10) << 20; break;
      case 'p': probes = std::strtoull(optarg, nullptr, 10); break;
      case 't': maxTable = std::strtoull(optarg, nullptr, 10) << 20; break;
      case 'm': mode = optarg; break;
      case 'f': families = optarg; break;
      case 'i': implFilter = optarg; break;
      case 'y': typeFilter = optarg; break;
      case 'q': paramFilter = optarg; break;
      case 'T': trials = std::max(1, std::atoi(optarg)); break;
      default:
         std::fprintf(stderr,
                      "usage: %s [-v vec] [-r reps] [-s streamMiB] [-p probes] "
                      "[-t maxTableMiB] [-m l1|stream|sweep|all] [-f fam,...]\n"
                      "       [-i impl,...] [-y type,...] [-q param,...] [-T trials]\n",
                      argv[0]);
         return 2;
      }
   }
   trace = std::getenv("NEONBENCH_TRACE") != nullptr;
   if (vecSize % 16 || vecSize == 0 || probes % vecSize) {
      std::fprintf(stderr, "vec must be a multiple of 16, probes of vec\n");
      return 2;
   }
#if defined(__aarch64__)
   std::fprintf(stderr, "run_neonbench: aarch64 neon=%d sve=%d (avx512_simde rows are SIMDe emulation)\n",
#ifdef VW_HAVE_NEON
                1,
#else
                0,
#endif
#ifdef VW_HAVE_SVE
                1
#else
                0
#endif
   );
#elif defined(SIMDE_X86_AVX512F_NATIVE)
   std::fprintf(stderr, "run_neonbench: x86 AVX-512 native (avx512_simde rows are the engine's kernels)\n");
#else
   std::fprintf(stderr, "run_neonbench: x86 without AVX-512 (avx512_simde rows are SIMDe emulation)\n");
#endif
   std::printf("family,impl,type,mode,param,vec,ns_per_elem,gb_per_s,check,autovec\n");
   const bool vec = wantMode("l1") || wantMode("stream");
   if (vec && wantFam("sel")) famSel<int32_t>("int32"), famSel<int64_t>("int64");
   if (vec && wantFam("selsel")) famSelsel<int32_t>("int32"), famSelsel<int64_t>("int64");
   if (vec && wantFam("hash")) famHash();
   if (vec && wantFam("proj")) famProj();
   if (vec && wantFam("runheads")) famRunHeads<uint16_t>("uint16", 64), famRunHeads<uint32_t>("uint32", 75);
   if (wantMode("sweep") && wantFam("bloom")) famBloom();
   if (wantMode("sweep") && wantFam("semi")) famSemi();
   if (wantMode("sweep") && wantFam("dirprobe")) famDirProbe();
   if (wantFam("roof")) famRoof();
   if (fails) std::fprintf(stderr, "run_neonbench: %d MISMATCH rows\n", fails);
   return fails ? 1 : 0;
}
