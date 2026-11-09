// run_hashbench: per-tuple cost of the hashing primitives. For every form
// (hash, hash_sel, rehash, rehash_sel, HashGroup packed keys) and key type it
// times
//   murmur_scalar : the scalar template with runtime::MurMurHash (default)
//   crc32_scalar  : the scalar template with runtime::CRC32Hash (VW_USE_CRC32)
//   murmur_avx512 : the SIMD MurmurHash64A tiers (VW_SIMD_HASH), every one
//   murmur_avx2     the target has (avx512 and avx2 on AVX-512 x86, avx2 on
//   murmur_neon     AVX2 x86, neon on aarch64)
//   crc32_vpclmul : VW_CRC32_VPCLMUL's kernels (hash, hash_sel, packed keys;
//                   AVX-512 + VPCLMULQDQ targets)
// crc32_scalar is named crc32_fast when built with VW_CRC32_FAST (one crc32
// per key with a fixed seed).
//   hash4_old     : the pre-existing hash4 / hash4_sel(ASM) kernels (int32
//                   only; SIMDhash=1 path; zero-extends keys)
// All impls are compiled into one binary regardless of the CMake options.
//
// Usage: run_hashbench [-v vecSize] [-m l1|stream|all] [-r reps] [-d density%]
//   l1     : one vecSize-long vector hashed reps times (cache resident)
//   stream : a 64 MiB column hashed vector by vector
// Output: CSV on stdout.
#include "common/runtime/Hash.hpp"
#include "common/runtime/Types.hpp"
#include "vectorwise/Primitives.hpp"
#include "vectorwise/SimdCrc.hpp"
#include "vectorwise/SimdHash.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

using namespace vectorwise;
using namespace vectorwise::primitives;
using runtime::CRC32Hash;
using runtime::MurMurHash;
using types::Date;
using hash_t = defs::hash_t;

namespace {

size_t vecSize = 1024;
size_t reps = 20000;
std::string mode = "all";
int density = 50;
constexpr size_t kStreamBytes = 64ull << 20;
volatile hash_t sink;

template <typename T> const char* tname();
template <> const char* tname<int32_t>() { return "int32"; }
template <> const char* tname<int64_t>() { return "int64"; }
template <> const char* tname<Date>() { return "Date"; }
template <> const char* tname<uint16_t>() { return "u16_key"; }
template <> const char* tname<uint64_t>() { return "u64_key"; }

bool want(const char* m) { return mode == "all" || mode == m; }

template <typename F> double bestNs(F&& body, size_t tuples) {
   double best = 1e300;
   for (int trial = 0; trial < 5; ++trial) {
      auto t0 = std::chrono::steady_clock::now();
      body();
      auto t1 = std::chrono::steady_clock::now();
      best = std::min(best,
                      std::chrono::duration<double, std::nano>(t1 - t0).count());
   }
   return best / double(tuples);
}

void row(const char* form, const char* impl, const char* type, const char* md,
         int dens, double ns) {
   std::printf("%s,%s,%s,%s,%zu,%d,%.4f\n", form, impl, type, md, vecSize, dens,
               ns);
   std::fflush(stdout);
}

struct ImplF2 {
   const char* name;
   F2 fn; // (n, result, input)
};
struct ImplF3 {
   const char* name;
   F3 fn; // (n, inSel, result, input)
};

/// hash / rehash: fn(n, result, input). rehash reuses result as seeds, which
/// the loop keeps valid (it only reads its own previous output).
template <typename T>
void benchDense(const char* form, const std::vector<ImplF2>& impls) {
   std::mt19937_64 rng(1);
   const size_t total = want("stream") ? kStreamBytes / sizeof(T) : vecSize;
   std::vector<T> in(std::max(total, vecSize));
   for (auto& x : in) x = T(int32_t(rng()));
   std::vector<hash_t> out(std::max(total, vecSize) + 16, 1);
   for (auto& im : impls) {
      if (!im.fn) continue;
      if (want("l1")) {
         double ns = bestNs([&] {
            for (size_t r = 0; r < reps; ++r) im.fn(vecSize, out.data(), in.data());
         }, reps * vecSize);
         sink = out[0];
         row(form, im.name, tname<T>(), "l1", 100, ns);
      }
      if (want("stream")) {
         const size_t chunks = total / vecSize;
         double ns = bestNs([&] {
            for (size_t c = 0; c < chunks; ++c)
               im.fn(vecSize, out.data() + c * vecSize, in.data() + c * vecSize);
         }, chunks * vecSize);
         sink = out[0];
         row(form, im.name, tname<T>(), "stream", 100, ns);
      }
   }
}

/// hash_sel / rehash_sel: fn(n, inSel, result, input); inSel keeps ~density%.
template <typename T>
void benchSel(const char* form, const std::vector<ImplF3>& impls) {
   std::mt19937_64 rng(2);
   const size_t total = want("stream") ? kStreamBytes / sizeof(T) : vecSize;
   std::vector<T> in(std::max(total, vecSize));
   for (auto& x : in) x = T(int32_t(rng()));
   std::vector<pos_t> sel;
   for (size_t i = 0; i < vecSize; ++i)
      if (int(rng() % 100) < density) sel.push_back(pos_t(i));
   const size_t k = sel.size();
   std::vector<hash_t> out(vecSize + 16, 1);
   for (auto& im : impls) {
      if (!im.fn) continue;
      if (want("l1")) {
         double ns = bestNs([&] {
            for (size_t r = 0; r < reps; ++r)
               im.fn(k, sel.data(), out.data(), in.data());
         }, reps * k);
         sink = out[0];
         row(form, im.name, tname<T>(), "l1", density, ns);
      }
      if (want("stream")) {
         const size_t chunks = total / vecSize;
         double ns = bestNs([&] {
            for (size_t c = 0; c < chunks; ++c)
               im.fn(k, sel.data(), out.data(), in.data() + c * vecSize);
         }, chunks * k);
         sink = out[0];
         row(form, im.name, tname<T>(), "stream", density, ns);
      }
   }
}

#ifdef VW_CRC32_FAST
constexpr const char* kCrcName = "crc32_fast";
#else
constexpr const char* kCrcName = "crc32_scalar";
#endif

/// the four primitive forms of one hash implementation
struct Forms {
   const char* name;
   F2 hash, rehash;
   F3 hash_sel, rehash_sel;
};
#define VW_MURMUR_FORMS(Name, NS)                                              \
   Forms {                                                                     \
      Name, (F2)&NS::hash<T, MurMurHash>, (F2)&NS::rehash<T, MurMurHash>,      \
          (F3)&NS::hash_sel<T, MurMurHash>, (F3)&NS::rehash_sel<T, MurMurHash> \
   }

template <typename T> std::vector<Forms> implsOf() {
   std::vector<Forms> f = {
       VW_MURMUR_FORMS("murmur_scalar", primitives),
       {kCrcName, (F2)&primitives::hash<T, CRC32Hash>,
        (F2)&primitives::rehash<T, CRC32Hash>,
        (F3)&primitives::hash_sel<T, CRC32Hash>,
        (F3)&primitives::rehash_sel<T, CRC32Hash>}};
#ifdef VW_HAVE_SIMD_HASH
   f.push_back(VW_MURMUR_FORMS("murmur_avx512", simd_hash::avx512));
#endif
#ifdef VW_HAVE_SIMD_HASH_AVX2
   f.push_back(VW_MURMUR_FORMS("murmur_avx2", simd_hash::avx2));
#endif
#ifdef VW_HAVE_SIMD_HASH_NEON
   f.push_back(VW_MURMUR_FORMS("murmur_neon", simd_hash::neon));
#endif
#ifdef VW_HAVE_SIMD_CRC
   // no rehash kernel (a per-row seed needs two reductions per key)
   f.push_back({"crc32_vpclmul", (F2)&simd_crc::hash<T>, nullptr,
                (F3)&simd_crc::hash_sel<T>, nullptr});
#endif
   return f;
}

template <typename T> void allForms() {
   const auto impls = implsOf<T>();
   std::vector<ImplF2> h, rh;
   std::vector<ImplF3> hs, rhs;
   for (auto& f : impls) {
      h.push_back({f.name, f.hash});
      rh.push_back({f.name, f.rehash});
      hs.push_back({f.name, f.hash_sel});
      rhs.push_back({f.name, f.rehash_sel});
   }
   benchDense<T>("hash", h);
   benchDense<T>("rehash", rh);
   benchSel<T>("hash_sel", hs);
   benchSel<T>("rehash_sel", rhs);
}

/// HashGroup packed keys: hashes[i] = hashFn.hashKey(key_i)
template <typename T> void packedKeys() {
   std::mt19937_64 rng(3);
   std::vector<T> keys(vecSize);
   for (auto& x : keys) x = T(rng());
   std::vector<hash_t> out(vecSize + 16);
   MurMurHash mm;
   CRC32Hash crc;
   auto scalarMM = [&] {
      for (size_t i = 0; i < vecSize; ++i) out[i] = mm.hashKey(keys[i]);
   };
   auto scalarCRC = [&] {
      for (size_t i = 0; i < vecSize; ++i) out[i] = crc.hashKey(keys[i]);
   };
   double ns = bestNs([&] { for (size_t r = 0; r < reps; ++r) scalarMM(); }, reps * vecSize);
   row("packed_keys", "murmur_scalar", tname<T>(), "l1", 100, ns);
   ns = bestNs([&] { for (size_t r = 0; r < reps; ++r) scalarCRC(); }, reps * vecSize);
   row("packed_keys", kCrcName, tname<T>(), "l1", 100, ns);
   const char* k = reinterpret_cast<const char*>(keys.data());
   auto simd = [&](const char* name, void (*fn)(size_t, const char*, hash_t*)) {
      const double t = bestNs([&] {
         for (size_t r = 0; r < reps; ++r) fn(vecSize, k, out.data());
      }, reps * vecSize);
      row("packed_keys", name, tname<T>(), "l1", 100, t);
   };
#ifdef VW_HAVE_SIMD_HASH
   simd("murmur_avx512", &simd_hash::avx512::hash_keys<T>);
#endif
#ifdef VW_HAVE_SIMD_HASH_AVX2
   simd("murmur_avx2", &simd_hash::avx2::hash_keys<T>);
#endif
#ifdef VW_HAVE_SIMD_HASH_NEON
   simd("murmur_neon", &simd_hash::neon::hash_keys<T>);
#endif
#ifdef VW_HAVE_SIMD_CRC
   simd("crc32_vpclmul", &simd_crc::hash_keys<T>);
#endif
   sink = out[0];
}

} // namespace

int main(int argc, char** argv) {
   int opt;
   while ((opt = getopt(argc, argv, "v:m:r:d:")) != -1) {
      switch (opt) {
      case 'v': vecSize = std::strtoull(optarg, nullptr, 10); break;
      case 'm': mode = optarg; break;
      case 'r': reps = std::strtoull(optarg, nullptr, 10); break;
      case 'd': density = std::atoi(optarg); break;
      default:
         std::fprintf(stderr,
                      "usage: %s [-v vec] [-m l1|stream|all] [-r reps] [-d density]\n",
                      argv[0]);
         return 1;
      }
   }
   std::fprintf(stderr, "run_hashbench: SIMD MurmurHash tier in the engine: %s\n",
                simd_hash::kTier);
   std::printf("form,impl,type,mode,vec,in_density_pct,ns_per_tuple\n");

   allForms<int32_t>();
   allForms<int64_t>();
   allForms<Date>();

#if defined(__AVX512F__) && !(defined(HASH_SIZE) && HASH_SIZE == 32)
   // pre-existing int32 SIMD kernels (SIMDhash=1)
   benchDense<int32_t>("hash", {{"hash4_old", hash4_int32_t_col}});
   benchDense<int32_t>("rehash", {{"hash4_old", rehash4_int32_t_col}});
#ifndef VW_POS_16 // the old *_sel kernels load inSel as 32-bit entries
   benchSel<int32_t>("hash_sel", {{"hash4_old", hash4_sel_int32_t_col}});
   benchSel<int32_t>("rehash_sel", {{"hash4_old", rehash4_sel_int32_t_col}});
#endif
#endif

   packedKeys<uint16_t>(); // Q1: two Char<1> keys
   packedKeys<uint64_t>();
   return 0;
}
