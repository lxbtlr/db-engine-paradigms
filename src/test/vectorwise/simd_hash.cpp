// Bit-exactness tests for the AVX-512 MurmurHash64A kernels (SimdHash.hpp)
// against the scalar templates with runtime::MurMurHash. Build and probe
// sides may hash through different paths, so any difference is a bug.
#include "common/runtime/Hash.hpp"
#include "common/runtime/Types.hpp"
#include "vectorwise/Primitives.hpp"
#include "vectorwise/SimdHash.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace vectorwise;
using namespace vectorwise::primitives;
using types::Date;
using runtime::MurMurHash;
using hash_t = defs::hash_t;

namespace {
#ifdef VW_HAVE_SIMD_HASH

const std::vector<size_t> kSizes = {0, 1, 7, 8, 9, 15, 16, 17, 31, 1000, 1024};
constexpr size_t kSlack = 8;
constexpr hash_t kCanary = 0x5a5a5a5a5a5a5a5aull;

template <typename T> T mk(int64_t v) { return T(v); }
template <> Date mk<Date>(int64_t v) { return Date(int32_t(v)); }

/// Random values including negatives and the type's extremes.
template <typename T> std::vector<T> values(size_t n, std::mt19937_64& rng) {
   using L = std::conditional_t<std::is_same<T, Date>::value, int32_t, T>;
   std::uniform_int_distribution<long long> d(-100000, 100000);
   std::vector<T> v(n + 1);
   for (auto& x : v) x = mk<T>(d(rng));
   if (n > 3) {
      v[0] = mk<T>((int64_t)std::numeric_limits<L>::min());
      v[1] = mk<T>((int64_t)std::numeric_limits<L>::max());
      v[2] = mk<T>(-1);
      v[3] = mk<T>(0);
   }
   return v;
}

std::vector<pos_t> subset(size_t n, std::mt19937_64& rng) {
   std::vector<pos_t> s;
   for (size_t i = 0; i < n; ++i)
      if (rng() % 3) s.push_back(pos_t(i));
   return s;
}

void expectSame(const std::vector<hash_t>& ref, const std::vector<hash_t>& got,
                size_t n, const std::string& where) {
   for (size_t i = 0; i < n; ++i) ASSERT_EQ(ref[i], got[i]) << where << " i=" << i;
   for (size_t i = n; i < got.size(); ++i)
      ASSERT_EQ(kCanary, got[i]) << where << " overwrite at " << i;
}

template <typename T> void checkType(const char* name) {
   std::mt19937_64 rng(7);
   for (size_t n : kSizes) {
      auto in = values<T>(n, rng);
      auto sel = subset(n, rng);
      const size_t k = sel.size();
      const std::string ctx = std::string(name) + " n=" + std::to_string(n);
      {  // hash
         std::vector<hash_t> ref(n + kSlack, kCanary), got(n + kSlack, kCanary);
         primitives::hash<T, MurMurHash>(n, ref.data(), in.data());
         simd_hash::hash<T, MurMurHash>(n, got.data(), in.data());
         expectSame(ref, got, n, "hash " + ctx);
      }
      {  // hash_sel
         std::vector<hash_t> ref(k + kSlack, kCanary), got(k + kSlack, kCanary);
         primitives::hash_sel<T, MurMurHash>(k, sel.data(), ref.data(), in.data());
         simd_hash::hash_sel<T, MurMurHash>(k, sel.data(), got.data(), in.data());
         expectSame(ref, got, k, "hash_sel " + ctx);
      }
      {  // rehash: random seeds in result
         std::vector<hash_t> ref(n + kSlack, kCanary);
         for (size_t i = 0; i < n; ++i) ref[i] = rng();
         auto got = ref;
         primitives::rehash<T, MurMurHash>(n, ref.data(), in.data());
         simd_hash::rehash<T, MurMurHash>(n, got.data(), in.data());
         expectSame(ref, got, n, "rehash " + ctx);
      }
      {  // rehash_sel
         std::vector<hash_t> ref(k + kSlack, kCanary);
         for (size_t i = 0; i < k; ++i) ref[i] = rng();
         auto got = ref;
         primitives::rehash_sel<T, MurMurHash>(k, sel.data(), ref.data(), in.data());
         simd_hash::rehash_sel<T, MurMurHash>(k, sel.data(), got.data(), in.data());
         expectSame(ref, got, k, "rehash_sel " + ctx);
      }
   }
}

/// HashGroup packed keys: hashFn.hashKey(key) with key zero-extended.
template <typename T> void checkPackedKeys(const char* name) {
   std::mt19937_64 rng(9);
   MurMurHash h;
   for (size_t n : kSizes) {
      std::vector<T> keys(n + 1);
      for (auto& x : keys) x = T(rng());
      if (n > 1) keys[0] = std::numeric_limits<T>::max();
      std::vector<hash_t> ref(n + kSlack, kCanary), got(n + kSlack, kCanary);
      for (size_t i = 0; i < n; ++i) ref[i] = h.hashKey(keys[i]);
      simd_hash::hash_keys<T>(n, reinterpret_cast<const char*>(keys.data()),
                              got.data());
      expectSame(ref, got, n, std::string("hash_keys ") + name + " n=" + std::to_string(n));
   }
}

#endif // VW_HAVE_SIMD_HASH
} // namespace

#ifdef VW_HAVE_SIMD_HASH
TEST(SimdHash, Int8) { checkType<int8_t>("int8"); }
TEST(SimdHash, Int16) { checkType<int16_t>("int16"); }
TEST(SimdHash, Int32) { checkType<int32_t>("int32"); }
TEST(SimdHash, Int64) { checkType<int64_t>("int64"); }
TEST(SimdHash, HashT) { checkType<uint64_t>("hash_t"); }
TEST(SimdHash, Date) { checkType<Date>("Date"); }
TEST(SimdHash, PackedKeys) {
   checkPackedKeys<uint8_t>("u8");
   checkPackedKeys<uint16_t>("u16");
   checkPackedKeys<uint32_t>("u32");
   checkPackedKeys<uint64_t>("u64");
}
#else
TEST(SimdHash, Skipped) { GTEST_SKIP() << "no AVX-512F/DQ or 32-bit hashes"; }
#endif

// The public primitive names must follow VW_SIMD_HASH.
TEST(SimdHashRedirect, Names) {
#if defined(VW_SIMD_HASH) && defined(VW_HAVE_SIMD_HASH) && !defined(VW_USE_CRC32)
   EXPECT_EQ((void*)hash_int32_t_col, (void*)(&simd_hash::hash<int32_t, MurMurHash>));
   EXPECT_EQ((void*)hash_sel_int32_t_col,
             (void*)(&simd_hash::hash_sel<int32_t, MurMurHash>));
   EXPECT_EQ((void*)rehash_Date_col, (void*)(&simd_hash::rehash<Date, MurMurHash>));
   EXPECT_EQ((void*)rehash_sel_int64_t_col,
             (void*)(&simd_hash::rehash_sel<int64_t, MurMurHash>));
   // Char<N> stays scalar
   EXPECT_EQ((void*)hash_Char_25_col,
             (void*)(&primitives::hash<Char_25, MurMurHash>));
#elif !defined(VW_USE_CRC32) && !(defined(HASH_SIZE) && HASH_SIZE == 32)
   EXPECT_EQ((void*)hash_int32_t_col,
             (void*)(&primitives::hash<int32_t, MurMurHash>));
#else
   GTEST_SKIP() << "non-MurMur default hash";
#endif
}
