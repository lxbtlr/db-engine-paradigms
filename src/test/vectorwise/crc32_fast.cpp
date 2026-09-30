// CRC32Hash must hash exactly as the original two-crc32 formula on every
// path, with or without VW_CRC32_FAST (which derives the second crc32 from
// the first for fixed seeds and interleaves wide packed keys). Build and
// probe sides may hash through different paths, so any difference is a bug.
#include "common/runtime/Hash.hpp"
#include "common/runtime/Types.hpp"
#include "vectorwise/Primitives.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using runtime::CRC32Hash;
using hash_t = defs::hash_t;

namespace {

/// the original CRC32Hash::hashKey(T, seed)
uint64_t refKey(uint64_t k, uint64_t seed) {
   uint64_t result1 = _mm_crc32_u64(seed, k);
   uint64_t result2 = _mm_crc32_u64(0x04c11db7, k);
   return ((result2 << 32) | result1) * 0x2545F4914F6CDD1Dull;
}

/// the original CRC32Hash::hashKey(const void*, len, seed)
uint64_t refBytes(const char* key, int len, uint64_t seed) {
   auto data = reinterpret_cast<const uint8_t*>(key);
   uint64_t s = seed;
   for (; len >= 8; data += 8, len -= 8) {
      uint64_t k;
      std::memcpy(&k, data, 8);
      s = refKey(k, s);
   }
   if (len >= 4) {
      uint32_t k;
      std::memcpy(&k, data, 4);
      s = refKey(k, s);
      data += 4;
      len -= 4;
   }
   if (len >= 3) s ^= ((uint64_t)data[2]) << 16;
   if (len >= 2) s ^= ((uint64_t)data[1]) << 8;
   if (len >= 1) s ^= data[0];
   return s;
}

const uint64_t kSeeds[] = {0, vectorwise::primitives::seed, 0x04c11db7, 0xffffffffffffffffull,
                           0x123456789abcdef0ull};

} // namespace

TEST(CRC32Fast, Keys) {
   std::mt19937_64 rng(21);
   CRC32Hash h;
   for (int t = 0; t < 20000; ++t) {
      const uint64_t k = (t < 3) ? (t == 0 ? 0 : (t == 1 ? ~0ull : 0x04c11db7)) : rng();
      for (uint64_t s : kSeeds) {
         // the Hash<> overloads pass each type through; signed keys sign-extend
         ASSERT_EQ(refKey(k, s), h((uint64_t)k, s)) << k << " " << s;
         ASSERT_EQ(refKey((uint64_t)(int64_t)(int32_t)k, s), h((int32_t)k, s)) << k << " " << s;
         ASSERT_EQ(refKey((uint32_t)k, s), h((uint32_t)k, s)) << k << " " << s;
         ASSERT_EQ(refKey((uint64_t)(int64_t)(int8_t)k, s), h((int8_t)k, s)) << k << " " << s;
      }
      // compile-time seeds, as HashGroup (0) and the hash primitives use them
      ASSERT_EQ(refKey(k, 0), h.hashKey(k)) << k;
      ASSERT_EQ(refKey(k, 0), h.hashKey(k, 0)) << k;
      ASSERT_EQ(refKey(k, vectorwise::primitives::seed), h.hashKey(k, vectorwise::primitives::seed)) << k;
   }
}

TEST(CRC32Fast, Primitives) {
   std::mt19937_64 rng(22);
   const size_t n = 1000;
   std::vector<int32_t> in(n);
   std::vector<int64_t> in64(n);
   for (size_t i = 0; i < n; ++i) in[i] = int32_t(rng()), in64[i] = int64_t(rng());
   std::vector<vectorwise::pos_t> sel;
   for (size_t i = 0; i < n; i += 3) sel.push_back(vectorwise::pos_t(i));
   std::vector<hash_t> out(n), ref(n);
   const uint64_t seed = vectorwise::primitives::seed;

   vectorwise::primitives::hash<int32_t, CRC32Hash>(n, out.data(), in.data());
   for (size_t i = 0; i < n; ++i) ref[i] = refKey((uint64_t)(int64_t)in[i], seed);
   ASSERT_EQ(ref, out) << "hash<int32_t>";

   vectorwise::primitives::hash<int64_t, CRC32Hash>(n, out.data(), in64.data());
   for (size_t i = 0; i < n; ++i) ref[i] = refKey((uint64_t)in64[i], seed);
   ASSERT_EQ(ref, out) << "hash<int64_t>";

   std::vector<hash_t> outSel(sel.size()), refSel(sel.size());
   vectorwise::primitives::hash_sel<int32_t, CRC32Hash>(sel.size(), sel.data(), outSel.data(), in.data());
   for (size_t i = 0; i < sel.size(); ++i) refSel[i] = refKey((uint64_t)(int64_t)in[sel[i]], seed);
   ASSERT_EQ(refSel, outSel) << "hash_sel<int32_t>";

   // rehash: per-row seeds from a previous hash
   std::vector<hash_t> chain = ref;
   vectorwise::primitives::rehash<int32_t, CRC32Hash>(n, chain.data(), in.data());
   for (size_t i = 0; i < n; ++i) ref[i] = refKey((uint64_t)(int64_t)in[i], ref[i]);
   ASSERT_EQ(ref, chain) << "rehash<int32_t>";
}

TEST(CRC32Fast, PackedKeys) {
   std::mt19937_64 rng(23);
   CRC32Hash h;
   for (uint32_t keySize = 1; keySize <= 48; ++keySize)
      for (size_t n : {size_t(0), size_t(1), size_t(3), size_t(4), size_t(5), size_t(13), size_t(1024)}) {
         std::vector<char> keys(n * keySize + 8);
         for (auto& c : keys) c = char(rng());
         for (uint64_t s : {uint64_t(0), uint64_t(0x123456789abcdef0ull)}) {
            std::vector<hash_t> ref(n);
            for (size_t i = 0; i < n; ++i) ref[i] = refBytes(keys.data() + i * keySize, keySize, s);
            for (size_t i = 0; i < n; ++i)
               ASSERT_EQ(ref[i], h.hashKey(keys.data() + i * keySize, (int)keySize, s))
                   << "hashKey(void*) keySize=" << keySize << " i=" << i;
#ifdef VW_CRC32_FAST
            std::vector<hash_t> got(n + 1, 0xdeadbeef);
            h.hashKeys(keys.data(), keySize, n, s, got.data());
            for (size_t i = 0; i < n; ++i)
               ASSERT_EQ(ref[i], got[i]) << "hashKeys keySize=" << keySize << " n=" << n << " i=" << i;
            ASSERT_EQ(hash_t(0xdeadbeef), got[n]) << "hashKeys wrote past n";
#endif
         }
      }
}

