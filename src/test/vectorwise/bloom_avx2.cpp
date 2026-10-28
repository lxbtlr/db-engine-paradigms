// VW_JOIN_BLOOM AVX2 filter test (include/vectorwise/BloomKernels.hpp): must
// return exactly what the scalar loop does (same survivors, hashes and order)
// for any pass rate and any length, tails included.
#include "vectorwise/BloomKernels.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <vector>

#ifdef GTEST_SKIP
#define VW_TEST_SKIP(msg) GTEST_SKIP() << msg
#else
#define VW_TEST_SKIP(msg)                                                      \
   do {                                                                        \
      std::cout << "[  SKIPPED ] " << msg << std::endl;                        \
      return;                                                                  \
   } while (0)
#endif

using namespace vectorwise;

TEST(BloomAvx2, MatchesScalarLoop) {
#if !defined(__AVX2__)
   VW_TEST_SKIP("no AVX2 on this target");
#else
   std::mt19937_64 rng(7);
   for (uint64_t words : {64u, 4096u}) {             // tiny and L1-sized filters
      for (int insertPct : {1, 30, 90}) {            // pass rate of the probes
         const uint64_t mask = words - 1;
         std::vector<uint64_t> filter(words, 0);
         std::vector<uint64_t> inserted;
         for (size_t k = 0; k < words; ++k) {
            const uint64_t h = rng();
            filter[(h >> 40) & mask] |= bloom::bits(h);
            inserted.push_back(h);
         }
         for (size_t n : {0u, 1u, 3u, 4u, 5u, 1023u, 1024u}) {
            std::vector<uint64_t> hs(n);
            for (auto& h : hs)
               h = int(rng() % 100) < insertPct ? inserted[rng() % inserted.size()] : rng();
            // reference: the scalar loop of Hashjoin::bloomFilter
            std::vector<uint32_t> refSel;
            std::vector<uint64_t> refOut;
            for (size_t i = 0; i < n; ++i)
               if (bloom::test(filter.data(), mask, hs[i]))
                  refSel.push_back(uint32_t(i)), refOut.push_back(hs[i]);
            std::vector<uint32_t> sel(n + 1);
            std::vector<uint64_t> out(n + 1);
            const size_t m = bloom::filter4(filter.data(), mask, hs.data(), n, sel.data(), out.data());
            ASSERT_EQ(m, refSel.size()) << "words " << words << " pass " << insertPct << " n " << n;
            for (size_t k = 0; k < m; ++k) {
               ASSERT_EQ(sel[k], refSel[k]);
               ASSERT_EQ(out[k], refOut[k]);
            }
         }
      }
   }
#endif
}
