// Differential tests for the VW_AGGR_DIRECT kernels in AggrDirect.hpp:
// sum and count must match a scalar reference for every group count, for
// lengths that end on and off a lane block, and with rows of no group.
#include <cstdint>
#include "vectorwise/AggrDirect.hpp"
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace vectorwise::primitives::aggr_direct;

TEST(AggrDirect, SumAndCountMatchReference) {
   std::mt19937_64 rng(42);
   for (unsigned groups = 1; groups <= kMaxGroups; ++groups)
      for (size_t n : {0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 1023, 1024}) {
         std::vector<uint16_t> ids(n);
         std::vector<int64_t> col(n);
         for (size_t r = 0; r < n; ++r) {
            // about one row in eight belongs to no group
            ids[r] = rng() % 8 == 0 ? kNone : uint16_t(rng() % groups);
            col[r] = int64_t(rng() % 2000001) - 1000000;
         }
         std::vector<int64_t> refSum(groups, 0), refCnt(groups, 0);
         for (size_t r = 0; r < n; ++r)
            if (ids[r] < groups) refSum[ids[r]] += col[r], ++refCnt[ids[r]];
         // the kernels add into acc / cnt
         std::vector<int64_t> sum(groups, 5), cnt(groups, 7);
         std::vector<int64_t> sum2(groups, 3), cnt2(groups, 9);
         sumN(groups, ids.data(), n, col.data(), sum.data());
         countN(groups, ids.data(), n, cnt.data());
         sumCountN(groups, ids.data(), n, col.data(), sum2.data(), cnt2.data());
         for (unsigned g = 0; g < groups; ++g) {
            EXPECT_EQ(sum[g], refSum[g] + 5)
                << kIsa << " groups=" << groups << " n=" << n << " g=" << g;
            EXPECT_EQ(cnt[g], refCnt[g] + 7)
                << kIsa << " groups=" << groups << " n=" << n << " g=" << g;
            EXPECT_EQ(sum2[g], refSum[g] + 3)
                << kIsa << " sumCount groups=" << groups << " n=" << n;
            EXPECT_EQ(cnt2[g], refCnt[g] + 9)
                << kIsa << " sumCount groups=" << groups << " n=" << n;
         }
      }
}
