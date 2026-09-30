// Differential tests for the grouped SUM kernels in SimdAggr.hpp: every
// kernel must match the scalar reference of the aggr_sel_col path it models.
#include <cstdint>
#include "vectorwise/SimdAggr.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// GTEST_SKIP needs googletest >= 1.10; the bundled revision is older.
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
using namespace vectorwise::primitives::simd_aggr;

namespace {

/// One vector of Q1-like input: rows, a selection vector over them, a group
/// id per row, and the per-group lists Lookup_T would build.
struct Input {
   size_t rows;
   std::vector<int64_t> col;
   std::vector<pos_t> sel;             // selected row indices, ascending
   std::vector<uint8_t> gid;           // per row; kNoGroup if not selected
   std::vector<std::vector<pos_t>> gpos; // per group: index into sel
   std::vector<std::vector<pos_t>> gsel; // per group: row index
};

Input make(size_t rows, unsigned groups, int selPct, std::mt19937_64& rng) {
   Input in;
   in.rows = rows;
   in.col.resize(rows + 8);
   // magnitudes keep every sum far from int64 overflow
   for (auto& v : in.col) v = int64_t(rng() % 2000001) - 1000000;
   in.gid.assign(rows + 8, kNoGroup);
   in.gpos.resize(groups);
   in.gsel.resize(groups);
   for (size_t r = 0; r < rows; ++r) {
      if (int(rng() % 100) >= selPct) continue;
      const unsigned g = rng() % groups;
      in.gid[r] = uint8_t(g);
      in.gpos[g].push_back(pos_t(in.sel.size()));
      in.gsel[g].push_back(pos_t(r));
      in.sel.push_back(pos_t(r));
   }
   return in;
}

std::string ctx(size_t rows, unsigned groups, int selPct) {
   return "rows=" + std::to_string(rows) + " groups=" + std::to_string(groups) +
          " sel%=" + std::to_string(selPct);
}

const size_t kRows[] = {0, 1, 7, 8, 9, 15, 16, 17, 31, 33, 100, 1000, 1023, 1024};
const int kSel[] = {0, 1, 50, 98, 100};

} // namespace

TEST(SimdAggr, DirectScalarMatchesLists) {
   std::mt19937_64 rng(11);
   for (size_t rows : kRows)
      for (unsigned groups = 1; groups <= 8; ++groups)
         for (int s : kSel) {
            auto in = make(rows, groups, s, rng);
            std::vector<int64_t> acc(groups, 7);
            sum_direct_scalar(in.gid.data(), rows, in.col.data(), acc.data());
            for (unsigned g = 0; g < groups; ++g) {
               int64_t ref = sum_sel(in.gsel[g].data(), in.gsel[g].size(),
                                     in.col.data(), 7);
               ASSERT_EQ(ref, acc[g]) << ctx(rows, groups, s) << " g=" << g;
               ASSERT_EQ(ref, sum_pos(in.gpos[g].data(), in.gpos[g].size(),
                                      in.sel.data(), in.col.data(), 7))
                   << ctx(rows, groups, s) << " g=" << g << " (sum_pos)";
            }
         }
}

#ifdef VW_HAVE_SIMD_AGGR

TEST(SimdAggr, Gather) {
   std::mt19937_64 rng(12);
   for (size_t rows : kRows)
      for (unsigned groups = 1; groups <= 8; ++groups)
         for (int s : kSel) {
            auto in = make(rows, groups, s, rng);
            for (unsigned g = 0; g < groups; ++g) {
               const auto& gs = in.gsel[g];
               const auto& gp = in.gpos[g];
               int64_t ref = sum_sel(gs.data(), gs.size(), in.col.data(), -3);
               ASSERT_EQ(ref, sum_sel_gather(gs.data(), gs.size(),
                                             in.col.data(), -3))
                   << ctx(rows, groups, s) << " g=" << g << " sum_sel_gather";
               ASSERT_EQ(ref, sum_pos_gather(gp.data(), gp.size(),
                                             in.sel.data(), in.col.data(), -3))
                   << ctx(rows, groups, s) << " g=" << g << " sum_pos_gather";
            }
         }
}

TEST(SimdAggr, DirectSimd) {
   std::mt19937_64 rng(13);
   for (size_t rows : kRows)
      for (unsigned groups = 1; groups <= 8; ++groups)
         for (int s : kSel) {
            auto in = make(rows, groups, s, rng);
            // garbage past the vector end must not leak into the sums
            for (size_t r = rows; r < in.gid.size(); ++r) in.gid[r] = 0;
            std::vector<int64_t> ref(groups, 5), got(groups, 5);
            sum_direct_scalar(in.gid.data(), rows, in.col.data(), ref.data());
            ASSERT_TRUE(sum_direct_simd_n(groups, in.gid.data(), rows,
                                          in.col.data(), got.data()));
            for (unsigned g = 0; g < groups; ++g)
               ASSERT_EQ(ref[g], got[g]) << ctx(rows, groups, s) << " g=" << g;
         }
   int64_t acc[9] = {};
   uint8_t gid[8] = {};
   int64_t col[8] = {};
   EXPECT_FALSE(sum_direct_simd_n(9, gid, 8, col, acc));
}

#else
TEST(SimdAggr, Skipped) { VW_TEST_SKIP("no AVX-512F/VL/BW in this build"); }
#endif
