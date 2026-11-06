// VW_GROUP_CONCAT_SIMD: differential tests for the one-pass compound key
// kernels (SimdConcat.hpp). Every compiled-in kernel, with either selection
// load, must write exactly the bytes of the per-column reference (Concat_T)
// for random key shapes, dense and selection-vector columns mixed, and
// vector sizes around the block boundaries; it may not write past
// n * K + kSlack, and may not read past a column's last value (columns end
// at a protected page).
#include "vectorwise/SimdConcat.hpp"
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

using namespace vectorwise;
using namespace vectorwise::primitives::simd_concat;

namespace {

constexpr pos_t kRows = 1024;

/// a column of rows * w bytes whose last byte is the last byte before a
/// PROT_NONE page
struct GuardedColumn {
   char* map = nullptr;
   size_t mapSize = 0;
   char* data = nullptr;
   GuardedColumn(size_t bytes, std::mt19937_64& rng) {
      const size_t page = size_t(sysconf(_SC_PAGESIZE));
      const size_t body = (bytes + page - 1) / page * page;
      mapSize = body + page;
      map = static_cast<char*>(mmap(nullptr, mapSize, PROT_READ | PROT_WRITE,
                                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
      EXPECT_NE(map, MAP_FAILED);
      mprotect(map + body, page, PROT_NONE);
      data = map + body - bytes;
      for (size_t b = 0; b < bytes; ++b) data[b] = char(rng());
   }
   ~GuardedColumn() { munmap(map, mapSize); }
   GuardedColumn(const GuardedColumn&) = delete;
};

struct Shape {
   std::vector<uint32_t> width, offset;
   std::vector<bool> sel;
   uint32_t keySize = 0;
};

Shape randomShape(std::mt19937_64& rng, uint32_t maxKey) {
   static const uint32_t widths[] = {1, 2, 4, 8};
   for (;;) {
      Shape s;
      const size_t cols = 1 + rng() % 6;
      for (size_t c = 0; c < cols; ++c) {
         const uint32_t w = widths[rng() % 4];
         s.width.push_back(w);
         s.offset.push_back(s.keySize);
         s.sel.push_back(rng() % 2);
         s.keySize += w;
      }
      if (s.keySize <= maxKey) return s;
   }
}

std::vector<Isa> kernels() {
   std::vector<Isa> v;
   for (Isa isa : {Isa::Avx2, Isa::Vbmi, Isa::Neon})
      if (available(isa)) v.push_back(isa);
   return v;
}

/// runs plan p and the reference on the same input; shared: every
/// selection-vector column reads the first one's selection vector (one
/// filter feeding several key columns, TPC-H Q1)
void check(const Plan& p, const Shape& s, pos_t n, bool ascending,
           std::mt19937_64& rng, bool shared = false) {
   const pos_t* first = nullptr;
   const size_t cols = s.width.size();
   std::vector<std::unique_ptr<GuardedColumn>> columns;
   std::vector<std::vector<pos_t>> sels(cols);
   const char* data[kMaxCols];
   const pos_t* sel[kMaxCols];
   for (size_t c = 0; c < cols; ++c) {
      // dense columns hold exactly n rows; sel columns kRows, with the last
      // row selected so a read past it would fault
      const size_t rows = s.sel[c] ? kRows : std::max<pos_t>(n, 1);
      columns.push_back(
          std::make_unique<GuardedColumn>(rows * s.width[c], rng));
      data[c] = columns.back()->data;
      sel[c] = nullptr;
      if (s.sel[c]) {
         auto& v = sels[c];
         for (pos_t i = 0; i < n; ++i) v.push_back(pos_t(rng() % kRows));
         if (n) v[rng() % n] = kRows - 1;
         if (n) v[rng() % n] = 0;
         if (ascending) std::sort(v.begin(), v.end());
         v.resize(n + 32); // sel buffers are vector sized
         sel[c] = shared && first ? first : v.data();
         if (!first) first = sel[c];
      }
   }
   const size_t bytes = size_t(n) * s.keySize;
   std::vector<char> want(bytes + kSlack, 0x5a), got(bytes + 2 * kSlack, 0x5a);
   runColumns(cols, s.width.data(), s.offset.data(), s.keySize, n, data, sel,
              want.data());
   run(p, n, data, sel, got.data());
   ASSERT_EQ(0, std::memcmp(want.data(), got.data(), bytes))
       << name(p.isa) << " K=" << s.keySize << " cols=" << cols << " n=" << n;
   for (size_t b = bytes + kSlack; b < got.size(); ++b)
      ASSERT_EQ(got[b], 0x5a) << name(p.isa) << ": wrote past the slack";
}

const pos_t kSizes[] = {0,  1,  2,  3,  5,  7,   8,   15,  16,  17,  31,
                        32, 33, 63, 64, 65, 100, 255, 513, 1000, 1024};

} // namespace

TEST(GroupConcat, MatchesPerColumnCopy) {
   std::mt19937_64 rng(7);
   for (Isa isa : kernels()) {
      const uint32_t maxKey = isa == Isa::Vbmi ? 64 : 16;
      for (SelLoad sl : {SelLoad::Stage, SelLoad::Gather}) {
         for (int shape = 0; shape < 60; ++shape) {
            const Shape s = randomShape(rng, maxKey);
            bool selFlags[kMaxCols];
            for (size_t c = 0; c < s.sel.size(); ++c) selFlags[c] = s.sel[c];
            Plan p;
            if (!build(p, isa, sl, s.width.size(), s.width.data(),
                       s.offset.data(), selFlags, s.keySize))
               continue; // AVX2 gather of K = 1
            for (pos_t n : kSizes)
               for (bool ascending : {true, false})
                  check(p, s, n, ascending, rng);
         }
      }
   }
}

// the shapes HashGroup sees in TPC-H and SSB
TEST(GroupConcat, QueryShapes) {
   struct Q {
      std::vector<uint32_t> width;
      std::vector<bool> sel;
   };
   const Q shapes[] = {
       {{1, 1}, {true, true}},          // Q1 returnflag, linestatus
       {{4, 4, 4}, {true, false, false}}, // Q3 orderkey, orderdate, prio
       {{4, 4}, {false, false}},
       {{8, 8}, {false, true}},
       {{1, 2, 4}, {true, false, true}},
       {{2, 2, 2, 2}, {false, true, false, true}},
   };
   std::mt19937_64 rng(11);
   for (Isa isa : kernels())
      for (SelLoad sl : {SelLoad::Stage, SelLoad::Gather})
         for (const Q& q : shapes) {
            Shape s;
            bool selFlags[kMaxCols];
            for (size_t c = 0; c < q.width.size(); ++c) {
               s.width.push_back(q.width[c]);
               s.offset.push_back(s.keySize);
               s.sel.push_back(q.sel[c]);
               selFlags[c] = q.sel[c];
               s.keySize += q.width[c];
            }
            Plan p;
            ASSERT_TRUE(build(p, isa, sl, s.width.size(), s.width.data(),
                              s.offset.data(), selFlags, s.keySize))
                << name(isa) << " K=" << s.keySize;
            for (pos_t n : kSizes) check(p, s, n, true, rng);
         }
}

TEST(GroupConcat, UncoveredShapes) {
   const uint32_t w3[] = {3, 4}, o3[] = {0, 3};
   const bool sel[] = {false, false};
   const uint32_t w16[] = {8, 8, 8}, o16[] = {0, 8, 16};
   const bool sel3[] = {false, false, false};
   for (Isa isa : kernels()) {
      Plan p;
      EXPECT_FALSE(build(p, isa, SelLoad::Stage, 2, w3, o3, sel, 7));
      if (isa != Isa::Vbmi)
         EXPECT_FALSE(build(p, isa, SelLoad::Stage, 3, w16, o16, sel3, 24));
   }
   Plan p;
   EXPECT_FALSE(build(p, Isa::Scalar, SelLoad::Stage, 2, w3, o3, sel, 7));
}

// VW_GROUP_CONCAT_ROW: the instantiated row kernels, with the selection
// vector shared (the kernel) and not (the per-column fallback)
TEST(GroupConcat, RowKernels) {
   std::mt19937_64 rng(13);
   Shape s;
   for (uint32_t w : {1u, 1u}) {
      s.width.push_back(w);
      s.offset.push_back(s.keySize);
      s.sel.push_back(true);
      s.keySize += w;
   }
   const bool sel[] = {true, true};
   Plan p;
   ASSERT_TRUE(build(p, Isa::Row, SelLoad::Stage, 2, s.width.data(),
                     s.offset.data(), sel, s.keySize));
   for (pos_t n : kSizes)
      for (bool shared : {true, false}) check(p, s, n, true, rng, shared);

   // shapes without an instantiation
   const uint32_t w22[] = {2, 2}, o22[] = {0, 2};
   EXPECT_FALSE(build(p, Isa::Row, SelLoad::Stage, 2, w22, o22, sel, 4));
   const bool mixed[] = {true, false};
   EXPECT_FALSE(build(p, Isa::Row, SelLoad::Stage, 2, s.width.data(),
                      s.offset.data(), mixed, 2));
}
