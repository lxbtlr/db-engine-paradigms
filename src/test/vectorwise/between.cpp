// Range selection (BetweenPrimitives.hpp): differential tests. Every kernel,
// scalar and AVX-512, must match the chain of two selection primitives it
// replaces, also when its output overwrites its input selection; under
// VW_SEL_BETWEEN, ExpressionBuilder must fuse exactly the pairs it may.
#include "vectorwise/BetweenPrimitives.hpp"
#include "common/runtime/Types.hpp"
#include "vectorwise/Operations.hpp"
#include "vectorwise/Primitives.hpp"
#include "vectorwise/QueryBuilder.hpp"
#include "vectorwise/SimdSelection.hpp"
#include <functional>
#include <gtest/gtest.h>
#include <random>
#include <stdexcept>
#include <vector>

using namespace vectorwise;
using namespace vectorwise::primitives;
using namespace vectorwise::primitives::between;
using types::Date;

namespace {

constexpr pos_t kRows = 1000;

template <typename T> T mk(int64_t v) { return T(v); }
template <> Date mk<Date>(int64_t v) { return Date(int32_t(v)); }

template <typename T> struct Data {
   std::vector<T> x;
   std::vector<pos_t> sel;
   Data() {
      std::mt19937_64 rng(7);
      for (pos_t i = 0; i < kRows; ++i) {
         x.push_back(mk<T>(int64_t(rng() % 40)));
         if (rng() % 10 < 6) sel.push_back(i); // ~60% pass, ascending
      }
   }
};

/// the two primitives the kernel replaces: upper bound, then lower bound
template <typename T, template <typename> class Lo, template <typename> class Hi>
std::vector<pos_t> chain(pos_t n, pos_t* inSel, T* col, T lo, T hi) {
   std::vector<pos_t> mid(kRows), out(kRows);
   pos_t k = inSel ? selsel_col_val_bf<T, Hi>(n, inSel, mid.data(), col, &hi)
                   : sel_col_val_bf<T, Hi>(n, mid.data(), col, &hi);
   k = selsel_col_val_bf<T, Lo>(k, mid.data(), out.data(), col, &lo);
   out.resize(k);
   return out;
}

template <typename T, template <typename> class Lo, template <typename> class Hi,
          typename F>
void expectChain(const char* what, bool selIn, F&& fn) {
   Data<T> d;
   const int64_t bounds[][2] = {{10, 25}, {0, 39}, {17, 17}, {30, 5}, {-5, 50}};
   for (auto& b : bounds) {
      T lo = mk<T>(b[0]), hi = mk<T>(b[1]);
      for (pos_t n : {0, 1, 7, 8, 15, 16, 17, 33, 300, int(kRows)}) {
         pos_t* in = nullptr;
         if (selIn) {
            n = std::min<pos_t>(n, pos_t(d.sel.size()));
            in = d.sel.data();
         }
         auto expected = chain<T, Lo, Hi>(n, in, d.x.data(), lo, hi);
         std::vector<pos_t> out(kRows + 16);
         pos_t got = fn(n, in, out.data(), d.x.data(), &lo, &hi);
         out.resize(got);
         ASSERT_EQ(out, expected) << what << " n=" << n << " [" << b[0] << ","
                                  << b[1] << "]";
         if (selIn) { // in place: the output overwrites the input selection
            std::vector<pos_t> buf(d.sel);
            buf.resize(kRows + 16);
            got = fn(n, buf.data(), buf.data(), d.x.data(), &lo, &hi);
            buf.resize(got);
            ASSERT_EQ(buf, expected) << what << " in place n=" << n;
         }
      }
   }
}

template <typename T, template <typename> class Lo, template <typename> class Hi>
void checkAll(between::Type t, bool loStrict, bool hiStrict) {
   for (bool selIn : {false, true}) {
      Fn k = kernel(t, loStrict, hiStrict, selIn);
      expectChain<T, Lo, Hi>("kernel()", selIn,
                       [k](pos_t n, pos_t* in, pos_t* out, T* c, T* l, T* h) {
                          return k(n, in, out, c, l, h);
                       });
      expectChain<T, Lo, Hi>("between_bf", selIn,
                       [selIn](pos_t n, pos_t* in, pos_t* out, T* c, T* l, T* h) {
                          return selIn ? between_bf<T, Lo, Hi, true>(n, in, out, c, l, h)
                                       : between_bf<T, Lo, Hi, false>(n, in, out, c, l, h);
                       });
   }
#ifdef VW_HAVE_SIMD_SEL
   using simd::Emit;
   expectChain<T, Lo, Hi>("sel_between_v reg", false,
                    [](pos_t n, pos_t*, pos_t* out, T* c, T* l, T* h) {
                       return simd::sel_between_v<T, Lo, Hi, Emit::Reg>(n, out, c, l, h);
                    });
   expectChain<T, Lo, Hi>("sel_between_v mem", false,
                    [](pos_t n, pos_t*, pos_t* out, T* c, T* l, T* h) {
                       return simd::sel_between_v<T, Lo, Hi, Emit::Mem>(n, out, c, l, h);
                    });
#ifdef VW_HAVE_SIMD_SEL_256
   expectChain<T, Lo, Hi>("sel_between_256 reg", false,
                    [](pos_t n, pos_t*, pos_t* out, T* c, T* l, T* h) {
                       return simd::sel_between_256<T, Lo, Hi, Emit::Reg>(n, out, c, l, h);
                    });
   expectChain<T, Lo, Hi>("sel_between_256 mem", false,
                    [](pos_t n, pos_t*, pos_t* out, T* c, T* l, T* h) {
                       return simd::sel_between_256<T, Lo, Hi, Emit::Mem>(n, out, c, l, h);
                    });
#endif
   expectChain<T, Lo, Hi>("selsel_between_scalarload", true,
                    [](pos_t n, pos_t* in, pos_t* out, T* c, T* l, T* h) {
                       return simd::selsel_between_scalarload<T, Lo, Hi>(n, in, out, c, l, h);
                    });
   expectChain<T, Lo, Hi>("selsel_between_hwgather", true,
                    [](pos_t n, pos_t* in, pos_t* out, T* c, T* l, T* h) {
                       return simd::selsel_between_hwgather<T, Lo, Hi>(n, in, out, c, l, h);
                    });
#endif
}

template <typename T> void checkType(between::Type t) {
   checkAll<T, std::greater_equal, std::less_equal>(t, false, false);
   checkAll<T, std::greater_equal, std::less>(t, false, true);
   checkAll<T, std::greater, std::less_equal>(t, true, false);
   checkAll<T, std::greater, std::less>(t, true, true);
}

} // namespace

TEST(Between, KernelsMatchChain) {
   checkType<int32_t>(Int32);
   checkType<int64_t>(Int64);
   checkType<Date>(Date32);
}

TEST(Between, DescriptorsMatchPrimitives) {
   Desc d;
   ASSERT_TRUE(describe((const void*)sel_less_int32_t_col_int32_t_val, d));
   EXPECT_EQ(d.type, Int32);
   EXPECT_TRUE(d.upper && d.strict && !d.selInput);
   ASSERT_TRUE(describe((const void*)selsel_greater_equal_Date_col_Date_val_bf, d));
   EXPECT_EQ(d.type, Date32);
   EXPECT_TRUE(!d.upper && !d.strict && d.selInput);
   ASSERT_TRUE(describe((const void*)selsel_less_equal_int64_t_col_int64_t_val, d));
   EXPECT_EQ(d.type, Int64);
   EXPECT_TRUE(d.upper && !d.strict && d.selInput);
   ASSERT_TRUE(describe((const void*)sel_greater_int64_t_col_int64_t_val_bf, d));
   EXPECT_TRUE(!d.upper && d.strict && !d.selInput);
   EXPECT_FALSE(describe((const void*)sel_equal_to_int32_t_col_int32_t_val, d));
   EXPECT_FALSE(describe((const void*)sel_less_int32_t_col_int32_t_col, d));
   EXPECT_FALSE(describe((const void*)proj_plus_int64_t_col_int64_t_val, d));
}

#ifdef VW_SEL_BETWEEN
namespace {
// the builder types are protected: re-export them
struct Exposed : QueryBuilder {
   using QueryBuilder::DS;
   using QueryBuilder::ExpressionBuilder;
};
using DS = Exposed::DS;
DS buffer(void* p, size_t size = sizeof(pos_t)) {
   DS r;
   r.buf = DS::BufferSpec::Buffer;
   r.dataSize = size;
   r.data = p;
   return r;
}
DS value(void* p) {
   DS r;
   r.buf = DS::BufferSpec::Value;
   r.data = p;
   return r;
}
Exposed::ExpressionBuilder builder() {
   Exposed::ExpressionBuilder b;
   b.expression = std::make_unique<Expression>();
   return b;
}
size_t betweenOps(const Expression& e) {
   size_t k = 0;
   for (auto& op : e.ops) k += dynamic_cast<BetweenOp*>(op.get()) != nullptr;
   return k;
}
} // namespace

TEST(Between, BuilderFusesQ6Ranges) {
   // Q6's Select: shipdate [c1, c2) upper bound first, quantity < c5,
   // discount [c3, c4] in place (sel_a -> sel_b -> sel_a)
   std::mt19937_64 rng(3);
   std::vector<int32_t> ship(kRows);
   std::vector<int64_t> qty(kRows), disc(kRows);
   for (pos_t i = 0; i < kRows; ++i) {
      ship[i] = int32_t(rng() % 100);
      qty[i] = int64_t(rng() % 50);
      disc[i] = int64_t(rng() % 11);
   }
   int32_t c1 = 20, c2 = 60;
   int64_t c3 = 5, c4 = 7, c5 = 24;
   std::vector<pos_t> selA(kRows), selB(kRows);
   auto b = builder();
   b.addOp(sel_less_int32_t_col_int32_t_val_bf, buffer(selA.data()),
           buffer(ship.data(), 4), value(&c2))
       .addOp(selsel_greater_equal_int32_t_col_int32_t_val_bf,
              buffer(selA.data()), buffer(selB.data()), buffer(ship.data(), 4),
              value(&c1))
       .addOp(selsel_less_int64_t_col_int64_t_val_bf, buffer(selB.data()),
              buffer(selA.data()), buffer(qty.data(), 8), value(&c5))
       .addOp(selsel_greater_equal_int64_t_col_int64_t_val_bf,
              buffer(selA.data()), buffer(selB.data()), buffer(disc.data(), 8),
              value(&c3))
       .addOp(selsel_less_equal_int64_t_col_int64_t_val_bf, buffer(selB.data()),
              buffer(selA.data()), buffer(disc.data(), 8), value(&c4));
   std::unique_ptr<Expression> e = b;
   ASSERT_EQ(e->ops.size(), 3u);
   EXPECT_NE(dynamic_cast<BetweenOp*>(e->ops[0].get()), nullptr);
   EXPECT_EQ(dynamic_cast<BetweenOp*>(e->ops[1].get()), nullptr);
   EXPECT_NE(dynamic_cast<BetweenOp*>(e->ops[2].get()), nullptr);
   std::vector<pos_t> expected;
   for (pos_t i = 0; i < kRows; ++i)
      if (ship[i] >= c1 && ship[i] < c2 && qty[i] < c5 && disc[i] >= c3 &&
          disc[i] <= c4)
         expected.push_back(i);
   const pos_t n = e->evaluate(kRows);
   ASSERT_EQ(n, pos_t(expected.size()));
   selA.resize(n);
   EXPECT_EQ(selA, expected);
}

TEST(Between, BuilderLeavesOtherPairs) {
   std::vector<int64_t> x(kRows), y(kRows);
   for (pos_t i = 0; i < kRows; ++i) {
      x[i] = i % 13;
      y[i] = i % 7;
   }
   int64_t lo = 3, hi = 9;
   std::vector<pos_t> s1(kRows), s2(kRows), s3(kRows);
   // different columns
   auto b1 = builder();
   b1.addOp(sel_greater_equal_int64_t_col_int64_t_val_bf, buffer(s1.data()),
            buffer(x.data(), 8), value(&lo))
       .addOp(selsel_less_int64_t_col_int64_t_val_bf, buffer(s1.data()),
              buffer(s2.data()), buffer(y.data(), 8), value(&hi));
   std::unique_ptr<Expression> e1 = b1;
   EXPECT_EQ(e1->ops.size(), 2u);
   EXPECT_EQ(betweenOps(*e1), 0u);
   // two lower bounds
   auto b2 = builder();
   b2.addOp(sel_greater_equal_int64_t_col_int64_t_val_bf, buffer(s1.data()),
            buffer(x.data(), 8), value(&lo))
       .addOp(selsel_greater_int64_t_col_int64_t_val_bf, buffer(s1.data()),
              buffer(s2.data()), buffer(x.data(), 8), value(&hi));
   std::unique_ptr<Expression> e2 = b2;
   EXPECT_EQ(betweenOps(*e2), 0u);
   // the second does not read the first's output
   auto b3 = builder();
   b3.addOp(sel_greater_equal_int64_t_col_int64_t_val_bf, buffer(s1.data()),
            buffer(x.data(), 8), value(&lo))
       .addOp(selsel_less_int64_t_col_int64_t_val_bf, buffer(s3.data()),
              buffer(s2.data()), buffer(x.data(), 8), value(&hi));
   std::unique_ptr<Expression> e3 = b3;
   EXPECT_EQ(betweenOps(*e3), 0u);
   // a later op reads the selection the fused pair does not write
   auto b4 = builder();
   b4.addOp(sel_greater_equal_int64_t_col_int64_t_val_bf, buffer(s1.data()),
            buffer(x.data(), 8), value(&lo))
       .addOp(selsel_less_int64_t_col_int64_t_val_bf, buffer(s1.data()),
              buffer(s2.data()), buffer(x.data(), 8), value(&hi));
   EXPECT_THROW(b4.addOp(selsel_less_int64_t_col_int64_t_val_bf,
                         buffer(s1.data()), buffer(s3.data()),
                         buffer(y.data(), 8), value(&hi)),
                std::runtime_error);
}
#endif
