// Differential tests for the AVX2 selection kernels (SimdSelAvx2.hpp)
// against the scalar templates in Primitives.hpp and BetweenPrimitives.hpp.
// The kernels are compiled whenever the target has AVX2, independent of the
// VW_SIMD_SEL options, so these tests run on every x86 machine. The kernels
// may write result entries between the last match and n (as the _bf
// templates do), never at or past n.
#include "common/runtime/Types.hpp"
#include "vectorwise/BetweenPrimitives.hpp"
#include "vectorwise/Primitives.hpp"
#include "vectorwise/SimdSelAvx2.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <climits>
#include <iostream>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace vectorwise;
using namespace vectorwise::primitives;
using types::Date;

namespace {
#ifdef VW_HAVE_SIMD_SEL_AVX2
namespace avx2 = simd::avx2;

const std::vector<size_t> kSizes = {0, 1, 7, 8, 9, 15, 16, 17, 1000, 1023, 1024};
constexpr size_t kSlack = 16; // result entries past n that must stay intact
const pos_t kCanary = pos_t(0xBEEF);

template <typename T> T mk(int64_t v) { return T(v); }
template <> Date mk<Date>(int64_t v) { return Date(int32_t(v)); }
template <typename T> int64_t num(const T& v) { return int64_t(v); }
template <> int64_t num<Date>(const Date& v) { return v.value; }
template <typename T> int64_t tmin() { return sizeof(T) == 8 ? INT64_MIN : INT32_MIN; }
template <typename T> int64_t tmax() { return sizeof(T) == 8 ? INT64_MAX : INT32_MAX; }

/// n values in [-1000, 1000] plus the type extremes; one leading element
/// so the data is not 32-byte aligned
template <typename T> std::vector<T> column(size_t n, std::mt19937_64& rng) {
   std::uniform_int_distribution<int64_t> d(-1000, 1000);
   std::vector<T> v(n + 1);
   for (auto& x : v) x = mk<T>(d(rng));
   if (n > 2) v[1] = mk<T>(tmin<T>()), v[n] = mk<T>(tmax<T>());
   return v;
}

/// constants for 0..100% selectivity, the type extremes and 0
template <typename T> std::vector<T> constants(const T* col, size_t n) {
   std::vector<T> out = {mk<T>(tmin<T>()), mk<T>(tmax<T>()), mk<T>(0)};
   std::vector<int64_t> v;
   for (size_t i = 0; i < n; ++i) v.push_back(num(col[i]));
   std::sort(v.begin(), v.end());
   for (int p : {0, 1, 50, 99, 100})
      if (n) out.push_back(mk<T>(v[std::min(n - 1, n * p / 100)]));
   return out;
}

std::vector<pos_t> subset(size_t n, int density, std::mt19937_64& rng) {
   std::uniform_int_distribution<int> d(0, 99);
   std::vector<pos_t> s;
   for (size_t i = 0; i < n; ++i)
      if (d(rng) < density) s.push_back(pos_t(i));
   return s;
}

/// same count and entries as the reference, nothing written at or past n
void expectSame(pos_t fRef, const std::vector<pos_t>& ref, pos_t f,
                const std::vector<pos_t>& out, size_t n, const std::string& w) {
   ASSERT_EQ(fRef, f) << w;
   for (pos_t i = 0; i < f; ++i) ASSERT_EQ(ref[i], out[i]) << w << " i=" << i;
   for (size_t i = n; i < out.size(); ++i)
      ASSERT_EQ(kCanary, out[i]) << w << " write past n at " << i;
}

template <typename T, template <typename> class Op> void checkSel() {
   std::mt19937_64 rng(7);
   for (size_t n : kSizes) {
      auto colv = column<T>(n, rng);
      T* col = colv.data() + 1;
      for (T c : constants(col, n)) {
         std::vector<pos_t> ref(n + kSlack, kCanary), out(n + kSlack, kCanary);
         const pos_t fr = sel_col_val_bf<T, Op>(pos_t(n), ref.data(), col, &c);
         const pos_t f = avx2::sel_col_val<T, Op>(pos_t(n), out.data(), col, &c);
         expectSame(fr, ref, f, out, n,
                    "sel n=" + std::to_string(n) + " c=" + std::to_string(num(c)));
         // selected input: both gather implementations
         for (int dens : {10, 50, 100}) {
            const auto in = subset(n, dens, rng);
            const pos_t k = pos_t(in.size());
            auto inv = in;
            std::vector<pos_t> r2(k + kSlack, kCanary), s(k + kSlack, kCanary),
                h(k + kSlack, kCanary);
            const pos_t f2 = selsel_col_val_bf<T, Op>(k, inv.data(), r2.data(), col, &c);
            const pos_t fs = avx2::selsel_col_val_g<T, Op, false>(k, inv.data(), s.data(), col, &c);
            const pos_t fh = avx2::selsel_col_val_g<T, Op, true>(k, inv.data(), h.data(), col, &c);
            const std::string w = "selsel n=" + std::to_string(n) + " d=" + std::to_string(dens);
            expectSame(f2, r2, fs, s, k, w + " scalarload");
            expectSame(f2, r2, fh, h, k, w + " hwgather");
         }
      }
   }
}

template <typename T> void checkSelAllOps() {
   checkSel<T, std::equal_to>();
   checkSel<T, std::less>();
   checkSel<T, std::less_equal>();
   checkSel<T, std::greater>();
   checkSel<T, std::greater_equal>();
}

template <typename T, template <typename> class Lo, template <typename> class Hi>
void checkBetween() {
   std::mt19937_64 rng(11);
   for (size_t n : kSizes) {
      auto colv = column<T>(n, rng);
      T* col = colv.data() + 1;
      const auto cs = constants(col, n);
      for (T l : cs)
         for (T h : cs) {
            const std::string w = "between n=" + std::to_string(n) + " l=" +
                                  std::to_string(num(l)) + " h=" + std::to_string(num(h));
            std::vector<pos_t> ref(n + kSlack, kCanary), out(n + kSlack, kCanary);
            const pos_t fr = between::between_bf<T, Lo, Hi, false>(
                pos_t(n), nullptr, ref.data(), col, &l, &h);
            const pos_t f = avx2::sel_between<T, Lo, Hi>(pos_t(n), out.data(), col, &l, &h);
            expectSame(fr, ref, f, out, n, w);
            // selected input, separate and in place (out == inSel)
            const auto in = subset(n, 60, rng);
            const pos_t k = pos_t(in.size());
            std::vector<pos_t> r2(k + kSlack, kCanary);
            auto inv = in;
            const pos_t f2 = between::between_bf<T, Lo, Hi, true>(k, inv.data(), r2.data(), col, &l, &h);
            for (bool hw : {false, true}) {
               std::vector<pos_t> o(k + kSlack, kCanary);
               auto iv = in;
               const pos_t fo = hw ? avx2::between_g<T, Lo, Hi, true, true>(k, iv.data(), o.data(), col, &l, &h)
                                   : avx2::between_g<T, Lo, Hi, true, false>(k, iv.data(), o.data(), col, &l, &h);
               expectSame(f2, r2, fo, o, k, w + (hw ? " selsel hw" : " selsel sl"));
               std::vector<pos_t> io(in);
               io.resize(k + kSlack, kCanary);
               const pos_t fi = hw ? avx2::between_g<T, Lo, Hi, true, true>(k, io.data(), io.data(), col, &l, &h)
                                   : avx2::between_g<T, Lo, Hi, true, false>(k, io.data(), io.data(), col, &l, &h);
               expectSame(f2, r2, fi, io, k, w + " in place");
            }
         }
   }
}

template <typename T> void checkBetweenAll() {
   checkBetween<T, std::greater_equal, std::less_equal>();
   checkBetween<T, std::greater_equal, std::less>();
   checkBetween<T, std::greater, std::less_equal>();
   checkBetween<T, std::greater, std::less>();
}

} // namespace

TEST(SimdSelAvx2, Int32) { checkSelAllOps<int32_t>(); }
TEST(SimdSelAvx2, Int64) { checkSelAllOps<int64_t>(); }
TEST(SimdSelAvx2, Date) { checkSelAllOps<Date>(); }
TEST(SimdSelAvx2, BetweenInt32) { checkBetweenAll<int32_t>(); }
TEST(SimdSelAvx2, BetweenInt64) { checkBetweenAll<int64_t>(); }
TEST(SimdSelAvx2, BetweenDate) { checkBetweenAll<Date>(); }

#else
} // namespace
TEST(SimdSelAvx2, Skipped) { std::cout << "[  SKIPPED ] no AVX2 in this build\n"; }
#endif
