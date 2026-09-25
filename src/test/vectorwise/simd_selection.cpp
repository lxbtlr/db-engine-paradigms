// Differential tests for the AVX-512 selection kernels (SimdSelection.hpp)
// against the scalar templates in Primitives.hpp. The kernels are compiled
// whenever the ISA is available, independent of the VW_SIMD_SEL options, so
// these tests always exercise them on AVX-512 machines. The SimdSelRedirect
// tests additionally check the CMake-option wiring in Selection.cpp.
#include "common/runtime/Types.hpp"
#include "vectorwise/Primitives.hpp"
#include "vectorwise/SimdSelection.hpp"
#include <gtest/gtest.h>
#include <iostream>

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
#include <algorithm>
#include <climits>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace vectorwise;
using namespace vectorwise::primitives;
using types::Char;
using types::Date;

namespace {
#ifdef VW_HAVE_SIMD_SEL

const std::vector<size_t> kSizes = {0,  1,  7,    15,   16,   17,
                                    31, 33, 1000, 1023, 1024, 4096};
const std::vector<int> kSelPct = {0, 1, 50, 99, 100};
constexpr size_t kSlack = 32; // result entries past n that must stay intact
const pos_t kCanary = pos_t(0xBEEF);

template <typename T> T mk(int64_t v) { return T(v); }
template <> Date mk<Date>(int64_t v) { return Date(int32_t(v)); }
template <typename T> int64_t num(const T& v) { return int64_t(v); }
template <> int64_t num<Date>(const Date& v) { return v.value; }

template <typename T> int64_t tmin() {
   return sizeof(T) == 8 ? INT64_MIN : INT32_MIN;
}
template <typename T> int64_t tmax() {
   return sizeof(T) == 8 ? INT64_MAX : INT32_MAX;
}

/// n values in [-1000, 1000], with extremes sprinkled in. The buffer has one
/// leading element so the returned pointer is not 64-byte aligned.
template <typename T> struct Column {
   std::vector<T> buf;
   T* data;
   Column(size_t n, std::mt19937_64& rng) : buf(n + 1) {
      std::uniform_int_distribution<int64_t> d(-1000, 1000);
      for (auto& x : buf) x = mk<T>(d(rng));
      if (n > 2) {
         buf[1] = mk<T>(tmin<T>());
         buf[n] = mk<T>(tmax<T>());
      }
      data = buf.data() + 1;
   }
};

/// Constants that give roughly the requested selectivity for `<` on `col`,
/// plus the column extremes and the type extremes.
template <typename T>
std::vector<T> constants(const T* col, size_t n) {
   std::vector<T> out = {mk<T>(tmin<T>()), mk<T>(tmax<T>()), mk<T>(0)};
   if (n == 0) return out;
   std::vector<int64_t> v(n);
   for (size_t i = 0; i < n; ++i) v[i] = num(col[i]);
   std::sort(v.begin(), v.end());
   for (int p : kSelPct) out.push_back(mk<T>(v[std::min(n - 1, n * p / 100)]));
   return out;
}

/// Random sorted subset of [0, n) with the given density (percent).
std::vector<pos_t> subset(size_t n, int density, std::mt19937_64& rng) {
   std::vector<pos_t> s;
   std::uniform_int_distribution<int> d(0, 99);
   for (size_t i = 0; i < n; ++i)
      if (d(rng) < density) s.push_back(pos_t(i));
   return s;
}

std::string ctx(const char* what, size_t n, int64_t c) {
   return std::string(what) + " n=" + std::to_string(n) +
          " c=" + std::to_string(c);
}

/// Compare a SIMD kernel's output (found, result[0..found)) with the scalar
/// reference, and check that nothing past result[found] was written.
void expectSame(pos_t fRef, const std::vector<pos_t>& ref, pos_t fSimd,
                const std::vector<pos_t>& simd, const std::string& where) {
   ASSERT_EQ(fRef, fSimd) << where;
   for (pos_t i = 0; i < fRef; ++i) ASSERT_EQ(ref[i], simd[i]) << where << " i=" << i;
   for (size_t i = fSimd; i < simd.size(); ++i)
      ASSERT_EQ(kCanary, simd[i]) << where << " overwrite at " << i;
}

/// All contiguous kernel variants: 512-bit reg/mem x u1/u2, plus the 256-bit
/// reg/mem kernels when AVX512VL is there.
template <typename T>
using SelKernel = pos_t (*)(pos_t, pos_t*, T*, T*);
template <typename T, template <typename> class Op>
std::vector<std::pair<const char*, SelKernel<T>>> colValKernels() {
   return {{"reg_u1", &simd::sel_col_val_v<T, Op, simd::Emit::Reg, 1>},
           {"reg_u2", &simd::sel_col_val_v<T, Op, simd::Emit::Reg, 2>},
           {"mem_u1", &simd::sel_col_val_v<T, Op, simd::Emit::Mem, 1>},
           {"mem_u2", &simd::sel_col_val_v<T, Op, simd::Emit::Mem, 2>},
#ifdef VW_HAVE_SIMD_SEL_256
           {"w256_reg", &simd::sel_col_val_256<T, Op, simd::Emit::Reg>},
           {"w256_mem", &simd::sel_col_val_256<T, Op, simd::Emit::Mem>},
#endif
   };
}
template <typename T, template <typename> class Op>
std::vector<std::pair<const char*, SelKernel<T>>> colColKernels() {
   return {{"reg_u1", &simd::sel_col_col_v<T, Op, simd::Emit::Reg, 1>},
           {"reg_u2", &simd::sel_col_col_v<T, Op, simd::Emit::Reg, 2>},
           {"mem_u1", &simd::sel_col_col_v<T, Op, simd::Emit::Mem, 1>},
           {"mem_u2", &simd::sel_col_col_v<T, Op, simd::Emit::Mem, 2>},
#ifdef VW_HAVE_SIMD_SEL_256
           {"w256_reg", &simd::sel_col_col_256<T, Op, simd::Emit::Reg>},
           {"w256_mem", &simd::sel_col_col_256<T, Op, simd::Emit::Mem>},
#endif
   };
}

template <typename T, template <typename> class Op> void checkColVal() {
   std::mt19937_64 rng(42);
   for (size_t n : kSizes) {
      Column<T> col(n, rng);
      for (T c : constants(col.data, n)) {
         std::vector<pos_t> ref(n + kSlack, kCanary);
         pos_t fr = sel_col_val_bf<T, Op>(n, ref.data(), col.data, &c);
         // every width x compress form x unroll, whichever the CMake
         // options select
         for (auto& [name, kernel] : colValKernels<T, Op>()) {
            std::vector<pos_t> got(n + kSlack, kCanary);
            pos_t fs = kernel(n, got.data(), col.data, &c);
            expectSame(fr, ref, fs, got, ctx(name, n, num(c)) + " sel_col_val");
         }
      }
   }
}

template <typename T, template <typename> class Op> void checkColCol() {
   std::mt19937_64 rng(43);
   for (size_t n : kSizes) {
      Column<T> a(n, rng), b(n, rng);
      // make ~1/4 of the pairs equal to exercise the <=/>=/== boundaries
      for (size_t i = 0; i < n; i += 4) b.data[i] = a.data[i];
      std::vector<pos_t> ref(n + kSlack, kCanary);
      pos_t fr = sel_col_col_bf<T, Op>(n, ref.data(), a.data, b.data);
      for (auto& [name, kernel] : colColKernels<T, Op>()) {
         std::vector<pos_t> got(n + kSlack, kCanary);
         pos_t fs = kernel(n, got.data(), a.data, b.data);
         expectSame(fr, ref, fs, got, ctx(name, n, 0) + " sel_col_col");
      }
   }
}

template <typename T, template <typename> class Op> void checkSelselColVal() {
   std::mt19937_64 rng(44);
   for (size_t n : kSizes) {
      Column<T> col(n, rng);
      for (int density : {10, 50, 100}) {
         auto in = subset(n, density, rng);
         for (T c : constants(col.data, n)) {
            const size_t k = in.size();
            std::vector<pos_t> ref(k + kSlack, kCanary);
            pos_t fr = selsel_col_val_bf<T, Op>(k, in.data(), ref.data(),
                                                 col.data, &c);
            // both gathered-input implementations, whichever one the
            // VW_SIMD_SEL_GATHER option selects
            for (auto kernel : {&simd::selsel_col_val_scalarload<T, Op>,
                                &simd::selsel_col_val_hwgather<T, Op>}) {
               std::vector<pos_t> got(k + kSlack, kCanary);
               pos_t fs = kernel(k, in.data(), got.data(), col.data, &c);
               expectSame(fr, ref, fs, got,
                          ctx(kernel == &simd::selsel_col_val_hwgather<T, Op>
                                  ? "selsel_col_val_hwgather"
                                  : "selsel_col_val_scalarload",
                              k, num(c)));
            }
         }
      }
   }
}

template <typename T, template <typename> class Op> void checkSelselColCol() {
   std::mt19937_64 rng(45);
   for (size_t n : kSizes) {
      Column<T> a(n, rng), b(n, rng);
      for (size_t i = 0; i < n; i += 3) b.data[i] = a.data[i];
      for (int density : {10, 50, 100}) {
         auto in = subset(n, density, rng);
         const size_t k = in.size();
         std::vector<pos_t> ref(k + kSlack, kCanary);
         pos_t fr = selsel_col_col_bf<T, Op>(k, in.data(), ref.data(), a.data,
                                              b.data);
         for (auto kernel : {&simd::selsel_col_col_scalarload<T, Op>,
                             &simd::selsel_col_col_hwgather<T, Op>}) {
            std::vector<pos_t> got(k + kSlack, kCanary);
            pos_t fs = kernel(k, in.data(), got.data(), a.data, b.data);
            expectSame(fr, ref, fs, got,
                       ctx(kernel == &simd::selsel_col_col_hwgather<T, Op>
                               ? "selsel_col_col_hwgather"
                               : "selsel_col_col_scalarload",
                           k, 0));
         }
      }
   }
}

template <typename T> void checkAllOps() {
   checkColVal<T, std::equal_to>();
   checkColVal<T, std::less>();
   checkColVal<T, std::less_equal>();
   checkColVal<T, std::greater>();
   checkColVal<T, std::greater_equal>();
   checkColCol<T, std::equal_to>();
   checkColCol<T, std::less>();
   checkColCol<T, std::less_equal>();
   checkColCol<T, std::greater>();
   checkColCol<T, std::greater_equal>();
   checkSelselColVal<T, std::equal_to>();
   checkSelselColVal<T, std::less>();
   checkSelselColVal<T, std::less_equal>();
   checkSelselColVal<T, std::greater>();
   checkSelselColVal<T, std::greater_equal>();
   checkSelselColCol<T, std::equal_to>();
   checkSelselColCol<T, std::less>();
   checkSelselColCol<T, std::less_equal>();
   checkSelselColCol<T, std::greater>();
   checkSelselColCol<T, std::greater_equal>();
}

#ifdef VW_HAVE_SIMD_SEL_CHAR

/// Random Char<N> values: lengths 0..N, random bytes, garbage padding. A few
/// values share the constant's prefix but differ in length.
template <unsigned N> void checkCharEq() {
   using C = Char<N>;
   std::mt19937_64 rng(46 + N);
   std::uniform_int_distribution<int> byte(0, 255), len(0, N);
   const char alphabet[] = "ABC"; // small alphabet -> real matches
   auto random_char = [&](unsigned l) {
      C c;
      std::memset(&c, 0, sizeof(c));
      for (unsigned j = 0; j < N; ++j) c.value[j] = char(byte(rng)); // padding
      c.len = l;
      for (unsigned j = 0; j < l; ++j) c.value[j] = alphabet[byte(rng) % 3];
      return c;
   };
   for (size_t n : kSizes) {
      std::vector<C> buf(n + 1);
      for (auto& x : buf) {
         const int kind = byte(rng) % 3; // full length / very short / any
         x = random_char(kind == 0 ? N : kind == 1 ? len(rng) % 3 : len(rng));
      }
      C* col = buf.data() + 1;
      std::vector<C> cons;
      cons.push_back(random_char(0)); // empty string
      cons.push_back(random_char(N)); // full length
      if (n > 0) {
         cons.push_back(col[n / 2]); // guaranteed match
         C shorter = col[n / 2];
         if (shorter.len > 0) shorter.len--; // same prefix, shorter
         cons.push_back(shorter);
      }
      for (unsigned l = 0; l <= 2; ++l) cons.push_back(random_char(l));
      for (C& c : cons) {
         std::vector<pos_t> ref(n + kSlack, kCanary), got(n + kSlack, kCanary);
         pos_t fr = sel_col_val_bf<C, std::equal_to>(n, ref.data(), col, &c);
         pos_t fs = simd::sel_char_eq_col_val<N>(n, got.data(), col, &c);
         expectSame(fr, ref, fs, got,
                    ctx("sel_char_eq", n, c.len) + " N=" + std::to_string(N));
      }
   }
}

#endif // VW_HAVE_SIMD_SEL_CHAR

#endif // VW_HAVE_SIMD_SEL
} // namespace

#ifdef VW_HAVE_SIMD_SEL
TEST(SimdSel, Int32) { checkAllOps<int32_t>(); }
TEST(SimdSel, Int64) { checkAllOps<int64_t>(); }
TEST(SimdSel, Date) { checkAllOps<Date>(); }
#else
TEST(SimdSel, Skipped) { VW_TEST_SKIP("no AVX-512F in this build"); }
#endif

#ifdef VW_HAVE_SIMD_SEL_CHAR
TEST(SimdSel, CharEq) {
   checkCharEq<2>();
   checkCharEq<6>();
   checkCharEq<10>();
   checkCharEq<25>();
   checkCharEq<55>();
   checkCharEq<63>();
}
#else
TEST(SimdSel, CharEqSkipped) { VW_TEST_SKIP("no AVX-512BW in this build"); }
#endif

// The CMake options must redirect the public primitive names (the ones query
// plans use) to the SIMD kernels, and leave them scalar otherwise.
TEST(SimdSelRedirect, Names) {
#if defined(VW_SIMD_SEL) && defined(VW_HAVE_SIMD_SEL)
   EXPECT_EQ((void*)sel_greater_Date_col_Date_val_bf,
             (void*)(&simd::sel_col_val<Date, std::greater>));
   EXPECT_EQ((void*)sel_less_Date_col_Date_val,
             (void*)(&simd::sel_col_val<Date, std::less>));
#if defined(VW_SIMD_SEL_GATHER_SCALARLOAD) || defined(VW_SIMD_SEL_GATHER_HWGATHER)
   EXPECT_EQ((void*)selsel_greater_equal_Date_col_Date_val_bf,
             (void*)(&simd::selsel_col_val<Date, std::greater_equal>));
   EXPECT_EQ((void*)selsel_equal_to_int64_t_col_int64_t_col,
             (void*)(&simd::selsel_col_col<int64_t, std::equal_to>));
#else // VW_SIMD_SEL_GATHER=scalar: gathered input stays scalar
   EXPECT_EQ((void*)selsel_greater_equal_Date_col_Date_val_bf,
             (void*)(&selsel_col_val_bf<Date, std::greater_equal>));
   EXPECT_EQ((void*)selsel_equal_to_int64_t_col_int64_t_col,
             (void*)(&selsel_col_col<int64_t, std::equal_to>));
#endif
   EXPECT_EQ((void*)sel_greater_int64_t_col_int64_t_val_bf,
             (void*)(&simd::sel_col_val<int64_t, std::greater>));
   EXPECT_EQ((void*)sel_less_int32_t_col_int32_t_col_bf,
             (void*)(&simd::sel_col_col<int32_t, std::less>));
   // unsupported types stay scalar
   EXPECT_EQ((void*)sel_less_int16_t_col_int16_t_val_bf,
             (void*)(&sel_col_val_bf<int16_t, std::less>));
#else
   EXPECT_EQ((void*)sel_greater_Date_col_Date_val_bf,
             (void*)(&sel_col_val_bf<Date, std::greater>));
#endif
#if defined(VW_SIMD_SEL_CHAR) && defined(VW_HAVE_SIMD_SEL_CHAR)
   EXPECT_EQ((void*)sel_equal_to_Char_10_col_Char_10_val_bf,
             (void*)(&simd::sel_char_eq_col_val<10>));
   EXPECT_EQ((void*)sel_equal_to_Char_25_col_Char_25_val_bf,
             (void*)(&simd::sel_char_eq_col_val<25>));
   // Char<1> and non-equality comparators stay scalar
   EXPECT_EQ((void*)sel_equal_to_Char_1_col_Char_1_val_bf,
             (void*)(&sel_col_val_bf<Char_1, std::equal_to>));
   EXPECT_EQ((void*)sel_less_Char_10_col_Char_10_val_bf,
             (void*)(&sel_col_val_bf<Char_10, std::less>));
#else
   EXPECT_EQ((void*)sel_equal_to_Char_10_col_Char_10_val_bf,
             (void*)(&sel_col_val_bf<Char_10, std::equal_to>));
#endif
}
