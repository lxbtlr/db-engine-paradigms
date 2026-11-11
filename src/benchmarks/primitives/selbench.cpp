// run_selbench: per-tuple cost of selection primitives. Compares the scalar
// branch-free templates (what queries use by default), the six pre-existing
// *_avx512 kernels, and the generic AVX-512 kernels from SimdSelection.hpp
// (for gathered input both the scalar-load and the vpgather implementation).
//
// Usage: run_selbench [-v vecSize] [-m l1|stream|all] [-r reps]
//                     [-s sel%,sel%,...] [-d inSel density %]
//                     [-R lo:hi,lo:hi,...] [-f family substring]
//   l1     : one vecSize-long vector, called reps times (cache resident)
//   stream : a 64 MiB column scanned vector by vector (as a Scan does)
//   -R     : range bounds for the between families; values are uniform in
//            [0, 100), so x >= lo passes 100-lo %, x < hi passes hi %
//   -f     : run only the families whose name contains the substring
// Output: CSV on stdout, one row per (kernel, impl, mode, selectivity). The
// between families (VW_SEL_COMPOUND) put the bounds in op ("ge_lt:lo:hi") and
// the overall pass rate in sel_pct.
#include "common/runtime/Types.hpp"
#include "vectorwise/BetweenPrimitives.hpp"
#include "vectorwise/Primitives.hpp"
#include "vectorwise/SimdSelection.hpp"
#if defined(__aarch64__) && !defined(VW_POS_16)
#include "vectorwise/SimdNeon.hpp"
#endif
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <vector>

using namespace vectorwise;
using namespace vectorwise::primitives;
using types::Char;
using types::Date;

namespace {

size_t vecSize = 1024;
size_t reps = 20000;
std::string mode = "all";
std::vector<int> sels = {1, 10, 50, 90, 99};
int inDensity = 50;
std::string onlyFamily;
// default ranges: overall 15% with each bound first (Q6 l_shipdate is ~15%
// of which the upper bound passes most), 1%, 27% (SSB Q1.1 lo_discount),
// 85%, all
std::vector<std::pair<int, int>> ranges = {{40, 55}, {0, 15}, {85, 100},
                                           {45, 46}, {20, 47}, {10, 95},
                                           {0, 100}};
constexpr size_t kStreamBytes = 64ull << 20;

volatile size_t sink; // keeps results alive

template <typename T> T mk(int64_t v) { return T(v); }
template <> Date mk<Date>(int64_t v) { return Date(int32_t(v)); }
template <typename T> const char* tname();
template <> const char* tname<int32_t>() { return "int32"; }
template <> const char* tname<int64_t>() { return "int64"; }
template <> const char* tname<Date>() { return "Date"; }

/// Values are uniform in [0, 100); this constant makes Op true for ~sel%.
template <template <typename> class Op> int64_t constFor(int sel);
template <> int64_t constFor<std::less>(int sel) { return sel; }
template <> int64_t constFor<std::less_equal>(int sel) { return sel - 1; }
template <> int64_t constFor<std::greater>(int sel) { return 99 - sel; }
template <> int64_t constFor<std::greater_equal>(int sel) { return 100 - sel; }
template <template <typename> class Op> const char* opname();
template <> const char* opname<std::less>() { return "less"; }
template <> const char* opname<std::less_equal>() { return "less_equal"; }
template <> const char* opname<std::greater>() { return "greater"; }
template <> const char* opname<std::greater_equal>() { return "greater_equal"; }

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

void row(const std::string& family, const char* impl, const char* type,
         const char* op, const std::string& md, int density, int sel,
         double ns, size_t found) {
   std::printf("%s,%s,%s,%s,%s,%zu,%d,%d,%.4f,%zu\n", family.c_str(), impl,
               type, op, md.c_str(), vecSize, density, sel, ns, found);
   std::fflush(stdout);
}

bool wantMode(const char* m) { return mode == "all" || mode == m; }
bool wantFamily(const std::string& f) {
   return onlyFamily.empty() || f.find(onlyFamily) != std::string::npos;
}

struct Impl3 {
   const char* name;
   F3 fn;
};
struct Impl4 {
   const char* name;
   F4 fn;
};

/// sel_col_val / sel_col_col: fn(n, result, a, b)
template <typename T, template <typename> class Op>
void benchF3(const std::string& family, bool colcol,
             const std::vector<Impl3>& impls) {
   if (!wantFamily(family)) return;
   std::mt19937_64 rng(1);
   std::uniform_int_distribution<int> d(0, 99);
   const size_t total = wantMode("stream") ? kStreamBytes / sizeof(T) : vecSize;
   std::vector<T> a(std::max(total, vecSize)), b(a.size());
   for (auto& x : a) x = mk<T>(d(rng));
   std::vector<pos_t> res(vecSize + 64);
   for (int sel : sels) {
      T c = mk<T>(constFor<Op>(sel));
      if (colcol)
         for (auto& x : b) x = c; // column of the "constant"
      for (auto& im : impls) {
         if (!im.fn) continue;
         void* bp = colcol ? (void*)b.data() : (void*)&c;
         if (wantMode("l1")) {
            size_t found = 0;
            double ns = bestNs(
                [&] {
                   for (size_t r = 0; r < reps; ++r)
                      found = im.fn(vecSize, res.data(), a.data(), bp);
                },
                reps * vecSize);
            sink = found;
            row(family, im.name, tname<T>(), opname<Op>(), "l1", 100, sel, ns,
                found);
         }
         if (wantMode("stream")) {
            size_t found = 0;
            const size_t chunks = total / vecSize;
            double ns = bestNs(
                [&] {
                   found = 0;
                   for (size_t ch = 0; ch < chunks; ++ch) {
                      void* bb = colcol ? (void*)(b.data() + ch * vecSize) : bp;
                      found += im.fn(vecSize, res.data(),
                                     a.data() + ch * vecSize, bb);
                   }
                },
                chunks * vecSize);
            sink = found;
            row(family, im.name, tname<T>(), opname<Op>(), "stream", 100, sel,
                ns, found);
         }
      }
   }
}

/// selsel_col_val / selsel_col_col: fn(n, inSel, result, a, b). The input
/// selection vector keeps ~inDensity% of each vector (positions within it).
template <typename T, template <typename> class Op>
void benchF4(const std::string& family, bool colcol,
             const std::vector<Impl4>& impls) {
   if (!wantFamily(family)) return;
   std::mt19937_64 rng(2);
   std::uniform_int_distribution<int> d(0, 99);
   const size_t total = wantMode("stream") ? kStreamBytes / sizeof(T) : vecSize;
   std::vector<T> a(std::max(total, vecSize)), b(a.size());
   for (auto& x : a) x = mk<T>(d(rng));
   std::vector<pos_t> inSel;
   for (size_t i = 0; i < vecSize; ++i)
      if (d(rng) < inDensity) inSel.push_back(pos_t(i));
   const size_t k = inSel.size();
   std::vector<pos_t> res(vecSize + 64);
   for (int sel : sels) {
      T c = mk<T>(constFor<Op>(sel));
      if (colcol)
         for (auto& x : b) x = c;
      for (auto& im : impls) {
         if (!im.fn) continue;
         void* bp = colcol ? (void*)b.data() : (void*)&c;
         if (wantMode("l1")) {
            size_t found = 0;
            double ns = bestNs(
                [&] {
                   for (size_t r = 0; r < reps; ++r)
                      found = im.fn(k, inSel.data(), res.data(), a.data(), bp);
                },
                reps * k);
            sink = found;
            row(family, im.name, tname<T>(), opname<Op>(), "l1", inDensity,
                sel, ns, found);
         }
         if (wantMode("stream")) {
            size_t found = 0;
            const size_t chunks = total / vecSize;
            double ns = bestNs(
                [&] {
                   found = 0;
                   for (size_t ch = 0; ch < chunks; ++ch) {
                      void* bb = colcol ? (void*)(b.data() + ch * vecSize) : bp;
                      found += im.fn(k, inSel.data(), res.data(),
                                     a.data() + ch * vecSize, bb);
                   }
                },
                chunks * k);
            sink = found;
            row(family, im.name, tname<T>(), opname<Op>(), "stream", inDensity,
                sel, ns, found);
         }
      }
   }
}

#ifdef VW_HAVE_SIMD_SEL
#define SIMD3(expr) (F3) & expr
#define SIMD4(expr) (F4) & expr
#else
#define SIMD3(expr) nullptr
#define SIMD4(expr) nullptr
#endif
// 256-bit (ymm, AVX512VL) contiguous kernels
#ifdef VW_HAVE_SIMD_SEL_256
#define SIMD256(expr) (F3) & expr
#else
#define SIMD256(expr) nullptr
#endif
#ifdef __AVX512F__
#define OLD(name) name
#else
#define OLD(name) nullptr
#endif
// The old selsel *_avx512 kernels load inSel as 32-bit entries, so they read
// garbage indices under VW_POS_16; skip them there.
#if defined(__AVX512F__) && !defined(VW_POS_16)
#define OLD_SELSEL(name) name
#else
#define OLD_SELSEL(name) nullptr
#endif

template <typename T, template <typename> class Op>
void selColVal(F3 old = nullptr) {
   benchF3<T, Op>("sel_col_val", false,
                  {{"scalar_bf", (F3)&sel_col_val_bf<T, Op>},
                   {"avx512_old", old},
                   {"simd_reg_u1", SIMD3((simd::sel_col_val_v<T, Op, simd::Emit::Reg, 1>))},
                   {"simd_reg_u2", SIMD3((simd::sel_col_val_v<T, Op, simd::Emit::Reg, 2>))},
                   {"simd_mem_u1", SIMD3((simd::sel_col_val_v<T, Op, simd::Emit::Mem, 1>))},
                   {"simd_mem_u2", SIMD3((simd::sel_col_val_v<T, Op, simd::Emit::Mem, 2>))},
                   {"simd256_reg", SIMD256((simd::sel_col_val_256<T, Op, simd::Emit::Reg>))},
                   {"simd256_mem", SIMD256((simd::sel_col_val_256<T, Op, simd::Emit::Mem>))}});
}
template <typename T, template <typename> class Op> void selColCol() {
   benchF3<T, Op>("sel_col_col", true,
                  {{"scalar_bf", (F3)&sel_col_col_bf<T, Op>},
                   {"simd_reg_u1", SIMD3((simd::sel_col_col_v<T, Op, simd::Emit::Reg, 1>))},
                   {"simd_reg_u2", SIMD3((simd::sel_col_col_v<T, Op, simd::Emit::Reg, 2>))},
                   {"simd_mem_u1", SIMD3((simd::sel_col_col_v<T, Op, simd::Emit::Mem, 1>))},
                   {"simd_mem_u2", SIMD3((simd::sel_col_col_v<T, Op, simd::Emit::Mem, 2>))},
                   {"simd256_reg", SIMD256((simd::sel_col_col_256<T, Op, simd::Emit::Reg>))},
                   {"simd256_mem", SIMD256((simd::sel_col_col_256<T, Op, simd::Emit::Mem>))}});
}
template <typename T, template <typename> class Op>
void selselColVal(F4 old = nullptr) {
   benchF4<T, Op>("selsel_col_val", false,
                  {{"scalar_bf", (F4)&selsel_col_val_bf<T, Op>},
                   {"avx512_old", old},
                   {"simd_scalarload",
                    SIMD4((simd::selsel_col_val_scalarload<T, Op>))},
                   {"simd_hwgather",
                    SIMD4((simd::selsel_col_val_hwgather<T, Op>))}});
}
template <typename T, template <typename> class Op> void selselColCol() {
   benchF4<T, Op>("selsel_col_col", true,
                  {{"scalar_bf", (F4)&selsel_col_col_bf<T, Op>},
                   {"simd_scalarload",
                    SIMD4((simd::selsel_col_col_scalarload<T, Op>))},
                   {"simd_hwgather",
                    SIMD4((simd::selsel_col_col_hwgather<T, Op>))}});
}

/// Char<N> == constant over a column of TPC-H-like segment names.
template <unsigned N> void charEq() {
   if (!wantFamily("char_eq_" + std::to_string(N))) return;
   using C = Char<N>;
   static const char* words[] = {"AUTOMOBILE", "BUILDING", "FURNITURE",
                                 "MACHINERY", "HOUSEHOLD"};
   std::mt19937_64 rng(3);
   const size_t total = wantMode("stream") ? kStreamBytes / sizeof(C) : vecSize;
   std::vector<C> a(std::max(total, vecSize));
   for (auto& x : a) {
      std::memset(&x, 0, sizeof(x));
      const char* w = words[rng() % 5];
      x.len = std::min<size_t>(std::strlen(w), N);
      std::memcpy(x.value, w, x.len);
   }
   C c = a[0];
   std::vector<Impl3> impls = {
       {"scalar_bf", (F3)&sel_col_val_bf<C, std::equal_to>},
#ifdef VW_HAVE_SIMD_SEL_CHAR
       {"simd", (F3)&simd::sel_char_eq_col_val<N>},
#endif
   };
   std::vector<pos_t> res(vecSize + 64);
   const std::string fam = "char_eq_" + std::to_string(N);
   for (auto& im : impls) {
      if (wantMode("l1")) {
         size_t found = 0;
         double ns = bestNs(
             [&] {
                for (size_t r = 0; r < reps; ++r)
                   found = im.fn(vecSize, res.data(), a.data(), &c);
             },
             reps * vecSize);
         sink = found;
         row(fam, im.name, "Char", "equal_to", "l1", 100, 20, ns, found);
      }
      if (wantMode("stream")) {
         size_t found = 0;
         const size_t chunks = total / vecSize;
         double ns = bestNs(
             [&] {
                found = 0;
                for (size_t ch = 0; ch < chunks; ++ch)
                   found += im.fn(vecSize, res.data(), a.data() + ch * vecSize,
                                  &c);
             },
             chunks * vecSize);
         sink = found;
         row(fam, im.name, "Char", "equal_to", "stream", 100, 20, ns, found);
      }
   }
}

template <typename T> auto rawOf(const T& v) {
   if constexpr (std::is_same<T, Date>::value)
      return v.value;
   else
      return v;
}

/// Range lo <= x < hi (HiStrict) or lo <= x <= hi: the two-primitive chain
/// (either bound first; the VW plans test the upper bound first) against the
/// fused kernels. Dense input (family between) and gathered input through a
/// selection vector of ~inDensity% (selsel_between). fused_unsigned is
/// (x - lo) < (hi - lo) in unsigned arithmetic, one compare per row; it is a
/// bound only (not in the engine: Typer's bounds are runtime values, so its
/// compiler cannot make that rewrite).
template <typename T, template <typename> class Hi> void benchBetween(bool gathered) {
   const std::string family = gathered ? "selsel_between" : "between";
   if (!wantFamily(family)) return;
   constexpr bool hiStrict = std::is_same<Hi<T>, std::less<T>>::value;
   using R = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;
   // fn(n, inSel or nullptr, result, scratch, col, lo, hi)
   using FB = pos_t (*)(pos_t, pos_t*, pos_t*, pos_t*, T*, T*, T*);
   struct ImplB {
      const char* name;
      FB fn;
   };
   std::vector<ImplB> impls = {
       {"chain_hilo_bf",
        +[](pos_t n, pos_t* in, pos_t* res, pos_t* tmp, T* a, T* lo, T* hi) {
           pos_t k = in ? selsel_col_val_bf<T, Hi>(n, in, tmp, a, hi)
                        : sel_col_val_bf<T, Hi>(n, tmp, a, hi);
           return selsel_col_val_bf<T, std::greater_equal>(k, tmp, res, a, lo);
        }},
       {"chain_lohi_bf",
        +[](pos_t n, pos_t* in, pos_t* res, pos_t* tmp, T* a, T* lo, T* hi) {
           pos_t k = in ? selsel_col_val_bf<T, std::greater_equal>(n, in, tmp, a, lo)
                        : sel_col_val_bf<T, std::greater_equal>(n, tmp, a, lo);
           return selsel_col_val_bf<T, Hi>(k, tmp, res, a, hi);
        }},
       {"fused_bf",
        +[](pos_t n, pos_t* in, pos_t* res, pos_t*, T* a, T* lo, T* hi) {
           return in ? between::between_bf<T, std::greater_equal, Hi, true>(n, in, res, a, lo, hi)
                     : between::between_bf<T, std::greater_equal, Hi, false>(n, in, res, a, lo, hi);
        }},
       {"fused_unsigned",
        +[](pos_t n, pos_t* in, pos_t* res, pos_t*, T* a, T* lo, T* hi) {
           const R l = R(rawOf(*lo)), w = R(R(rawOf(*hi)) - l);
           pos_t found = 0;
           for (pos_t i = 0; i < n; i++) {
              const pos_t idx = in ? in[i] : i;
              const R d = R(R(rawOf(a[idx])) - l);
              res[found] = idx;
              found += hiStrict ? d < w : d <= w;
           }
           return found;
        }},
#ifdef VW_HAVE_SIMD_SEL
       // what fair_sel runs: SIMD on dense input, scalar _bf on gathered
       {"chain_hilo_simd",
        +[](pos_t n, pos_t* in, pos_t* res, pos_t* tmp, T* a, T* lo, T* hi) {
           pos_t k = in ? selsel_col_val_bf<T, Hi>(n, in, tmp, a, hi)
                        : simd::sel_col_val<T, Hi>(n, tmp, a, hi);
           return selsel_col_val_bf<T, std::greater_equal>(k, tmp, res, a, lo);
        }},
       {"chain_hilo_simd_sl",
        +[](pos_t n, pos_t* in, pos_t* res, pos_t* tmp, T* a, T* lo, T* hi) {
           pos_t k = in ? simd::selsel_col_val_scalarload<T, Hi>(n, in, tmp, a, hi)
                        : simd::sel_col_val<T, Hi>(n, tmp, a, hi);
           return simd::selsel_col_val_scalarload<T, std::greater_equal>(
               k, tmp, res, a, lo);
        }},
#endif
   };
#ifdef VW_HAVE_SIMD_SEL
   using simd::Emit;
   if (!gathered) {
      impls.push_back({"fused_simd512_reg", +[](pos_t n, pos_t*, pos_t* res, pos_t*, T* a, T* lo, T* hi) {
                          return simd::sel_between_v<T, std::greater_equal, Hi, Emit::Reg>(n, res, a, lo, hi);
                       }});
      impls.push_back({"fused_simd512_mem", +[](pos_t n, pos_t*, pos_t* res, pos_t*, T* a, T* lo, T* hi) {
                          return simd::sel_between_v<T, std::greater_equal, Hi, Emit::Mem>(n, res, a, lo, hi);
                       }});
#ifdef VW_HAVE_SIMD_SEL_256
      impls.push_back({"fused_simd256_reg", +[](pos_t n, pos_t*, pos_t* res, pos_t*, T* a, T* lo, T* hi) {
                          return simd::sel_between_256<T, std::greater_equal, Hi, Emit::Reg>(n, res, a, lo, hi);
                       }});
#endif
   } else {
      impls.push_back({"fused_scalarload", +[](pos_t n, pos_t* in, pos_t* res, pos_t*, T* a, T* lo, T* hi) {
                          return simd::selsel_between_scalarload<T, std::greater_equal, Hi>(n, in, res, a, lo, hi);
                       }});
      impls.push_back({"fused_hwgather", +[](pos_t n, pos_t* in, pos_t* res, pos_t*, T* a, T* lo, T* hi) {
                          return simd::selsel_between_hwgather<T, std::greater_equal, Hi>(n, in, res, a, lo, hi);
                       }});
   }
#endif
#ifdef VW_HAVE_NEON
   if (!gathered)
      impls.push_back({"fused_neon16", +[](pos_t n, pos_t*, pos_t* res, pos_t*, T* a, T* lo, T* hi) {
                          return neon::sel_between_x16<T, std::greater_equal, Hi>(n, res, a, lo, hi);
                       }});
   else
      impls.push_back({"fused_neon_lanes", +[](pos_t n, pos_t* in, pos_t* res, pos_t*, T* a, T* lo, T* hi) {
                          return neon::selsel_between<T, std::greater_equal, Hi>(n, in, res, a, lo, hi);
                       }});
   // the NEON chain, as a fair_sel would run it if sel_col_val_x16 were wired
   impls.push_back({"chain_hilo_neon", +[](pos_t n, pos_t* in, pos_t* res, pos_t* tmp, T* a, T* lo, T* hi) {
                       pos_t k = in ? selsel_col_val_bf<T, Hi>(n, in, tmp, a, hi)
                                    : neon::sel_col_val_x16<T, Hi>(n, tmp, a, hi);
                       return selsel_col_val_bf<T, std::greater_equal>(k, tmp, res, a, lo);
                    }});
#endif

   std::mt19937_64 rng(4);
   std::uniform_int_distribution<int> d(0, 99);
   const size_t total = wantMode("stream") ? kStreamBytes / sizeof(T) : vecSize;
   std::vector<T> a(std::max(total, vecSize));
   for (auto& x : a) x = mk<T>(d(rng));
   std::vector<pos_t> inSel;
   for (size_t i = 0; i < vecSize; ++i)
      if (!gathered || d(rng) < inDensity) inSel.push_back(pos_t(i));
   const size_t k = inSel.size();
   pos_t* in = gathered ? inSel.data() : nullptr;
   const int density = gathered ? inDensity : 100;
   std::vector<pos_t> res(vecSize + 64), tmp(vecSize + 64);
   for (auto [l, h] : ranges) {
      // x <= h-1 passes the same rows as x < h
      T lo = mk<T>(l), hi = mk<T>(hiStrict ? h : h - 1);
      const std::string op = std::string(hiStrict ? "ge_lt:" : "ge_le:") +
                             std::to_string(l) + ":" + std::to_string(h);
      const int pct = std::max(0, h - l);
      for (auto& im : impls) {
         if (wantMode("l1")) {
            size_t found = 0;
            double ns = bestNs(
                [&] {
                   for (size_t r = 0; r < reps; ++r)
                      found = im.fn(k, in, res.data(), tmp.data(), a.data(),
                                    &lo, &hi);
                },
                reps * k);
            sink = found;
            row(family, im.name, tname<T>(), op.c_str(), "l1", density, pct,
                ns, found);
         }
         if (wantMode("stream")) {
            size_t found = 0;
            const size_t chunks = total / vecSize;
            double ns = bestNs(
                [&] {
                   found = 0;
                   for (size_t ch = 0; ch < chunks; ++ch)
                      found += im.fn(k, in, res.data(), tmp.data(),
                                     a.data() + ch * vecSize, &lo, &hi);
                },
                chunks * k);
            sink = found;
            row(family, im.name, tname<T>(), op.c_str(), "stream", density,
                pct, ns, found);
         }
      }
   }
}

std::vector<int> parseList(const char* s) {
   std::vector<int> out;
   std::stringstream ss(s);
   std::string tok;
   while (std::getline(ss, tok, ',')) out.push_back(std::atoi(tok.c_str()));
   return out;
}

} // namespace

int main(int argc, char** argv) {
   int opt;
   while ((opt = getopt(argc, argv, "v:m:r:s:d:R:f:")) != -1) {
      switch (opt) {
      case 'v': vecSize = std::strtoull(optarg, nullptr, 10); break;
      case 'm': mode = optarg; break;
      case 'r': reps = std::strtoull(optarg, nullptr, 10); break;
      case 's': sels = parseList(optarg); break;
      case 'd': inDensity = std::atoi(optarg); break;
      case 'f': onlyFamily = optarg; break;
      case 'R': {
         ranges.clear();
         std::stringstream ss(optarg);
         std::string tok;
         while (std::getline(ss, tok, ',')) {
            const auto c = tok.find(':');
            ranges.push_back({std::atoi(tok.substr(0, c).c_str()),
                              std::atoi(tok.substr(c + 1).c_str())});
         }
         break;
      }
      default:
         std::fprintf(stderr,
                      "usage: %s [-v vec] [-m l1|stream|all] [-r reps] "
                      "[-s sel,..] [-d density] [-R lo:hi,..] [-f family]\n",
                      argv[0]);
         return 1;
      }
   }
#ifndef VW_HAVE_SIMD_SEL
   std::fprintf(stderr, "run_selbench: no AVX-512F, only scalar rows\n");
#endif
   std::printf("family,impl,type,op,mode,vec,in_density_pct,sel_pct,"
               "ns_per_tuple,found\n");

   // continuous -> compressed (Q1/Q6 int32 l_shipdate, Q3/Q5 Date, Q18 int64)
   selColVal<int32_t, std::less>(OLD(sel_less_int32_t_col_int32_t_val_avx512));
   selColVal<int32_t, std::less_equal>(
       OLD(sel_less_equal_int32_t_col_int32_t_val_avx512));
   selColVal<int32_t, std::greater>();
   selColVal<Date, std::less>();
   selColVal<Date, std::greater>();
   selColVal<int64_t, std::greater>();
   selColCol<int32_t, std::less>();
   selColCol<int64_t, std::less>();

   // gather -> compressed (Q5 Date, Q6 int32/int64)
   selselColVal<int32_t, std::greater_equal>(
       OLD_SELSEL(selsel_greater_equal_int32_t_col_int32_t_val_avx512));
   selselColVal<int64_t, std::less>(
       OLD_SELSEL(selsel_less_int64_t_col_int64_t_val_avx512));
   selselColVal<int64_t, std::greater_equal>(
       OLD_SELSEL(selsel_greater_equal_int64_t_col_int64_t_val_avx512));
   selselColVal<int64_t, std::less_equal>(
       OLD_SELSEL(selsel_less_equal_int64_t_col_int64_t_val_avx512));
   selselColVal<Date, std::greater_equal>();
   selselColCol<int64_t, std::less>();

   // Char<N> == constant (Q3 c_mktsegment is Char<10>)
   charEq<10>();
   charEq<25>();

   // ranges (VW_SEL_COMPOUND): Q6 l_shipdate int32 [lo, hi), Q5 o_orderdate
   // Date [lo, hi), Q6 / SSB Q1.x l_discount, lo_quantity int64 [lo, hi]
   for (bool gathered : {false, true}) {
      benchBetween<int32_t, std::less>(gathered);
      benchBetween<Date, std::less>(gathered);
      benchBetween<int64_t, std::less_equal>(gathered);
   }
   return 0;
}
