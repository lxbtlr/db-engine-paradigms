// run_concatbench: per-row cost of HashGroup's compound key concatenation
// (packing a row's key columns into one K-byte key), see SimdConcat.hpp.
//
// Variants:
//   scalar         today's Concat_T: one strided pass per column (memcpy of
//                  the column width; fixed-size for 1/2/4/8 bytes)
//   fused_scalar   one pass over the rows, every column per row
//   row_gpr        one pass, the row's key built in a 64-bit register from
//                  one shared index load, one store per row (K <= 8)
//   row_tmpl       row_gpr with the widths fixed at compile time (preset
//                  shapes whose columns are all s or all d)
//   wide_fixed     scalar per column, widths > 8 copied by two overlapping
//                  fixed-size moves instead of a memcpy call (wide shapes)
//   <isa>_stage    VW_GROUP_CONCAT_SIMD, selection-vector columns staged by
//                  scalar loads (isa: vbmi, avx2, neon; those compiled in)
//   <isa>_gather   the same with vpgather / NEON lane loads
//
// Shapes are <width><s|d>..., s = read through the vector's selection
// vector (a scanned base column), d = a dense per-vector buffer (a join
// output). Presets: q1 = 1s1s (Q1 returnflag, linestatus), q3 = 4s4d4d
// (Q3 orderkey, orderdate, shippriority), q18 = 25d4d4d8d4s (Q18 c_name,
// custkey, orderdate, totalprice, l_orderkey).
//
// Usage: run_concatbench [-k shape,...] [-s selPct,...] [-m l1|stream|both]
//                        [-v vecSize] [-N rows] [-r reps]
//   l1: one vector of rows, repeated (vecs = N / vecSize); stream: N rows
//   of each column walked vector by vector.
// Output: CSV on stdout, ns per selected row (best of reps), speedup over
// scalar; every variant's packed keys are checked against scalar (`ok`).
#include "vectorwise/SimdConcat.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace vectorwise;
using namespace vectorwise::primitives::simd_concat;

namespace {

struct Shape {
   std::string name, spec;
   std::vector<uint32_t> width, offset;
   std::vector<bool> sel;
   uint32_t keySize = 0;
};

bool parseShape(const std::string& name, Shape& s) {
   std::string spec = name;
   if (name == "q1") spec = "1s1s";
   if (name == "q3") spec = "4s4d4d";
   if (name == "q18") spec = "25d4d4d8d4s";
   s = Shape();
   s.name = name;
   s.spec = spec;
   size_t i = 0;
   while (i < spec.size()) {
      size_t j = i;
      while (j < spec.size() && isdigit(spec[j])) ++j;
      if (j == i || j == spec.size() || (spec[j] != 's' && spec[j] != 'd'))
         return false;
      const uint32_t w = uint32_t(std::stoul(spec.substr(i, j - i)));
      if (w == 0) return false;
      s.width.push_back(w);
      s.offset.push_back(s.keySize);
      s.sel.push_back(spec[j] == 's');
      s.keySize += w;
      i = j + 1;
   }
   return !s.width.empty() && s.width.size() <= kMaxCols;
}

std::vector<std::string> split(const char* arg) {
   std::vector<std::string> v;
   std::string cur;
   for (const char* p = arg; *p; ++p) {
      if (*p == ',') {
         v.push_back(cur);
         cur.clear();
      } else {
         cur += *p;
      }
   }
   if (!cur.empty()) v.push_back(cur);
   return v;
}

/// every column, per row
void fusedScalar(const Shape& s, pos_t n, const char* const* data,
                 const pos_t* const* sel, char* out) {
   const size_t cols = s.width.size();
   const uint32_t K = s.keySize;
   for (pos_t i = 0; i < n; ++i) {
      char* d = out + size_t(i) * K;
      for (size_t c = 0; c < cols; ++c) {
         const size_t r = sel[c] ? sel[c][i] : i;
         const uint32_t w = s.width[c];
         const char* src = data[c] + r * w;
         switch (w) {
         case 1: std::memcpy(d + s.offset[c], src, 1); break;
         case 2: std::memcpy(d + s.offset[c], src, 2); break;
         case 4: std::memcpy(d + s.offset[c], src, 4); break;
         case 8: std::memcpy(d + s.offset[c], src, 8); break;
         default: std::memcpy(d + s.offset[c], src, w);
         }
      }
   }
}

/// copy of w bytes (8 < w <= 32) by two overlapping fixed-size moves
inline void copyOverlap(char* d, const char* src, uint32_t w) {
   if (w <= 16) {
      uint64_t a, b;
      std::memcpy(&a, src, 8);
      std::memcpy(&b, src + w - 8, 8);
      std::memcpy(d, &a, 8);
      std::memcpy(d + w - 8, &b, 8);
   } else {
      char a[16], b[16];
      std::memcpy(a, src, 16);
      std::memcpy(b, src + w - 16, 16);
      std::memcpy(d, a, 16);
      std::memcpy(d + w - 16, b, 16);
   }
}

/// per column; widths > 8 by copyOverlap (> 32: memcpy)
void wideFixed(const Shape& s, pos_t n, const char* const* data,
               const pos_t* const* sel, char* out) {
   const uint32_t K = s.keySize;
   for (size_t c = 0; c < s.width.size(); ++c) {
      const uint32_t w = s.width[c];
      if (w <= 8 || w > 32) {
         runColumns(1, &w, &s.offset[c], K, n, &data[c], &sel[c], out);
         continue;
      }
      char* dest = out + s.offset[c];
      const char* src = data[c];
      if (sel[c]) {
         for (pos_t i = 0; i < n; ++i)
            copyOverlap(dest + size_t(i) * K, src + size_t(sel[c][i]) * w, w);
      } else {
         for (pos_t i = 0; i < n; ++i)
            copyOverlap(dest + size_t(i) * K, src + size_t(i) * w, w);
      }
   }
}

/// row_gpr: per row, the key assembled in a 64-bit register (K <= 8), one
/// 8-byte store (overlapping the next row's, kSlack covers the last). Each
/// value is read by the 8 bytes that end with it (or a column's first 8)
/// and shifted into place; columns sharing one selection vector (Shared)
/// load the row's index once.
template <int C, bool Shared>
void rowGpr(const Shape& s, pos_t n, const char* const* data,
            const pos_t* const* sel, char* out) {
   const uint32_t K = s.keySize;
   uint32_t w[C], shift[C];
   uint64_t mask[C];
   bool isSel[C];
   const pos_t* shared = nullptr;
   for (int c = 0; c < C; ++c) {
      w[c] = s.width[c];
      shift[c] = 8 * s.offset[c];
      mask[c] = w[c] == 8 ? ~uint64_t(0) : (uint64_t(1) << (8 * w[c])) - 1;
      isSel[c] = sel[c] != nullptr;
      if (sel[c]) shared = sel[c];
   }
   for (pos_t i = 0; i < n; ++i) {
      const size_t si = Shared ? shared[i] : 0;
      uint64_t key = 0;
      for (int c = 0; c < C; ++c) {
         const size_t idx = isSel[c] ? (Shared ? si : sel[c][i]) : i;
         const size_t o = idx * w[c];
         const size_t a = o + w[c] >= 8 ? o + w[c] - 8 : 0;
         uint64_t v;
         std::memcpy(&v, data[c] + a, 8);
         key |= ((v >> (8 * (o - a))) & mask[c]) << shift[c];
      }
      std::memcpy(out + size_t(i) * K, &key, 8);
   }
}

/// row_tmpl: row_gpr with the column widths fixed at compile time (the
/// plan picks an instantiation per width tuple); every column read through
/// the one selection vector (Sel) or dense; exact-width loads
template <size_t W> inline uint64_t loadW(const char* p) {
   if constexpr (W == 1) {
      uint8_t v;
      std::memcpy(&v, p, 1);
      return v;
   } else if constexpr (W == 2) {
      uint16_t v;
      std::memcpy(&v, p, 2);
      return v;
   } else if constexpr (W == 4) {
      uint32_t v;
      std::memcpy(&v, p, 4);
      return v;
   } else {
      uint64_t v;
      std::memcpy(&v, p, 8);
      return v;
   }
}
template <bool Sel, size_t... W, size_t... C>
void rowTmplImpl(pos_t n, const char* const* data, const pos_t* const* sel,
                 char* out, std::index_sequence<C...>) {
   constexpr size_t K = (W + ...);
   constexpr size_t ws[] = {W...};
   constexpr auto offs = [] {
      std::array<size_t, sizeof...(W)> o{};
      size_t acc = 0;
      for (size_t c = 0; c < sizeof...(W); ++c) o[c] = acc, acc += ws[c];
      return o;
   }();
   const char* d[] = {data[C]...};
   const pos_t* s = nullptr;
   if constexpr (Sel) s = sel[0];
   for (pos_t i = 0; i < n; ++i) {
      const size_t idx = Sel ? s[i] : i;
      const uint64_t key =
          ((loadW<ws[C]>(d[C] + idx * ws[C]) << (8 * offs[C])) | ...);
      std::memcpy(out + i * K, &key,
                  K == 1 || K == 2 || K == 4 || K == 8 ? K : 8);
   }
}
template <bool Sel, size_t... W>
void rowTmpl(pos_t n, const char* const* data, const pos_t* const* sel,
             char* out) {
   rowTmplImpl<Sel, W...>(n, data, sel, out,
                          std::make_index_sequence<sizeof...(W)>());
}
using TmplFn = void (*)(pos_t, const char* const*, const pos_t* const*, char*);
/// the instantiations for the bench's preset shapes
TmplFn rowTmplFor(const std::string& spec) {
   static const std::pair<const char*, TmplFn> table[] = {
       {"1s1s", rowTmpl<true, 1, 1>},
       {"2s2s", rowTmpl<true, 2, 2>},
       {"4s4s", rowTmpl<true, 4, 4>},
       {"1s2s4s", rowTmpl<true, 1, 2, 4>},
       {"2s2s2s2s", rowTmpl<true, 2, 2, 2, 2>},
       {"1s1s1s1s4s", rowTmpl<true, 1, 1, 1, 1, 4>},
       {"2d2d", rowTmpl<false, 2, 2>},
       {"4d4d", rowTmpl<false, 4, 4>},
       {"1d2d4d", rowTmpl<false, 1, 2, 4>},
   };
   for (auto& e : table)
      if (spec == e.first) return e.second;
   return nullptr;
}

template <bool Shared>
void rowGprN(const Shape& s, pos_t n, const char* const* data,
             const pos_t* const* sel, char* out) {
   switch (s.width.size()) {
   case 1: rowGpr<1, Shared>(s, n, data, sel, out); break;
   case 2: rowGpr<2, Shared>(s, n, data, sel, out); break;
   case 3: rowGpr<3, Shared>(s, n, data, sel, out); break;
   case 4: rowGpr<4, Shared>(s, n, data, sel, out); break;
   case 5: rowGpr<5, Shared>(s, n, data, sel, out); break;
   case 6: rowGpr<6, Shared>(s, n, data, sel, out); break;
   case 7: rowGpr<7, Shared>(s, n, data, sel, out); break;
   default: rowGpr<8, Shared>(s, n, data, sel, out); break;
   }
}

using Fn = std::function<void(pos_t, const char* const*, const pos_t* const*,
                              char*)>;
struct Variant {
   std::string name;
   Fn fn;
};

struct Input {
   std::vector<std::vector<char>> columns; // per column
   std::vector<pos_t> sel;                 // per vector, concatenated
   std::vector<pos_t> selN;                // selected rows per vector
   size_t vecs = 0;
   size_t selected = 0;
};

/// columns of `rows` rows; sel vectors over vectors of vecSize rows. l1:
/// one vector, repeated.
Input makeInput(const Shape& s, size_t rows, size_t vecSize, bool l1,
                unsigned selPct, std::mt19937_64& rng) {
   Input in;
   const size_t colRows = l1 ? vecSize : rows;
   for (uint32_t w : s.width) {
      std::vector<char> col(colRows * w + 64);
      for (auto& b : col) b = char(rng());
      in.columns.push_back(std::move(col));
   }
   in.vecs = rows / vecSize;
   const size_t selVecs = l1 ? 1 : in.vecs;
   in.sel.resize(selVecs * vecSize + 64);
   for (size_t v = 0; v < selVecs; ++v) {
      pos_t n = 0;
      for (size_t i = 0; i < vecSize; ++i)
         if (rng() % 100 < selPct) in.sel[v * vecSize + n++] = pos_t(i);
      in.selN.push_back(n);
   }
   for (size_t v = 0; v < in.vecs; ++v) in.selected += in.selN[l1 ? 0 : v];
   return in;
}

/// one pass over every vector; returns ns
double pass(const Shape& s, const Input& in, size_t vecSize, bool l1,
            const Fn& fn, std::vector<char>& out, uint64_t& sink) {
   const size_t cols = s.width.size();
   const char* data[kMaxCols];
   const pos_t* sel[kMaxCols];
   auto t0 = std::chrono::steady_clock::now();
   size_t dense = 0; // dense buffers are compact: n rows per vector
   for (size_t v = 0; v < in.vecs; ++v) {
      const size_t sv = l1 ? 0 : v;
      const pos_t n = in.selN[sv];
      for (size_t c = 0; c < cols; ++c) {
         const char* col = in.columns[c].data();
         if (s.sel[c]) {
            data[c] = col + (l1 ? 0 : v * vecSize * s.width[c]);
            sel[c] = in.sel.data() + sv * vecSize;
         } else {
            data[c] = col + (l1 ? 0 : dense * s.width[c]);
            sel[c] = nullptr;
         }
      }
      dense += n;
      fn(n, data, sel, out.data());
      sink += uint64_t(out[size_t(v % (n ? n : 1)) * s.keySize]);
   }
   auto t1 = std::chrono::steady_clock::now();
   return std::chrono::duration<double, std::nano>(t1 - t0).count();
}

/// fn's packed keys equal scalar's for the first vectors
bool verify(const Shape& s, const Input& in, size_t vecSize, bool l1,
            const Fn& fn, const Fn& ref) {
   const size_t cols = s.width.size();
   std::vector<char> a(vecSize * s.keySize + kSlack),
       b(vecSize * s.keySize + kSlack);
   const char* data[kMaxCols];
   const pos_t* sel[kMaxCols];
   size_t dense = 0;
   for (size_t v = 0; v < std::min<size_t>(in.vecs, 8); ++v) {
      const size_t sv = l1 ? 0 : v;
      const pos_t n = in.selN[sv];
      for (size_t c = 0; c < cols; ++c) {
         const char* col = in.columns[c].data();
         data[c] = s.sel[c] ? col + (l1 ? 0 : v * vecSize * s.width[c])
                            : col + (l1 ? 0 : dense * s.width[c]);
         sel[c] = s.sel[c] ? in.sel.data() + sv * vecSize : nullptr;
      }
      dense += n;
      ref(n, data, sel, a.data());
      fn(n, data, sel, b.data());
      if (std::memcmp(a.data(), b.data(), size_t(n) * s.keySize)) return false;
   }
   return true;
}

} // namespace

int main(int argc, char** argv) {
   std::vector<std::string> shapes = {"q1",   "q3",   "q18",    "2s2s",
                                      "4s4s", "8s8s", "1s2s4s", "2s2s2s2s",
                                      "1s1s1s1s4s", "2d2d", "4d4d", "1d2d4d"};
   std::vector<unsigned> sels = {98, 50};
   std::string modes = "both";
   size_t vecSize = 1024, rows = size_t(1) << 23;
   int reps = 5;
   int opt;
   while ((opt = getopt(argc, argv, "k:s:m:v:N:r:")) != -1) {
      switch (opt) {
      case 'k': shapes = split(optarg); break;
      case 's':
         sels.clear();
         for (auto& x : split(optarg)) sels.push_back(unsigned(std::stoul(x)));
         break;
      case 'm': modes = optarg; break;
      case 'v': vecSize = std::stoul(optarg); break;
      case 'N': rows = std::stoul(optarg); break;
      case 'r': reps = std::stoi(optarg); break;
      default:
         std::fprintf(stderr, "usage: %s [-k shape,..] [-s selPct,..] "
                              "[-m l1|stream|both] [-v vec] [-N rows] [-r reps]\n",
                      argv[0]);
         return 1;
      }
   }
   std::vector<bool> l1Modes;
   if (modes == "l1" || modes == "both") l1Modes.push_back(true);
   if (modes == "stream" || modes == "both") l1Modes.push_back(false);

   std::printf("shape,key_bytes,cols,variant,mode,sel_pct,vec,ns_per_row,"
               "speedup,ok\n");
   uint64_t sink = 0;
   std::mt19937_64 rng(42);
   for (const auto& shapeName : shapes) {
      Shape s;
      if (!parseShape(shapeName, s)) {
         std::fprintf(stderr, "bad shape %s\n", shapeName.c_str());
         return 1;
      }
      const size_t cols = s.width.size();
      std::vector<Variant> variants;
      variants.push_back({"scalar", [&s](pos_t n, const char* const* d,
                                          const pos_t* const* sl, char* o) {
                             runColumns(s.width.size(), s.width.data(),
                                        s.offset.data(), s.keySize, n, d, sl,
                                        o);
                          }});
      variants.push_back({"fused_scalar",
                          [&s](pos_t n, const char* const* d,
                               const pos_t* const* sl, char* o) {
                             fusedScalar(s, n, d, sl, o);
                          }});
      // the bench's sel columns all read the vector's one selection vector
      if (s.keySize <= 8)
         variants.push_back({"row_gpr", [&s](pos_t n, const char* const* d,
                                              const pos_t* const* sl,
                                              char* o) {
                                if (std::count(s.sel.begin(), s.sel.end(), true))
                                   rowGprN<true>(s, n, d, sl, o);
                                else
                                   rowGprN<false>(s, n, d, sl, o);
                             }});
      if (TmplFn f = rowTmplFor(s.spec))
         variants.push_back({"row_tmpl", f});
      if (std::any_of(s.width.begin(), s.width.end(),
                      [](uint32_t w) { return w > 8; }))
         variants.push_back({"wide_fixed",
                             [&s](pos_t n, const char* const* d,
                                  const pos_t* const* sl, char* o) {
                                wideFixed(s, n, d, sl, o);
                             }});
      std::vector<Plan> plans; // stable addresses for the lambdas
      plans.reserve(8);
      for (Isa isa : {Isa::Vbmi, Isa::Avx2, Isa::Neon})
         for (SelLoad sl : {SelLoad::Stage, SelLoad::Gather}) {
            bool selFlags[kMaxCols];
            for (size_t c = 0; c < cols; ++c) selFlags[c] = s.sel[c];
            Plan p;
            if (!build(p, isa, sl, cols, s.width.data(), s.offset.data(),
                       selFlags, s.keySize))
               continue;
            plans.push_back(p);
            const Plan* pp = &plans.back();
            variants.push_back(
                {std::string(name(isa)) +
                     (sl == SelLoad::Stage ? "_stage" : "_gather"),
                 [pp](pos_t n, const char* const* d, const pos_t* const* sel,
                      char* o) { run(*pp, n, d, sel, o); }});
         }

      for (bool l1 : l1Modes)
         for (unsigned selPct : sels) {
            Input in = makeInput(s, rows, vecSize, l1, selPct, rng);
            std::vector<char> out(vecSize * s.keySize + kSlack);
            double scalarNs = 0;
            for (const auto& v : variants) {
               const bool ok = verify(s, in, vecSize, l1, v.fn, variants[0].fn);
               double best = 1e300;
               pass(s, in, vecSize, l1, v.fn, out, sink); // warm
               for (int r = 0; r < reps; ++r)
                  best = std::min(best, pass(s, in, vecSize, l1, v.fn, out, sink));
               const double perRow = best / double(in.selected ? in.selected : 1);
               if (&v == &variants[0]) scalarNs = perRow;
               std::printf("%s,%u,%zu,%s,%s,%u,%zu,%.4f,%.3f,%d\n",
                           s.name.c_str(), s.keySize, cols, v.name.c_str(),
                           l1 ? "l1" : "stream", selPct, vecSize, perRow,
                           scalarNs / perRow, ok ? 1 : 0);
               std::fflush(stdout);
            }
         }
   }
   std::fprintf(stderr, "sink %llu\n", (unsigned long long)sink);
   return 0;
}
