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
//   wide_fixed     VW_GROUP_CONCAT_WIDE: columns other than 1/2/4/8 bytes
//                  copied by copyWideColumn (two overlapping fixed moves)
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
// Hash mode (-H): concat plus HashGroup's hash of the packed keys
// (MurmurHash64A, every variant's hashes checked against scalar+hash):
//   scalar+hash      today's Concat_T, then Hash_T's scalar pass
//   scalar+simdhash  Concat_T, then simd_hash::hash_keys (VW_SIMD_HASH)
//   row+hash, row+simdhash   row_tmpl, then either pass
//   fused, fused_simd        VW_GROUP_CONCAT_HASH: key and hash in one pass
//                            (the library's row shapes; simd: AVX-512)
//   fused_block              the row kernel over 128 rows, then hash_keys
//                            over them (VW_GROUP_CONCAT_HASH_MODE=blocked)
//
// Padding mode (-P), shapes whose compound key is 3, 5..7 or 9..15 bytes:
// concat, HashGroup's hash and a whole-key comparison per row
//   unpadded         the K-byte key: variable-length hash and memcmp
//   paddedP          VW_GROUP_KEY_PAD: padded to P = 4/8 (integer hash and
//                    compare; +simdhash: the AVX-512 pass) or 16 (hash and
//                    memcmp with the length fixed)
//
// Usage: run_concatbench [-k shape,...] [-s selPct,...] [-m l1|stream|both]
//                        [-v vecSize] [-N rows] [-r reps] [-H] [-P]
//   l1: one vector of rows, repeated (vecs = N / vecSize); stream: N rows
//   of each column walked vector by vector.
// Output: CSV on stdout, ns per selected row (best of reps), speedup over
// scalar; every variant's packed keys are checked against scalar (`ok`).
#include "common/runtime/Hash.hpp"
#include "vectorwise/SimdConcat.hpp"
#include "vectorwise/SimdHash.hpp"
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

/// per column; widths other than 1/2/4/8 by copyWideColumn
/// (VW_GROUP_CONCAT_WIDE)
void wideFixed(const Shape& s, pos_t n, const char* const* data,
               const pos_t* const* sel, char* out) {
   const uint32_t K = s.keySize;
   for (size_t c = 0; c < s.width.size(); ++c) {
      const uint32_t w = s.width[c];
      if (w == 1 || w == 2 || w == 4 || w == 8)
         runColumns(1, &w, &s.offset[c], K, n, &data[c], &sel[c], out);
      else
         copyWideColumn(n, data[c], sel[c], out + s.offset[c], K, w);
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

/// -H: the hashes of the variant last run (HashGroup's groupHashes)
std::vector<uint64_t> gHashes;

template <typename T>
void hashKeysT(pos_t n, const char* keys, uint64_t* out) {
   const runtime::MurMurHash h;
   for (pos_t i = 0; i < n; ++i) {
      T k;
      std::memcpy(&k, keys + size_t(i) * sizeof(T), sizeof(T));
      out[i] = h.hashKey(uint64_t(k));
   }
}
/// HashGroup::Hash_T without VW_SIMD_HASH: scalar MurmurHash64A of each
/// packed key, zero-extended for 1/2/4/8 bytes
void hashPass(const Shape& s, pos_t n, const char* keys) {
   uint64_t* out = gHashes.data();
   switch (s.keySize) {
   case 1: hashKeysT<uint8_t>(n, keys, out); break;
   case 2: hashKeysT<uint16_t>(n, keys, out); break;
   case 4: hashKeysT<uint32_t>(n, keys, out); break;
   case 8: hashKeysT<uint64_t>(n, keys, out); break;
   default: {
      const runtime::MurMurHash h;
      for (pos_t i = 0; i < n; ++i)
         out[i] = h.hashKey(keys + size_t(i) * s.keySize, int(s.keySize), 0);
   }
   }
}
/// HashGroup::Hash_T with VW_SIMD_HASH (simd_hash::hash_keys); false where
/// it is not compiled in or the key is not 1/2/4/8 bytes
bool simdHashPass(const Shape& s, pos_t n, const char* keys, bool dryRun) {
#ifdef VW_HAVE_SIMD_HASH
   namespace sh = vectorwise::primitives::simd_hash;
   uint64_t* out = gHashes.data();
   switch (s.keySize) {
   case 1: if (!dryRun) sh::hash_keys<uint8_t>(n, keys, out); return true;
   case 2: if (!dryRun) sh::hash_keys<uint16_t>(n, keys, out); return true;
   case 4: if (!dryRun) sh::hash_keys<uint32_t>(n, keys, out); return true;
   case 8: if (!dryRun) sh::hash_keys<uint64_t>(n, keys, out); return true;
   default: return false;
   }
#else
   (void)s, (void)n, (void)keys, (void)dryRun;
   return false;
#endif
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

/// fn's packed keys (-H: and hashes) equal ref's for the first vectors
bool verify(const Shape& s, const Input& in, size_t vecSize, bool l1,
            const Fn& fn, const Fn& ref, bool hashes) {
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
      const std::vector<uint64_t> want(gHashes.begin(), gHashes.begin() + n);
      fn(n, data, sel, b.data());
      if (std::memcmp(a.data(), b.data(), size_t(n) * s.keySize)) return false;
      if (hashes && !std::equal(want.begin(), want.end(), gHashes.begin()))
         return false;
   }
   return true;
}


/// -P: whole keys found equal (HashGroup's key comparison after a hash
/// match), and the padded variants' output (zero-initialized, so the
/// padding bytes stay zero as in HashGroup's packedKeys)
uint64_t gEqual = 0;
std::vector<char> gPadded;

/// compare n keys of K bytes with themselves through a pointer the compiler
/// cannot see is the same, so every comparison runs to the end
template <typename Eq>
void compareKeys(pos_t n, const char* keys, size_t K, Eq eq) {
   const char* other = keys;
   asm volatile("" : "+r"(other));
   uint64_t equal = 0;
   for (pos_t i = 0; i < n; ++i)
      equal += eq(keys + size_t(i) * K, other + size_t(i) * K);
   gEqual += equal;
}
/// unpadded K-byte key (not 1/2/4/8): Hash_T<char*> and Lookup_T<char*>'s
/// variable-length hash and memcmp
void hashEqBytes(const Shape& s, pos_t n, const char* keys) {
   hashPass(s, n, keys);
   const size_t K = s.keySize;
   compareKeys(n, keys, K, [K](const char* a, const char* b) {
      return std::memcmp(a, b, K) == 0;
   });
}
/// VW_GROUP_KEY_PAD: the key padded to P = 4/8 (integer hash, simd: the
/// AVX-512 pass, and integer compare) or 16 (byte hash and memcmp with the
/// length fixed)
template <size_t P>
void hashEqPadded(pos_t n, const char* keys, bool simd) {
   uint64_t* out = gHashes.data();
   if constexpr (P == 16) {
      const runtime::MurMurHash h;
      for (pos_t i = 0; i < n; ++i)
         out[i] = h.hashKey(keys + size_t(i) * 16, 16, 0);
      compareKeys(n, keys, 16, [](const char* a, const char* b) {
         return std::memcmp(a, b, 16) == 0;
      });
   } else {
      using T = std::conditional_t<P == 4, uint32_t, uint64_t>;
#ifdef VW_HAVE_SIMD_HASH
      if (simd)
         vectorwise::primitives::simd_hash::hash_keys<T>(n, keys, out);
      else
#endif
         hashKeysT<T>(n, keys, out);
      (void)simd;
      compareKeys(n, keys, P, [](const char* a, const char* b) {
         T x, y;
         std::memcpy(&x, a, P);
         std::memcpy(&y, b, P);
         return x == y;
      });
   }
}
/// the padded size of a compound key (VW_GROUP_KEY_PAD), or 0 if unpadded
uint32_t paddedSize(const Shape& s) {
   const uint32_t k = s.keySize;
   if (s.width.size() < 2 || k >= 16) return 0;
   const uint32_t to = k <= 2 ? k : k <= 4 ? 4 : k <= 8 ? 8 : 16;
   return to == k ? 0 : to;
}
/// -P: the padded variant's keys (gPadded, stride P) hold the unpadded
/// ones' bytes and zero padding, for the first vectors
bool verifyPadded(const Shape& s, const Input& in, size_t vecSize, bool l1,
                  const Fn& padded, uint32_t P) {
   const size_t cols = s.width.size();
   const uint32_t K = s.keySize;
   std::vector<char> a(vecSize * K + kSlack);
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
      runColumns(cols, s.width.data(), s.offset.data(), K, n, data, sel,
                 a.data());
      padded(n, data, sel, nullptr);
      for (pos_t i = 0; i < n; ++i) {
         const char* row = gPadded.data() + size_t(i) * P;
         if (std::memcmp(row, a.data() + size_t(i) * K, K)) return false;
         for (uint32_t b = K; b < P; ++b)
            if (row[b]) return false;
      }
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
   bool hashMode = false, padMode = false;
   while ((opt = getopt(argc, argv, "k:s:m:v:N:r:HP")) != -1) {
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
      case 'H': hashMode = true; break;
      case 'P': padMode = true; break;
      default:
         std::fprintf(stderr, "usage: %s [-k shape,..] [-s selPct,..] "
                              "[-m l1|stream|both] [-v vec] [-N rows] [-r reps]\n",
                      argv[0]);
         return 1;
      }
   }
   gHashes.assign(vecSize + 8, 0);
   gPadded.assign(vecSize * 16 + kSlack, 0);
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
      if (hashMode) {
         // concat + HashGroup's hash of the packed keys
         variants.clear();
         variants.push_back({"scalar+hash", [&s](pos_t n, const char* const* d,
                                                  const pos_t* const* sl,
                                                  char* o) {
                                runColumns(s.width.size(), s.width.data(),
                                           s.offset.data(), s.keySize, n, d,
                                           sl, o);
                                hashPass(s, n, o);
                             }});
         if (simdHashPass(s, 0, nullptr, true))
            variants.push_back({"scalar+simdhash",
                                [&s](pos_t n, const char* const* d,
                                     const pos_t* const* sl, char* o) {
                                   runColumns(s.width.size(), s.width.data(),
                                              s.offset.data(), s.keySize, n,
                                              d, sl, o);
                                   simdHashPass(s, n, o, false);
                                }});
         if (TmplFn f = rowTmplFor(s.spec)) {
            variants.push_back({"row+hash", [&s, f](pos_t n, const char* const* d,
                                                     const pos_t* const* sl,
                                                     char* o) {
                                   f(n, d, sl, o);
                                   hashPass(s, n, o);
                                }});
            if (simdHashPass(s, 0, nullptr, true))
               variants.push_back({"row+simdhash",
                                   [&s, f](pos_t n, const char* const* d,
                                           const pos_t* const* sl, char* o) {
                                      f(n, d, sl, o);
                                      simdHashPass(s, n, o, false);
                                   }});
         }
         bool selFlags[kMaxCols];
         for (size_t c = 0; c < cols; ++c) selFlags[c] = s.sel[c];
         Plan rp;
         if (build(rp, Isa::Row, SelLoad::Stage, cols, s.width.data(),
                   s.offset.data(), selFlags, s.keySize)) {
            plans.push_back(rp);
            const Plan* pp = &plans.back();
            const std::pair<const char*, HashMode> modes[] = {
                {"fused", HashMode::Scalar},
                {"fused_simd", HashMode::Simd},
                {"fused_block", HashMode::Blocked}};
            for (const auto& [label, mode] : modes) {
               if (mode != HashMode::Scalar && !simdHashAvailable()) continue;
               variants.push_back(
                   {label, [pp, mode = mode](pos_t n, const char* const* d,
                                             const pos_t* const* sl, char* o) {
                       runHash(*pp, mode, n, d, sl, o, gHashes.data());
                    }});
            }
         }
      }
      if (padMode) {
         // concat, hash and whole-key comparison per row: the key as it is,
         // and padded (VW_GROUP_KEY_PAD) to 4, 8 or 16 bytes
         const uint32_t P = paddedSize(s);
         if (!P) continue;
         // padding bytes start zero for this shape's stride
         std::fill(gPadded.begin(), gPadded.end(), 0);
         variants.clear();
         variants.push_back({"unpadded", [&s](pos_t n, const char* const* d,
                                              const pos_t* const* sl,
                                              char* o) {
                                runColumns(s.width.size(), s.width.data(),
                                           s.offset.data(), s.keySize, n, d,
                                           sl, o);
                                hashEqBytes(s, n, o);
                             }});
         for (bool simd : {false, true}) {
#ifndef VW_HAVE_SIMD_HASH
            if (simd) continue;
#endif
            if (simd && P == 16) continue;
            variants.push_back(
                {std::string("padded") + std::to_string(P) +
                     (simd ? "+simdhash" : ""),
                 [&s, P, simd](pos_t n, const char* const* d,
                               const pos_t* const* sl, char*) {
                    char* o = gPadded.data();
                    runColumns(s.width.size(), s.width.data(),
                               s.offset.data(), P, n, d, sl, o);
                    if (P == 4) hashEqPadded<4>(n, o, simd);
                    else if (P == 8) hashEqPadded<8>(n, o, simd);
                    else hashEqPadded<16>(n, o, simd);
                 }});
         }
      }

      for (bool l1 : l1Modes)
         for (unsigned selPct : sels) {
            Input in = makeInput(s, rows, vecSize, l1, selPct, rng);
            // -P pads keys up to 16 bytes
            std::vector<char> out(vecSize * std::max(16u, s.keySize) + kSlack);
            double scalarNs = 0;
            for (const auto& v : variants) {
               const bool ok =
                   padMode && &v != &variants[0]
                       ? verifyPadded(s, in, vecSize, l1, v.fn, paddedSize(s))
                       : verify(s, in, vecSize, l1, v.fn, variants[0].fn,
                                hashMode);
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
