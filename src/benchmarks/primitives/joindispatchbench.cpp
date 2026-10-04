// run_joindispatchbench: the cost questions behind a plan-resolved
// Hashjoin::resolvePaths (JOIN_DISPATCH_STUDY.md). Every variant of a family
// must produce the same output as the family's first variant (count +
// checksum); exit status 1 on any MISMATCH.
//
//   elide     TPC-H Q18 filters the inner group-by's output (1.5M groups,
//             l_quantity sum > 300, 57 pass) with a Select before the semi
//             join's build. Group entries are 32 B (EntryHeader 16, key int32
//             at 16, sum int64 at 20; QueryBuilder group layout). Variants,
//             per output block of vec entries, each ending with the survivors'
//             keys and build hashes:
//               today      gatherGroups (gather_val key + sum), Select
//                          (sel_greater_int64 _bf), build hash_sel + key copy
//               lazy_key   gather the sum only, filter, read the survivors'
//                          keys from the entries through the selection
//               fused_bf   one pass over the entries: branch-free filter on
//                          the sum read in place, then survivors' keys
//               fused_br   same, branching filter
//             mode cache: one partition's groups (24K entries, 768 KB) again
//             and again; stream: all groups once (1.5M, 48 MB).
//   semi      what a semi join (HashJoinBuilder::semi(), one int32 key)
//             should probe: VW_JOIN_SEMI's exact bitmap over the build keys'
//             range or the runtime::Hashmap path (probe hash pass, tagged
//             directory + chain, key-equality pass: Hashjoin's passes).
//             *_setup variants include the per-query structure set-up
//             (bitmap: allocate + zero + set bits; hash: Hashmap::setSize +
//             insert), as each query execution pays it. Shapes from TPC-H SF1
//             (q18 J1, q3 J1, q9 J2) and a density sweep.
//   bloom     VW_JOIN_BLOOM's filter kept on for the whole operator (no
//             per-vector adaptive switch) against the plain hash path, by
//             hit rate: whether "filter iff build >= MIN_KEYS" can be fixed
//             once per operator.
//   bloom_size the same by build size (256 .. 4M keys, 1% hits): where the
//             filter starts to pay, against the cache sizes printed first
//             (family cache, param = bytes); hit1_stream32 also streams 32 B
//             per probe row through the caches, as a real probe pipeline
//             does with its other columns
//   dispatch  per probe vector: today's decision chain in Hashjoin::next
//             (fusedReady, Bloom skip counter, join pointer compares) against
//             one member-function pointer resolved once. ns per vector.
//
// Usage: run_joindispatchbench [-f elide,semi,bloom,bloom_size,dispatch] [-r reps] [-v vec]
// CSV on stdout: family,variant,shape,param,mode,ns_per_row,check
#include "common/runtime/CacheInfo.hpp"
#include "common/runtime/Hash.hpp"
#include "common/runtime/Hashmap.hpp"
#include "vectorwise/Primitives.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_set>
#include <vector>

using namespace vectorwise;
using namespace vectorwise::primitives;
using runtime::Hashmap;
using hash_t = defs::hash_t;
static_assert(sizeof(pos_t) == 4, "32-bit pos_t expected");

namespace {

size_t vecSize = 1024;
int reps = 5;
std::string families = "all";
int fails = 0;
volatile uint64_t sink;

bool want(const char* f) {
   if (families == "all") return true;
   std::stringstream ss(families);
   for (std::string s; std::getline(ss, s, ',');)
      if (s == f) return true;
   return false;
}
uint64_t mix(uint64_t h, uint64_t v) { return (h ^ v) * 0x9E3779B97F4A7C15ull; }

template <typename F> double bestNs(F&& f) {
   double best = 1e300;
   for (int t = 0; t < reps; ++t) {
      const auto t0 = std::chrono::steady_clock::now();
      sink = f();
      const auto t1 = std::chrono::steady_clock::now();
      best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
   }
   return best;
}

struct Variant {
   const char* name;
   std::function<uint64_t(bool check)> run; // whole workload; check: checksum
};

void runFamily(const char* fam, const std::string& shape, const std::string& param,
               const char* mode, std::vector<Variant>& vs, double rows) {
   uint64_t ref = 0;
   for (size_t k = 0; k < vs.size(); ++k) {
      const uint64_t c = vs[k].run(true);
      if (k == 0) ref = c;
      const double ns = bestNs([&] { return vs[k].run(false); }) / rows;
      const bool ok = c == ref;
      fails += !ok;
      std::printf("%s,%s,%s,%s,%s,%.4f,%s\n", fam, vs[k].name, shape.c_str(), param.c_str(),
                  mode, ns, ok ? "ok" : "MISMATCH");
      std::fflush(stdout);
   }
}

inline hash_t buildHash(int32_t k) { return runtime::CRC32Hash()(k, primitives::seed); }

//=============================================================================
// elide: Q18's Select over the group-by output
//=============================================================================
constexpr size_t kStride = 32, kKeyOff = 16, kSumOff = 20;

struct Groups {
   std::vector<char> mem;
   size_t n;
   int64_t bound;
   char* at(size_t i) { return mem.data() + i * kStride; }
};

/// n group entries; pass = fraction passing sum > bound, or exactly `exact`
/// passing groups when exact > 0 (Q18: 57 of 1.5M)
Groups makeGroups(size_t n, double pass, size_t exact, std::mt19937_64& rng) {
   Groups g{std::vector<char>(n * kStride + 64), n, 300};
   for (size_t i = 0; i < n; ++i) {
      const int32_t key = int32_t(4 * i + 1); // o_orderkey-like
      int64_t sum = int64_t(rng() % 300);     // <= bound: filtered out
      if (!exact && double(rng() % 1000000) < pass * 1e6) sum = 301 + int64_t(rng() % 100);
      std::memcpy(g.at(i) + kKeyOff, &key, 4);
      std::memcpy(g.at(i) + kSumOff, &sum, 8);
   }
   for (size_t j = 0; j < exact; ++j) {
      const int64_t sum = 301 + int64_t(j);
      std::memcpy(g.at(rng() % n) + kSumOff, &sum, 8);
   }
   return g;
}

struct ElideBufs {
   std::vector<int32_t> kc;
   std::vector<int64_t> qc;
   std::vector<pos_t> sel;
   std::vector<hash_t> h;
   std::vector<int32_t> keys;
   explicit ElideBufs(size_t v) : kc(v), qc(v), sel(v), h(v), keys(v) {}
};

/// the survivors of one block: keys[0..m), h[0..m)
inline uint64_t emitSurvivors(ElideBufs& b, size_t m, bool check, uint64_t acc) {
   if (!check) return acc + m;
   for (size_t j = 0; j < m; ++j) acc = mix(mix(acc, uint64_t(uint32_t(b.keys[j]))), b.h[j]);
   return acc;
}

void famElide() {
   std::mt19937_64 rng(1);
   struct Shape { const char* name; double pass; size_t exact; };
   const Shape shapes[] = {{"q18", 0, 57},   {"p0.1", 0.001, 0}, {"p1", 0.01, 0},
                           {"p10", 0.1, 0}, {"p50", 0.5, 0},    {"p90", 0.9, 0}};
   for (const char* mode : {"cache", "stream"}) {
      const bool cache = !std::strcmp(mode, "cache");
      const size_t n = cache ? 24 * 1024 : 1500000;
      const int passes = cache ? 64 : 1; // cache: same partition again
      for (const auto& sh : shapes) {
         // q18 in cache mode: 57 of 1.5M scaled to the partition: 1 group
         Groups g = makeGroups(n, sh.pass, sh.exact ? (cache ? 1 : sh.exact) : 0, rng);
         ElideBufs b(vecSize);
         size_t stride = kStride;
         const int64_t bound = g.bound;
         auto blocks = [&](auto&& body, bool check) {
            uint64_t acc = 0;
            for (int p = 0; p < passes; ++p)
               for (size_t s = 0; s < n; s += vecSize)
                  acc = body(g.at(s), pos_t(std::min(vecSize, n - s)), check, acc);
            return acc;
         };
         std::vector<Variant> vs = {
             {"today", [&](bool check) {
                 return blocks([&](char* blk, pos_t m, bool ck, uint64_t acc) {
                    // gatherGroups: one GatherOpVal per group column
                    int32_t* kb = reinterpret_cast<int32_t*>(blk);
                    int64_t* qb = reinterpret_cast<int64_t*>(blk);
                    gather_val<int32_t>(m, &kb, kKeyOff, &stride, b.kc.data());
                    gather_val<int64_t>(m, &qb, kSumOff, &stride, b.qc.data());
                    // Select: sel_greater_int64_t_col_int64_t_val (BF)
                    int64_t bd = bound;
                    const pos_t c = sel_col_val_bf<int64_t, std::greater>(
                        m, b.sel.data(), b.qc.data(), &bd);
                    // join build: hash_sel_int32_t_col + scatter_sel key
                    hash_sel<int32_t, runtime::CRC32Hash>(c, b.sel.data(), b.h.data(), b.kc.data());
                    for (pos_t j = 0; j < c; ++j) b.keys[j] = b.kc[b.sel[j]];
                    return emitSurvivors(b, c, ck, acc);
                 }, check); }},
             {"lazy_key", [&](bool check) {
                 return blocks([&](char* blk, pos_t m, bool ck, uint64_t acc) {
                    int64_t* qb = reinterpret_cast<int64_t*>(blk);
                    gather_val<int64_t>(m, &qb, kSumOff, &stride, b.qc.data());
                    int64_t bd = bound;
                    const pos_t c = sel_col_val_bf<int64_t, std::greater>(
                        m, b.sel.data(), b.qc.data(), &bd);
                    for (pos_t j = 0; j < c; ++j) {
                       int32_t k;
                       std::memcpy(&k, blk + b.sel[j] * kStride + kKeyOff, 4);
                       b.keys[j] = k;
                       b.h[j] = buildHash(k);
                    }
                    return emitSurvivors(b, c, ck, acc);
                 }, check); }},
             {"fused_bf", [&](bool check) {
                 return blocks([&](char* blk, pos_t m, bool ck, uint64_t acc) {
                    pos_t* __restrict__ sel = b.sel.data();
                    pos_t c = 0;
                    for (pos_t i = 0; i < m; ++i) {
                       int64_t q;
                       std::memcpy(&q, blk + i * kStride + kSumOff, 8);
                       sel[c] = i;
                       c += q > bound;
                    }
                    for (pos_t j = 0; j < c; ++j) {
                       int32_t k;
                       std::memcpy(&k, blk + sel[j] * kStride + kKeyOff, 4);
                       b.keys[j] = k;
                       b.h[j] = buildHash(k);
                    }
                    return emitSurvivors(b, c, ck, acc);
                 }, check); }},
             {"fused_br", [&](bool check) {
                 return blocks([&](char* blk, pos_t m, bool ck, uint64_t acc) {
                    pos_t c = 0;
                    for (pos_t i = 0; i < m; ++i) {
                       int64_t q;
                       std::memcpy(&q, blk + i * kStride + kSumOff, 8);
                       if (q > bound) {
                          int32_t k;
                          std::memcpy(&k, blk + i * kStride + kKeyOff, 4);
                          b.keys[c] = k;
                          b.h[c++] = buildHash(k);
                       }
                    }
                    return emitSurvivors(b, c, ck, acc);
                 }, check); }},
         };
         runFamily("elide", sh.name, std::to_string(n), mode, vs, double(n) * passes);
      }
   }
}

//=============================================================================
// semi: exact bitmap vs runtime::Hashmap for a semi join's probe
//=============================================================================
struct SemiShape {
   std::string name;
   std::vector<int32_t> build; // distinct keys
   std::vector<int32_t> probe;
};

/// k distinct keys drawn with pick(), in random order
template <typename Pick>
std::vector<int32_t> distinctKeys(size_t k, Pick pick, std::mt19937_64& rng) {
   std::unordered_set<int32_t> seen;
   std::vector<int32_t> out;
   out.reserve(k);
   while (out.size() < k) {
      const int32_t v = pick();
      if (seen.insert(v).second) out.push_back(v);
   }
   std::shuffle(out.begin(), out.end(), rng);
   return out;
}

struct SEntry {
   Hashmap::EntryHeader h{nullptr, 0};
   int32_t key = 0;
};

/// VW_JOIN_SEMI's structure: bitmap over [lo, hi] of the build keys
struct Bitmap {
   std::unique_ptr<uint32_t[]> bits;
   int64_t lo = 0;
   uint64_t span = 0;
   void build(const std::vector<int32_t>& keys) {
      int64_t mn = INT64_MAX, mx = INT64_MIN;
      for (auto k : keys) mn = std::min<int64_t>(mn, k), mx = std::max<int64_t>(mx, k);
      lo = mn, span = uint64_t(mx - mn);
      bits.reset(new uint32_t[(span >> 5) + 1]()); // allocate + zero, as semiSetBits' setup
      for (auto k : keys) {
         const uint64_t d = uint64_t(int64_t(k) - lo);
         bits[d >> 5] |= uint32_t(1) << (d & 31);
      }
   }
   /// Operators.cpp semiProbe's scalar loop, dense probe keys
   size_t probe(const int32_t* k, size_t n, pos_t* out) const {
      size_t found = 0;
      for (size_t i = 0; i < n; ++i) {
         const uint64_t d = uint64_t(int64_t(k[i]) - lo);
         if (d <= span && (bits[d >> 5] >> (d & 31) & 1)) out[found++] = pos_t(i);
      }
      return found;
   }
};

/// the hash path: Hashmap::setSize + tagged inserts; probe as Hashjoin does
/// (probe hash pass, joinAllParallel-style first pass over the chains,
/// keys_equal pass)
struct HashSide {
   std::unique_ptr<Hashmap> ht;
   std::vector<SEntry> entries;
   std::vector<hash_t> hashes;
   std::vector<Hashmap::EntryHeader*> bm;
   std::vector<pos_t> pm;
   void build(const std::vector<int32_t>& keys) {
      ht.reset(new Hashmap());
      ht->setSize(keys.size());
      entries.assign(keys.size(), SEntry());
      for (size_t i = 0; i < keys.size(); ++i) {
         entries[i].key = keys[i];
         entries[i].h.hash = buildHash(keys[i]);
         ht->insert_tagged<false>(&entries[i].h, entries[i].h.hash);
      }
   }
   size_t probe(const int32_t* k, size_t n, pos_t* out) {
      if (hashes.size() < n) hashes.resize(n), bm.resize(n * 2), pm.resize(n * 2);
      for (size_t i = 0; i < n; ++i) hashes[i] = buildHash(k[i]);
      size_t c = 0;
      for (size_t i = 0; i < n; ++i)
         for (auto e = ht->find_chain_tagged(hashes[i]); e != Hashmap::end(); e = e->next)
            if (e->hash == hashes[i]) bm[c] = e, pm[c++] = pos_t(i);
      size_t found = 0;
      for (size_t j = 0; j < c; ++j) {
         out[found] = pm[j];
         found += reinterpret_cast<SEntry*>(bm[j])->key == k[pm[j]];
      }
      return found;
   }
};

void famSemi() {
   std::mt19937_64 rng(2);
   std::vector<SemiShape> shapes;
   auto distinctFrom = [&](size_t k, auto pick) { return distinctKeys(k, pick, rng); };
   {
      // Q18 J1: 57 order keys from the group-by, probe orders (1.5M, o_orderkey
      // order, sparse: 4 * i + 1)
      SemiShape s{"q18", {}, {}};
      for (size_t i = 0; i < 1500000; ++i) s.probe.push_back(int32_t(4 * i + 1));
      s.build = distinctFrom(57, [&] { return s.probe[rng() % s.probe.size()]; });
      shapes.push_back(std::move(s));
   }
   {
      // Q3 J1: 30K of 150K custkeys, probe o_custkey of 729K orders (random)
      SemiShape s{"q3", {}, {}};
      s.build = distinctFrom(30000, [&] { return int32_t(1 + rng() % 150000); });
      for (size_t i = 0; i < 729000; ++i) s.probe.push_back(int32_t(1 + rng() % 150000));
      shapes.push_back(std::move(s));
   }
   {
      // Q9 J2: 10.8K of 200K partkeys, probe ps_partkey (800K, sorted, 4 each)
      SemiShape s{"q9", {}, {}};
      s.build = distinctFrom(10800, [&] { return int32_t(1 + rng() % 200000); });
      for (size_t i = 0; i < 800000; ++i) s.probe.push_back(int32_t(1 + i / 4));
      shapes.push_back(std::move(s));
   }
   // density sweep: 4096 build keys over a growing range, 1M random probes;
   // bitmap 0.5 KiB .. 32 MiB (up to a whole unshared LLC on our machines)
   for (int64_t range : {4096ll, 65536ll, 1ll << 20, 1ll << 23, 1ll << 24, 1ll << 25,
                         1ll << 26, 1ll << 27, 3ll << 26, 1ll << 28}) {
      SemiShape s{"sweep_r" + std::to_string(range), {}, {}};
      s.build = distinctFrom(4096, [&] { return int32_t(rng() % range); });
      for (size_t i = 0; i < 1000000; ++i) s.probe.push_back(int32_t(rng() % range));
      shapes.push_back(std::move(s));
   }

   std::vector<pos_t> out(vecSize);
   for (auto& s : shapes) {
      Bitmap bmPre;
      bmPre.build(s.build);
      HashSide hsPre;
      hsPre.build(s.build);
      const size_t P = s.probe.size();
      auto probeAll = [&](auto& side, bool check) {
         uint64_t acc = 0;
         for (size_t off = 0; off < P; off += vecSize) {
            const size_t n = std::min(vecSize, P - off);
            const size_t f = side.probe(s.probe.data() + off, n, out.data());
            if (check)
               for (size_t j = 0; j < f; ++j) acc = mix(acc, off + out[j]);
            else
               acc += f;
         }
         return acc;
      };
      std::vector<Variant> vs = {
          {"bitmap", [&](bool ck) { return probeAll(bmPre, ck); }},
          {"bitmap_setup", [&](bool ck) {
              Bitmap b;
              b.build(s.build);
              return probeAll(b, ck); }},
          {"hash", [&](bool ck) { return probeAll(hsPre, ck); }},
          {"hash_setup", [&](bool ck) {
              HashSide h;
              h.build(s.build);
              return probeAll(h, ck); }},
      };
      char param[64];
      std::snprintf(param, sizeof param, "K%zu_bitmapKiB%llu", s.build.size(),
                    (unsigned long long)(((bmPre.span >> 5) + 1) * 4 / 1024));
      runFamily("semi", s.name, param, "probe", vs, double(P));
   }
}

//=============================================================================
// bloom: VW_JOIN_BLOOM's filter, always on, against the hash path, by hit rate
// (decides whether a once-per-operator rule can replace the filter's
// per-vector adaptive switch)
//=============================================================================
inline uint64_t bloomBits(hash_t h) {
   return (uint64_t(1) << (h & 63)) | (uint64_t(1) << ((h >> 6) & 63)) |
          (uint64_t(1) << ((h >> 12) & 63)) | (uint64_t(1) << ((h >> 18) & 63));
}

/// Operators.cpp bloomInsert / bloomFilter (scalar loop; VW_JOIN_BLOOM_BITS
/// 16), then the hash path's first pass and key check on the survivors only
struct BloomSide {
   HashSide hs;
   std::vector<uint64_t> bloom;
   uint64_t mask = 0;
   std::vector<pos_t> sel;
   std::vector<hash_t> sh;
   void build(const std::vector<int32_t>& keys) {
      hs.build(keys);
      size_t words = 1;
      while (words * 64 < keys.size() * 16) words <<= 1;
      bloom.assign(words, 0);
      mask = words - 1;
      for (auto& e : hs.entries) bloom[(e.h.hash >> 40) & mask] |= bloomBits(e.h.hash);
   }
   size_t probe(const int32_t* k, size_t n, pos_t* out) {
      if (sel.size() < n) sel.resize(n), sh.resize(n);
      if (hs.hashes.size() < n) hs.hashes.resize(n), hs.bm.resize(n * 2), hs.pm.resize(n * 2);
      for (size_t i = 0; i < n; ++i) hs.hashes[i] = buildHash(k[i]);
      size_t m = 0;
      for (size_t i = 0; i < n; ++i) {
         const hash_t h = hs.hashes[i];
         const uint64_t b = bloomBits(h);
         sel[m] = pos_t(i), sh[m] = h;
         m += (bloom[(h >> 40) & mask] & b) == b;
      }
      size_t c = 0;
      for (size_t j = 0; j < m; ++j)
         for (auto e = hs.ht->find_chain_tagged(sh[j]); e != Hashmap::end(); e = e->next)
            if (e->hash == sh[j]) hs.bm[c] = e, hs.pm[c++] = sel[j];
      size_t found = 0;
      for (size_t j = 0; j < c; ++j) {
         out[found] = hs.pm[j];
         found += reinterpret_cast<SEntry*>(hs.bm[j])->key == k[hs.pm[j]];
      }
      return found;
   }
};

void famBloom() {
   std::mt19937_64 rng(3);
   struct Shape { const char* name; size_t K; size_t P; double hit; };
   const Shape shapes[] = {
       {"q3j2", 146000, 2000000, 0.01},  {"q9j4", 43000, 2000000, 0.054},
       {"q9j5", 325000, 1500000, 0.20},  {"h50", 100000, 2000000, 0.50},
       {"h90", 100000, 2000000, 0.90},   {"fk100", 150000, 2000000, 1.0}};
   std::vector<pos_t> out(vecSize);
   for (const auto& sh : shapes) {
      // build keys: even numbers; misses: odd numbers (never in the build)
      const auto build = distinctKeys(sh.K, [&] { return int32_t(2 * (rng() % (1u << 29))); }, rng);
      std::vector<int32_t> probe(sh.P);
      for (auto& p : probe)
         p = double(rng() % 1000000) < sh.hit * 1e6 ? build[rng() % build.size()]
                                                     : int32_t(2 * (rng() % (1u << 29)) + 1);
      HashSide hs;
      hs.build(build);
      BloomSide bs;
      bs.build(build);
      auto probeAll = [&](auto& side, bool check) {
         uint64_t acc = 0;
         for (size_t off = 0; off < sh.P; off += vecSize) {
            const size_t n = std::min(vecSize, sh.P - off);
            const size_t f = side.probe(probe.data() + off, n, out.data());
            if (check)
               for (size_t j = 0; j < f; ++j) acc = mix(acc, off + out[j]);
            else
               acc += f;
         }
         return acc;
      };
      std::vector<Variant> vs = {
          {"hash", [&](bool ck) { return probeAll(hs, ck); }},
          {"bloom_hash", [&](bool ck) { return probeAll(bs, ck); }},
      };
      char param[64];
      std::snprintf(param, sizeof param, "K%zu_hit%g_filterKiB%zu", sh.K, sh.hit * 100,
                    bs.bloom.size() * 8 / 1024);
      runFamily("bloom", sh.name, param, "probe", vs, double(sh.P));
   }
}

//=============================================================================
// bloom_size: the same comparison by build size at a low hit rate (1%), to
// find where the filter starts to pay: the directory (8 B per slot) leaving
// a cache level. param carries the directory and filter sizes; compare with
// the cache rows at the top of the output (VW_JOIN_DISPATCH's rule: filter
// iff the directory is at least L1)
//=============================================================================
void famBloomSize() {
   std::mt19937_64 rng(5);
   const size_t P = 2000000;
   std::vector<pos_t> out(vecSize);
   // competing stream: the other columns a probe pipeline reads per row
   // (TPC-H Q9 J4's lineitem scan: ~32 B per row across its columns),
   // streamed through the caches alongside the probes; 64 MiB, so it never
   // stays resident
   constexpr size_t kStreamRowBytes = 32;
   std::vector<uint64_t> stream((64ull << 20) / 8, 1);
   volatile uint64_t streamSink = 0;
   for (size_t K : {256ul, 1024ul, 4096ul, 16384ul, 65536ul, 262144ul, 1048576ul,
                    4194304ul}) {
      const auto build = distinctKeys(K, [&] { return int32_t(2 * (rng() % (1u << 29))); }, rng);
      std::vector<int32_t> probe(P);
      for (auto& p : probe)
         p = rng() % 100 == 0 ? build[rng() % build.size()]
                              : int32_t(2 * (rng() % (1u << 29)) + 1);
      HashSide hs;
      hs.build(build);
      BloomSide bs;
      bs.build(build);
      auto probeAll = [&](auto& side, bool check, size_t rowBytes) {
         uint64_t acc = 0;
         const size_t words = stream.size();
         size_t pos = 0;
         for (size_t off = 0; off < P; off += vecSize) {
            const size_t n = std::min(vecSize, P - off);
            if (rowBytes) {
               // touch every cache line of this vector's share of the stream
               uint64_t s = 0;
               const size_t lines = n * rowBytes / 64;
               for (size_t l = 0; l < lines; ++l, pos = (pos + 8) % words) s += stream[pos];
               streamSink = streamSink + s;
            }
            const size_t f = side.probe(probe.data() + off, n, out.data());
            if (check)
               for (size_t j = 0; j < f; ++j) acc = mix(acc, off + out[j]);
            else
               acc += f;
         }
         return acc;
      };
      char param[96];
      std::snprintf(param, sizeof param, "K%zu_dirKiB%llu_filterKiB%zu", K,
                    (unsigned long long)((hs.ht->mask + 1) * 8 / 1024),
                    bs.bloom.size() * 8 / 1024);
      for (size_t rowBytes : {size_t(0), kStreamRowBytes}) {
         std::vector<Variant> vs = {
             {"hash", [&](bool ck) { return probeAll(hs, ck, rowBytes); }},
             {"bloom_hash", [&](bool ck) { return probeAll(bs, ck, rowBytes); }},
         };
         runFamily("bloom_size", rowBytes ? "hit1_stream32" : "hit1", param, "probe", vs,
                   double(P));
      }
   }
}

//=============================================================================
// dispatch: per-vector decision chain vs a resolved member-function pointer
//=============================================================================
struct FakeJoin {
   bool fusedOk = true;
   void* fusedDense = nullptr;
   void* fusedSel = this;
   size_t probeOps = 1;
   uint64_t* bloom = nullptr;
   uint32_t bloomSkip = 0;
   bool bloomOn = false, fusedCompute = false;
   uint64_t work = 0;
   size_t (FakeJoin::*join)() = &FakeJoin::joinA;
   void (FakeJoin::*probeStep)() = &FakeJoin::hashStep;
   bool fusedReady() const { return fusedOk && (fusedDense || fusedSel) && probeOps == 1; }
   __attribute__((noinline)) void hashStep() { work += 1; }
   __attribute__((noinline)) void filterStep() { work += 2; }
   __attribute__((noinline)) void fusedStep() { work += 3; }
   __attribute__((noinline)) size_t joinA() { return work & 7; }
   __attribute__((noinline)) size_t joinB() { return work & 3; }
   __attribute__((noinline)) size_t joinNew() { return work & 1; }
   /// Hashjoin::next's per-vector chain (VW_JOIN_FUSED_PROBE + VW_JOIN_BLOOM
   /// + VW_NEW_JOIN compiled in; fused, no filter, not the new join)
   size_t chain() {
      fusedCompute = false;
      if (fusedReady()) {
         bloomOn = false;
         if (bloom && !bloomSkip) {
            fusedStep();
            goto probe;
         }
         if (bloom) --bloomSkip;
         if (join == &FakeJoin::joinNew) {
            fusedCompute = true;
            goto probe;
         }
      }
      hashStep();
      bloomOn = false;
      if (bloom) {
         if (bloomSkip)
            --bloomSkip;
         else
            filterStep();
      }
   probe:
      return (this->*join)();
   }
   size_t resolved() {
      (this->*probeStep)();
      return (this->*join)();
   }
};

void famDispatch() {
   const size_t vectors = 1 << 22;
   FakeJoin j;
   std::vector<Variant> vs = {
       {"chain", [&](bool) {
           uint64_t a = 0;
           for (size_t v = 0; v < vectors; ++v) a += j.chain();
           return a + j.work; }},
       {"resolved", [&](bool) {
           uint64_t a = 0;
           for (size_t v = 0; v < vectors; ++v) a += j.resolved();
           return a + j.work; }},
   };
   // the step bodies mutate state, so outputs differ by design: no check
   for (auto& v : vs) {
      const double ns = bestNs([&] { return v.run(false); }) / double(vectors);
      std::printf("dispatch,%s,per_vector,-,-,%.4f,n/a\n", v.name, ns);
   }
}

} // namespace

int main(int argc, char** argv) {
   for (int opt; (opt = getopt(argc, argv, "f:r:v:")) != -1;) {
      if (opt == 'f') families = optarg;
      else if (opt == 'r') reps = std::max(1, std::atoi(optarg));
      else if (opt == 'v') vecSize = std::strtoull(optarg, nullptr, 10);
      else {
         std::fprintf(stderr, "usage: %s [-f elide,semi,bloom,bloom_size,dispatch] [-r reps] [-v vec]\n", argv[0]);
         return 2;
      }
   }
   std::printf("family,variant,shape,param,mode,ns_per_row,check\n");
   // this machine's caches (runtime::cacheBytes), param = bytes
   for (unsigned l = 1; l <= 3; ++l)
      std::printf("cache,L%u,cpu0,%zu,-,0,n/a\n", l, runtime::cacheBytes(l));
   if (want("elide")) famElide();
   if (want("semi")) famSemi();
   if (want("bloom")) famBloom();
   if (want("bloom_size")) famBloomSize();
   if (want("dispatch")) famDispatch();
   if (fails) std::fprintf(stderr, "run_joindispatchbench: %d MISMATCH rows\n", fails);
   return fails ? 1 : 0;
}
