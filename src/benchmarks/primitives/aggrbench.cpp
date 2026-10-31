// run_aggrbench: per-row cost of grouped SUM (aggr_sel_col) variants on
// Q1-shaped input, see SimdAggr.hpp.
//
// Aggregation side, one int64 column summed into every group per vector:
//   list_sel_scalar   VW_GROUP_AGGR_SEL today: acc += col[sel[i]] per group
//   list_pos_scalar   VW_GROUP_AGGR today:     acc += col[selParam1[pos[i]]]
//   scatter_scalar    no VW_GROUP_AGGR: *entries[i] += col[sel[i]] per row
//   list_sel_gather   B: AVX-512 gather over the group's sel list
//   list_pos_gather   B: AVX-512 gather over selParam1[pos[i]]
//   direct_scalar     D: acc[gid[r]] += col[r] over the vector's rows
//   direct_simd       D: masked AVX-512 accumulate per group (groups <= 8)
// Lookup side, the per-vector bookkeeping each approach needs from Lookup_T:
//   build_lists_sel   per-group pos + sel lists (VW_GROUP_AGGR_SEL)
//   build_lists_pos   per-group pos lists (VW_GROUP_AGGR)
//   build_gid         a group id per row (direct aggregation)
//
// Usage: run_aggrbench [-v vecSize] [-m l1|stream|all] [-r reps]
//                      [-g groups,...] [-s selPct,...] [-d q1|uniform]
//   -d q1 (default) uses Q1's group mix for 4 groups (A/F 25%, N/F 0.7%,
//   N/O 49%, R/F 25%) and uniform groups otherwise.
// Output: CSV on stdout; ns per selected row. Every variant's sums are
// checked against list_sel_scalar and a mismatch is reported in `ok`.
//
// Crossover mode: run_aggrbench -x [-g groups,..] [-a aggs,..] [-s selPct,..]
//                               [-f maxFill] [-d q1|uniform|zipf|runs] [-z skew]
//   HashGroup phase 1 per vector, swept over the number of distinct groups:
//   lookup_only   find/create the entry per row (shared by both, subtract it)
//   scatter       + aggr_sel_col without VW_GROUP_AGGR, once per aggregate
//   group_aggr    + VW_GROUP_AGGR: Group record per entry, size reset over all
//                 groups since the flush, pos append, per-group sum loop
//   Entries are 64 B-padded bump allocations (ResetableAllocator), the table
//   flushes at maxFill (default 1433, HashGroup's), -r counts vectors. Each
//   aggregate is its own 8 MB int64 column. Default groups 1..262144.
//   -d runs: keys ascending in runs of 1..7 rows (lineitem in l_orderkey
//   order, as Q3 and Q18 see it), wrapping at groups.
//   Output: impl,groups,dist,sel_pct,aggs,vec,max_fill,vecs_per_flush,
//           ns_per_selected_row,ok
//
// Partitioned mode: run_aggrbench -P [-g groups,..] [-a aggs,..] [-s selPct,..]
//                                    [-p partitions] [-G on|off|both]
//                                    [-d q1|uniform|zipf|runs] [-r vecs]
//   The whole group-by end to end: preagg, preagg_fused (VW_AGGR_FUSED),
//   partitioned (no pre-aggregation) and partitioned_opt (fused write and
//   merge); -G picks the phase 2 lookup (VW_GROUP_GLOBAL_DIRECT on/off).
//   -p is a power of two (HashGroup: 4 x threads, rounded up). aggs 1..8.
//   Output: impl,global_direct,groups,dist,sel_pct,aggs,vec,partitions,
//           spilled_per_row,phase1_ns,phase2_ns,total_ns,ok (ns per
//           selected input row; phase 2 includes the output walk)
#include "common/runtime/Hash.hpp"
#include "common/runtime/Hashmap.hpp"
#include "vectorwise/SimdAggr.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace vectorwise;
using namespace vectorwise::primitives::simd_aggr;

namespace {

size_t vecSize = 1024;
size_t reps = 2000;
std::string mode = "all";
std::string dist = "q1";
std::vector<unsigned> groupList = {1, 2, 4, 8, 16};
std::vector<int> selList = {100, 98, 50, 10};
constexpr size_t kStreamBytes = 64ull << 20;
constexpr size_t kPatterns = 64; // distinct selection/group patterns
volatile int64_t sink;

/// Lookup_T's output for one vector
struct Pattern {
   std::vector<pos_t> sel;                 // selected rows, ascending
   std::vector<uint8_t> gidSel;            // group per selected row
   std::vector<uint8_t> gid;               // group per row (kNoGroup)
   std::vector<std::vector<pos_t>> gpos, gsel;
};

std::vector<double> weights(unsigned groups) {
   if (dist == "q1" && groups == 4) return {0.25, 0.0066, 0.49, 0.2534};
   return std::vector<double>(groups, 1.0 / groups);
}

Pattern makePattern(unsigned groups, int selPct, std::mt19937_64& rng) {
   Pattern p;
   std::vector<double> w = weights(groups);
   std::discrete_distribution<unsigned> pickG(w.begin(), w.end());
   p.gid.assign(vecSize + 8, kNoGroup);
   p.gpos.resize(groups);
   p.gsel.resize(groups);
   for (size_t r = 0; r < vecSize; ++r) {
      if (int(rng() % 100) >= selPct) continue;
      const unsigned g = pickG(rng);
      p.gid[r] = uint8_t(g);
      p.gidSel.push_back(uint8_t(g));
      p.gpos[g].push_back(pos_t(p.sel.size()));
      p.gsel[g].push_back(pos_t(r));
      p.sel.push_back(pos_t(r));
   }
   return p;
}

void row(const char* side, const char* impl, unsigned groups, int selPct,
         const char* md, double ns, bool ok) {
   std::printf("%s,%s,%u,%s,%d,%s,%zu,%.4f,%s\n", side, impl, groups,
               dist.c_str(), selPct, md, vecSize, ns, ok ? "ok" : "MISMATCH");
   std::fflush(stdout);
}

/// fn(pattern, column base, acc[groups]) sums one vector
using AggrFn = std::function<void(const Pattern&, const int64_t*, int64_t*)>;

template <typename F> double bestNs(F&& body, size_t rowsDone) {
   double best = 1e300;
   for (int trial = 0; trial < 5; ++trial) {
      auto t0 = std::chrono::steady_clock::now();
      body();
      auto t1 = std::chrono::steady_clock::now();
      best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
   }
   return best / double(rowsDone);
}

void benchConfig(unsigned groups, int selPct) {
   std::mt19937_64 rng(groups * 1000 + selPct);
   std::vector<Pattern> pats;
   size_t selTotal = 0;
   for (size_t i = 0; i < kPatterns; ++i) {
      pats.push_back(makePattern(groups, selPct, rng));
      selTotal += pats.back().sel.size();
   }
   const double selPerVec = double(selTotal) / kPatterns;
   if (selPerVec == 0) return;
   const size_t streamVecs = kStreamBytes / sizeof(int64_t) / vecSize;
   std::vector<int64_t> col(std::max(streamVecs, size_t(1)) * vecSize + 8);
   for (auto& v : col) v = int64_t(rng() % 2000001) - 1000000;

   // aggregation variants
   std::vector<std::pair<const char*, AggrFn>> aggr;
   aggr.push_back({"list_sel_scalar", [groups](const Pattern& p, const int64_t* c, int64_t* acc) {
      for (unsigned g = 0; g < groups; ++g)
         acc[g] = sum_sel(p.gsel[g].data(), p.gsel[g].size(), c, acc[g]);
   }});
   aggr.push_back({"list_pos_scalar", [groups](const Pattern& p, const int64_t* c, int64_t* acc) {
      for (unsigned g = 0; g < groups; ++g)
         acc[g] = sum_pos(p.gpos[g].data(), p.gpos[g].size(), p.sel.data(), c, acc[g]);
   }});
   aggr.push_back({"scatter_scalar", [](const Pattern& p, const int64_t* c, int64_t* acc) {
      // entries[i] = the selected row's aggregate slot
      const size_t n = p.sel.size();
      const pos_t* sel = p.sel.data();
      const uint8_t* g = p.gidSel.data();
      for (size_t i = 0; i < n; ++i) acc[g[i]] += c[sel[i]];
   }});
   aggr.push_back({"direct_scalar", [](const Pattern& p, const int64_t* c, int64_t* acc) {
      sum_direct_scalar(p.gid.data(), vecSize, c, acc);
   }});
#ifdef VW_HAVE_SIMD_AGGR
   aggr.push_back({"list_sel_gather", [groups](const Pattern& p, const int64_t* c, int64_t* acc) {
      for (unsigned g = 0; g < groups; ++g)
         acc[g] = sum_sel_gather(p.gsel[g].data(), p.gsel[g].size(), c, acc[g]);
   }});
   aggr.push_back({"list_pos_gather", [groups](const Pattern& p, const int64_t* c, int64_t* acc) {
      for (unsigned g = 0; g < groups; ++g)
         acc[g] = sum_pos_gather(p.gpos[g].data(), p.gpos[g].size(), p.sel.data(), c, acc[g]);
   }});
   if (groups <= 8)
      aggr.push_back({"direct_simd", [groups](const Pattern& p, const int64_t* c, int64_t* acc) {
         sum_direct_simd_n(groups, p.gid.data(), vecSize, c, acc);
      }});
#endif

   // reference sums: list_sel_scalar over the first pattern / vector
   std::vector<int64_t> ref(groups, 0);
   aggr[0].second(pats[0], col.data(), ref.data());

   for (auto& [name, fn] : aggr) {
      std::vector<int64_t> acc(groups, 0);
      fn(pats[0], col.data(), acc.data());
      const bool ok = acc == ref;
      if (mode == "all" || mode == "l1") {
         std::vector<int64_t> a(groups, 0);
         double ns = bestNs([&] {
            for (size_t r = 0; r < reps; ++r) fn(pats[r % kPatterns], col.data(), a.data());
         }, size_t(reps * selPerVec));
         sink = a[0];
         row("aggr", name, groups, selPct, "l1", ns, ok);
      }
      if (mode == "all" || mode == "stream") {
         std::vector<int64_t> a(groups, 0);
         double ns = bestNs([&] {
            for (size_t v = 0; v < streamVecs; ++v)
               fn(pats[v % kPatterns], col.data() + v * vecSize, a.data());
         }, size_t(streamVecs * selPerVec));
         sink = a[0];
         row("aggr", name, groups, selPct, "stream", ns, ok);
      }
   }

   // lookup side: what each approach needs Lookup_T to write per vector
   std::vector<std::vector<pos_t>> lp(groups, std::vector<pos_t>(vecSize)),
       ls(groups, std::vector<pos_t>(vecSize));
   std::vector<pos_t> lsize(groups);
   std::vector<uint8_t> gid(vecSize + 8);
   auto buildSel = [&](const Pattern& p) {
      std::fill(lsize.begin(), lsize.end(), 0);
      for (size_t i = 0, n = p.sel.size(); i < n; ++i) {
         const unsigned g = p.gidSel[i];
         const pos_t k = lsize[g]++;
         lp[g][k] = pos_t(i);
         ls[g][k] = p.sel[i];
      }
   };
   auto buildPos = [&](const Pattern& p) {
      std::fill(lsize.begin(), lsize.end(), 0);
      for (size_t i = 0, n = p.sel.size(); i < n; ++i) {
         const unsigned g = p.gidSel[i];
         lp[g][lsize[g]++] = pos_t(i);
      }
   };
   auto buildGid = [&](const Pattern& p) {
      std::memset(gid.data(), kNoGroup, vecSize);
      for (size_t i = 0, n = p.sel.size(); i < n; ++i) gid[p.sel[i]] = p.gidSel[i];
   };
   std::pair<const char*, std::function<void(const Pattern&)>> builds[] = {
       {"build_lists_sel", buildSel}, {"build_lists_pos", buildPos}, {"build_gid", buildGid}};
   for (auto& [name, fn] : builds) {
      double ns = bestNs([&] {
         for (size_t r = 0; r < reps; ++r) {
            fn(pats[r % kPatterns]);
            asm volatile("" ::: "memory");
         }
      }, size_t(reps * selPerVec));
      sink = lsize[0] + gid[0];
      row("lookup", name, groups, selPct, "l1", ns, true);
   }
}

template <typename T> std::vector<T> parseList(const char* s) {
   std::vector<T> out;
   std::stringstream ss(s);
   std::string item;
   while (std::getline(ss, item, ',')) out.push_back(T(std::stol(item)));
   return out;
}

//--- crossover mode (-x) -----------------------------------------------------
// Baseline scatter vs VW_GROUP_AGGR as HashGroup phase 1 runs them, swept over
// the number of distinct groups. Unlike the kernels above, entries are real
// hash-entry-shaped allocations, the VW_GROUP_AGGR side pays its Group
// records, reset loop and walk over every group since the last flush, and the
// pre-aggregation table flushes at maxFill like flushAndClear.
namespace xover {

size_t maxFill = 1433; // HashGroup: ht.setSize(initialMapSize = 1024) * 0.7
double zipfS = 1.0;
std::vector<unsigned> aggList = {1, 4, 8};
constexpr size_t kColVecs = 1024; // rows per aggregate column / vecSize

/// runtime::Hashmap::Group under VW_GROUP_AGGR (without _SEL)
struct Group {
   pos_t pos[1024];
   pos_t size;
};

enum class Impl { LookupOnly, Scatter, GroupAggr };
const char* implName(Impl i) {
   switch (i) {
   case Impl::LookupOnly: return "lookup_only";
   case Impl::Scatter: return "scatter";
   default: return "group_aggr";
   }
}

/// ResetableAllocator: bump allocation, every allocation skips 1..64 bytes to
/// the next 64 B line; reset on flush (clearHashtable)
struct Arena {
   uint8_t* base = nullptr;
   uint8_t* start = nullptr;
   uint8_t* end = nullptr;
   explicit Arena(size_t bytes) {
      bytes = (bytes + 4095) & ~size_t(4095);
      base = static_cast<uint8_t*>(std::aligned_alloc(4096, bytes));
      if (!base) throw std::bad_alloc();
      std::memset(base, 0, bytes); // fault pages outside the timed region
      start = base, end = base + bytes;
   }
   ~Arena() { std::free(base); }
   uint8_t* allocate(size_t size) {
      const size_t aligndiff = 64 - (uintptr_t(start) % 64);
      uint8_t* alloc = start + aligndiff;
      start += size + aligndiff;
      if (start > end) throw std::runtime_error("arena exhausted");
      return alloc;
   }
   void reset() { start = base; }
};

/// Pre-aggregation entry: [next 8][hash 8]([Group* 8])[key 8][aggregate 8 x A].
/// The Group pointer exists only under VW_GROUP_AGGR (EntryHeader::group).
struct Table {
   size_t groupOff = 0, keyOff, aggOff, entrySize;
   unsigned aggs;
   bool withGroup;
   Arena arena;
   std::vector<uint8_t*> slot; // key -> entry, stands in for the hash lookup
   std::vector<uint8_t*> live; // allocations / VW_GROUP_AGGR's groups
   int64_t checksum = 0;       // stands in for spilling entries on flush
   size_t flushes = 0;

   Table(unsigned groups, unsigned aggs, bool withGroup)
       : aggs(aggs), withGroup(withGroup),
         arena((maxFill + vecSize) *
               (16 + 8 + 8 + 8 * aggs + 64 + (withGroup ? sizeof(Group) + 64 : 0))),
         slot(groups, nullptr) {
      size_t off = 16;
      if (withGroup) groupOff = off, off += 8;
      keyOff = off, off += 8;
      aggOff = off;
      entrySize = off + 8 * size_t(aggs);
      live.reserve(maxFill + vecSize);
   }
   Group*& group(uint8_t* e) const {
      return *reinterpret_cast<Group**>(e + groupOff);
   }
   uint32_t& key(uint8_t* e) const {
      return *reinterpret_cast<uint32_t*>(e + keyOff);
   }
   int64_t* agg(uint8_t* e) const {
      return reinterpret_cast<int64_t*>(e + aggOff);
   }
   uint8_t* create(uint32_t k) {
      uint8_t* e = arena.allocate(entrySize);
      reinterpret_cast<uint64_t*>(e)[0] = 0;                         // next
      reinterpret_cast<uint64_t*>(e)[1] = k * 0x9E3779B97F4A7C15ull; // hash
      key(e) = k;
      std::memset(agg(e), 0, 8 * size_t(aggs)); // aggr_init
      if (withGroup) {
         auto* g = reinterpret_cast<Group*>(arena.allocate(sizeof(Group)));
         g->size = 0;
         group(e) = g;
      }
      slot[k] = e;
      live.push_back(e);
      return e;
   }
   void flush() {
      for (uint8_t* e : live) {
         for (unsigned a = 0; a < aggs; ++a) checksum += agg(e)[a];
         slot[key(e)] = nullptr;
      }
      live.clear();
      arena.reset();
      ++flushes;
   }
};

/// aggr_sel_col without VW_GROUP_AGGR
__attribute__((noinline)) void aggrScatter(size_t n, uint8_t** __restrict__ entries,
                                           const pos_t* __restrict__ selParam1,
                                           const int64_t* __restrict__ param1,
                                           size_t offset) {
   for (size_t i = 0; i < n; i++) {
      int64_t* aggregate = reinterpret_cast<int64_t*>(entries[i] + offset);
      *aggregate += param1[selParam1[i]];
   }
}

/// aggr_sel_col under VW_GROUP_AGGR
__attribute__((noinline)) void aggrGroups(uint8_t* const* __restrict__ groups,
                                          size_t nGroups, size_t groupOff,
                                          const pos_t* __restrict__ selParam1,
                                          const int64_t* __restrict__ param1,
                                          size_t offset) {
   for (size_t j = 0; j < nGroups; j++) {
      uint8_t* entry = groups[j];
      Group* __restrict__ group = *reinterpret_cast<Group**>(entry + groupOff);
      pos_t size = group->size;
      if (size == 0) continue;
      pos_t* __restrict__ pos = group->pos;
      int64_t* __restrict__ aggregate = reinterpret_cast<int64_t*>(entry + offset);
      int64_t value = *aggregate;
      for (pos_t i = 0; i < size; i++) value += param1[selParam1[pos[i]]];
      *aggregate = value;
   }
}

/// Lookup_T's input: per vector, the selected rows and each one's group key
struct Stream {
   std::vector<uint32_t> off{0};
   std::vector<pos_t> sel;
   std::vector<uint32_t> key;
   size_t vecs() const { return off.size() - 1; }
};

Stream makeStream(unsigned groups, int selPct, std::mt19937_64& rng) {
   std::vector<double> w;
   if (dist == "q1" && groups == 4) w = {0.25, 0.0066, 0.49, 0.2534};
   else if (dist == "zipf")
      for (unsigned k = 1; k <= groups; ++k) w.push_back(1.0 / std::pow(double(k), zipfS));
   else if (dist != "runs") w.assign(groups, 1.0);
   if (w.empty()) w.assign(1, 1.0); // runs: pick is unused
   std::discrete_distribution<uint32_t> pick(w.begin(), w.end());
   // runs: lineitem in l_orderkey order, keys ascending (wrapping at groups),
   // each repeated 1..7 times uniformly like dbgen's lines per order
   const bool runs = dist == "runs";
   uint32_t runKey = 0, runLeft = 1 + rng() % 7;
   // enough vectors that every group shows up several times
   const size_t vecs = std::clamp<size_t>(8 * size_t(groups) / vecSize + 1, 256, 8192);
   Stream s;
   for (size_t v = 0; v < vecs; ++v) {
      for (size_t r = 0; r < vecSize; ++r) {
         uint32_t k;
         if (runs) {
            if (runLeft == 0) runKey = (runKey + 1) % groups, runLeft = 1 + rng() % 7;
            k = runKey, --runLeft;
         } else {
            k = pick(rng);
         }
         if (int(rng() % 100) >= selPct) continue; // after the run, like a filter
         s.sel.push_back(pos_t(r));
         s.key.push_back(k);
      }
      s.off.push_back(uint32_t(s.sel.size()));
   }
   return s;
}

/// One trial: `reps` vectors through lookup, aggregation, and flush.
/// Returns selected rows processed.
size_t runTrial(Impl impl, Table& t, const Stream& s,
                const std::vector<std::vector<int64_t>>& cols,
                std::vector<uint8_t*>& matches) {
   size_t rows = 0;
   t.checksum = 0, t.flushes = 0;
   for (size_t v = 0; v < reps; ++v) {
      const size_t sv = v % s.vecs();
      const size_t n = s.off[sv + 1] - s.off[sv];
      const pos_t* sel = s.sel.data() + s.off[sv];
      const uint32_t* keys = s.key.data() + s.off[sv];
      const size_t colBase = (v % kColVecs) * vecSize;
      rows += n;

      if (impl == Impl::GroupAggr)
         for (uint8_t* e : t.live) t.group(e)->size = 0;

      for (size_t i = 0; i < n; ++i) {
         const uint32_t k = keys[i];
         uint8_t* e = t.slot[k];
         if (e == nullptr || t.key(e) != k) e = t.create(k); // touches the entry, like the key compare
         if (impl == Impl::GroupAggr) {
            Group* g = t.group(e);
            g->pos[g->size++] = pos_t(i);
         } else {
            matches[i] = e;
         }
      }

      for (unsigned a = 0; a < t.aggs; ++a) {
         const int64_t* col = cols[a].data() + colBase;
         const size_t offset = t.aggOff + 8 * a;
         if (impl == Impl::Scatter)
            aggrScatter(n, matches.data(), sel, col, offset);
         else if (impl == Impl::GroupAggr)
            aggrGroups(t.live.data(), t.live.size(), t.groupOff, sel, col, offset);
      }

      if (t.live.size() >= maxFill) t.flush();
   }
   t.flush();
   return rows;
}

void benchConfig(unsigned groups, int selPct, unsigned aggs) {
   std::mt19937_64 rng(groups * 1000003ull + selPct * 101 + aggs);
   const Stream s = makeStream(groups, selPct, rng);
   std::vector<std::vector<int64_t>> cols(aggs, std::vector<int64_t>(kColVecs * vecSize));
   for (auto& c : cols)
      for (auto& v : c) v = int64_t(rng() % 2000001) - 1000000;

   int64_t expected = 0;
   for (size_t v = 0; v < reps; ++v) {
      const size_t sv = v % s.vecs();
      const size_t colBase = (v % kColVecs) * vecSize;
      for (size_t i = s.off[sv]; i < s.off[sv + 1]; ++i)
         for (unsigned a = 0; a < aggs; ++a) expected += cols[a][colBase + s.sel[i]];
   }

   std::vector<uint8_t*> matches(vecSize);
   for (Impl impl : {Impl::LookupOnly, Impl::Scatter, Impl::GroupAggr}) {
      Table t(groups, aggs, impl == Impl::GroupAggr);
      size_t rows = runTrial(impl, t, s, cols, matches); // warm-up
      const bool ok = impl == Impl::LookupOnly || t.checksum == expected;
      const double ns = bestNs([&] { rows = runTrial(impl, t, s, cols, matches); }, rows);
      sink = t.checksum;
      std::printf("%s,%u,%s,%d,%u,%zu,%zu,%.3f,%.4f,%s\n", implName(impl), groups,
                  dist.c_str(), selPct, aggs, vecSize, maxFill,
                  double(reps) / double(t.flushes), ns, ok ? "ok" : "MISMATCH");
      std::fflush(stdout);
   }
}

} // namespace xover

//--- partitioned mode (-P) ---------------------------------------------------
// The whole HashGroup pipeline on one thread, timed end to end: phase 1 into
// P hash partitions, phase 2 per partition into the global table, then a walk
// over every group (stands in for gatherGroups). Uses runtime::Hashmap and
// CRC32Hash as HashGroup does. Entry: [next][hash][key 8][aggregate 8 x K];
// a spilled row is the entry without next.
//   preagg           no options: local table (Lookup_T, one allocation and
//                    buildScatter per new group), K aggr_sel_col passes, spill
//                    on flush; phase 2 merges with K aggr_row passes
//   preagg_fused     the same, with VW_AGGR_FUSED's one-pass update
//   partitioned      no local table: every row is written to its partition,
//                    one pass per field (hash, key, each aggregate); phase 2
//                    merges with K aggr_row passes
//   partitioned_opt  one fused pass writes the row (VW_AGGR_FUSED's loop);
//                    phase 2 merges with one fused pass
// Phase 2 finds groups through globalFindOrCreate (VW_GROUP_GLOBAL_DIRECT,
// -G on) or findGroups + createMissingGroups (-G off), for every variant.
namespace part {

using runtime::Hashmap;
using Entry = Hashmap::EntryHeader;
using hash_t = Hashmap::hash_t;

size_t nrPartitions = 256;
std::string globalDirect = "both";
constexpr unsigned kMaxAggs = 8; // VW_AGGR_FUSED's kernel: 1..8 aggregates
constexpr size_t kKeyOff = 16;   // in the entry; in the row it is 8
constexpr size_t kAggOff = 24;   // in the entry; in the row it is 16

enum class Impl { Preagg, PreaggFused, Partitioned, PartitionedOpt };
const char* implName(Impl i) {
   switch (i) {
   case Impl::Preagg: return "preagg";
   case Impl::PreaggFused: return "preagg_fused";
   case Impl::Partitioned: return "partitioned";
   default: return "partitioned_opt";
   }
}

inline uint64_t& keyOf(Entry* e) {
   return *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(e) + kKeyOff);
}
inline int64_t* aggsOf(Entry* e) {
   return reinterpret_cast<int64_t*>(reinterpret_cast<uint8_t*>(e) + kAggOff);
}
inline hash_t rowHash(const uint8_t* r) {
   hash_t h;
   std::memcpy(&h, r, sizeof(h));
   return h;
}
inline uint64_t rowKey(const uint8_t* r) {
   uint64_t k;
   std::memcpy(&k, r + 8, sizeof(k));
   return k;
}
/// HashGroup's global table key: the partition bits are the hash's top bits,
/// so tag and slot come from the hash rotated right by 16
inline hash_t globalKey(hash_t h) {
   return (h >> 16) | (h << (sizeof(h) * 8 - 16));
}

/// ResetableAllocator: chunks of growing size, every allocation skips 1..64
/// bytes to the next 64 B line, reset() reuses the chunks from the first.
/// Chunks are faulted when created (the warm-up trial creates them all).
class ChunkArena {
   struct Chunk {
      uint8_t* base;
      size_t size;
   };
   std::vector<Chunk> chunks;
   size_t cur = 0, nextSize = 2 << 20;
   uint8_t* start = nullptr;
   uint8_t* end = nullptr;

 public:
   ChunkArena() = default;
   ChunkArena(const ChunkArena&) = delete;
   ~ChunkArena() {
      for (auto& c : chunks) std::free(c.base);
   }
   void* allocate(size_t size) {
      if (start) {
         uint8_t* a = start + (64 - uintptr_t(start) % 64);
         if (a + size <= end) return start = a + size, a;
      }
      size_t i = start ? cur + 1 : 0;
      while (i < chunks.size() && chunks[i].size < size + 64) ++i;
      if (i == chunks.size()) {
         const size_t s = std::max(nextSize, size + 64);
         nextSize = s * 2;
         auto* b = static_cast<uint8_t*>(std::malloc(s));
         if (!b) throw std::bad_alloc();
         std::memset(b, 0, s);
         chunks.push_back({b, s});
      }
      cur = i, start = chunks[i].base, end = start + chunks[i].size;
      uint8_t* a = start + (64 - uintptr_t(start) % 64);
      return start = a + size, a;
   }
   void reset() { start = nullptr; }
};

/// PartitionedDeque<1024>: per partition a list of 1024-row chunks, the
/// partition is the hash's top bits
class Deque {
 public:
   static constexpr size_t kChunkRows = 1024;
   struct Chunk {
      Chunk* next;
      uint64_t pad;
      uint8_t* data() { return reinterpret_cast<uint8_t*>(this + 1); }
   };
   struct Partition {
      Chunk* first = nullptr;
      Chunk* last = nullptr;
      uint8_t* current = nullptr;
      uint8_t* end = nullptr;
   };
   std::vector<Partition> parts;
   size_t rowSize;
   unsigned shift;

   Deque(size_t nParts, size_t rowSize)
       : parts(nParts), rowSize(rowSize),
         shift(unsigned(sizeof(hash_t) * 8) - unsigned(__builtin_ctzll(nParts))) {}
   /// the slot the next row of hash h's partition goes to
   uint8_t* slot(hash_t h) {
      Partition& p = parts[h >> shift];
      if (p.current == p.end) grow(p);
      uint8_t* s = p.current;
      p.current += rowSize;
      return s;
   }
   /// Partition::push_back without VW_SPILL_WORD_COPY: memcpy, run-time size
   void push_back(const void* row, hash_t h) { std::memcpy(slot(h), row, rowSize); }
   size_t rows(const Partition& p, Chunk* c) const {
      return c == p.last ? size_t(p.current - c->data()) / rowSize : kChunkRows;
   }
   void clear() {
      for (auto& p : parts) p = Partition();
      store.reset();
   }

 private:
   ChunkArena store;
   void grow(Partition& p) {
      auto* c = static_cast<Chunk*>(store.allocate(sizeof(Chunk) + kChunkRows * rowSize));
      c->next = nullptr;
      if (p.first) p.last->next = c;
      else p.first = c;
      p.last = c;
      p.current = c->data();
      p.end = p.current + kChunkRows * rowSize;
   }
};

//--- kernels: one function per primitive call, as the interpreter runs them --

/// Hash(n) with VW_USE_CRC32
__attribute__((noinline)) void hashKeys(size_t n, const uint32_t* __restrict__ keys,
                                        hash_t* __restrict__ hashes) {
   runtime::CRC32Hash hasher;
   for (size_t i = 0; i < n; i++) hashes[i] = hasher.hashKey(keys[i], 0);
}

/// aggr_sel_col, one aggregate
__attribute__((noinline)) void aggrSelCol(size_t n, Entry** __restrict__ entries,
                                          const pos_t* __restrict__ sel,
                                          const int64_t* __restrict__ col, size_t offset) {
   for (size_t i = 0; i < n; i++) {
      auto* aggregate = reinterpret_cast<int64_t*>(reinterpret_cast<uint8_t*>(entries[i]) + offset);
      *aggregate += col[sel[i]];
   }
}

/// VW_AGGR_FUSED's aggrFusedBlock<K>: every aggregate as col[d][sel[d][i]]
template <unsigned K>
__attribute__((noinline)) void aggrFused(size_t n, Entry** __restrict__ entries,
                                         const pos_t* const* sels,
                                         const int64_t* const* cols) {
   const pos_t* s[K];
   const int64_t* c[K];
   for (unsigned d = 0; d < K; ++d) s[d] = sels[d], c[d] = cols[d];
   for (size_t i = 0; i < n; i++) {
      int64_t* e = aggsOf(entries[i]);
      for (unsigned d = 0; d < K; ++d) e[d] += c[d][s[d][i]];
   }
}

/// aggr_row, one aggregate: add a spilled row's partial into its entry
__attribute__((noinline)) void aggrRow(size_t n, Entry** __restrict__ entries,
                                       size_t offset, const uint8_t* __restrict__ data,
                                       size_t inSize, size_t inOffset) {
   data += inOffset;
   for (size_t i = 0; i < n; i++, data += inSize) {
      auto* aggregate = reinterpret_cast<int64_t*>(reinterpret_cast<uint8_t*>(entries[i]) + offset);
      int64_t v;
      std::memcpy(&v, data, sizeof(v));
      *aggregate += v;
   }
}

/// fused merge: all K partials of a row into its entry in one pass
template <unsigned K>
__attribute__((noinline)) void mergeFused(size_t n, Entry** __restrict__ entries,
                                          const uint8_t* __restrict__ data, size_t rowSize) {
   for (size_t i = 0; i < n; i++) {
      int64_t* e = aggsOf(entries[i]);
      const uint8_t* r = data + i * rowSize + 16;
      for (unsigned d = 0; d < K; ++d) {
         int64_t v;
         std::memcpy(&v, r + 8 * d, sizeof(v));
         e[d] += v;
      }
   }
}

/// partitioned: one scatter pass per row field into the partition slots
__attribute__((noinline)) void scatterHash(size_t n, uint8_t** __restrict__ dst,
                                           const hash_t* __restrict__ hashes) {
   for (size_t i = 0; i < n; i++) std::memcpy(dst[i], &hashes[i], sizeof(hash_t));
}
__attribute__((noinline)) void scatterKey(size_t n, uint8_t** __restrict__ dst,
                                          const uint32_t* __restrict__ keys) {
   for (size_t i = 0; i < n; i++) {
      const uint64_t k = keys[i];
      std::memcpy(dst[i] + 8, &k, sizeof(k));
   }
}
__attribute__((noinline)) void scatterSelCol(size_t n, uint8_t** __restrict__ dst,
                                             const pos_t* __restrict__ sel,
                                             const int64_t* __restrict__ col, size_t offset) {
   for (size_t i = 0; i < n; i++) std::memcpy(dst[i] + offset, &col[sel[i]], sizeof(int64_t));
}

/// partitioned_opt: the whole row in one pass, aggregates read as in
/// aggrFusedBlock
template <unsigned K>
__attribute__((noinline)) void writeFused(size_t n, uint8_t** __restrict__ dst,
                                          const hash_t* __restrict__ hashes,
                                          const uint32_t* __restrict__ keys,
                                          const pos_t* const* sels,
                                          const int64_t* const* cols) {
   const pos_t* s[K];
   const int64_t* c[K];
   for (unsigned d = 0; d < K; ++d) s[d] = sels[d], c[d] = cols[d];
   for (size_t i = 0; i < n; i++) {
      uint8_t* r = dst[i];
      const uint64_t k = keys[i];
      std::memcpy(r, &hashes[i], sizeof(hash_t));
      std::memcpy(r + 8, &k, sizeof(k));
      for (unsigned d = 0; d < K; ++d) std::memcpy(r + 16 + 8 * d, &c[d][s[d][i]], sizeof(int64_t));
   }
}

using AggrFusedFn = void (*)(size_t, Entry**, const pos_t* const*, const int64_t* const*);
using MergeFusedFn = void (*)(size_t, Entry**, const uint8_t*, size_t);
using WriteFusedFn = void (*)(size_t, uint8_t**, const hash_t*, const uint32_t*,
                              const pos_t* const*, const int64_t* const*);
template <unsigned... K> struct FusedTable {
   static constexpr AggrFusedFn aggr[] = {&aggrFused<K>...};
   static constexpr MergeFusedFn merge[] = {&mergeFused<K>...};
   static constexpr WriteFusedFn write[] = {&writeFused<K>...};
};
using Fused = FusedTable<1, 2, 3, 4, 5, 6, 7, 8>; // index K - 1

//--- the operator ------------------------------------------------------------

class Pipeline {
   const Impl impl;
   const bool direct;
   const unsigned K;
   const size_t entrySize, rowSize;
   Hashmap ht;
   size_t maxFill = 0;
   ChunkArena groupStore;
   std::vector<std::pair<uint8_t*, size_t>> allocations;
   size_t entriesInHt = 0;
   Deque spill;
   std::vector<hash_t> hashes;
   std::vector<Entry*> matches;
   std::vector<uint8_t*> dst;
   std::vector<const pos_t*> sels;
   std::vector<const int64_t*> colPtrs;
   // findGroups / createMissingGroups
   std::vector<pos_t> found, notFound, keysNEq, reps, groupOf;
   std::vector<int32_t> smallHt; // HashmapSmall stand-in: slot -> new group

 public:
   uint64_t checksum = 0;
   size_t groupsOut = 0, spilledRows = 0;

   Pipeline(Impl impl, bool direct, unsigned aggs)
       : impl(impl), direct(direct), K(aggs), entrySize(kAggOff + 8 * size_t(aggs)),
         rowSize(entrySize - sizeof(Entry::next)), spill(nrPartitions, rowSize),
         hashes(vecSize), matches(vecSize), dst(vecSize), sels(aggs), colPtrs(aggs),
         found(vecSize), notFound(vecSize), keysNEq(vecSize), reps(vecSize),
         groupOf(vecSize), smallHt(4 * vecSize, -1) {}

   /// a fresh operator: HashGroup's constructor sizes the table for 1024
   void reset() {
      maxFill = ht.setSize(1024);
      checksum = 0, groupsOut = 0, spilledRows = 0;
   }

   /// phase 1, one vector of n selected rows
   void consume(size_t n, const pos_t* sel, const uint32_t* keys,
                const std::vector<const int64_t*>& cols) {
      for (unsigned d = 0; d < K; ++d) sels[d] = sel, colPtrs[d] = cols[d];
      hashKeys(n, keys, hashes.data());
      if (impl == Impl::Partitioned || impl == Impl::PartitionedOpt) {
         for (size_t i = 0; i < n; i++) dst[i] = spill.slot(hashes[i]);
         spilledRows += n;
         if (impl == Impl::PartitionedOpt) {
            Fused::write[K - 1](n, dst.data(), hashes.data(), keys, sels.data(), colPtrs.data());
         } else {
            scatterHash(n, dst.data(), hashes.data());
            scatterKey(n, dst.data(), keys);
            for (unsigned d = 0; d < K; ++d)
               scatterSelCol(n, dst.data(), sel, cols[d], 16 + 8 * d);
         }
         return;
      }
      lookupLocal(n, keys);
      if (impl == Impl::PreaggFused) {
         Fused::aggr[K - 1](n, matches.data(), sels.data(), colPtrs.data());
      } else {
         for (unsigned d = 0; d < K; ++d)
            aggrSelCol(n, matches.data(), sel, cols[d], kAggOff + 8 * d);
      }
      if (entriesInHt >= maxFill) flushAndClear();
   }

   /// end of phase 1
   void finishLocal() {
      if (impl == Impl::Preagg || impl == Impl::PreaggFused) flushAndClear();
   }

   /// phase 2 and the output walk
   void aggregatePartitions() {
      for (auto& p : spill.parts) {
         for (auto* c = p.first; c; c = c->next) {
            const size_t rows = spill.rows(p, c);
            for (size_t pos = 0; pos < rows; pos += vecSize) {
               const size_t n = std::min(vecSize, rows - pos);
               const uint8_t* data = c->data() + pos * rowSize;
               if (direct) {
                  globalFindOrCreate(data, n);
               } else {
                  findGroups(data, n);
                  createMissingGroups(data, n);
               }
               if (impl == Impl::PartitionedOpt) {
                  Fused::merge[K - 1](n, matches.data(), data, rowSize);
               } else {
                  for (unsigned d = 0; d < K; ++d)
                     aggrRow(n, matches.data(), kAggOff + 8 * d, data, rowSize, 16 + 8 * d);
               }
            }
         }
         // gatherGroups stand-in: read every group once
         for (auto& [block, cnt] : allocations)
            for (size_t j = 0; j < cnt; ++j) {
               auto* e = reinterpret_cast<Entry*>(block + j * entrySize);
               const uint64_t k = keyOf(e);
               for (unsigned d = 0; d < K; ++d)
                  checksum += uint64_t(aggsOf(e)[d]) * (k + 1 + d);
               ++groupsOut;
            }
         clearHashtable();
      }
      spill.clear();
   }

 private:
   void clearHashtable() {
      ht.clear();
      entriesInHt = 0;
      groupStore.reset();
      allocations.clear();
   }

   /// buildScatter for new entries: hash, key, then each aggregate's
   /// neutral value, one pass each
   void initEntries(uint8_t* block, size_t cnt, const uint8_t* data, const pos_t* rows) {
      for (size_t j = 0; j < cnt; ++j) {
         auto* e = reinterpret_cast<Entry*>(block + j * entrySize);
         e->hash = rowHash(data + size_t(rows[j]) * rowSize);
      }
      for (size_t j = 0; j < cnt; ++j)
         keyOf(reinterpret_cast<Entry*>(block + j * entrySize)) =
             rowKey(data + size_t(rows[j]) * rowSize);
      for (unsigned d = 0; d < K; ++d)
         for (size_t j = 0; j < cnt; ++j)
            aggsOf(reinterpret_cast<Entry*>(block + j * entrySize))[d] = 0;
   }

   /// Lookup_T without VW_GROUP_BATCH_CREATE
   void lookupLocal(size_t n, const uint32_t* keys) {
      for (size_t i = 0; i < n; i++) {
         const hash_t h = hashes[i];
         Entry* e = ht.find_chain(h);
         for (; e != nullptr; e = e->next)
            if (e->hash == h && keyOf(e) == keys[i]) break;
         if (!e) {
            auto* block = static_cast<uint8_t*>(groupStore.allocate(entrySize));
            allocations.emplace_back(block, 1);
            e = reinterpret_cast<Entry*>(block);
            // buildScatter.evaluate(1)
            e->hash = h;
            keyOf(e) = keys[i];
            for (unsigned d = 0; d < K; ++d) aggsOf(e)[d] = 0;
            ht.insert<false>(e, h);
            ++entriesInHt;
         }
         matches[i] = e;
      }
   }

   void flushAndClear() {
      for (auto& [block, cnt] : allocations)
         for (size_t j = 0; j < cnt; ++j) {
            auto* e = reinterpret_cast<Entry*>(block + j * entrySize);
            spill.push_back(&e->hash, e->hash);
         }
      for (auto& a : allocations) spilledRows += a.second;
      clearHashtable();
   }

   /// VW_GROUP_GLOBAL_DIRECT
   void globalFindOrCreate(const uint8_t* data, size_t n) {
      for (size_t i = 0; i < n; ++i) ht.prefetch_slot(globalKey(rowHash(data + i * rowSize)));
      uint8_t* block = nullptr;
      size_t created = 0;
      for (size_t i = 0; i < n; ++i) {
         const uint8_t* row = data + i * rowSize;
         const hash_t h = rowHash(row);
         const uint64_t k = rowKey(row);
         Entry* e = ht.find_chain_tagged(globalKey(h));
         for (; e != nullptr; e = e->next)
            if (e->hash == h && keyOf(e) == k) break;
         if (!e) {
            if (!block) block = static_cast<uint8_t*>(groupStore.allocate(n * entrySize));
            e = reinterpret_cast<Entry*>(block + created * entrySize);
            e->hash = h;
            keyOf(e) = k;
            ht.insert_tagged<false>(e, globalKey(h));
            reps[created++] = pos_t(i);
         }
         matches[i] = e;
      }
      if (!created) return;
      allocations.emplace_back(block, created);
      entriesInHt += created;
      initEntries(block, created, data, reps.data());
      if (entriesInHt > maxFill) {
         // as createMissingGroups(allowResize): grow and reinsert all entries
         maxFill = ht.setSize(entriesInHt * 2);
         for (auto& [b, cnt] : allocations)
            for (size_t j = 0; j < cnt; ++j) {
               auto* x = reinterpret_cast<Entry*>(b + j * entrySize);
               ht.insert_tagged<false>(x, globalKey(x->hash));
            }
      }
   }

   /// findGroups: htLookup (hash only), keyEquality, htFollow
   void findGroups(const uint8_t* data, size_t n) {
      size_t nFound = 0, nNotFound = 0, nNeq = 0;
      for (size_t i = 0; i < n; ++i) {
         const hash_t h = rowHash(data + i * rowSize);
         Entry* e = ht.find_chain(h);
         while (e != nullptr && e->hash != h) e = e->next;
         if (e) matches[i] = e, found[nFound++] = pos_t(i);
         else notFound[nNotFound++] = pos_t(i);
      }
      for (size_t j = 0; j < nFound; ++j) {
         const size_t i = found[j];
         if (keyOf(matches[i]) != rowKey(data + i * rowSize)) keysNEq[nNeq++] = pos_t(i);
      }
      for (size_t j = 0; j < nNeq; ++j) {
         const size_t i = keysNEq[j];
         const uint8_t* row = data + i * rowSize;
         Entry* e = matches[i]->next;
         while (e != nullptr && !(e->hash == rowHash(row) && keyOf(e) == rowKey(row))) e = e->next;
         if (e) matches[i] = e;
         else notFound[nNotFound++] = pos_t(i);
      }
      missing = nNotFound;
   }

   /// createMissingGroups(allowResize): one group per distinct missing key
   /// (partition_by_key_row through a small hash table), one block, one
   /// buildScatter, insert or grow and reinsert, then htMatches
   void createMissingGroups(const uint8_t* data, size_t) {
      if (!missing) return;
      const size_t mask = smallHt.size() - 1;
      size_t groups = 0;
      for (size_t j = 0; j < missing; ++j) {
         const uint8_t* row = data + size_t(notFound[j]) * rowSize;
         const uint64_t k = rowKey(row);
         size_t s = rowHash(row) & mask;
         for (;; s = (s + 1) & mask) {
            const int32_t g = smallHt[s];
            if (g < 0) {
               smallHt[s] = int32_t(groups);
               reps[groups] = notFound[j];
               groupOf[j] = pos_t(groups++);
               break;
            }
            if (rowKey(data + size_t(reps[g]) * rowSize) == k) {
               groupOf[j] = pos_t(g);
               break;
            }
         }
      }
      // clear the slots used
      for (size_t g = 0; g < groups; ++g) {
         size_t s = rowHash(data + size_t(reps[g]) * rowSize) & mask;
         while (smallHt[s] >= 0) smallHt[s] = -1, s = (s + 1) & mask;
      }
      entriesInHt += groups;
      auto* block = static_cast<uint8_t*>(groupStore.allocate(groups * entrySize));
      allocations.emplace_back(block, groups);
      initEntries(block, groups, data, reps.data());
      if (entriesInHt > maxFill) {
         maxFill = ht.setSize(entriesInHt * 2);
         for (auto& [b, cnt] : allocations)
            ht.insertAll<false>(reinterpret_cast<Entry*>(b), cnt, entrySize);
      } else {
         ht.insertAll<false>(reinterpret_cast<Entry*>(block), groups, entrySize);
      }
      for (size_t j = 0; j < missing; ++j)
         matches[notFound[j]] = reinterpret_cast<Entry*>(block + size_t(groupOf[j]) * entrySize);
   }
   size_t missing = 0;
};

void benchConfig(unsigned groups, int selPct, unsigned aggs) {
   std::mt19937_64 rng(groups * 1000003ull + selPct * 101 + aggs);
   const xover::Stream s = xover::makeStream(groups, selPct, rng);
   std::vector<std::vector<int64_t>> cols(aggs, std::vector<int64_t>(xover::kColVecs * vecSize));
   for (auto& c : cols)
      for (auto& v : c) v = int64_t(rng() % 2000001) - 1000000;

   // expected result: sum over groups of aggregate d times (key + 1 + d)
   uint64_t expected = 0;
   size_t inputRows = 0;
   std::vector<bool> seen(groups);
   for (size_t v = 0; v < reps; ++v) {
      const size_t sv = v % s.vecs();
      const size_t colBase = (v % xover::kColVecs) * vecSize;
      for (size_t i = s.off[sv]; i < s.off[sv + 1]; ++i) {
         seen[s.key[i]] = true;
         for (unsigned d = 0; d < aggs; ++d)
            expected += uint64_t(cols[d][colBase + s.sel[i]]) * (uint64_t(s.key[i]) + 1 + d);
      }
      inputRows += s.off[sv + 1] - s.off[sv];
   }
   if (inputRows == 0) return;
   const size_t distinct = size_t(std::count(seen.begin(), seen.end(), true));

   std::vector<bool> modes;
   if (globalDirect != "off") modes.push_back(true);
   if (globalDirect != "on") modes.push_back(false);
   std::vector<const int64_t*> colv(aggs);

   for (bool direct : modes)
      for (Impl impl : {Impl::Preagg, Impl::PreaggFused, Impl::Partitioned, Impl::PartitionedOpt}) {
         Pipeline p(impl, direct, aggs);
         double best1 = 0, best2 = 0;
         bool ok = true;
         for (int trial = 0; trial < 6; ++trial) { // trial 0 warms up
            p.reset();
            auto t0 = std::chrono::steady_clock::now();
            for (size_t v = 0; v < reps; ++v) {
               const size_t sv = v % s.vecs();
               const size_t colBase = (v % xover::kColVecs) * vecSize;
               for (unsigned d = 0; d < aggs; ++d) colv[d] = cols[d].data() + colBase;
               p.consume(s.off[sv + 1] - s.off[sv], s.sel.data() + s.off[sv],
                         s.key.data() + s.off[sv], colv);
            }
            p.finishLocal();
            auto t1 = std::chrono::steady_clock::now();
            p.aggregatePartitions();
            auto t2 = std::chrono::steady_clock::now();
            ok = ok && p.checksum == expected && p.groupsOut == distinct;
            const double d1 = std::chrono::duration<double, std::nano>(t1 - t0).count();
            const double d2 = std::chrono::duration<double, std::nano>(t2 - t1).count();
            if (trial == 1 || (trial > 1 && d1 + d2 < best1 + best2)) best1 = d1, best2 = d2;
         }
         sink = int64_t(p.checksum);
         const double r = double(inputRows);
         std::printf("%s,%s,%u,%s,%d,%u,%zu,%zu,%.4f,%.4f,%.4f,%.4f,%s\n", implName(impl),
                     direct ? "on" : "off", groups, dist.c_str(), selPct, aggs, vecSize,
                     nrPartitions, double(p.spilledRows) / r, best1 / r, best2 / r,
                     (best1 + best2) / r, ok ? "ok" : "MISMATCH");
         std::fflush(stdout);
      }
}

} // namespace part

} // namespace

int main(int argc, char** argv) {
   int opt;
   bool crossover = false, partitioned = false, groupsSet = false;
   while ((opt = getopt(argc, argv, "v:m:r:g:s:d:xa:f:z:Pp:G:")) != -1) {
      switch (opt) {
      case 'v': vecSize = std::strtoull(optarg, nullptr, 10); break;
      case 'm': mode = optarg; break;
      case 'r': reps = std::strtoull(optarg, nullptr, 10); break;
      case 'g': groupList = parseList<unsigned>(optarg), groupsSet = true; break;
      case 's': selList = parseList<int>(optarg); break;
      case 'd': dist = optarg; break;
      case 'x': crossover = true; break;
      case 'a': xover::aggList = parseList<unsigned>(optarg); break;
      case 'f': xover::maxFill = std::strtoull(optarg, nullptr, 10); break;
      case 'z': xover::zipfS = std::strtod(optarg, nullptr); break;
      case 'P': partitioned = true; break;
      case 'p': part::nrPartitions = std::strtoull(optarg, nullptr, 10); break;
      case 'G': part::globalDirect = optarg; break;
      default:
         std::fprintf(stderr,
                      "usage: %s [-v vec] [-m l1|stream|all] [-r reps] [-g groups,..] "
                      "[-s selPct,..] [-d q1|uniform]\n"
                      "       %s -x [-g groups,..] [-a aggs,..] [-s selPct,..] "
                      "[-f maxFill] [-d q1|uniform|zipf|runs] [-z skew] [-r vecs] [-v vec]\n"
                      "       %s -P [-g groups,..] [-a aggs,..] [-s selPct,..] [-p partitions] "
                      "[-G on|off|both] [-d q1|uniform|zipf|runs] [-z skew] [-r vecs] [-v vec]\n",
                      argv[0], argv[0], argv[0]);
         return 1;
      }
   }
   if (partitioned) {
      const size_t p = part::nrPartitions;
      if (p < 2 || (p & (p - 1))) {
         std::fprintf(stderr, "-p: partitions must be a power of two >= 2\n");
         return 1;
      }
      if (part::globalDirect != "on" && part::globalDirect != "off" &&
          part::globalDirect != "both") {
         std::fprintf(stderr, "-G: on, off or both\n");
         return 1;
      }
      for (unsigned a : xover::aggList)
         if (a == 0 || a > part::kMaxAggs) {
            std::fprintf(stderr, "-P: aggs must be 1..%u\n", part::kMaxAggs);
            return 1;
         }
      if (!groupsSet)
         groupList = {1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1500000};
      for (unsigned g : groupList)
         if (g == 0) {
            std::fprintf(stderr, "groups must be >= 1\n");
            return 1;
         }
      std::printf("impl,global_direct,groups,dist,sel_pct,aggs,vec,partitions,"
                  "spilled_per_row,phase1_ns,phase2_ns,total_ns,ok\n");
      for (unsigned a : xover::aggList)
         for (int s : selList)
            for (unsigned g : groupList) part::benchConfig(g, s, a);
      return 0;
   }
   if (crossover) {
      if (vecSize == 0 || vecSize > 1024) {
         std::fprintf(stderr, "-x: vec must be 1..1024 (Group::pos[1024])\n");
         return 1;
      }
      if (!groupsSet)
         groupList = {1,   2,    4,    8,    16,   32,    64,    128,   256,
                      512, 1024, 1433, 2048, 4096, 16384, 65536, 262144};
      for (unsigned g : groupList)
         if (g == 0) {
            std::fprintf(stderr, "groups must be >= 1\n");
            return 1;
         }
      std::printf("impl,groups,dist,sel_pct,aggs,vec,max_fill,vecs_per_flush,"
                  "ns_per_selected_row,ok\n");
      for (unsigned a : xover::aggList)
         for (int s : selList)
            for (unsigned g : groupList) xover::benchConfig(g, s, a);
      return 0;
   }
   for (unsigned g : groupList)
      if (g == 0 || g > 255) {
         std::fprintf(stderr, "groups must be 1..255\n");
         return 1;
      }
#ifndef VW_HAVE_SIMD_AGGR
   std::fprintf(stderr, "run_aggrbench: no AVX-512F/VL/BW, gather/direct_simd rows skipped\n");
#endif
   std::printf("side,impl,groups,dist,sel_pct,mode,vec,ns_per_selected_row,ok\n");
   for (unsigned g : groupList)
      for (int s : selList) benchConfig(g, s);
   return 0;
}
