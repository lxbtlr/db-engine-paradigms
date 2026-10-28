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
#include "vectorwise/SimdAggr.hpp"
#include <algorithm>
#include <chrono>
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

} // namespace

int main(int argc, char** argv) {
   int opt;
   while ((opt = getopt(argc, argv, "v:m:r:g:s:d:")) != -1) {
      switch (opt) {
      case 'v': vecSize = std::strtoull(optarg, nullptr, 10); break;
      case 'm': mode = optarg; break;
      case 'r': reps = std::strtoull(optarg, nullptr, 10); break;
      case 'g': groupList = parseList<unsigned>(optarg); break;
      case 's': selList = parseList<int>(optarg); break;
      case 'd': dist = optarg; break;
      default:
         std::fprintf(stderr,
                      "usage: %s [-v vec] [-m l1|stream|all] [-r reps] [-g groups,..] "
                      "[-s selPct,..] [-d q1|uniform]\n",
                      argv[0]);
         return 1;
      }
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
