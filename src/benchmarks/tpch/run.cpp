#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <iterator>
#include <sstream>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <unordered_set>

#include "benchmarks/tpch/Queries.hpp"
#ifdef CANARY_IN_BENCH
#include "canary.hpp"
#endif
#include "common/runtime/Import.hpp"
#include "common/runtime/Concurrency.hpp"
#include "profile.hpp"
#include "tbb/tbb.h"
#include <tbb/global_control.h>
#include <tbb/task_arena.h>
#include <tbb/task_scheduler_observer.h>
#if defined(NUMA_ALLOC) || defined(NUMA_SHARD)
#include "common/runtime/NumaAlloc.hpp"
#endif
#include "hyper/NumaTrace.hpp"

#ifndef HYPER_FLOAT
/// Pins each TBB worker thread to a CPU using the same policy as
/// WorkerGroup::run (spread by default, packed with THREAD_PIN_PACKED).
class PinningObserver : public tbb::task_scheduler_observer {
public:
   explicit PinningObserver(tbb::task_arena& arena)
       : tbb::task_scheduler_observer(arena) {
      observe(true);
   }
   void on_scheduler_entry(bool /*is_worker*/) override {
      int slot = tbb::this_task_arena::current_thread_index();
      if (slot < 0) return;
      size_t target = runtime::cpuOfThread(static_cast<size_t>(slot));
#ifdef NUMA_SHARD
      // Skip the re-pin when the thread is already pinned to exactly the
      // target CPU. Keeping any CPU of the target's region instead would let
      // a thread keep its CPU after a slot change while the thread taking its
      // old slot is pinned there too: on a 1-region host this put the main
      // thread (still on Tectorwise's cpuOfThread(n-1)) and slot n-1 on one
      // CPU, and Q1's scan took 3x longer (trace: cpu_shared).
      cpu_set_t current;
      if (pthread_getaffinity_np(pthread_self(), sizeof(current), &current) ==
              0 &&
          CPU_COUNT(&current) == 1 && CPU_ISSET(target, &current))
         return;
#endif
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(target, &cpuset);
      pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
   }
   ~PinningObserver() override { observe(false); }
};
#endif // !HYPER_FLOAT

// NOTE: this was helpful for debuging, but breaks if we dont use the thread arg, disable for now
// lets force this thing to use one thread
//tbb::global_control control(
//    tbb::global_control::max_allowed_parallelism, 1
//);



using namespace std;
using namespace runtime;

static void escape(void* p) { asm volatile("" : : "g"(p) : "memory"); }

static void dumpQ1Result(const char* label, runtime::Query* query) {
   if (!query || !query->result) return;
   auto& rel = *query->result;
   auto retAttr = rel.getAttribute("l_returnflag");
   auto statusAttr = rel.getAttribute("l_linestatus");
   auto qtyAttr = rel.getAttribute("sum_qty");
   auto basePriceAttr = rel.getAttribute("sum_base_price");
   auto discPriceAttr = rel.getAttribute("sum_disc_price");
   auto chargeAttr = rel.getAttribute("sum_charge");
   auto countAttr = rel.getAttribute("count_order");

   fprintf(stderr, "\n=== Q1 Results [%s] ===\n", label);
   fprintf(stderr, "%-4s %-4s %20s %20s %20s %20s %15s\n",
           "ret", "stat", "sum_qty", "sum_base_price", "sum_disc_price",
           "sum_charge", "count_order");
   for (auto& block : rel) {
      auto n = block.size();
      auto ret = reinterpret_cast<types::Char<1>*>(block.data(retAttr));
      auto status = reinterpret_cast<types::Char<1>*>(block.data(statusAttr));
      auto qty = reinterpret_cast<int64_t*>(block.data(qtyAttr));
      auto basePrice = reinterpret_cast<int64_t*>(block.data(basePriceAttr));
      auto discPrice = reinterpret_cast<int64_t*>(block.data(discPriceAttr));
      auto charge = reinterpret_cast<int64_t*>(block.data(chargeAttr));
      auto count = reinterpret_cast<int64_t*>(block.data(countAttr));
      for (size_t i = 0; i < n; ++i) {
         fprintf(stderr, "%-4c %-4c %20ld %20ld %20ld %20ld %15ld\n",
                 ret[i].value, status[i].value,
                 qty[i], basePrice[i], discPrice[i], charge[i], count[i]);
      }
   }
   fprintf(stderr, "=== END ===\n\n");
}

/// Order-independent result digest (env digest=1): per row a hash over its
/// columns in name order; row hashes and per-column value hashes are summed,
/// so block and row order (thread count, dispatch) do not change it.
static uint64_t mix64(uint64_t x) {
   x ^= x >> 30;
   x *= 0xbf58476d1ce4e5b9ull;
   x ^= x >> 27;
   x *= 0x94d049bb133111ebull;
   return x ^ (x >> 31);
}
static uint64_t hashBytes(const char* p, size_t n) {
   uint64_t h = 0xcbf29ce484222325ull;
   for (size_t i = 0; i < n; ++i) h = (h ^ uint8_t(p[i])) * 0x100000001b3ull;
   return h;
}
/// "q1 h  t8  " -> "q1h_t8"
static std::string compactLabel(const std::string& label) {
   std::string s;
   for (char c : label)
      if (c != ' ') s += c;
   auto t = s.find('t', 1);
   if (t != std::string::npos) s.insert(t, "_");
   return s;
}
struct DigestColumn {
   std::string name;
   size_t size;
   uint64_t sum = 0;
};
static void printDigest(const std::string& label, size_t rows, uint64_t sum,
                        const std::vector<DigestColumn>& cols) {
   fprintf(stderr, "[RESULT] q=%s rows=%zu row=%016llx cols=",
           compactLabel(label).c_str(), rows, (unsigned long long)sum);
   for (size_t c = 0; c < cols.size(); ++c)
      fprintf(stderr, "%s%s:%zu:%016llx", c ? "," : "", cols[c].name.c_str(),
              cols[c].size, (unsigned long long)cols[c].sum);
   fprintf(stderr, "\n");
}
static void printDigest(const std::string& label,
                        std::unique_ptr<runtime::Query>& q) {
   auto& rel = *q->result;
   std::vector<DigestColumn> cols;
   std::vector<BlockRelation::Attribute> ids;
   for (auto& a : rel.attributeSizes()) {
      cols.push_back({a.first, a.second});
      ids.push_back(rel.getAttribute(a.first));
   }
   size_t rows = 0;
   uint64_t sum = 0;
   std::vector<const char*> base(cols.size());
   for (auto& block : rel) {
      for (size_t c = 0; c < cols.size(); ++c)
         base[c] = static_cast<const char*>(block.data(ids[c]));
      for (size_t i = 0, n = block.size(); i < n; ++i, ++rows) {
         uint64_t h = 0;
         for (size_t c = 0; c < cols.size(); ++c) {
            auto v = hashBytes(base[c] + i * cols[c].size, cols[c].size);
            cols[c].sum += mix64(v);
            h = mix64(h ^ v);
         }
         sum += h;
      }
   }
   printDigest(label, rows, sum, cols);
}
static void printDigest(const std::string& label, runtime::Relation& rel) {
   std::vector<std::pair<std::string, runtime::Attribute*>> attrs;
   for (auto& kv : rel.attributes) attrs.emplace_back(kv.first, &kv.second);
   std::sort(attrs.begin(), attrs.end(),
             [](auto& a, auto& b) { return a.first < b.first; });
   std::vector<DigestColumn> cols;
   for (auto& a : attrs) cols.push_back({a.first, a.second->type->rt_size()});
   uint64_t sum = 0;
   for (size_t i = 0; i < rel.nrTuples; ++i) {
      uint64_t h = 0;
      for (size_t c = 0; c < cols.size(); ++c) {
         auto p = static_cast<const char*>(attrs[c].second->data());
         auto v = hashBytes(p + i * cols[c].size, cols[c].size);
         cols[c].sum += mix64(v);
         h = mix64(h ^ v);
      }
      sum += h;
   }
   printDigest(label, rel.nrTuples, sum, cols);
}

#if defined(NUMA_DEBUG) && defined(NUMA_SHARD)
/// Placement check: while Typer runs, the base columns of the sharded tables
/// are PROT_NONE, so a Typer pipeline that reads a base column instead of its
/// shard faults. Columns are file mmaps (page aligned); a column that is not
/// (loaded without the binary cache) is skipped and counted.
static void protectBaseColumns(Database& db, const char* const* tables,
                               size_t nTables, int prot) {
   const uintptr_t page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
   size_t cols = 0, skipped = 0, bytes = 0;
   for (size_t t = 0; t < nTables; ++t) {
      if (!db.hasRelation(tables[t])) continue;
      auto& rel = db[tables[t]];
      for (auto& kv : rel.attributes) {
         auto p = reinterpret_cast<uintptr_t>(kv.second.data());
         size_t n = rel.nrTuples * kv.second.type->rt_size();
         if (!p || n == 0) continue;
         if (p % page) {
            skipped++;
            continue;
         }
         size_t len = (n + page - 1) / page * page;
         if (mprotect(reinterpret_cast<void*>(p), len, prot) != 0) {
            perror("mprotect");
            std::abort();
         }
         cols++;
         bytes += len;
      }
   }
   fprintf(stderr,
           "[PLACEMENT] base columns %s: %zu columns, %zu MB, %zu skipped\n",
           prot == PROT_NONE ? "PROT_NONE during Typer" : "readable again",
           cols, bytes >> 20, skipped);
}
#endif

size_t nrTuples(Database& db, std::vector<std::string> tables) {
   size_t sum = 0;
   for (auto& table : tables) sum += db[table].nrTuples;
   return sum;
}

/// Clears Linux page cache.
/// This function only works on Linux.
void clearOsCaches() {
   if (system("sync; echo 3 > /proc/sys/vm/drop_caches")) {
      throw std::runtime_error("Could not flush system caches: " +
                               std::string(std::strerror(errno)));
   }
}

int main(int argc, char* argv[]) {
    PerfEvents e;
    Database tpch;
    tl("proc_start");
    // load tpch data
//importTPCH(argv[2], tpch);

    bool clearCaches = false;
#ifdef CANARY_IN_BENCH
    bool canaryOnly = false;
#endif

    // Defaults
    int repetitions = 1;
    std::string tpchPath = "";
    std::string threadArg = "";
    size_t vectorSize = 1024;
    int settleSeconds = 10;
    std::string selectedQuery = "";  // e.g., "1" or "1,3,6"
    std::string selectedEngine = ""; // e.g., "h" or "v"

    int opt;
#ifdef CANARY_IN_BENCH
    const char* optstring = "q:e:r:p:t:v:s:cC";
#else
    const char* optstring = "q:e:r:p:t:v:s:";
#endif
    // q: query, e: engine, r: reps, p: path, t: threads, v: vectorSize, s: settle
    while ((opt = getopt(argc, argv, optstring)) != -1) {
        switch (opt) {
            case 'q': selectedQuery = optarg; break;
            case 'e': selectedEngine = optarg; break;
            case 'r': repetitions = atoi(optarg); break;
            case 'p': tpchPath = optarg; break;
            case 't': threadArg = optarg; break;
            case 'v': vectorSize = atoi(optarg); break;
            case 's': settleSeconds = atoi(optarg); break;
#ifdef CANARY_IN_BENCH
            case 'c': canary::g_enabled = true; break;
            case 'C': canaryOnly = true; break;
#endif
            default:
#ifdef CANARY_IN_BENCH
                std::cerr << "Usage: " << argv[0] << " -p <path> [-q query] [-e engine] [-r reps] [-t threads] [-v vSize] [-s settleSeconds] [-c canary] [-C canary-only]\n";
#else
                std::cerr << "Usage: " << argv[0] << " -p <path> [-q query] [-e engine] [-r reps] [-t threads] [-v vSize] [-s settleSeconds]\n";
#endif
                exit(1);
        }
    }

    // Parse comma-separated thread counts (e.g. "1,4,12,44" or just "44")
    std::vector<size_t> threadCounts;
    if (!threadArg.empty()) {
        std::istringstream ts(threadArg);
        std::string tok;
        while (std::getline(ts, tok, ',')) {
            if (!tok.empty()) threadCounts.push_back(std::atoi(tok.c_str()));
        }
    }
    if (threadCounts.empty())
        threadCounts.push_back(std::thread::hardware_concurrency());

#ifdef CANARY_IN_BENCH
    // Canary-only self-test / calibration mode: run both frozen kernels once
    // (emit as phase "before") and exit. No TPC-H data required.
    if (canaryOnly) {
        fprintf(stderr, "canary version: %s\n", canary::version());
        canary::g_enabled = true;
        canary::before("canary-only");
        return 0;
    }
#endif

    if (tpchPath.empty()) {
        std::cerr << "Error: Path to TPC-H directory (-p) is required.\n";
        exit(1);
    }
    importTPCH(tpchPath, tpch);

    runtime::assertTopology();

#ifdef NUMA_ALLOC
    runtime::numaReplicateRelation(tpch["lineitem"]);
    fprintf(stderr, "NUMA replication: lineitem replicated to %zu regions\n",
            runtime::NUM_NUMA_REGIONS);
#ifdef NUMA_DEBUG
    fprintf(stderr, "--- NUMA placement verification ---\n");
    runtime::verifyNumaPlacement(tpch["lineitem"]);
    fprintf(stderr, "--- end verification ---\n");
#endif
#endif

#ifdef NUMA_SHARD
    size_t maxThreads = *std::max_element(threadCounts.begin(), threadCounts.end());
    size_t nActive = runtime::activeRegions(maxThreads);
    const char* shardTables[] = {"lineitem", "orders", "customer",
                                 "supplier", "part", "partsupp",
                                 "nation", "region"};
    for (auto& name : shardTables) {
       if (!tpch.hasRelation(name)) continue;
       runtime::numaShardRelation(tpch[name], nActive);
       fprintf(stderr, "NUMA sharding: %s sharded across %zu of %zu regions\n",
               name, nActive, runtime::NUM_NUMA_REGIONS);
       for (size_t r = 0; r < runtime::NUM_NUMA_REGIONS; ++r) {
          auto& s = tpch[name].numaShards[r];
          if (s.tupleEnd > s.tupleBegin)
             fprintf(stderr, "  shard %zu: tuples [%zu, %zu)\n",
                     r, s.tupleBegin, s.tupleEnd);
       }
    }
#endif

#ifdef WARM_PAGES
    // Touch every page of every relation to warm page tables and TLB caches.
    // Equalizes page-walk costs across configs (sharding/replication does this
    // implicitly via memcpy; baseline does not).
    {
        volatile char sink = 0;
        const char* tables[] = {"lineitem", "orders", "customer", "part",
                                "supplier", "partsupp", "nation", "region"};
        for (auto& name : tables) {
            if (!tpch.hasRelation(name)) continue;
            auto& rel = tpch[name];
            for (auto& attrKv : rel.attributes) {
                auto& attr = attrKv.second;
                const char* p = static_cast<const char*>(attr.data());
                size_t bytes = rel.nrTuples * attr.type->rt_size();
                for (size_t off = 0; off < bytes; off += 4096)
                    sink += p[off];
            }
        }
        fprintf(stderr, "Page warmup complete.\n");
    }
#endif

    // Settle: let THP compaction / khugepaged finish before measurement
    if (settleSeconds > 0) {
        fprintf(stderr, "Settling for %d seconds...\n", settleSeconds);
        std::this_thread::sleep_for(std::chrono::seconds(settleSeconds));
        fprintf(stderr, "Done settling.\n");
    }

    tl("load_end");

#if defined(NUMA_DEBUG) && defined(NUMA_ALLOC)
    fprintf(stderr, "--- NUMA placement verification ---\n");
    runtime::verifyNumaPlacement(tpch["lineitem"]);
    fprintf(stderr, "--- end verification ---\n");
#endif
#if defined(NUMA_DEBUG) && defined(NUMA_SHARD)
    fprintf(stderr, "--- NUMA placement verification ---\n");
    for (auto& name : shardTables) {
       if (!tpch.hasRelation(name)) continue;
       fprintf(stderr, "  verifying %s...\n", name);
       runtime::verifyNumaPlacement(tpch[name]);
    }
    fprintf(stderr, "--- end verification ---\n");
#endif

    // Now, filter the master query set
    std::unordered_set<std::string> allQueries = {"1h", "1v", "3h", "3v", "5h", "5v", "6h", "6b", "6v", "18h", "18v", "9h", "9v"};
    std::unordered_set<std::string> q;

    // Parse comma-separated query list (e.g. "1,3,6" or just "1")
    std::vector<std::string> queryNumbers;
    if (!selectedQuery.empty()) {
        std::istringstream ss(selectedQuery);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (!token.empty()) queryNumbers.push_back(token);
        }
    }

    // Extract the query number from a key like "18h" -> "18"
    auto queryNum = [](const std::string& key) {
        return key.substr(0, key.size() - 1);
    };

    if (!queryNumbers.empty() && !selectedEngine.empty()) {
        // Run specific query+engine pairs
        for (auto& qn : queryNumbers) {
            std::string target = qn + selectedEngine;
            if (allQueries.count(target)) q.insert(target);
        }
    } else if (!queryNumbers.empty()) {
        // Run all engines for specified queries
        for (auto& qn : queryNumbers) {
            for (auto& aq : allQueries) {
                if (queryNum(aq) == qn)
                    q.insert(aq);
            }
        }
    } else if (!selectedEngine.empty()) {
        // Run all queries for specified engine
        for (auto& aq : allQueries) {
            if (aq.back() == selectedEngine[0])
                q.insert(aq);
        }
    } else {
        // Default: Run everything
        q = allQueries;
    }

    // Diagnostics
    const char* numaMode = "baseline";
#ifdef NUMA_ALLOC
    numaMode = "NUMA_ALLOC";
#endif
#ifdef NUMA_SHARD
    numaMode = "NUMA_SHARD";
#endif
    {
        std::string tcStr;
        for (size_t i = 0; i < threadCounts.size(); ++i) {
            if (i > 0) tcStr += ",";
            tcStr += std::to_string(threadCounts[i]);
        }
        fprintf(stderr, "Config: %s | Engine: %s | Query: %s | Threads: %s | VectorSize: %zu | Settle: %ds"
#ifdef NUMA_DEBUG
                " | NUMA_DEBUG"
#endif
                "\n",
                numaMode, selectedEngine.c_str(), selectedQuery.c_str(),
                tcStr.c_str(), vectorSize, settleSeconds);
    }

    if (auto v = std::getenv("SIMDhash")) conf.useSimdHash = atoi(v);
    if (auto v = std::getenv("SIMDjoin")) conf.useSimdJoin = atoi(v);
    if (auto v = std::getenv("SIMDsel")) conf.useSimdSel = atoi(v);
    if (auto v = std::getenv("SIMDproj")) conf.useSimdProj = atoi(v);
    if (auto v = std::getenv("clearCaches")) clearCaches = atoi(v);
    bool digest = false;
    if (auto v = std::getenv("digest")) digest = atoi(v);

    // Helper to build query label with thread count suffix
    auto label = [](const char* base, size_t t) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%s t%-3zu", base, t);
        return std::string(buf);
    };

   for (size_t nrThreads : threadCounts) {
    fprintf(stderr, "--- threads: %zu ---\n", nrThreads);
    writeHeader = true;

    // --- VW queries first (no TBB thread pool interference) ---
   if (q.count("1v")) {
      e.timeAndProfile(label("q1 v ", nrThreads), nrTuples(tpch, {"lineitem"}),
                       [&]() {
                          if (clearCaches) clearOsCaches();
                          auto result =
                              q1_vectorwise(tpch, nrThreads, vectorSize);
                          escape(&result);
                       },
                       repetitions);
      auto vResult = q1_vectorwise(tpch, nrThreads, vectorSize);
      //dumpQ1Result("vectorwise", vResult.get());
   }
   if (q.count("3v"))
      e.timeAndProfile(
          label("q3 v ", nrThreads), nrTuples(tpch, {"customer", "orders", "lineitem"}),
          [&]() {
             if (clearCaches) clearOsCaches();
             auto result = q3_vectorwise(tpch, nrThreads, vectorSize);
             escape(&result);
          },
          repetitions);
   if (q.count("5v"))
      e.timeAndProfile(label("q5 v ", nrThreads),
                       nrTuples(tpch, {"supplier", "region", "nation",
                                       "customer", "orders", "lineitem"}),
                       [&]() {
                          if (clearCaches) clearOsCaches();
                          auto result =
                              q5_vectorwise(tpch, nrThreads, vectorSize);
                          escape(&result);
                       },
                       repetitions);
   if (q.count("6v"))
      e.timeAndProfile(label("q6 v ", nrThreads), tpch["lineitem"].nrTuples,
                       [&]() {
                          if (clearCaches) clearOsCaches();
                          auto result =
                              q6_vectorwise(tpch, nrThreads, vectorSize);
                          escape(&result);
                       },
                       repetitions);
   if (q.count("9v"))
      e.timeAndProfile(label("q9 v ", nrThreads),
                       nrTuples(tpch, {"nation", "supplier", "part", "partsupp",
                                       "lineitem", "orders"}),
                       [&]() {
                          if (clearCaches) clearOsCaches();
                          auto result =
                              q9_vectorwise(tpch, nrThreads, vectorSize);
                          escape(&result);
                       },
                       repetitions);
   if (q.count("18v"))
      e.timeAndProfile(
          label("q18 v", nrThreads),
          nrTuples(tpch, {"customer", "lineitem", "orders", "lineitem"}),
          [&]() {
             if (clearCaches) clearOsCaches();
             auto result = q18_vectorwise(tpch, nrThreads, vectorSize);
             escape(&result);
          },
          repetitions);

   if (digest) {
      // one untimed run per query, result digest to stderr
      auto dv = [&](const char* name, auto fn) {
         auto r = fn(tpch, nrThreads, vectorSize);
         printDigest(label(name, nrThreads), r);
      };
      if (q.count("1v")) dv("q1 v ", q1_vectorwise);
      if (q.count("3v")) dv("q3 v ", q3_vectorwise);
      if (q.count("5v")) dv("q5 v ", q5_vectorwise);
      if (q.count("6v")) dv("q6 v ", q6_vectorwise);
      if (q.count("9v")) dv("q9 v ", q9_vectorwise);
      if (q.count("18v")) dv("q18 v", q18_vectorwise);
   }

    // --- Hyper queries after VW (TBB threads can't interfere) ---
    {
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, nrThreads);
    tbb::task_arena arena(static_cast<int>(nrThreads));
#ifndef HYPER_FLOAT
    PinningObserver pinner(arena);
#endif
#if defined(NUMA_DEBUG) && defined(NUMA_SHARD)
    protectBaseColumns(tpch, shardTables, std::size(shardTables), PROT_NONE);
#endif

   if (q.count("1h")) {
      e.timeAndProfile(label("q1 h ", nrThreads), nrTuples(tpch, {"lineitem"}),
                       [&]() {
                          NUMA_TRACE_QUERY(label("q1 h ", nrThreads));
                          if (clearCaches) clearOsCaches();
                          arena.execute([&] {
                             auto result = q1_hyper(tpch, nrThreads);
                             escape(&result);
                          });
                       },
                       repetitions);
      std::unique_ptr<runtime::Query> hResult;
      arena.execute([&] { hResult = q1_hyper(tpch, nrThreads); });
      //dumpQ1Result("hyper", hResult.get());
   }
   if (q.count("3h"))
      e.timeAndProfile(label("q3 h ", nrThreads),
                       nrTuples(tpch, {"customer", "orders", "lineitem"}),
                       [&]() {
                          NUMA_TRACE_QUERY(label("q3 h ", nrThreads));
                          if (clearCaches) clearOsCaches();
                          arena.execute([&] {
                             auto result = q3_hyper(tpch, nrThreads);
                             escape(&result);
                          });
                       },
                       repetitions);
   if (q.count("5h"))
      e.timeAndProfile(label("q5 h ", nrThreads),
                       nrTuples(tpch, {"supplier", "region", "nation",
                                       "customer", "orders", "lineitem"}),
                       [&]() {
                          NUMA_TRACE_QUERY(label("q5 h ", nrThreads));
                          if (clearCaches) clearOsCaches();
                          arena.execute([&] {
                             auto result = q5_hyper(tpch, nrThreads);
                             escape(&result);
                          });
                       },
                       repetitions);
   if (q.count("6h"))
      e.timeAndProfile(label("q6 h ", nrThreads), tpch["lineitem"].nrTuples,
                       [&]() {
                          NUMA_TRACE_QUERY(label("q6 h ", nrThreads));
                          if (clearCaches) clearOsCaches();
                          arena.execute([&] {
                             auto result = q6_hyper(tpch, nrThreads);
                             escape(&result);
                          });
                       },
                       repetitions);
   if (q.count("6b"))
      e.timeAndProfile(label("q6 b ", nrThreads), tpch["lineitem"].nrTuples,
                       [&]() {
                          NUMA_TRACE_QUERY(label("q6 b ", nrThreads));
                          if (clearCaches) clearOsCaches();
                          arena.execute([&] {
                             auto result = q6_hyper_branching(tpch, nrThreads);
                             escape(&result);
                          });
                       },
                       repetitions);
   if (q.count("9h"))
      e.timeAndProfile(label("q9 h ", nrThreads),
                       nrTuples(tpch, {"nation", "supplier", "part", "partsupp",
                                       "lineitem", "orders"}),
                       [&]() {
                          NUMA_TRACE_QUERY(label("q9 h ", nrThreads));
                          if (clearCaches) clearOsCaches();
                          arena.execute([&] {
                             auto result = q9_hyper(tpch, nrThreads);
                             escape(&result);
                          });
                       },
                       repetitions);
   if (q.count("18h"))
      e.timeAndProfile(
          label("q18 h", nrThreads),
          nrTuples(tpch, {"customer", "lineitem", "orders", "lineitem"}),
          [&]() {
             NUMA_TRACE_QUERY(label("q18 h", nrThreads));
             if (clearCaches) clearOsCaches();
             arena.execute([&] {
                auto result = q18_hyper(tpch, nrThreads);
                escape(&result);
             });
          },
          repetitions);
   if (digest) {
      auto dh = [&](const char* name, auto fn) {
         arena.execute([&] {
            auto r = fn(tpch, nrThreads);
            printDigest(label(name, nrThreads), r);
         });
      };
      if (q.count("1h")) dh("q1 h ", q1_hyper);
      if (q.count("3h")) dh("q3 h ", q3_hyper);
      if (q.count("5h")) dh("q5 h ", q5_hyper);
      if (q.count("6h")) dh("q6 h ", q6_hyper);
      if (q.count("6b")) dh("q6 b ", q6_hyper_branching);
      if (q.count("9h")) dh("q9 h ", q9_hyper);
      if (q.count("18h")) dh("q18 h", q18_hyper);
   }
#if defined(NUMA_DEBUG) && defined(NUMA_SHARD)
    protectBaseColumns(tpch, shardTables, std::size(shardTables), PROT_READ);
#endif
    } // TBB arena + global_control destroyed here

   } // end thread count loop
   tl("proc_end");
   return 0;
}

