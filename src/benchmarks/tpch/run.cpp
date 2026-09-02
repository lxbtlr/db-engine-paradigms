#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <iterator>
#include <sstream>
#include <thread>
#include <unordered_set>

#include "benchmarks/tpch/Queries.hpp"
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
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
#ifdef THREAD_PIN_PACKED
      size_t tps = runtime::CORES_PER_SOCKET * runtime::SMT_PER_CORE;
      size_t socket = static_cast<size_t>(slot) / tps;
      size_t j = static_cast<size_t>(slot) % tps;
      size_t cpu = j * runtime::SOCKETS_COUNT + socket;
      CPU_SET(cpu, &cpuset);
#else
      CPU_SET(slot, &cpuset);
#endif
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

/*
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
*/

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
    // load tpch data
//importTPCH(argv[2], tpch);
 
    bool clearCaches = false;
   
    // Defaults
    int repetitions = 1;
    std::string tpchPath = "";
    std::string threadArg = "";
    size_t vectorSize = 1024;
    int settleSeconds = 10;
    std::string selectedQuery = "";  // e.g., "1" or "1,3,6"
    std::string selectedEngine = ""; // e.g., "h" or "v"

    int opt;
    // q: query, e: engine, r: reps, p: path, t: threads, v: vectorSize, s: settle
    while ((opt = getopt(argc, argv, "q:e:r:p:t:v:s:")) != -1) {
        switch (opt) {
            case 'q': selectedQuery = optarg; break;
            case 'e': selectedEngine = optarg; break;
            case 'r': repetitions = atoi(optarg); break;
            case 'p': tpchPath = optarg; break;
            case 't': threadArg = optarg; break;
            case 'v': vectorSize = atoi(optarg); break;
            case 's': settleSeconds = atoi(optarg); break;
            default:
                std::cerr << "Usage: " << argv[0] << " -p <path> [-q query] [-e engine] [-r reps] [-t threads] [-v vSize] [-s settleSeconds]\n";
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

    if (tpchPath.empty()) {
        std::cerr << "Error: Path to TPC-H directory (-p) is required.\n";
        exit(1);
    }
    importTPCH(tpchPath, tpch);

#ifdef NUMA_ALLOC
    runtime::assertTopology();
    runtime::numaReplicateRelation(tpch["lineitem"]);
    fprintf(stderr, "NUMA replication: lineitem replicated to %zu regions\n",
            runtime::NUM_NUMA_REGIONS);
#endif

#ifdef NUMA_SHARD
    runtime::assertTopology();
    runtime::numaShardRelation(tpch["lineitem"]);
    fprintf(stderr, "NUMA sharding: lineitem sharded across %zu regions\n",
            runtime::NUM_NUMA_REGIONS);
    for (size_t r = 0; r < runtime::NUM_NUMA_REGIONS; ++r) {
       auto& s = tpch["lineitem"].numaShards[r];
       fprintf(stderr, "  shard %zu: tuples [%zu, %zu)\n",
               r, s.tupleBegin, s.tupleEnd);
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

#if defined(NUMA_DEBUG) && (defined(NUMA_ALLOC) || defined(NUMA_SHARD))
    fprintf(stderr, "--- NUMA placement verification ---\n");
    runtime::verifyNumaPlacement(tpch["lineitem"]);
    fprintf(stderr, "--- end verification ---\n");
#endif

    // Now, filter the master query set
    std::unordered_set<std::string> allQueries = {"1h", "1v", "3h", "3v", "5h", "5v", "6h", "6v", "18h", "18v", "9h", "9v"};
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
   if (q.count("1v"))
      e.timeAndProfile(label("q1 v ", nrThreads), nrTuples(tpch, {"lineitem"}),
                       [&]() {
                          if (clearCaches) clearOsCaches();
                          auto result =
                              q1_vectorwise(tpch, nrThreads, vectorSize);
                          escape(&result);
                       },
                       repetitions);
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

    // --- Hyper queries after VW (TBB threads can't interfere) ---
    {
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, nrThreads);
    tbb::task_arena arena(static_cast<int>(nrThreads));
#ifndef HYPER_FLOAT
    PinningObserver pinner(arena);
#endif

   if (q.count("1h"))
      e.timeAndProfile(label("q1 h ", nrThreads), nrTuples(tpch, {"lineitem"}),
                       [&]() {
                          if (clearCaches) clearOsCaches();
                          arena.execute([&] {
                             auto result = q1_hyper(tpch, nrThreads);
                             escape(&result);
                          });
                       },
                       repetitions);
   if (q.count("3h"))
      e.timeAndProfile(label("q3 h ", nrThreads),
                       nrTuples(tpch, {"customer", "orders", "lineitem"}),
                       [&]() {
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
                          if (clearCaches) clearOsCaches();
                          arena.execute([&] {
                             auto result = q6_hyper(tpch, nrThreads);
                             escape(&result);
                          });
                       },
                       repetitions);
   if (q.count("9h"))
      e.timeAndProfile(label("q9 h ", nrThreads),
                       nrTuples(tpch, {"nation", "supplier", "part", "partsupp",
                                       "lineitem", "orders"}),
                       [&]() {
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
             if (clearCaches) clearOsCaches();
             arena.execute([&] {
                auto result = q18_hyper(tpch, nrThreads);
                escape(&result);
             });
          },
          repetitions);
    } // TBB arena + global_control destroyed here

   } // end thread count loop
   return 0;
}

