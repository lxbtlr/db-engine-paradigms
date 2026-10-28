#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "benchmarks/Pinning.hpp"
#include "benchmarks/ssb/Queries.hpp"
#include "common/runtime/Import.hpp"
#include "profile.hpp"
#include "tbb/tbb.h"
#include <tbb/global_control.h>
#include <tbb/task_arena.h>

// Same command line and output as run_tpch, so the ablation scripts can drive
// either binary. SSB query 1.1 is "11" (the label is "q11 v  t1"), and -q
// also accepts the dotted form ("1.1,3.4").

using namespace std;
using namespace runtime;

static void escape(void* p) { asm volatile("" : : "g"(p) : "memory"); }

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

using HyperFn = std::unique_ptr<runtime::Query> (*)(Database&, size_t);
using VectorwiseFn = std::unique_ptr<runtime::Query> (*)(Database&, size_t,
                                                         size_t);

struct SsbQuery {
   const char* id; // "11" for SSB 1.1
   std::vector<std::string> tables;
   HyperFn hyper;
   VectorwiseFn vectorwise;
};

static const std::vector<SsbQuery>& allQueries() {
   static const std::vector<std::string> q1 = {"date", "lineorder"};
   static const std::vector<std::string> q2 = {"date", "lineorder", "supplier",
                                               "part"};
   static const std::vector<std::string> q3 = {"date", "lineorder", "supplier",
                                               "customer"};
   static const std::vector<std::string> q4 = {"date", "lineorder", "supplier",
                                               "customer", "part"};
   static const std::vector<SsbQuery> queries = {
       {"11", q1, ssb::q11_hyper, ssb::q11_vectorwise},
       {"12", q1, ssb::q12_hyper, ssb::q12_vectorwise},
       {"13", q1, ssb::q13_hyper, ssb::q13_vectorwise},
       {"21", q2, ssb::q21_hyper, ssb::q21_vectorwise},
       {"22", q2, ssb::q22_hyper, ssb::q22_vectorwise},
       {"23", q2, ssb::q23_hyper, ssb::q23_vectorwise},
       {"31", q3, ssb::q31_hyper, ssb::q31_vectorwise},
       {"32", q3, ssb::q32_hyper, ssb::q32_vectorwise},
       {"33", q3, ssb::q33_hyper, ssb::q33_vectorwise},
       {"34", q3, ssb::q34_hyper, ssb::q34_vectorwise},
       {"41", q4, ssb::q41_hyper, ssb::q41_vectorwise},
       {"42", q4, ssb::q42_hyper, ssb::q42_vectorwise},
       {"43", q4, ssb::q43_hyper, ssb::q43_vectorwise},
   };
   return queries;
}

static void usage(const char* argv0) {
   std::cerr << "Usage: " << argv0
             << " -p <path> [-q queries] [-e engine] [-r reps] [-t threads]"
                " [-v vSize] [-s settleSeconds]\n"
                "  -q  comma list of SSB queries, 11..43 or 1.1..4.3 (default:"
                " all 13)\n"
                "  -e  h (Hyper) or v (Vectorwise) (default: both)\n";
   exit(1);
}

int main(int argc, char* argv[]) {
   PerfEvents e;
   Database ssb;
   tl("proc_start");

   bool clearCaches = false;

   // Defaults
   int repetitions = 1;
   std::string ssbPath = "";
   std::string threadArg = "";
   size_t vectorSize = 1024;
   int settleSeconds = 10;
   std::string selectedQuery = "";  // e.g., "11" or "11,2.1,34"
   std::string selectedEngine = ""; // "h" or "v"

   int opt;
   // q: query, e: engine, r: reps, p: path, t: threads, v: vectorSize, s: settle
   while ((opt = getopt(argc, argv, "q:e:r:p:t:v:s:")) != -1) {
      switch (opt) {
      case 'q': selectedQuery = optarg; break;
      case 'e': selectedEngine = optarg; break;
      case 'r': repetitions = atoi(optarg); break;
      case 'p': ssbPath = optarg; break;
      case 't': threadArg = optarg; break;
      case 'v': vectorSize = atoi(optarg); break;
      case 's': settleSeconds = atoi(optarg); break;
      default: usage(argv[0]);
      }
   }
   if (optind < argc) usage(argv[0]);

   if (!selectedEngine.empty() && selectedEngine != "h" &&
       selectedEngine != "v") {
      std::cerr << "Error: engine (-e) must be h or v, got '" << selectedEngine
                << "'\n";
      exit(1);
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

   // Selected queries, in SSB order; an unknown query is an error rather than
   // silently skipped, so a typo cannot produce an empty run
   std::vector<const SsbQuery*> queries;
   if (selectedQuery.empty()) {
      for (auto& q : allQueries()) queries.push_back(&q);
   } else {
      std::vector<std::string> wanted;
      std::istringstream ss(selectedQuery);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
         tok.erase(std::remove(tok.begin(), tok.end(), '.'), tok.end());
         if (tok.empty()) continue;
         auto it = std::find_if(allQueries().begin(), allQueries().end(),
                                [&](const SsbQuery& q) { return tok == q.id; });
         if (it == allQueries().end()) {
            std::cerr << "Error: unknown SSB query '" << tok
                      << "' (11..13, 21..23, 31..34, 41..43)\n";
            exit(1);
         }
         wanted.push_back(tok);
      }
      for (auto& q : allQueries())
         if (std::find(wanted.begin(), wanted.end(), q.id) != wanted.end())
            queries.push_back(&q);
   }
   if (queries.empty()) usage(argv[0]);
   const bool runV = selectedEngine.empty() || selectedEngine == "v";
   const bool runH = selectedEngine.empty() || selectedEngine == "h";

   if (ssbPath.empty()) {
      std::cerr << "Error: Path to SSB directory (-p) is required.\n";
      exit(1);
   }
   importSSB(ssbPath, ssb);

   // Settle: let THP compaction / khugepaged finish before measurement
   if (settleSeconds > 0) {
      fprintf(stderr, "Settling for %d seconds...\n", settleSeconds);
      std::this_thread::sleep_for(std::chrono::seconds(settleSeconds));
      fprintf(stderr, "Done settling.\n");
   }

   tl("load_end");

   // Diagnostics
   {
      std::string tcStr;
      for (size_t i = 0; i < threadCounts.size(); ++i) {
         if (i > 0) tcStr += ",";
         tcStr += std::to_string(threadCounts[i]);
      }
      fprintf(stderr,
              "Config: baseline | Engine: %s | Query: %s | Threads: %s | "
              "VectorSize: %zu | Settle: %ds\n",
              selectedEngine.c_str(), selectedQuery.c_str(), tcStr.c_str(),
              vectorSize, settleSeconds);
   }

   if (auto v = std::getenv("SIMDhash")) conf.useSimdHash = atoi(v);
   if (auto v = std::getenv("SIMDjoin")) conf.useSimdJoin = atoi(v);
   if (auto v = std::getenv("SIMDsel")) conf.useSimdSel = atoi(v);
   if (auto v = std::getenv("SIMDproj")) conf.useSimdProj = atoi(v);
   if (auto v = std::getenv("clearCaches")) clearCaches = atoi(v);

   // Query label with thread count suffix, as in run_tpch: "q11 v  t4  "
   auto label = [](const SsbQuery& q, char engine, size_t t) {
      char buf[32];
      snprintf(buf, sizeof(buf), "q%s %c t%-3zu", q.id, engine, t);
      return std::string(buf);
   };

   for (size_t nrThreads : threadCounts) {
      fprintf(stderr, "--- threads: %zu ---\n", nrThreads);
      writeHeader = true;

      // --- VW queries first (no TBB thread pool interference) ---
      if (runV)
         for (auto* q : queries)
            e.timeAndProfile(label(*q, 'v', nrThreads), nrTuples(ssb, q->tables),
                             [&]() {
                                if (clearCaches) clearOsCaches();
                                auto result =
                                    q->vectorwise(ssb, nrThreads, vectorSize);
                                escape(&result);
                             },
                             repetitions);

      // --- Hyper queries after VW (TBB threads can't interfere) ---
      if (runH) {
         tbb::global_control gc(tbb::global_control::max_allowed_parallelism,
                                nrThreads);
         tbb::task_arena arena(static_cast<int>(nrThreads));
         PinningObserver pinner(arena);
         for (auto* q : queries)
            e.timeAndProfile(label(*q, 'h', nrThreads), nrTuples(ssb, q->tables),
                             [&]() {
                                if (clearCaches) clearOsCaches();
                                arena.execute([&] {
                                   auto result = q->hyper(ssb, nrThreads);
                                   escape(&result);
                                });
                             },
                             repetitions);
      } // TBB arena + global_control destroyed here
   }
   tl("proc_end");
   return 0;
}
