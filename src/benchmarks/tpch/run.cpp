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
#include "profile.hpp"
#include "tbb/tbb.h"
#include <tbb/global_control.h>

// NOTE: this was helpful for debuging, but breaks if we dont use the thread arg, disable for now
// lets force this thing to use one thread
//tbb::global_control control(
//    tbb::global_control::max_allowed_parallelism, 1
//);



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

int main(int argc, char* argv[]) {
    PerfEvents e;
    Database tpch;
    // load tpch data
//importTPCH(argv[2], tpch);
 
    bool clearCaches = false;
   
    // Defaults
    int repetitions = 1;
    std::string tpchPath = "";
    size_t nrThreads = std::thread::hardware_concurrency();
    size_t vectorSize = 1024;
    std::string selectedQuery = "";  // e.g., "1"
    std::string selectedEngine = ""; // e.g., "h" or "v"

    int opt;
    // q: query, e: engine, r: reps, p: path, t: threads, v: vectorSize
    while ((opt = getopt(argc, argv, "q:e:r:p:t:v:")) != -1) {
        switch (opt) {
            case 'q': selectedQuery = optarg; break;
            case 'e': selectedEngine = optarg; break;
            case 'r': repetitions = atoi(optarg); break;
            case 'p': tpchPath = optarg; break;
            case 't': nrThreads = atoi(optarg); break;
            case 'v': vectorSize = atoi(optarg); break;
            default:
                std::cerr << "Usage: " << argv[0] << " -p <path> [-q query] [-e engine] [-r reps] [-t threads] [-v vSize]\n";
                exit(1);
        }
    }

    if (tpchPath.empty()) {
        std::cerr << "Error: Path to TPC-H directory (-p) is required.\n";
        exit(1);
    }
    importTPCH(tpchPath, tpch);

    // Now, filter the master query set
    std::unordered_set<std::string> allQueries = {"1h", "1v", "3h", "3v", "5h", "5v", "6h", "6v" ,"18h", "18v", "9h", "9v"};
    std::unordered_set<std::string> q;

    if (!selectedQuery.empty() && !selectedEngine.empty()) {
        // Run specific pair, e.g., "1" + "h" -> "1h"
        std::string target = selectedQuery + selectedEngine;
        if (allQueries.count(target)) q.insert(target);
    } else if (!selectedQuery.empty()) {
        // Run all engines for one query, e.g., "1h" and "1v"
        if (allQueries.count(selectedQuery + "h")) q.insert(selectedQuery + "h");
        if (allQueries.count(selectedQuery + "v")) q.insert(selectedQuery + "v");
    } else {
        // Default: Run everything
        q = allQueries;
    }

    // Diagnostics
    fprintf(stderr, "Engine: %s | Query: %s | Threads: %ld | VectorSize: %ld\n", 
            selectedEngine.c_str(), selectedQuery.c_str(), nrThreads, vectorSize);

    tbb::global_control scheduler(tbb::global_control::max_allowed_parallelism, nrThreads);

   if (q.count("1h"))
      e.timeAndProfile("q1 hyper     ", nrTuples(tpch, {"lineitem"}),
                       [&]() {
                          if (clearCaches) clearOsCaches();
                          auto result = q1_hyper(tpch, nrThreads);
                          escape(&result);
                       },
                       repetitions);
   if (q.count("1v"))
      e.timeAndProfile("q1 vectorwise", nrTuples(tpch, {"lineitem"}),
                       [&]() {
                          if (clearCaches) clearOsCaches();
                          auto result =
                              q1_vectorwise(tpch, nrThreads, vectorSize);
                          escape(&result);
                       },
                       repetitions);
   if (q.count("3h"))
      e.timeAndProfile("q3 hyper     ",
                       nrTuples(tpch, {"customer", "orders", "lineitem"}),
                       [&]() {
                          if (clearCaches) clearOsCaches();
                          auto result = q3_hyper(tpch, nrThreads);
                          escape(&result);
                       },
                       repetitions);
   if (q.count("3v"))
      e.timeAndProfile(
          "q3 vectorwise", nrTuples(tpch, {"customer", "orders", "lineitem"}),
          [&]() {
             if (clearCaches) clearOsCaches();
             auto result = q3_vectorwise(tpch, nrThreads, vectorSize);
             escape(&result);
          },
          repetitions);
   if (q.count("5h"))
      e.timeAndProfile("q5 hyper     ",
                       nrTuples(tpch, {"supplier", "region", "nation",
                                       "customer", "orders", "lineitem"}),
                       [&]() {
                          if (clearCaches) clearOsCaches();
                          auto result = q5_hyper(tpch, nrThreads);
                          escape(&result);
                       },
                       repetitions);
   if (q.count("5v"))
      e.timeAndProfile("q5 vectorwise",
                       nrTuples(tpch, {"supplier", "region", "nation",
                                       "customer", "orders", "lineitem"}),
                       [&]() {
                          if (clearCaches) clearOsCaches();
                          auto result =
                              q5_vectorwise(tpch, nrThreads, vectorSize);
                          escape(&result);
                       },
                       repetitions);
   if (q.count("6h"))
      e.timeAndProfile("q6 hyper     ", tpch["lineitem"].nrTuples,
                       [&]() {
                          if (clearCaches) clearOsCaches();
                          auto result = q6_hyper(tpch, nrThreads);
                          escape(&result);
                       },
                       repetitions);
   if (q.count("6v"))
      e.timeAndProfile("q6 vectorwise", tpch["lineitem"].nrTuples,
                       [&]() {
                          if (clearCaches) clearOsCaches();
                          auto result =
                              q6_vectorwise(tpch, nrThreads, vectorSize);
                          escape(&result);
                       },
                       repetitions);
   if (q.count("9h"))
      e.timeAndProfile("q9 hyper     ",
                       nrTuples(tpch, {"nation", "supplier", "part", "partsupp",
                                       "lineitem", "orders"}),
                       [&]() {
                          if (clearCaches) clearOsCaches();
                          auto result = q9_hyper(tpch, nrThreads);
                          escape(&result);
                       },
                       repetitions);
   if (q.count("9v"))
      e.timeAndProfile("q9 vectorwise",
                       nrTuples(tpch, {"nation", "supplier", "part", "partsupp",
                                       "lineitem", "orders"}),
                       [&]() {
                          if (clearCaches) clearOsCaches();
                          auto result =
                              q9_vectorwise(tpch, nrThreads, vectorSize);
                          escape(&result);
                       },
                       repetitions);
   if (q.count("18h"))
      e.timeAndProfile(
          "q18 hyper     ",
          nrTuples(tpch, {"customer", "lineitem", "orders", "lineitem"}),
          [&]() {
             if (clearCaches) clearOsCaches();
             auto result = q18_hyper(tpch, nrThreads);
             escape(&result);
          },
          repetitions);
   if (q.count("18v"))
      e.timeAndProfile(
          "q18 vectorwise",
          nrTuples(tpch, {"customer", "lineitem", "orders", "lineitem"}),
          [&]() {
             if (clearCaches) clearOsCaches();
             auto result = q18_vectorwise(tpch, nrThreads, vectorSize);
             escape(&result);
          },
          repetitions);
   return 0;
}
