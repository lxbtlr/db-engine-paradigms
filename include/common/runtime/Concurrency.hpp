#pragma once
#include "Barrier.hpp"
#include "common/Compat.hpp"
#include "common/runtime/Barrier.hpp"
#include "common/runtime/MemoryPool.hpp"
#include "tbb/task_group.h"
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <thread>
#include <unordered_map>
#include <utility>

namespace runtime {

class Worker;
class WorkerGroup;

// -------------------------------------------------------------------------
// NUMA topology constants (compile-time; must match the hardware this binary
// is built for).  These drive both the scheduling policies below and the
// regionOf() helper used for per-NUMA work distribution.
// -------------------------------------------------------------------------
static constexpr size_t SOCKETS_COUNT    = 4;
static constexpr size_t CORES_PER_SOCKET = 22;
static constexpr size_t SMT_PER_CORE     = 2;
static constexpr size_t NUM_NUMA_REGIONS = SOCKETS_COUNT; // one region per socket

// regionOf(thread_id) — single source of truth for thread → NUMA region mapping.
//
// Under NUMA_BANDWIDTH (fan-out schedule): threads are interleaved across
// sockets, so thread t lands on socket t % SOCKETS_COUNT.
//
// Under NUMA_POOLS (consolidated schedule): threads are packed per socket,
// so the first CORES_PER_SOCKET*SMT_PER_CORE threads are on socket 0, etc.
//
// Memory-order note: this is a pure arithmetic mapping; no atomics involved.
#if defined(NUMA_BANDWIDTH)
inline size_t regionOf(size_t thread_id) {
    return thread_id % SOCKETS_COUNT;
}
#elif defined(NUMA_POOLS)
inline size_t regionOf(size_t thread_id) {
    return thread_id / (CORES_PER_SOCKET * SMT_PER_CORE);
}
#else
#error "Neither NUMA_BANDWIDTH nor NUMA_POOLS is defined. \
 Exactly one must be set at compile time."
#endif

extern thread_local Worker* this_worker;
#ifdef NUMA_POOLS
constexpr size_t MAX_NUMA_NODES = 4;
extern GlobalPool numaPools[MAX_NUMA_NODES];
extern size_t activeNumaNodes;

// NUMA utility functions - declared before Worker class
inline size_t detectNumaNodes() {
#ifdef __linux__
    // Read from /sys/devices/system/node to detect NUMA nodes
    size_t nodeCount = 0;
    for (size_t i = 0; i < MAX_NUMA_NODES; ++i) {
        std::string path = "/sys/devices/system/node/node" + std::to_string(i);
        std::ifstream nodeCheck(path);
        if (nodeCheck.good()) {
            nodeCount = i + 1;
        }
    }
    return nodeCount > 0 ? nodeCount : 1;
#else
    return 1; // Non-Linux systems default to single node
#endif
}

inline size_t cpuToNumaNode(size_t cpu, size_t socketsCount, size_t coresPerSocket) {
    // For typical Intel/AMD systems: NUMA node corresponds to socket
    // Physical layout: cpu = socket * coresPerSocket * smtPerCore + ...
    return cpu / (coresPerSocket * 2); // Assuming 2-way SMT
}

inline GlobalPool* getNumaPool(size_t numaNode) {
    if (numaNode >= activeNumaNodes) {
        numaNode = 0; // Fallback to node 0 if invalid
    }
    return &numaPools[numaNode];
}
#else
extern GlobalPool defaultPool;
#endif
extern thread_local bool currentBarrier;


// Used to control thread scheduling, by default we use the provided method
#define NEW_POLICY 1
#define NUMA_BANDWIDTH 1
//#define NUMA_LATENCY 1
class Worker
/// information about the worker thread.
/// accessible via thread local 'this_worker'
{
 public:
   Worker* previousWorker;
   WorkerGroup* group;
   Allocator allocator;
   HierarchicBarrier* barrier;
   size_t worker_id = 0; // logical index assigned by WorkerGroup::run (0-based)
#ifdef NUMA_POOLS
   size_t numaNode = 0;
#endif

   void start() {
      // set reference to worker in this thread
      this_worker = this;
      currentBarrier = 0;

      function();
   };
   Worker(WorkerGroup* g, std::function<void()> f, HierarchicBarrier* b)
       : group(g), barrier(b), function(f){};
#ifdef NUMA_POOLS
   Worker(WorkerGroup* g, HierarchicBarrier* b, size_t node)
       : group(g), barrier(b), numaNode(node) {
      this_worker = this;
      allocator.setSource(getNumaPool(node));
   };
#else
   Worker(WorkerGroup* g, HierarchicBarrier* b, GlobalPool& p)
       : group(g), barrier(b) {
      this_worker = this;
      allocator.setSource(&p);
   };
#endif
   explicit Worker() {}
   // void join() { t->join(); }

 private:
   std::function<void()> function;
};

class WorkerGroup
/// Group of worker threads which work on the same task, share a barrier etc.
{
   std::deque<Worker> threads;

 public:
   std::deque<Barrier> barriers;
   size_t size = std::thread::hardware_concurrency();
   WorkerGroup(WorkerGroup&) = delete;
   WorkerGroup() {
      barriers.emplace_back(size);
      barriers.emplace_back(size);
   };
   WorkerGroup(size_t nrWorkers) : size(nrWorkers) {
      barriers.emplace_back(size);
      barriers.emplace_back(size);
   };
   /// Spawn workers
   /// Calling thread is used as one worker
   inline void run(std::function<void()> f);
};

// #############################################################################
// static scheduling policies
// #############################################################################
inline std::vector<size_t> buildNumaPhysicalCoreMap(size_t socketsCount, 
                                                     size_t coresPerSocket,
                                                     size_t smtPerCore) {
    // Physical PU IDs are laid out as:
    // First all physical cores: core * socketsCount + socket
    // Then HT siblings offset by (socketsCount * coresPerSocket * smtIndex)
    size_t totalPhysical = socketsCount * coresPerSocket;
    std::vector<size_t> physicalCores;
    physicalCores.reserve(totalPhysical * smtPerCore);

    for (size_t smt = 0; smt < smtPerCore; ++smt)
        for (size_t socket = 0; socket < socketsCount; ++socket)
            for (size_t core = 0; core < coresPerSocket; ++core)
                physicalCores.push_back(smt * totalPhysical + core * socketsCount + socket);

    return physicalCores;
}

// Bandwidth-optimized: fan out across NUMA nodes as early as possible
// Physical cores exhausted first, then hyperthreads in the same order
inline std::vector<size_t> numaFanOutSchedule(size_t numThreads,
                                               size_t socketsCount,
                                               size_t coresPerSocket,
                                               size_t smtPerCore = 1) {
    auto physicalMap = buildNumaPhysicalCoreMap(socketsCount, coresPerSocket, smtPerCore);
    size_t totalPUs = socketsCount * coresPerSocket * smtPerCore;

    if (numThreads > totalPUs)
        throw std::runtime_error("numThreads " + std::to_string(numThreads) +
                                 " exceeds available PUs " + std::to_string(totalPUs));

    std::vector<size_t> schedule;
    schedule.reserve(numThreads);

    // Walk: smt -> core -> socket, so physical cores across all sockets
    // are exhausted before any hyperthread is used
    for (size_t smt = 0; smt < smtPerCore && schedule.size() < numThreads; ++smt)
        for (size_t core = 0; core < coresPerSocket && schedule.size() < numThreads; ++core)
            for (size_t socket = 0; socket < socketsCount && schedule.size() < numThreads; ++socket)
                schedule.push_back(physicalMap[smt * socketsCount * coresPerSocket +
                                               socket * coresPerSocket + core]);

    return schedule;
}

// Latency-optimized: consolidate within NUMA node before spilling over
// Physical cores exhausted per socket first, then hyperthreads per socket
inline std::vector<size_t> numaConsolidatedSchedule(size_t numThreads,
                                                     size_t socketsCount,
                                                     size_t coresPerSocket,
                                                     size_t smtPerCore = 1) {
    auto physicalMap = buildNumaPhysicalCoreMap(socketsCount, coresPerSocket, smtPerCore);
    size_t totalPUs = socketsCount * coresPerSocket * smtPerCore;

    if (numThreads > totalPUs)
        throw std::runtime_error("numThreads " + std::to_string(numThreads) +
                                 " exceeds available PUs " + std::to_string(totalPUs));

    std::vector<size_t> schedule;
    schedule.reserve(numThreads);

    // Walk: socket -> smt -> core, so all SMT siblings on a socket
    // are used before moving to the next socket
    for (size_t socket = 0; socket < socketsCount && schedule.size() < numThreads; ++socket)
        for (size_t smt = 0; smt < smtPerCore && schedule.size() < numThreads; ++smt)
            for (size_t core = 0; core < coresPerSocket && schedule.size() < numThreads; ++core)
                schedule.push_back(physicalMap[smt * socketsCount * coresPerSocket +
                                               socket * coresPerSocket + core]);

    return schedule;
}

inline void WorkerGroup::run(std::function<void()> f) {
   tbb::task_group g;
   auto barriers = HierarchicBarrier::create(size);
   int64_t group = -1;


   
   size_t nprocs = 88;
   size_t smt = 2;
#ifdef NUMA_BANDWIDTH
   // fan-out: good for Q1, Q6 (bandwidth bound)
   auto schedule = numaFanOutSchedule(size - 1, 4, 22, smt);

#elif NUMA_LATENCY
   // consolidated: good for Q3, Q9 (join/hash table heavy)
   auto schedule = numaConsolidatedSchedule(size - 1, 4, 22,smt);
#else
   std::vector<size_t> schedule;
#endif

   for (size_t i = 0; i < schedule.size(); ++i) {
       if (schedule[i] >= nprocs*smt) {
           throw std::runtime_error("Schedule entry " + std::to_string(i) + 
                   " maps to invalid CPU " + std::to_string(schedule[i]));
       }
   }




   for (size_t i = 0; i < size - 1; ++i) {
      // lets carefully use the mapping to barriers that the system was already
      // using then ask for the cpu that we want based on our schedule
      if (i % HierarchicBarrier::threadsPerBarrier == 0) ++group;
#ifdef NUMA_POOLS
      // Determine CPU assignment first, then compute NUMA node
      size_t selection;
#ifdef NEW_POLICY
      selection = schedule[i];
#else
      selection = i;
#endif
      size_t node = cpuToNumaNode(selection, 4, 22); // 4 sockets, 22 cores per socket
      threads.emplace_back(this, f, barriers[group]);
      threads.back().worker_id = i;
      threads.back().numaNode = node;
      threads.back().allocator.setSource(getNumaPool(node));

      // Debug output (will be compiled out in release if desired)
      static std::once_flag debug_flag;
      std::call_once(debug_flag, []() {
         std::cerr << "[NUMA] Worker assignment (first few): thread -> CPU -> NUMA node" << std::endl;
      });
      if (i < 4) {
         std::cerr << "[NUMA] Worker " << i << " -> CPU " << selection << " -> Node " << node << std::endl;
      }
#else
      threads.emplace_back(this, f, barriers[group]);
      threads.back().worker_id = i;
#endif
      auto worker = &threads.back();
#ifndef NUMA_POOLS
#ifdef NEW_POLICY
      size_t selection = schedule[i];
#else
      size_t selection = i;
#endif
#endif

      g.run([worker, i,selection]() {


#ifndef __APPLE__
         pthread_t currentThread = pthread_self();
         pthread_setname_np(currentThread,
                            ("workerPool " + std::to_string(selection)).c_str());
         cpu_set_t cpuset;
         CPU_ZERO(&cpuset);

         CPU_SET(selection, &cpuset);
         if (pthread_setaffinity_np(currentThread, sizeof(cpu_set_t),
                                    &cpuset) != 0) {
            throw std::runtime_error("Could not pin thread " +
                                     std::to_string(i) + " to thread " +
                                     std::to_string(selection));
         }
#else
         compat::unused(i);
#endif
         worker->start();
      });
   }
   // calling worker temporarily joins this group
   // TODO: generalize joining of other groups with a stack
   auto prevGroup = this_worker->group;
   auto prevBarrier = currentBarrier;
   auto prevBarrierPtr = this_worker->barrier;
   auto prevWorkerId = this_worker->worker_id;
   this_worker->group = this;
   this_worker->barrier = barriers.back();
   this_worker->worker_id = size - 1; // calling thread is the last logical worker
   currentBarrier = 0;
   f();
   this_worker->group = prevGroup;
   currentBarrier = prevBarrier;
   this_worker->barrier = prevBarrierPtr;
   this_worker->worker_id = prevWorkerId;

   g.wait();

   HierarchicBarrier::destroy(barriers);
}

template <typename T> class thread_specific {
   std::mutex m;

 public:
   std::unordered_map<std::thread::id, T> threadData;
   T& local();
   T& put(T t);
   template <typename... Args> T& create(Args&&... args);
};

template <typename T> T& thread_specific<T>::local() {
   std::lock_guard<std::mutex> lock(m);
   auto t = threadData.find(std::this_thread::get_id());
   if (t == threadData.end())
      throw std::runtime_error("Thread specific element not found.");
   return t->second;
}
template <typename T> T& thread_specific<T>::put(T t) {
   std::lock_guard<std::mutex> lock(m);
   return threadData
       .emplace(move(std::make_pair(std::this_thread::get_id(), std::move(t))))
       .first->second;
}

template <typename T>
template <typename... Args>
T& thread_specific<T>::create(Args&&... args) {
   std::lock_guard<std::mutex> lock(m);
   return threadData
       .emplace(std::piecewise_construct,
                std::forward_as_tuple(std::this_thread::get_id()),
                std::forward_as_tuple(args...))
       .first->second;
}

inline bool __attribute__((noinline)) barrier()
/// Shorthand for using the current thread groups barrier
{
   return this_worker->barrier->wait();
}

template <typename F>
inline bool __attribute__ ((noinline)) barrier(F finalizer)
/// Shorthand for using the current thread groups barrier
{
   return this_worker->barrier->wait(finalizer);
}
} // namespace runtime
