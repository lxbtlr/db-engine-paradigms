#pragma once
#include "Barrier.hpp"
#include "common/Compat.hpp"
#include "common/runtime/Barrier.hpp"
#include "common/runtime/MemoryPool.hpp"
#include "tbb/task_group.h"
#include <deque>
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

extern thread_local Worker* this_worker;
extern GlobalPool defaultPool;
extern thread_local bool currentBarrier;


// Used to control thread scheduling, by default we use the provided method
//#define NEW_POLICY 1
//#define NUMA_BANDWIDTH 1
//#define NUMA_LATENCY 1
// #define ARM_BISECT 1

class Worker
/// information about the worker thread.
/// accessible via thread local 'this_worker'
{
 public:
   Worker* previousWorker;
   WorkerGroup* group;
   Allocator allocator;
   HierarchicBarrier* barrier;

   void start() {
      // set reference to worker in this thread
      this_worker = this;
      currentBarrier = 0;

      function();
   };
   Worker(WorkerGroup* g, std::function<void()> f, HierarchicBarrier* b)
       : group(g), barrier(b), function(f){};
   Worker(WorkerGroup* g, HierarchicBarrier* b, GlobalPool& p)
       : group(g), barrier(b) {
      this_worker = this;
      allocator.setSource(&p);
   };
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

struct ArmMeshConfig {
    size_t totalCores;
    size_t coresPerL2Group;
    size_t smtPerCore;

    size_t totalL2Groups() const { return totalCores / coresPerL2Group; }
    size_t totalPUs()      const { return totalCores * smtPerCore; }
};

inline ArmMeshConfig graviton4()   { return { 96,  1, 1 }; }
inline ArmMeshConfig ampereAltra() { return { 128, 4, 1 }; }

inline std::vector<size_t> bisectSchedule(size_t lo, size_t hi) {
    std::vector<size_t> schedule;
    std::function<void(size_t, size_t)> bisect = [&](size_t lo, size_t hi) {
        if (lo > hi) return;
        size_t mid = lo + (hi - lo) / 2;
        schedule.push_back(mid);
        if (mid > lo) bisect(lo, mid - 1);
        bisect(mid + 1, hi);
    };
    bisect(lo, hi);
    return schedule;
}

inline std::vector<size_t> armMeshBisect(size_t numThreads, const ArmMeshConfig& cfg) {
    if (numThreads > cfg.totalPUs())
        throw std::runtime_error("numThreads " + std::to_string(numThreads) +
                                 " exceeds available PUs " + std::to_string(cfg.totalPUs()));

    auto groupOrder = bisectSchedule(0, cfg.totalL2Groups() - 1);

    std::vector<size_t> schedule;
    schedule.reserve(numThreads);

    for (size_t coreInGroup = 0; coreInGroup < cfg.coresPerL2Group
                                 && schedule.size() < numThreads; ++coreInGroup)
        for (size_t group : groupOrder) {
            if (schedule.size() >= numThreads) break;
            schedule.push_back(group * cfg.coresPerL2Group + coreInGroup);
        }

    return schedule;
}

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
#elif ARM_BISECT
   auto schedule = armMeshBisect(size - 1, graviton4());
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
      threads.emplace_back(this, f, barriers[group]);
      auto worker = &threads.back();
#ifdef NEW_POLICY
      size_t selection = schedule[i];
#else
      size_t selection = i;
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
   this_worker->group = this;
   this_worker->barrier = barriers.back();
   currentBarrier = 0;
   f();
   this_worker->group = prevGroup;
   currentBarrier = prevBarrier;
   this_worker->barrier = prevBarrierPtr;

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
