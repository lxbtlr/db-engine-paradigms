#pragma once
#include "Barrier.hpp"
#include "common/Compat.hpp"
#include "common/runtime/Barrier.hpp"
#include "common/runtime/MemoryPool.hpp"
#include <vector>
#include <array>
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

// Topology constants — set via -DCFG_SOCKETS_COUNT etc. from CMake
// (TARGET_MACHINE or manual cache variables).  Defaults for standalone builds.
#ifndef CFG_SOCKETS_COUNT
#define CFG_SOCKETS_COUNT 4
#endif
#ifndef CFG_CORES_PER_SOCKET
#define CFG_CORES_PER_SOCKET 22
#endif
#ifndef CFG_SMT_PER_CORE
#define CFG_SMT_PER_CORE 2
#endif
// NUM_REGIONS: number of memory/L3 domains for sharding.
// Equals SOCKETS_COUNT on multi-socket (dubliner=4, manchego=2),
// or CCDS_PER_SOCKET on single-socket multi-CCD (roquefort=4).
// Set explicitly per machine preset in CMakeLists.txt.
#ifndef CFG_NUM_REGIONS
#define CFG_NUM_REGIONS CFG_SOCKETS_COUNT
#endif

constexpr size_t SOCKETS_COUNT = CFG_SOCKETS_COUNT;
constexpr size_t CORES_PER_SOCKET = CFG_CORES_PER_SOCKET;
constexpr size_t SMT_PER_CORE = CFG_SMT_PER_CORE;
constexpr size_t NUM_NUMA_REGIONS = CFG_NUM_REGIONS;
constexpr size_t THREADS_PER_SOCKET = CORES_PER_SOCKET * SMT_PER_CORE;
constexpr size_t THREADS_PER_REGION =
    THREADS_PER_SOCKET * SOCKETS_COUNT / NUM_NUMA_REGIONS;
constexpr size_t CORES_PER_REGION =
    CORES_PER_SOCKET * SOCKETS_COUNT / NUM_NUMA_REGIONS;

/// Map a thread id to the CPU id it should be pinned to.
///
/// Three layout families:
///
///   1. Interleaved (default): CPU c -> node (c % SOCKETS_COUNT)
///        NUM_NUMA_REGIONS == SOCKETS_COUNT (regions are sockets)
///
///   2. Contiguous (CPU_LAYOUT_CONTIGUOUS): cores numbered contiguously per
///        socket, siblings follow primaries within each socket block.
///        NUM_NUMA_REGIONS == SOCKETS_COUNT (regions are sockets)
///
///   3. CCD (CPU_LAYOUT_CONTIGUOUS + NUM_REGIONS > SOCKETS_COUNT):
///        Sub-socket L3 domains.  Primaries grouped first across all CCDs,
///        then siblings.  Example (roquefort, 4 CCDs of 6 cores):
///          CCD 0: primaries 0-5,   siblings 24-29
///          CCD 1: primaries 6-11,  siblings 30-35
///          CCD 2: primaries 12-17, siblings 36-41
///          CCD 3: primaries 18-23, siblings 42-47
///
inline size_t cpuOfThread(size_t tid) {
#if defined(CPU_LAYOUT_CONTIGUOUS)
   // Contiguous: sub-socket regions (CCD) or socket-level regions
#ifdef THREAD_PIN_PACKED
   // Packed: fill one region before moving to next, primaries before siblings
   size_t region = tid / THREADS_PER_REGION;
   size_t j = tid % THREADS_PER_REGION;
   size_t core_in_region = j % CORES_PER_REGION;
   size_t smt = j / CORES_PER_REGION;
   // Which socket this region belongs to, and offset within that socket
   size_t socket = (region * CORES_PER_REGION) / CORES_PER_SOCKET;
   size_t core_in_socket =
       (region * CORES_PER_REGION) % CORES_PER_SOCKET + core_in_region;
   return socket * THREADS_PER_SOCKET + core_in_socket +
          smt * CORES_PER_SOCKET;
#else
   // Spread across regions: thread 0->region0, 1->region1, ..., N->region0, ...
   size_t region = tid % NUM_NUMA_REGIONS;
   size_t j = tid / NUM_NUMA_REGIONS;
   size_t core_in_region = j % CORES_PER_REGION;
   size_t smt = j / CORES_PER_REGION;
   size_t socket = (region * CORES_PER_REGION) / CORES_PER_SOCKET;
   size_t core_in_socket =
       (region * CORES_PER_REGION) % CORES_PER_SOCKET + core_in_region;
   return socket * THREADS_PER_SOCKET + core_in_socket +
          smt * CORES_PER_SOCKET;
#endif
#else // Interleaved (default): NUM_NUMA_REGIONS == SOCKETS_COUNT
#ifdef THREAD_PIN_PACKED
   size_t socket = tid / THREADS_PER_SOCKET;
   size_t j = tid % THREADS_PER_SOCKET;
   return j * SOCKETS_COUNT + socket;
#else
   return tid;
#endif
#endif
}

/// Map a CPU id to its region (NUMA node, CCD, or socket — depending on config).
inline size_t regionOfCpu(size_t cpu) {
#ifdef CPU_LAYOUT_CONTIGUOUS
   // Contiguous: strip SMT layer, then divide by cores-per-region
   size_t socket = cpu / THREADS_PER_SOCKET;
   size_t within_socket = cpu % THREADS_PER_SOCKET;
   size_t core_in_socket = within_socket % CORES_PER_SOCKET;
   return (socket * CORES_PER_SOCKET + core_in_socket) / CORES_PER_REGION;
#else
   return cpu % NUM_NUMA_REGIONS;
#endif
}

/// Legacy alias — assertTopology() uses this name.
inline size_t nodeOfCpu(size_t cpu) { return regionOfCpu(cpu); }

/// Map a thread id to its region (convenience: regionOfCpu(cpuOfThread(tid))).
inline size_t regionOf(size_t tid) {
#ifdef THREAD_PIN_PACKED
   return tid / THREADS_PER_REGION;
#else
   return tid % NUM_NUMA_REGIONS;
#endif
}

/// Count how many distinct regions are occupied by nrThreads threads.
inline size_t activeRegions(size_t nrThreads) {
#ifdef THREAD_PIN_PACKED
   size_t full = nrThreads / THREADS_PER_REGION;
   size_t partial = (nrThreads % THREADS_PER_REGION) > 0 ? 1 : 0;
   size_t n = full + partial;
   return n < NUM_NUMA_REGIONS ? n : NUM_NUMA_REGIONS;
#else
   return nrThreads < NUM_NUMA_REGIONS ? nrThreads : NUM_NUMA_REGIONS;
#endif
}

/// Validate at startup that the compile-time topology constants match the
/// actual hardware.  Aborts if nodeOfCpu(c) doesn't match numa_node_of_cpu(c)
/// for any CPU, or if the node count differs from SOCKETS_COUNT.
void assertTopology();

class Worker;
class WorkerGroup;

extern thread_local Worker* this_worker;
extern GlobalPool defaultPool;
extern thread_local bool currentBarrier;

class Worker
/// information about the worker thread.
/// accessible via thread local 'this_worker'
{
 public:
   Worker* previousWorker;
   WorkerGroup* group;
   Allocator allocator;
   HierarchicBarrier* barrier;
   size_t worker_id = 0;

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

inline void WorkerGroup::run(std::function<void()> f) {
   auto barriers = HierarchicBarrier::create(size);
   int64_t group = -1;
   std::vector<std::thread> pool;
   pool.reserve(size - 1);
   for (size_t i = 0; i < size - 1; ++i) {
      if (i % HierarchicBarrier::threadsPerBarrier == 0) ++group;
      threads.emplace_back(this, f, barriers[group]);
      auto worker = &threads.back();
      worker->worker_id = i;
      pool.emplace_back([worker, i]() {

#ifndef __APPLE__
         pthread_t currentThread = pthread_self();
         pthread_setname_np(currentThread,
                            ("workerPool " + std::to_string(i)).c_str());
         cpu_set_t cpuset;
         CPU_ZERO(&cpuset);
         CPU_SET(cpuOfThread(i), &cpuset);
         if (pthread_setaffinity_np(currentThread, sizeof(cpu_set_t),
                                    &cpuset) != 0) {
            throw std::runtime_error("Could not pin thread " +
                                     std::to_string(i) + " to thread " +
                                     std::to_string(i));
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
   this_worker->worker_id = size - 1;
#ifndef __APPLE__
   // Pin the calling thread to its designated CPU, just like spawned workers
   cpu_set_t callerCpuset;
   CPU_ZERO(&callerCpuset);
   CPU_SET(cpuOfThread(size - 1), &callerCpuset);
   pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &callerCpuset);
#endif
   f();
   this_worker->group = prevGroup;
   currentBarrier = prevBarrier;
   this_worker->barrier = prevBarrierPtr;

   for (auto& t : pool) t.join();

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
