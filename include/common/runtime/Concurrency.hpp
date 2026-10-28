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

#ifndef CFG_NUM_REGIONS
#define CFG_NUM_REGIONS CFG_SOCKETS_COUNT
#endif

constexpr size_t SOCKETS_COUNT = CFG_SOCKETS_COUNT;
constexpr size_t CORES_PER_SOCKET = CFG_CORES_PER_SOCKET;
constexpr size_t SMT_PER_CORE = CFG_SMT_PER_CORE;
/// Regions are what NUMA_SHARD / NUMA_ALLOC place data and schedule by: the
/// sockets by default, or sub-socket L3 domains (CCDs) with NUM_REGIONS >
/// SOCKETS_COUNT, which needs CPU_LAYOUT_CONTIGUOUS.
constexpr size_t NUM_NUMA_REGIONS = CFG_NUM_REGIONS;
constexpr size_t THREADS_PER_SOCKET = CORES_PER_SOCKET * SMT_PER_CORE;
constexpr size_t PHYSICAL_CORES = SOCKETS_COUNT * CORES_PER_SOCKET;
constexpr size_t CORES_PER_REGION = PHYSICAL_CORES / NUM_NUMA_REGIONS;
constexpr bool SUB_SOCKET_REGIONS = NUM_NUMA_REGIONS > SOCKETS_COUNT;
static_assert(NUM_NUMA_REGIONS % SOCKETS_COUNT == 0 &&
                  CORES_PER_SOCKET % (NUM_NUMA_REGIONS / SOCKETS_COUNT) == 0,
              "regions must split each socket's cores evenly");
#ifndef CPU_LAYOUT_CONTIGUOUS
static_assert(!SUB_SOCKET_REGIONS,
              "sub-socket regions need CPU_LAYOUT_CONTIGUOUS");
#endif

/// Map a thread id to the CPU id it should be pinned to.
/// Two layouts:
///   Interleaved (default): CPU c -> node (c % SOCKETS_COUNT); the SMT
///   siblings of CPUs 0..P-1 (P = SOCKETS_COUNT * CORES_PER_SOCKET physical
///   cores) are P..2P-1, as on dubliner and manchego
///     thread i in packed mode: the physical cores of socket 0, then of
///       socket 1, ...; SMT siblings only from thread P on, in the same order
///     thread i in spread mode: cpu = i  (sequential IDs round-robin across nodes)
///   Contiguous (CPU_LAYOUT_CONTIGUOUS): CPU c -> node (c / THREADS_PER_SOCKET)
///     thread i in packed mode: cpu = i  (contiguous blocks already match)
///     thread i in spread mode: cpu = (i % SOCKETS_COUNT) * TPS + (i / SOCKETS_COUNT)
///   Contiguous with sub-socket regions (NUM_REGIONS > SOCKETS_COUNT): a
///   region is a block of CORES_PER_REGION cores, and a socket's SMT siblings
///   follow its primaries (CPU s*TPS + smt*C + core), as on roquefort's CCDs
///     packed: the physical cores of region 0, then of region 1, ...; SMT
///       siblings only from thread P on, in the same order
///     spread: round-robin across regions, physical cores before siblings
inline size_t cpuOfThread(size_t tid) {
#if defined(CPU_LAYOUT_CONTIGUOUS)
   if constexpr (SUB_SOCKET_REGIONS) {
#ifdef THREAD_PIN_PACKED
      size_t smt = tid / PHYSICAL_CORES, core = tid % PHYSICAL_CORES;
#else
      size_t region = tid % NUM_NUMA_REGIONS, j = tid / NUM_NUMA_REGIONS;
      size_t smt = j / CORES_PER_REGION;
      size_t core = region * CORES_PER_REGION + j % CORES_PER_REGION;
#endif
      return core / CORES_PER_SOCKET * THREADS_PER_SOCKET +
             smt * CORES_PER_SOCKET + core % CORES_PER_SOCKET;
   }
#ifdef THREAD_PIN_PACKED
   return tid;
#else
   // Spread across nodes: thread 0->node0, 1->node1, ..., N->node0, ...
   size_t node = tid % SOCKETS_COUNT;
   size_t j = tid / SOCKETS_COUNT;
   return node * THREADS_PER_SOCKET + j;
#endif
#else // Interleaved (default)
#ifdef THREAD_PIN_PACKED
   size_t smt = tid / PHYSICAL_CORES, t = tid % PHYSICAL_CORES;
   size_t socket = t / CORES_PER_SOCKET, core = t % CORES_PER_SOCKET;
   return smt * PHYSICAL_CORES + core * SOCKETS_COUNT + socket;
#else
   return tid;
#endif
#endif
}

/// Map a CPU id to its NUMA node / memory domain.
inline size_t nodeOfCpu(size_t cpu) {
#ifdef CPU_LAYOUT_CONTIGUOUS
   return cpu / THREADS_PER_SOCKET;
#else
   return cpu % SOCKETS_COUNT;
#endif
}

/// Map a CPU id to its region (its node, or its CCD with sub-socket regions).
inline size_t regionOfCpu(size_t cpu) {
   if constexpr (SUB_SOCKET_REGIONS) {
      size_t socket = cpu / THREADS_PER_SOCKET;
      size_t core = cpu % THREADS_PER_SOCKET % CORES_PER_SOCKET;
      return (socket * CORES_PER_SOCKET + core) / CORES_PER_REGION;
   }
   return nodeOfCpu(cpu);
}

/// The NUMA node that holds a region's memory.
inline size_t nodeOfRegion(size_t region) {
   return region / (NUM_NUMA_REGIONS / SOCKETS_COUNT);
}

/// Map a thread id to its region (convenience: regionOfCpu(cpuOfThread(tid))).
inline size_t regionOf(size_t tid) {
   if constexpr (SUB_SOCKET_REGIONS) {
#ifdef THREAD_PIN_PACKED
      return tid % PHYSICAL_CORES / CORES_PER_REGION;
#else
      return tid % NUM_NUMA_REGIONS;
#endif
   }
#if defined(THREAD_PIN_PACKED) && defined(CPU_LAYOUT_CONTIGUOUS)
   return tid / THREADS_PER_SOCKET;
#elif defined(THREAD_PIN_PACKED)
   return tid % PHYSICAL_CORES / CORES_PER_SOCKET;
#else
   return tid % SOCKETS_COUNT;
#endif
}

/// Rank of a thread among the threads of its region, in thread id order:
/// the number of lower thread ids with the same regionOf. Work stealing
/// (NUMA_SHARD) uses it to stagger the steal order of a region's threads.
inline size_t rankInRegion(size_t tid) {
   if constexpr (SUB_SOCKET_REGIONS) {
#ifdef THREAD_PIN_PACKED
      return tid / PHYSICAL_CORES * CORES_PER_REGION +
             tid % PHYSICAL_CORES % CORES_PER_REGION;
#else
      return tid / NUM_NUMA_REGIONS;
#endif
   }
#if defined(THREAD_PIN_PACKED) && defined(CPU_LAYOUT_CONTIGUOUS)
   return tid % THREADS_PER_SOCKET;
#elif defined(THREAD_PIN_PACKED)
   return tid / PHYSICAL_CORES * CORES_PER_SOCKET +
          tid % PHYSICAL_CORES % CORES_PER_SOCKET;
#else
   return tid / SOCKETS_COUNT;
#endif
}

/// Count how many distinct regions are occupied by nrThreads threads.
inline size_t activeRegions(size_t nrThreads) {
   bool used[NUM_NUMA_REGIONS] = {};
   size_t n = 0;
   for (size_t t = 0; t < nrThreads && n < NUM_NUMA_REGIONS; ++t)
      if (!used[regionOf(t)]) used[regionOf(t)] = true, ++n;
   return n;
}

#if defined(NUMA_ALLOC) || defined(NUMA_SHARD)
/// Validate at startup that the compile-time topology constants match the
/// hardware (libnuma node of every CPU; L3 sets for sub-socket regions).
/// Aborts on a mismatch.
void assertTopology();
#endif

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
#if (defined(NUMA_ALLOC) || defined(NUMA_SHARD)) && !defined(__APPLE__)
   // NUMA builds: the calling thread runs worker size-1, so it reads that
   // worker's region; pin it there like the spawned workers, and restore
   // its previous affinity afterwards
   cpu_set_t callerPrev, callerCpuset;
   bool callerPinned = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t),
                                              &callerPrev) == 0;
   CPU_ZERO(&callerCpuset);
   CPU_SET(cpuOfThread(size - 1), &callerCpuset);
   pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &callerCpuset);
#endif
   f();
#if (defined(NUMA_ALLOC) || defined(NUMA_SHARD)) && !defined(__APPLE__)
   if (callerPinned)
      pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &callerPrev);
#endif
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
