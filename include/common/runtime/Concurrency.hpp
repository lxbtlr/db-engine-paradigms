#pragma once
#include "Barrier.hpp"
#include "common/Compat.hpp"
#include "common/runtime/Barrier.hpp"
#include "common/runtime/MemoryPool.hpp"
#include "tbb/task_group.h"
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

constexpr size_t SOCKETS_COUNT = CFG_SOCKETS_COUNT;
constexpr size_t CORES_PER_SOCKET = CFG_CORES_PER_SOCKET;
constexpr size_t SMT_PER_CORE = CFG_SMT_PER_CORE;
constexpr size_t NUM_NUMA_REGIONS = SOCKETS_COUNT;
constexpr size_t THREADS_PER_SOCKET = CORES_PER_SOCKET * SMT_PER_CORE;

/// Map a thread id to the CPU id it should be pinned to.
/// Two layouts:
///   Interleaved (default): CPU c -> node (c % SOCKETS_COUNT)
///     thread i in packed mode: cpu = (i % TPS) * SOCKETS_COUNT + (i / TPS)
///     thread i in spread mode: cpu = i  (sequential IDs round-robin across nodes)
///   Contiguous (CPU_LAYOUT_CONTIGUOUS): CPU c -> node (c / THREADS_PER_SOCKET)
///     thread i in packed mode: cpu = i  (contiguous blocks already match)
///     thread i in spread mode: cpu = (i % SOCKETS_COUNT) * TPS + (i / SOCKETS_COUNT)
inline size_t cpuOfThread(size_t tid) {
#if defined(CPU_LAYOUT_CONTIGUOUS)
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
   size_t socket = tid / THREADS_PER_SOCKET;
   size_t j = tid % THREADS_PER_SOCKET;
   return j * SOCKETS_COUNT + socket;
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

/// Map a thread id to its NUMA region (convenience: nodeOfCpu(cpuOfThread(tid))).
inline size_t regionOf(size_t tid) {
#ifdef THREAD_PIN_PACKED
   return tid / THREADS_PER_SOCKET;
#else
   return tid % SOCKETS_COUNT;
#endif
}

#if defined(NUMA_ALLOC) || defined(NUMA_SHARD) || defined(NUMA_DEBUG)
/// Validate at startup that the compile-time topology constants match the
/// actual hardware.  Aborts if nodeOfCpu(c) doesn't match numa_node_of_cpu(c)
/// for any CPU, or if the node count differs from SOCKETS_COUNT.
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
   tbb::task_group g;
   auto barriers = HierarchicBarrier::create(size);
   int64_t group = -1;
   for (size_t i = 0; i < size - 1; ++i) {
      if (i % HierarchicBarrier::threadsPerBarrier == 0) ++group;
      threads.emplace_back(this, f, barriers[group]);
      auto worker = &threads.back();
      worker->worker_id = i;
      g.run([worker, i]() {

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
