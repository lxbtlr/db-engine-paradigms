#include "common/runtime/Query.hpp"
#include <deque>
#include <tbb/tbb.h>
#ifdef NUMA_SHARD
#include "common/runtime/Concurrency.hpp"
#include "common/runtime/Database.hpp"
#include <array>
#include <atomic>
#endif

static const size_t morselSize = 100000;

struct ProcessingResources {
   std::vector<runtime::Worker> workers;
   std::unique_ptr<runtime::Query> query;
};

inline ProcessingResources initQuery(size_t nrThreads) {
   ProcessingResources r;
   r.query = std::make_unique<runtime::Query>();
   r.query->result = std::make_unique<runtime::BlockRelation>();
   runtime::Barrier b(nrThreads);
   r.workers.resize(nrThreads);

   tbb::parallel_for(size_t(0), nrThreads, size_t(1), [&](auto i) {
      auto& worker = r.workers[i];
      // save thread local worker pointer
      worker.previousWorker = runtime::this_worker;
      // use this worker resource
      runtime::this_worker = &worker;
      r.query->participate();
      b.wait();
   });
   return r;
}

inline void leaveQuery(size_t nrThreads) {
   runtime::Barrier b(nrThreads);
   tbb::parallel_for(size_t(0), nrThreads, size_t(1), [&](auto) {
      // reset thread local worker pointer
      runtime::this_worker = runtime::this_worker->previousWorker;
      b.wait();
   });
}

#define PARALLEL_SCAN(N, ENTRIES, BLOCK)                                       \
   tbb::parallel_for(tbb::blocked_range<size_t>(0, N, morselSize),             \
                     [&](const tbb::blocked_range<size_t>& r) {                \
                        auto& entries = ENTRIES.local();                       \
                        for (auto i = r.begin(), end = r.end(); i != end; ++i) \
                           BLOCK                                               \
                     })

template <typename E, typename L>
void parallel_scan(size_t n, E& entriesGlobal, L& cb) {
   tbb::parallel_for(tbb::blocked_range<size_t>(0, n, morselSize),
                     [&](const tbb::blocked_range<size_t>& r) {
                        auto& entries = entriesGlobal.local();
                        for (auto i = r.begin(), end = r.end(); i != end; ++i)
                           cb(i, entries);
                     });
}

#define PARALLEL_SELECT(N, ENTRIES, BLOCK)                                     \
   tbb::parallel_reduce(                                                       \
       tbb::blocked_range<size_t>(0, N, morselSize), 0,                        \
       [&](const tbb::blocked_range<size_t>& r, const size_t& f) {             \
          auto& entries = ENTRIES.local();                                     \
          auto found = f;                                                      \
          for (size_t i = r.begin(), end = r.end(); i != end; ++i) BLOCK       \
          return found;                                                        \
       },                                                                      \
       [](const size_t& a, const size_t& b) { return a + b; })

template <typename E, typename HT> void parallel_insert(E& entries, HT& ht) {
   tbb::parallel_for(entries.range(), [&ht](const auto& r) {
      for (auto& entries : r) ht.insertAll(entries);
   });
}

#ifdef NUMA_SHARD
/// Extract per-shard column pointers for a NUMA-sharded relation.
template <typename T>
std::array<const T*, runtime::NUM_NUMA_REGIONS>
numaShardPtrs(const runtime::Relation& rel, const std::string& attr) {
   std::array<const T*, runtime::NUM_NUMA_REGIONS> ptrs;
   for (size_t r = 0; r < runtime::NUM_NUMA_REGIONS; ++r)
      ptrs[r] = static_cast<const T*>(rel.numaShards[r].columns.at(attr));
   return ptrs;
}

/// Per-node morsel counter, cache-line padded to avoid false sharing.
struct alignas(64) PaddedAtomicPos {
   std::atomic<size_t> pos{0};
};

/// NUMA-aware parallel scan with ring-and-rank work stealing.
/// Replaces tbb::parallel_for for sharded scan loops.
/// Body signature: void(size_t begin, size_t end, size_t node)
///   begin/end = shard-local tuple range (0-based within shard)
///   node      = NUMA region the morsel belongs to
template <typename Body>
void numa_parallel_scan(size_t nrThreads, const runtime::Relation& rel,
                        size_t morselSz, Body body) {
   constexpr size_t N = runtime::NUM_NUMA_REGIONS;

   std::array<size_t, N> shardTuples;
   for (size_t r = 0; r < N; ++r)
      shardTuples[r] = rel.numaShards[r].tupleEnd - rel.numaShards[r].tupleBegin;

   std::array<PaddedAtomicPos, N> nodePos;

   tbb::parallel_for(size_t(0), nrThreads, size_t(1), [&](size_t /*tid*/) {
      // Use the actual TBB slot (which PinningObserver pinned to a specific
      // CPU) rather than the loop variable, which TBB may assign to any worker.
      size_t slot = static_cast<size_t>(
          tbb::this_task_arena::current_thread_index());
      size_t home = runtime::regionOf(slot);

      size_t rank;
#ifdef THREAD_PIN_PACKED
      rank = slot % runtime::THREADS_PER_SOCKET;
#else
      rank = slot / runtime::SOCKETS_COUNT;
#endif

      std::array<size_t, N> stealOrder;
      stealOrder[0] = home;
      for (size_t k = 0; k < N - 1; ++k)
         stealOrder[k + 1] = (home + 1 + ((rank + k) % (N - 1))) % N;

      for (size_t si = 0; si < N; ++si) {
         size_t node = stealOrder[si];
         size_t nodeTup = shardTuples[node];
         if (nodeTup == 0) continue;

         while (true) {
            size_t chunk = nodePos[node].pos.fetch_add(1, std::memory_order_relaxed);
            size_t begin = chunk * morselSz;
            if (begin >= nodeTup) break;
            size_t end = std::min(begin + morselSz, nodeTup);
            body(begin, end, node);
         }
      }
   });
}

/// NUMA-aware parallel reduce with ring-and-rank work stealing.
/// Replaces tbb::parallel_reduce for sharded scan+aggregate loops.
/// Body signature: void(size_t begin, size_t end, size_t node, T& acc)
/// Combine signature: T(T a, T b)
template <typename T, typename Body, typename Combine>
T numa_parallel_reduce(size_t nrThreads, const runtime::Relation& rel,
                       size_t morselSz, T identity, Body body,
                       Combine combine) {
   constexpr size_t N = runtime::NUM_NUMA_REGIONS;

   std::array<size_t, N> shardTuples;
   for (size_t r = 0; r < N; ++r)
      shardTuples[r] = rel.numaShards[r].tupleEnd - rel.numaShards[r].tupleBegin;

   std::array<PaddedAtomicPos, N> nodePos;

   // One task per thread; each resolves its NUMA identity from the actual
   // TBB slot (pinned by PinningObserver), not from the iteration variable.
   tbb::enumerable_thread_specific<T> locals(identity);

   tbb::parallel_for(size_t(0), nrThreads, size_t(1), [&](size_t /*tid*/) {
      size_t slot = static_cast<size_t>(
          tbb::this_task_arena::current_thread_index());
      size_t home = runtime::regionOf(slot);

      size_t rank;
#ifdef THREAD_PIN_PACKED
      rank = slot % runtime::THREADS_PER_SOCKET;
#else
      rank = slot / runtime::SOCKETS_COUNT;
#endif

      std::array<size_t, N> stealOrder;
      stealOrder[0] = home;
      for (size_t k = 0; k < N - 1; ++k)
         stealOrder[k + 1] = (home + 1 + ((rank + k) % (N - 1))) % N;

      T& acc = locals.local();
      for (size_t si = 0; si < N; ++si) {
         size_t node = stealOrder[si];
         size_t nodeTup = shardTuples[node];
         if (nodeTup == 0) continue;

         while (true) {
            size_t chunk =
                nodePos[node].pos.fetch_add(1, std::memory_order_relaxed);
            size_t begin = chunk * morselSz;
            if (begin >= nodeTup) break;
            size_t end = std::min(begin + morselSz, nodeTup);
            body(begin, end, node, acc);
         }
      }
   });

   T result = identity;
   for (auto& local : locals)
      result = combine(result, local);
   return result;
}
#endif // NUMA_SHARD
