#include "common/runtime/Query.hpp"
#include <deque>
#include <tbb/tbb.h>
#include "hyper/NumaTrace.hpp"
#ifdef NUMA_SHARD
#include "common/runtime/Concurrency.hpp"
#include "common/runtime/Database.hpp"
#include <array>
#include <atomic>
#endif

#ifndef CFG_MORSEL_SIZE
#define CFG_MORSEL_SIZE 10000
#endif
static const size_t morselSize = CFG_MORSEL_SIZE;

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

   NT_SCOPE("initQuery");
   tbb::parallel_for(size_t(0), nrThreads, size_t(1), [&](auto i) {
      NT_TASK();
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
   NT_SCOPE("leaveQuery");
   tbb::parallel_for(size_t(0), nrThreads, size_t(1), [&](auto) {
      NT_TASK();
      // reset thread local worker pointer
      runtime::this_worker = runtime::this_worker->previousWorker;
      b.wait();
   });
}

#define PARALLEL_SCAN(N, ENTRIES, BLOCK)                                       \
   NT_SCOPE_TMP("PARALLEL_SCAN"),                                              \
   tbb::parallel_for(tbb::blocked_range<size_t>(0, N, morselSize),             \
                     [&](const tbb::blocked_range<size_t>& r) {                \
                        NT_TASK();                                             \
                        auto& entries = ENTRIES.local();                       \
                        for (auto i = r.begin(), end = r.end(); i != end; ++i) \
                           BLOCK                                               \
                     })

template <typename E, typename L>
void parallel_scan(size_t n, E& entriesGlobal, L& cb) {
   NT_SCOPE("parallel_scan");
   tbb::parallel_for(tbb::blocked_range<size_t>(0, n, morselSize),
                     [&](const tbb::blocked_range<size_t>& r) {
                        NT_TASK();
                        auto& entries = entriesGlobal.local();
                        for (auto i = r.begin(), end = r.end(); i != end; ++i)
                           cb(i, entries);
                     });
}

#define PARALLEL_SELECT(N, ENTRIES, BLOCK)                                     \
   (NT_SCOPE_TMP("PARALLEL_SELECT"),                                           \
   tbb::parallel_reduce(                                                       \
       tbb::blocked_range<size_t>(0, N, morselSize), 0,                        \
       [&](const tbb::blocked_range<size_t>& r, const size_t& f) {             \
          NT_TASK();                                                           \
          auto& entries = ENTRIES.local();                                     \
          auto found = f;                                                      \
          for (size_t i = r.begin(), end = r.end(); i != end; ++i) BLOCK       \
          return found;                                                        \
       },                                                                      \
       [](const size_t& a, const size_t& b) { return a + b; }))

template <typename E, typename HT> void parallel_insert(E& entries, HT& ht) {
   NT_SCOPE("parallel_insert");
   tbb::parallel_for(entries.range(), [&ht](const auto& r) {
      NT_TASK();
      for (auto& entries : r) ht.insertAll(entries);
   });
}

#ifdef NUMA_SHARD
/// Extract per-shard column pointers for a NUMA-sharded relation.
/// Inactive regions (empty shards) get nullptr — the scan helpers never
/// dereference them because shardTuples[r] == 0 for those regions.
template <typename T>
std::array<const T*, runtime::NUM_NUMA_REGIONS>
numaShardPtrs(const runtime::Relation& rel, const std::string& attr) {
   std::array<const T*, runtime::NUM_NUMA_REGIONS> ptrs;
   for (size_t r = 0; r < runtime::NUM_NUMA_REGIONS; ++r) {
      auto it = rel.numaShards[r].columns.find(attr);
      ptrs[r] = (it != rel.numaShards[r].columns.end())
                    ? static_cast<const T*>(it->second)
                    : nullptr;
   }
   return ptrs;
}

/// Per-node morsel counter, cache-line padded to avoid false sharing.
struct alignas(64) PaddedAtomicPos {
   std::atomic<size_t> pos{0};
};

/// Dispatch state of one launcher task of the numa_* helpers: its home
/// region and intra-region rank, the ring-and-rank steal order (as
/// vectorwise::Scan), and claiming of items (morsels, stacks) from per-region
/// counters. items[r] = number of items of region r.
class NumaCursor {
   static constexpr size_t N = runtime::NUM_NUMA_REGIONS;
   std::array<PaddedAtomicPos, N>& pos;
   const std::array<size_t, N>& items;
   std::array<size_t, N> stealOrder;
   size_t si = 0;
#ifdef NUMA_DEBUG
   numa_trace::Task ntTask;
   std::array<size_t, N> ntTaken{};
#endif

 public:
   size_t home;

   NumaCursor(std::array<PaddedAtomicPos, N>& p,
              const std::array<size_t, N>& it)
       : pos(p), items(it) {
      // Use the actual TBB slot (which PinningObserver pinned to a specific
      // CPU) rather than the loop variable, which TBB may assign to any worker.
      size_t slot =
          static_cast<size_t>(tbb::this_task_arena::current_thread_index());
      home = runtime::regionOf(slot);
#ifdef NUMA_DEBUG
      ntTask = numa_trace::task();
#endif
      size_t rank;
#ifdef THREAD_PIN_PACKED
      rank = slot % runtime::THREADS_PER_REGION;
#else
      rank = slot / runtime::NUM_NUMA_REGIONS;
#endif
      stealOrder[0] = home;
      if constexpr (N > 1) {
         for (size_t k = 0; k < N - 1; ++k)
            stealOrder[k + 1] = (home + 1 + ((rank + k) % (N - 1))) % N;
      }
   }
#ifdef NUMA_DEBUG
   ~NumaCursor() { numa_trace::morsels(nullptr, home, ntTask.region, ntTaken); }
#endif

   /// Claim the next item: home region first, then steal in ring order.
   bool next(size_t& node, size_t& item) {
      for (; si < N; ++si) {
         size_t n = stealOrder[si];
         if (items[n] == 0) continue;
         size_t i = pos[n].pos.fetch_add(1, std::memory_order_relaxed);
         if (i < items[n]) {
            node = n;
            item = i;
#ifdef NUMA_DEBUG
            ntTaken[n]++;
#endif
            return true;
         }
      }
      return false;
   }
};

/// Morsels per shard of a NUMA-sharded relation.
inline std::array<size_t, runtime::NUM_NUMA_REGIONS>
numaMorsels(const runtime::Relation& rel, size_t morselSz) {
   std::array<size_t, runtime::NUM_NUMA_REGIONS> m;
   for (size_t r = 0; r < runtime::NUM_NUMA_REGIONS; ++r) {
      size_t t = rel.numaShards[r].tupleEnd - rel.numaShards[r].tupleBegin;
      m[r] = (t + morselSz - 1) / morselSz;
   }
   return m;
}

inline size_t numaShardTuples(const runtime::Relation& rel, size_t node) {
   return rel.numaShards[node].tupleEnd - rel.numaShards[node].tupleBegin;
}

/// NUMA-aware parallel scan with ring-and-rank work stealing.
/// Replaces tbb::parallel_for for sharded scan loops.
/// Body signature: void(size_t begin, size_t end, size_t node)
///   begin/end = shard-local tuple range (0-based within shard)
///   node      = NUMA region the morsel belongs to
template <typename Body>
void numa_parallel_scan(size_t nrThreads, const runtime::Relation& rel,
                        size_t morselSz, Body body) {
   constexpr size_t N = runtime::NUM_NUMA_REGIONS;
   auto morsels = numaMorsels(rel, morselSz);
   std::array<PaddedAtomicPos, N> nodePos;
#ifdef NUMA_DEBUG
   numa_trace::Scope ntScope("numa_parallel_scan", true);
#endif

   tbb::parallel_for(size_t(0), nrThreads, size_t(1), [&](size_t /*tid*/) {
      NumaCursor cursor(nodePos, morsels);
      size_t node, m;
      while (cursor.next(node, m)) {
         size_t begin = m * morselSz;
         size_t end = std::min(begin + morselSz, numaShardTuples(rel, node));
         body(begin, end, node);
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
   auto morsels = numaMorsels(rel, morselSz);
   std::array<PaddedAtomicPos, N> nodePos;
#ifdef NUMA_DEBUG
   numa_trace::Scope ntScope("numa_parallel_reduce", true);
#endif

   // One task per thread; each resolves its NUMA identity from the actual
   // TBB slot (pinned by PinningObserver), not from the iteration variable.
   tbb::enumerable_thread_specific<T> locals(identity);

   tbb::parallel_for(size_t(0), nrThreads, size_t(1), [&](size_t /*tid*/) {
      NumaCursor cursor(nodePos, morsels);
      T& acc = locals.local();
      size_t node, m;
      while (cursor.next(node, m)) {
         size_t begin = m * morselSz;
         size_t end = std::min(begin + morselSz, numaShardTuples(rel, node));
         body(begin, end, node, acc);
      }
   });

   T result = identity;
   for (auto& local : locals)
      result = combine(result, local);
   return result;
}
#endif // NUMA_SHARD
