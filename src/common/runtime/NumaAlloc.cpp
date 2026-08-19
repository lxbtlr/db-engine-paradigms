#if defined(NUMA_ALLOC) || defined(NUMA_SHARD)

#include "common/runtime/NumaAlloc.hpp"
#include "common/runtime/Concurrency.hpp"
#include "common/runtime/Memory.hpp"
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <pthread.h>
#include <thread>
#include <vector>
#ifdef NUMA_DEBUG
#include <cstdio>
#include <numaif.h>
#endif

namespace runtime {

// Pin the calling thread to a specific CPU on NUMA region r.
// CPU topology: CPU c is on NUMA node (c % SOCKETS_COUNT).
// logical index j on socket r -> CPU = j * SOCKETS_COUNT + r
static void pinToRegion(size_t r, size_t j = 0) {
   cpu_set_t cpuset;
   CPU_ZERO(&cpuset);
   CPU_SET(j * SOCKETS_COUNT + r, &cpuset);
   pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

// A pool of pinned worker threads for a single NUMA region.
// Created once, reused across all columns, then joined.
struct RegionPool {
   struct Work {
      void* dst;
      const void* src;
      size_t off;
      size_t len;
   };

   size_t nWorkers;
   size_t region;
   std::vector<std::thread> threads;

   // Synchronisation: main sets work items then increments generation;
   // workers spin-wait on generation, do their chunk, then signal done.
   std::mutex mu;
   std::condition_variable cv_work;   // workers wait for new work
   std::condition_variable cv_done;   // main waits for workers to finish
   std::vector<Work> items;           // one per worker
   size_t generation = 0;            // bumped each dispatch
   std::atomic<size_t> doneCount{0};
   bool shutdown = false;

   RegionPool(size_t nw, size_t r) : nWorkers(nw), region(r) {
      items.resize(nw);
      threads.reserve(nw);
      for (size_t w = 0; w < nw; ++w) {
         threads.emplace_back([this, w]() { workerLoop(w); });
      }
   }

   void workerLoop(size_t w) {
      pinToRegion(region, w);
      size_t myGen = 0;
      for (;;) {
         {
            std::unique_lock<std::mutex> lk(mu);
            cv_work.wait(lk, [&] { return generation > myGen || shutdown; });
            if (shutdown) return;
            myGen = generation;
         }
         auto& item = items[w];
         if (item.len > 0)
            std::memcpy(static_cast<char*>(item.dst) + item.off,
                        static_cast<const char*>(item.src) + item.off,
                        item.len);
         if (doneCount.fetch_add(1, std::memory_order_acq_rel) + 1 == nWorkers)
            cv_done.notify_one();
      }
   }

   void dispatch(void* dst, const void* src, size_t bytes) {
      constexpr size_t PAGE = 4096;
      size_t chunkBase = (bytes / nWorkers) & ~(PAGE - 1);
      {
         std::lock_guard<std::mutex> lk(mu);
         for (size_t w = 0; w < nWorkers; ++w) {
            size_t off = w * chunkBase;
            size_t len = (w + 1 == nWorkers) ? (bytes - off) : chunkBase;
            items[w] = {dst, src, off, len};
         }
         doneCount.store(0, std::memory_order_relaxed);
         ++generation;
      }
      cv_work.notify_all();
      // Wait for all workers to finish
      std::unique_lock<std::mutex> lk(mu);
      cv_done.wait(lk, [&] {
         return doneCount.load(std::memory_order_acquire) == nWorkers;
      });
   }

   void join() {
      {
         std::lock_guard<std::mutex> lk(mu);
         shutdown = true;
      }
      cv_work.notify_all();
      for (auto& t : threads) t.join();
   }
};

#ifdef NUMA_ALLOC
void numaReplicateRelation(Relation& rel) {
   constexpr size_t workersPerRegion = CORES_PER_SOCKET;
   std::vector<std::thread> threads;
   threads.reserve(NUM_NUMA_REGIONS);

   for (size_t r = 0; r < NUM_NUMA_REGIONS; ++r) {
      threads.emplace_back([&rel, r]() {
         pinToRegion(r, 0);
         auto& replica = rel.numaReplicas[r];

         RegionPool pool(workersPerRegion, r);

         for (auto& kv : rel.attributes) {
            auto& attrName = kv.first;
            auto& attr = kv.second;
            size_t bytes = rel.nrTuples * attr.type->rt_size();

            void* p = mem::malloc_numa(bytes, static_cast<int>(r));
            size_t allocSize = mem::malloc_numa_size(bytes);
            pool.dispatch(p, attr.data(), bytes);
            mem::populate_pages(p, bytes);

            replica.columns[attrName] = p;
            replica.mmaps.emplace_back(p, allocSize);
         }

         pool.join();
      });
   }

   for (auto& t : threads) t.join();
   rel.hasNumaReplicas = true;
}

void numaFreeReplicas(Relation& rel) {
   for (size_t r = 0; r < NUM_NUMA_REGIONS; ++r) {
      for (auto& m : rel.numaReplicas[r].mmaps) {
         munmap(m.first, m.second); // m.second is already malloc_numa_size(bytes)
      }
      rel.numaReplicas[r].columns.clear();
      rel.numaReplicas[r].mmaps.clear();
   }
   rel.hasNumaReplicas = false;
}
#endif // NUMA_ALLOC

#ifdef NUMA_SHARD
void numaShardRelation(Relation& rel) {
   constexpr size_t N = NUM_NUMA_REGIONS;
   constexpr size_t workersPerRegion = CORES_PER_SOCKET;
   size_t totalTuples = rel.nrTuples;

   // Compute shard tuple boundaries (exact split, no alignment requirement
   // on tuple boundaries — mmap sizes are rounded up per column)
   size_t baseShard = totalTuples / N;
   size_t remainder = totalTuples % N;

   // Set up ranges: first 'remainder' shards get baseShard+1 tuples
   size_t cursor = 0;
   for (size_t r = 0; r < N; ++r) {
      size_t count = baseShard + (r < remainder ? 1 : 0);
      rel.numaShards[r].tupleBegin = cursor;
      rel.numaShards[r].tupleEnd = cursor + count;
      cursor += count;
   }

   // Allocate, bind, and load each shard in parallel across regions
   std::vector<std::thread> threads;
   threads.reserve(N);

   for (size_t r = 0; r < N; ++r) {
      threads.emplace_back([&rel, r, workersPerRegion]() {
         auto& shard = rel.numaShards[r];
         size_t shardTuples = shard.tupleEnd - shard.tupleBegin;
         if (shardTuples == 0) return;

         pinToRegion(r, 0);
         RegionPool pool(workersPerRegion, r);

         for (auto& kv : rel.attributes) {
            auto& attrName = kv.first;
            auto& attr = kv.second;
            size_t elemSize = attr.type->rt_size();
            size_t bytes = shardTuples * elemSize;

            void* p = mem::malloc_numa(bytes, static_cast<int>(r));
            size_t allocSize = mem::malloc_numa_size(bytes);

            const char* src =
                static_cast<const char*>(attr.data()) +
                shard.tupleBegin * elemSize;
            pool.dispatch(p, src, bytes);
            mem::populate_pages(p, bytes);

            shard.columns[attrName] = p;
            shard.mmaps.emplace_back(p, allocSize);
         }

         pool.join();
      });
   }

   for (auto& t : threads) t.join();
   rel.hasNumaShards = true;
}

void numaFreeShards(Relation& rel) {
   for (size_t r = 0; r < NUM_NUMA_REGIONS; ++r) {
      for (auto& m : rel.numaShards[r].mmaps) {
         munmap(m.first, m.second); // m.second is already malloc_numa_size(bytes)
      }
      rel.numaShards[r].columns.clear();
      rel.numaShards[r].mmaps.clear();
   }
   rel.hasNumaShards = false;
}
#endif // NUMA_SHARD

#ifdef NUMA_DEBUG
void verifyNumaPlacement(Relation& rel) {
   constexpr size_t STRIDE = 2 * 1024 * 1024; // 2MB stride
   constexpr size_t BATCH = 512;               // pages per move_pages call
   bool anyFail = false;

   auto verifyShard = [&](size_t r, const char* label,
                          const std::unordered_map<std::string, void*>& columns,
                          size_t nrTuples,
                          const std::unordered_map<std::string, Attribute>& attrs) {
      for (auto& kv : columns) {
         auto& attrName = kv.first;
         void* base = kv.second;
         auto it = attrs.find(attrName);
         if (it == attrs.end()) continue;
         size_t elemSize = it->second.type->rt_size();
         size_t bytes = nrTuples * elemSize;

         // Sample pages at STRIDE intervals across the full extent
         size_t nSamples = 0;
         for (size_t off = 0; off < bytes; off += STRIDE) nSamples++;
         if (nSamples == 0) continue;

         size_t onHome = 0;
         size_t total = 0;
         size_t counts[64] = {};

         // Process in batches
         for (size_t batchStart = 0; batchStart < nSamples;
              batchStart += BATCH) {
            size_t batchN = std::min(BATCH, nSamples - batchStart);
            void* pages[BATCH];
            int status[BATCH];
            for (size_t i = 0; i < batchN; ++i)
               pages[i] =
                   static_cast<char*>(base) + (batchStart + i) * STRIDE;
            move_pages(0, static_cast<long>(batchN), pages, nullptr,
                       status, 0);
            for (size_t i = 0; i < batchN; ++i) {
               if (status[i] >= 0 &&
                   static_cast<size_t>(status[i]) < 64)
                  counts[status[i]]++;
               total++;
            }
         }
         onHome = counts[r];
         double pct = total > 0 ? 100.0 * onHome / total : 0.0;
         fprintf(stderr,
                 "%s %zu col %-20s: %zu/%zu pages on home node %zu (%.1f%%)",
                 label, r, attrName.c_str(), onHome, total, r, pct);
         // Print non-home nodes if any
         for (size_t n = 0; n < 64; ++n) {
            if (n != r && counts[n] > 0)
               fprintf(stderr, " [node%zu=%zu]", n, counts[n]);
         }
         fprintf(stderr, "\n");
         if (pct < 95.0) {
            fprintf(stderr,
                    "ERROR: %s %zu col %s has only %.1f%% on home node "
                    "(threshold 95%%)\n",
                    label, r, attrName.c_str(), pct);
            anyFail = true;
         }
      }
   };

#ifdef NUMA_ALLOC
   if (rel.hasNumaReplicas) {
      for (size_t r = 0; r < NUM_NUMA_REGIONS; ++r) {
         verifyShard(r, "replica", rel.numaReplicas[r].columns,
                     rel.nrTuples, rel.attributes);
      }
   }
#endif
#ifdef NUMA_SHARD
   if (rel.hasNumaShards) {
      for (size_t r = 0; r < NUM_NUMA_REGIONS; ++r) {
         size_t shardTuples =
             rel.numaShards[r].tupleEnd - rel.numaShards[r].tupleBegin;
         verifyShard(r, "shard", rel.numaShards[r].columns, shardTuples,
                     rel.attributes);
      }
   }
#endif

   if (anyFail) {
      fprintf(stderr, "NUMA placement verification FAILED — aborting\n");
      abort();
   }
}
#endif // NUMA_DEBUG

} // namespace runtime

#endif // NUMA_ALLOC || NUMA_SHARD
