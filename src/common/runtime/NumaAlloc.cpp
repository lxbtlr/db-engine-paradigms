#ifdef NUMA_ALLOC

#include "common/runtime/NumaAlloc.hpp"
#include "common/runtime/Concurrency.hpp"
#include "common/runtime/Memory.hpp"
#include <cstring>
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

// Parallel memcpy: split the byte range across nWorkers threads, each pinned
// to a different core on NUMA region r so first-touch places pages locally.
static void parallelMemcpyOnRegion(void* dst, const void* src, size_t bytes,
                                   size_t r, size_t nWorkers) {
   if (nWorkers <= 1) {
      pinToRegion(r, 0);
      std::memcpy(dst, src, bytes);
      return;
   }
   std::vector<std::thread> workers;
   workers.reserve(nWorkers);
   // Align chunk boundaries to 4KB pages for clean first-touch placement
   constexpr size_t PAGE = 4096;
   size_t chunkBase = (bytes / nWorkers) & ~(PAGE - 1);
   for (size_t w = 0; w < nWorkers; ++w) {
      size_t off = w * chunkBase;
      size_t len = (w + 1 == nWorkers) ? (bytes - off) : chunkBase;
      workers.emplace_back([dst, src, off, len, r, w]() {
         pinToRegion(r, w);
         std::memcpy(static_cast<char*>(dst) + off,
                     static_cast<const char*>(src) + off, len);
      });
   }
   for (auto& t : workers) t.join();
}

void numaReplicateRelation(Relation& rel) {
   // Use all cores on each socket for parallel first-touch memcpy
   constexpr size_t workersPerRegion = CORES_PER_SOCKET;
   std::vector<std::thread> threads;
   threads.reserve(NUM_NUMA_REGIONS);

   for (size_t r = 0; r < NUM_NUMA_REGIONS; ++r) {
      threads.emplace_back([&rel, r]() {
         pinToRegion(r, 0);
         auto& replica = rel.numaReplicas[r];

         for (auto& kv : rel.attributes) {
            auto& attrName = kv.first;
            auto& attr = kv.second;
            size_t bytes = rel.nrTuples * attr.type->rt_size();

            void* p = mem::malloc_huge(bytes);
            parallelMemcpyOnRegion(p, attr.data(), bytes, r, workersPerRegion);

            replica.columns[attrName] = p;
            replica.mmaps.emplace_back(p, bytes);
         }
      });
   }

   for (auto& t : threads) t.join();
   rel.hasNumaReplicas = true;
}

void numaFreeReplicas(Relation& rel) {
   for (size_t r = 0; r < NUM_NUMA_REGIONS; ++r) {
      for (auto& m : rel.numaReplicas[r].mmaps) {
         mem::free_huge(m.first, m.second);
      }
      rel.numaReplicas[r].columns.clear();
      rel.numaReplicas[r].mmaps.clear();
   }
   rel.hasNumaReplicas = false;
}

#ifdef NUMA_DEBUG
void verifyNumaPlacement(Relation& rel) {
   for (size_t r = 0; r < NUM_NUMA_REGIONS; ++r) {
      for (auto& kv : rel.numaReplicas[r].columns) {
         constexpr size_t pageSize = 4096;
         constexpr size_t nPages = 4;
         void* pages[nPages];
         int status[nPages];
         for (size_t p = 0; p < nPages; ++p)
            pages[p] = (char*)kv.second + p * pageSize;
         move_pages(0, nPages, pages, nullptr, status, 0);
         fprintf(stderr, "region %zu col %-20s: pages on nodes %d %d %d %d\n",
                 r, kv.first.c_str(), status[0], status[1], status[2], status[3]);
      }
   }
}
#endif

} // namespace runtime

#endif // NUMA_ALLOC
