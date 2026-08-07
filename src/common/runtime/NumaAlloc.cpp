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

void numaReplicateRelation(Relation& rel) {
   std::vector<std::thread> threads;
   threads.reserve(NUM_NUMA_REGIONS);

   for (size_t r = 0; r < NUM_NUMA_REGIONS; ++r) {
      threads.emplace_back([&rel, r]() {
         // Pin this thread to the first core on NUMA region r
         // CPU topology: CPU c is on NUMA node (c % SOCKETS_COUNT)
         // First CPU on socket r is simply r (i.e. j=0 -> cpu = 0*4 + r = r)
         cpu_set_t cpuset;
         CPU_ZERO(&cpuset);
         CPU_SET(r, &cpuset);
         pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

         auto& replica = rel.numaReplicas[r];

         for (auto& kv : rel.attributes) {
            auto& attrName = kv.first;
            auto& attr = kv.second;
            size_t bytes = rel.nrTuples * attr.type->rt_size();

            void* p = mem::malloc_huge(bytes);
            // First-touch: memcpy triggers page faults on this NUMA node
            std::memcpy(p, attr.data(), bytes);

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
