#ifdef NUMA_ALLOC

#include "common/runtime/NumaAlloc.hpp"
#include "common/runtime/Concurrency.hpp"
#include "common/runtime/Memory.hpp"
#include <cstring>
#include <pthread.h>
#include <thread>
#include <vector>

namespace runtime {

void numaReplicateRelation(Relation& rel) {
   std::vector<std::thread> threads;
   threads.reserve(NUM_NUMA_REGIONS);

   for (size_t r = 0; r < NUM_NUMA_REGIONS; ++r) {
      threads.emplace_back([&rel, r]() {
         // Pin this thread to the first core on NUMA region r
         cpu_set_t cpuset;
         CPU_ZERO(&cpuset);
         size_t cpu = r * CORES_PER_SOCKET; // first physical core on socket r
         CPU_SET(cpu, &cpuset);
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

} // namespace runtime

#endif // NUMA_ALLOC
