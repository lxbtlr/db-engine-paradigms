#include "common/runtime/Concurrency.hpp"

#if defined(NUMA_ALLOC) || defined(NUMA_SHARD) || defined(NUMA_DEBUG)
#include <numa.h>
#include <cstdio>
#include <cstdlib>
#endif

namespace runtime {

thread_local Worker* this_worker;
thread_local bool currentBarrier = false;

WorkerGroup mainGroup(1);
HierarchicBarrier mainBarrier(1, nullptr);
GlobalPool defaultPool;
Worker mainWorker(&mainGroup, &mainBarrier, defaultPool);

#if defined(NUMA_ALLOC) || defined(NUMA_SHARD) || defined(NUMA_DEBUG)
void assertTopology() {
   if (numa_available() < 0) {
      fprintf(stderr, "assertTopology: libnuma reports NUMA not available\n");
      abort();
   }
   int maxNode = numa_max_node();
   if (static_cast<size_t>(maxNode + 1) != SOCKETS_COUNT) {
      fprintf(stderr,
              "assertTopology: numa_max_node()=%d but SOCKETS_COUNT=%zu\n",
              maxNode, SOCKETS_COUNT);
      abort();
   }
   size_t nCpus = SOCKETS_COUNT * CORES_PER_SOCKET * SMT_PER_CORE;
   for (size_t c = 0; c < nCpus; ++c) {
      int node = numa_node_of_cpu(static_cast<int>(c));
      size_t expected = nodeOfCpu(c);
      if (node < 0 || static_cast<size_t>(node) != expected) {
         fprintf(stderr,
                 "assertTopology: cpu %zu on node %d, expected node %zu\n",
                 c, node, expected);
         abort();
      }
   }
   fprintf(stderr, "assertTopology: verified %zu CPUs across %zu nodes OK\n",
           nCpus, SOCKETS_COUNT);
}
#endif

} // namespace runtime
