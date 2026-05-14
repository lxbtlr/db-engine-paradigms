#include "common/runtime/Concurrency.hpp"

namespace runtime {

thread_local Worker* this_worker;
thread_local bool currentBarrier = false;

WorkerGroup mainGroup(1);
HierarchicBarrier mainBarrier(1, nullptr);

#ifdef NUMA_POOLS
// Initialize NUMA pool array
GlobalPool numaPools[MAX_NUMA_NODES];
size_t activeNumaNodes = detectNumaNodes();
Worker mainWorker(&mainGroup, &mainBarrier, 0); // Main worker on NUMA node 0
#else
GlobalPool defaultPool;
Worker mainWorker(&mainGroup, &mainBarrier, defaultPool);
#endif

} // namespace runtime
