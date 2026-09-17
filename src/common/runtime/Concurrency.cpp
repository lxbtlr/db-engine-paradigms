#include "common/runtime/Concurrency.hpp"

#include <numa.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace runtime {

thread_local Worker* this_worker;
thread_local bool currentBarrier = false;

WorkerGroup mainGroup(1);
HierarchicBarrier mainBarrier(1, nullptr);
GlobalPool defaultPool;
Worker mainWorker(&mainGroup, &mainBarrier, defaultPool);

/// Parse a CPU list string like "0-5,24-29" into a set of CPU ids.
static std::set<int> parseCpuList(const std::string& s) {
   std::set<int> result;
   std::istringstream ss(s);
   std::string token;
   while (std::getline(ss, token, ',')) {
      auto dash = token.find('-');
      if (dash != std::string::npos) {
         int lo = std::stoi(token.substr(0, dash));
         int hi = std::stoi(token.substr(dash + 1));
         for (int i = lo; i <= hi; ++i) result.insert(i);
      } else {
         result.insert(std::stoi(token));
      }
   }
   return result;
}

void assertTopology() {
   if (numa_available() < 0) {
      fprintf(stderr, "assertTopology: libnuma reports NUMA not available\n");
      abort();
   }
   int maxNode = numa_max_node();
   size_t nCpus = SOCKETS_COUNT * CORES_PER_SOCKET * SMT_PER_CORE;

   // 1. Verify NUMA node count matches SOCKETS_COUNT
   if (static_cast<size_t>(maxNode + 1) != SOCKETS_COUNT) {
      fprintf(stderr,
              "assertTopology: numa_max_node()=%d but SOCKETS_COUNT=%zu\n",
              maxNode, SOCKETS_COUNT);
      abort();
   }

   // 2. When regions == sockets, verify cpu→node mapping against kernel
   if (NUM_NUMA_REGIONS == SOCKETS_COUNT) {
      for (size_t c = 0; c < nCpus; ++c) {
         int node = numa_node_of_cpu(static_cast<int>(c));
         size_t expected = regionOfCpu(c);
         if (node < 0 || static_cast<size_t>(node) != expected) {
            fprintf(stderr,
                    "assertTopology: cpu %zu on NUMA node %d, "
                    "expected region %zu\n",
                    c, node, expected);
            abort();
         }
      }
   }

   // 3. When regions > sockets (sub-socket domains like CCDs), verify that
   //    our regionOfCpu mapping is consistent with L3 cache sharing sets.
   if (NUM_NUMA_REGIONS > SOCKETS_COUNT) {
      // Build expected sharing sets from regionOfCpu
      std::vector<std::set<int>> expectedSets(NUM_NUMA_REGIONS);
      for (size_t c = 0; c < nCpus; ++c)
         expectedSets[regionOfCpu(c)].insert(static_cast<int>(c));

      // Read kernel's L3 sharing sets
      std::vector<std::set<int>> kernelSets;
      std::set<int> seen;
      for (size_t c = 0; c < nCpus; ++c) {
         if (seen.count(static_cast<int>(c))) continue;
         char path[256];
         snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%zu/cache/index3/shared_cpu_list",
                  c);
         std::ifstream f(path);
         if (!f.is_open()) {
            fprintf(stderr,
                    "assertTopology: cannot open %s — skipping L3 check\n",
                    path);
            goto skip_l3;
         }
         {
            std::string line;
            std::getline(f, line);
            auto cpus = parseCpuList(line);
            kernelSets.push_back(cpus);
            for (int id : cpus) seen.insert(id);
         }
      }

      if (kernelSets.size() != NUM_NUMA_REGIONS) {
         fprintf(stderr,
                 "assertTopology: kernel reports %zu L3 domains but "
                 "NUM_NUMA_REGIONS=%zu\n",
                 kernelSets.size(), NUM_NUMA_REGIONS);
         abort();
      }

      // Check that each kernel L3 set matches one of our expected sets
      for (auto& kset : kernelSets) {
         if (!kset.empty()) {
            size_t region = regionOfCpu(static_cast<size_t>(*kset.begin()));
            if (kset != expectedSets[region]) {
               fprintf(stderr,
                       "assertTopology: L3 sharing set for cpu %d doesn't "
                       "match region %zu mapping\n",
                       *kset.begin(), region);
               // Print expected vs actual
               fprintf(stderr, "  kernel: ");
               for (int c : kset) fprintf(stderr, "%d ", c);
               fprintf(stderr, "\n  expected: ");
               for (int c : expectedSets[region])
                  fprintf(stderr, "%d ", c);
               fprintf(stderr, "\n");
               abort();
            }
         }
      }
   skip_l3:;
   }

   // 4. Internal consistency: regionOf(tid) == regionOfCpu(cpuOfThread(tid))
   for (size_t t = 0; t < nCpus; ++t) {
      size_t cpu = cpuOfThread(t);
      if (regionOf(t) != regionOfCpu(cpu)) {
         fprintf(stderr,
                 "assertTopology: regionOf(%zu)=%zu but "
                 "regionOfCpu(cpuOfThread(%zu))=regionOfCpu(%zu)=%zu\n",
                 t, regionOf(t), t, cpu, regionOfCpu(cpu));
         abort();
      }
   }

   fprintf(stderr,
           "assertTopology: verified %zu CPUs, %zu NUMA nodes, "
           "%zu regions OK\n",
           nCpus, SOCKETS_COUNT, NUM_NUMA_REGIONS);
}

} // namespace runtime
