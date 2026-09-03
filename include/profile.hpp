#include "common/Compat.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#include <asm/unistd.h>
#include <linux/perf_event.h>
#if !defined(__aarch64__) && !defined(__arm__)
extern "C" {
#include "jevents.h"
}
#else
// Stubs for ARM — jevents is x86-only
inline char* get_cpu_str() {
   static char buf[] = "ARM";
   return buf;
}
inline int resolve_event(const char*, struct perf_event_attr*) { return -1; }
#endif

// ARM CPU part identification (from /proc/cpuinfo)
#ifdef __aarch64__
#include <fstream>
inline std::string get_arm_part() {
   std::ifstream cpuinfo("/proc/cpuinfo");
   std::string line;
   std::string implementer, part;
   while (std::getline(cpuinfo, line)) {
      if (line.find("CPU implementer") != std::string::npos) {
         auto pos = line.find(':');
         if (pos != std::string::npos) implementer = line.substr(pos + 2);
      }
      if (line.find("CPU part") != std::string::npos) {
         auto pos = line.find(':');
         if (pos != std::string::npos) { part = line.substr(pos + 2); break; }
      }
   }
   return implementer + "-" + part;
}
#endif
#endif

#define GLOBAL 1

extern bool writeHeader;

struct PerfEvents {
   const size_t printFieldWidth = 10;
   size_t counters;

#ifdef __linux__
   struct read_format {
      uint64_t value = 0;        /* The value of the event */
      uint64_t time_enabled = 0; /* if PERF_FORMAT_TOTAL_TIME_ENABLED */
      uint64_t time_running = 0; /* if PERF_FORMAT_TOTAL_TIME_RUNNING */
      uint64_t id = 0;           /* if PERF_FORMAT_ID */
   };
#endif
   struct event {
#ifdef __linux__
      struct perf_event_attr pe;
      int fd;
      read_format prev;
      read_format data;
#endif
      double readCounter() {

#ifdef __linux__
         return (data.value - prev.value) *
                (double)(data.time_enabled - prev.time_enabled) /
                (data.time_running - prev.time_running);
#else
         return 0;
#endif
      }
   };
   std::unordered_map<std::string, std::vector<event>> events;
   std::vector<std::string> ordered_names;

   PerfEvents() {
      if (GLOBAL)
         counters = 1;
      else {
         counters = std::thread::hardware_concurrency();
      }
#ifdef __linux__
#ifdef __aarch64__
      {
      // ARM CPU implementer-part pairs:
      //   0x41-0xd49 = Cortex-A77 (Neoverse N1 family, e.g. Graviton 2)
      //   0x41-0xd0c = Neoverse N1
      //   0x41-0xd40 = Neoverse V1 (Graviton 3)
      //   0x41-0xd4f = Neoverse V2 (Graviton 4)
      //   0xc0-0xac3 = Ampere Altra (ARMv8.2)
      std::string armPart = get_arm_part();

      // Common counters available on all ARMv8+
      add("cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
      add("instr.", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
      add("br. misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);

      // PERF_COUNT_HW_CACHE_LL via PERF_TYPE_HW_CACHE is unreliable on ARM —
      // many PMU drivers don't map it. Use generic PERF_COUNT_HW_CACHE_MISSES
      // for LLC-misses on all ARM targets.
      add("LLC-misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
      add("l1-misses", PERF_TYPE_HW_CACHE,
          PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
              (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));

      if (armPart == "0x41-0xd49" || armPart == "0x41-0xd0c") {
         // Cortex-A77 / Neoverse N1 (Graviton 2)
         add("mem_stall", PERF_TYPE_HARDWARE,
             PERF_COUNT_HW_STALLED_CYCLES_BACKEND);
      } else if (armPart == "0x41-0xd40" || armPart == "0x41-0xd4f") {
         // Neoverse V1/V2 (Graviton 3/4)
         add("mem_stall", PERF_TYPE_HARDWARE,
             PERF_COUNT_HW_STALLED_CYCLES_BACKEND);
      }
      }
#else
      char* cpustr = get_cpu_str();
      std::string cpu(cpustr);
      // see https://download.01.org/perfmon/mapfile.csv for cpu strings
      if (cpu == "GenuineIntel-6-57-core") {
         // Knights Landing
         add("cycles", "cpu/cpu-cycles/");
         add("LLC-misses", "cpu/cache-misses/");
         add("l1-misses", "MEM_UOPS_RETIRED.L1_MISS_LOADS");
         // e.add("l1-hits", "mem_load_retired.l1_hit");
         add("stores", "MEM_UOPS_RETIRED.ALL_STORES");
         add("loads", "MEM_UOPS_RETIRED.ALL_LOADS");
         add("instr.", "instructions");
      } else if (cpu == "GenuineIntel-6-55-core") {
         // Skylake X
         add("cycles", "cpu/cpu-cycles/");
         add("LLC-misses", "cpu/cache-misses/");
         add("LLC-misses2", "mem_load_retired.l3_miss");
         add("l1-misses", PERF_TYPE_HW_CACHE,
             PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                 (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
         add("instr.", "instructions");
         add("br. misses", "cpu/branch-misses/");
         add("all_rd", "offcore_requests.all_data_rd");
         add("stores", "mem_inst_retired.all_stores");
         add("loads", "mem_inst_retired.all_loads");
         add("mem_stall", "cycle_activity.stalls_mem_any");
         //add("page-faults", "page-faults");
      } else if (cpu == "AuthenticAMD-25-1-core" ||
                 cpu == "AuthenticAMD-25-11-core") {
         // AMD Zen3 (25-1) / Zen4 (25-11)
         // Core-side counters that actually open on AMD. The generic L1D
         // cache events and PERF_COUNT_HW_CACHE_MISSES work; the Intel-only
         // cpu/mem-loads|stores (PEBS) and the PERF_TYPE_HW_CACHE LL-miss
         // generic do NOT open on AMD (LL generic unsupported, L3 events are
         // on the L3PMC uncore PMU). All named events resolve via the AMD
         // perfmon JSON through jevents and were verified to open.
         add("cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
         add("LLC-misses", "cpu/cache-misses/");
         add("l1-misses", PERF_TYPE_HW_CACHE,
             PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                 (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
         add("l1-hits", PERF_TYPE_HW_CACHE,
             PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                 (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16));
         add("stores", "ls_dispatch.store_dispatch");
         add("loads", "ls_dispatch.ld_dispatch");
         // Loads that miss L1 (off-core read requests) as a bandwidth proxy
         // (64B/request assumption in the bandwidth formula). Closest core-side
         // analog to Intel's offcore_requests.all_data_rd. True DRAM bandwidth
         // needs the amd_df uncore PMU, which is out of scope here.
         add("all_rd", "ls_mab_alloc.loads");
         add("instr.", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
         add("br. misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
      } else {
         add("cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
         add("LLC-misses", PERF_TYPE_HW_CACHE,
             PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
             (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
         add("l1-misses", PERF_TYPE_HW_CACHE,
             PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                 (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
         add("l1-hits", PERF_TYPE_HW_CACHE,
             PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
             (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16));
         add("stores", "cpu/mem-stores/");
         add("loads", "cpu/mem-loads/");
         add("instr.", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
         add("br. misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
      }
#endif // __aarch64__
      add("task-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK);
#endif

      registerAll();
   }

   void add(std::string name, uint64_t type, uint64_t eventID) {
      if (getenv("EXTERNALPROFILE")) return;
#ifdef __linux__

      ordered_names.push_back(name);
      auto& eventsPerThread = events[name];
      eventsPerThread.assign(counters, event());
      for (auto& event : eventsPerThread) {
         auto& pe = event.pe;
         memset(&pe, 0, sizeof(struct perf_event_attr));
         pe.type = type;
         pe.size = sizeof(struct perf_event_attr);
         pe.config = eventID;
         pe.disabled = true;
         pe.inherit = 1;
         pe.inherit_stat = 0;
         pe.exclude_kernel = true;
         pe.exclude_hv = true;
         pe.read_format =
             PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
      }
#else
      compat::unused(name, type, eventID);
#endif
   }
   void add(std::string name, std::string str) {
      if (getenv("EXTERNALPROFILE")) return;
#ifdef __linux__
      ordered_names.push_back(name);
      auto& eventsPerThread = events[name];
      eventsPerThread.assign(counters, event());
      for (auto& event : eventsPerThread) {
         auto& pe = event.pe;
         memset(&pe, 0, sizeof(struct perf_event_attr));
         if (resolve_event(const_cast<char*>(str.c_str()), &pe) < 0)
            std::cerr << "Error resolving perf event " << str << std::endl;
         pe.disabled = true;
         pe.inherit = 1;
         pe.inherit_stat = 0;
         pe.exclude_kernel = true;
         pe.exclude_hv = true;
         pe.read_format =
             PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
      }
#else
      compat::unused(name, str);
#endif
   }

   void registerAll() {
      for (auto& ev : events) {
         size_t i = 0;
         for (auto& event : ev.second) {

#ifdef __linux__
            if (GLOBAL)
               event.fd =
                   syscall(__NR_perf_event_open, &event.pe, 0, -1, -1, 0);
            else
               event.fd = syscall(__NR_perf_event_open, &event.pe, 0, i, -1, 0);
            if (event.fd < 0)
               std::cerr << "Error opening perf event " << ev.first
                         << std::endl;
#else
            compat::unused(event);
#endif
            ++i;
         }
      }
   }

   void startAll() {
      for (auto& ev : events) {
         for (auto& event : ev.second) {
#ifdef __linux__
           ioctl(event.fd, PERF_EVENT_IOC_ENABLE, 0);
           if (read(event.fd, &event.prev, sizeof(uint64_t) * 3) !=
               sizeof(uint64_t) * 3)
             std::cerr << "Error reading counter " << ev.first << std::endl;
#else
            compat::unused(event);
#endif
         }
      }
   }

   ~PerfEvents() {
      for (auto& ev : events)
         for (auto& event : ev.second)
#ifdef __linux__
            close(event.fd);
#else
            compat::unused(event);
#endif
   }

   void readAll() {
      for (auto& ev : events)
         for (auto& event : ev.second) {
#ifdef __linux__
            if (read(event.fd, &event.data, sizeof(uint64_t) * 3) !=
                sizeof(uint64_t) * 3)
               std::cerr << "Error reading counter " << ev.first << std::endl;
            ioctl(event.fd, PERF_EVENT_IOC_DISABLE, 0);
#else
            compat::unused(event);
#endif
         }
   }

   void printHeader(std::ostream& out) {
      for (auto& name : ordered_names)
         out << std::setw(printFieldWidth) << name << ",";
   }

   void printAll(std::ostream& out, double n) {
      for (auto& name : ordered_names) {
         double aggr = 0;
         for (auto& event : events[name]) aggr += event.readCounter();
         out << std::setw(printFieldWidth) << aggr / n << ",";
      }
   }

   double operator[](std::string index) {
      double aggr = 0;
      for (auto& event : events[index]) aggr += event.readCounter();
      return aggr;
   };

   void timeAndProfile(std::string s, uint64_t count, std::function<void()> fn,
                       uint64_t repetitions = 1, bool mem = false);
};

inline double gettime() {
   struct timeval now_tv;
   gettimeofday(&now_tv, NULL);
   return ((double)now_tv.tv_sec) + ((double)now_tv.tv_usec) / 1000000.0;
}

size_t getCurrentRSS() {
   long rss = 0L;
   FILE* fp = NULL;
   if ((fp = fopen("/proc/self/statm", "r")) == NULL)
      return (size_t)0L; /* Can't open? */
   if (fscanf(fp, "%*s%ld", &rss) != 1) {
      fclose(fp);
      return (size_t)0L; /* Can't read? */
   }
   fclose(fp);
   return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);
}

void PerfEvents::timeAndProfile(std::string s, uint64_t count,
                                std::function<void()> fn, uint64_t repetitions,
                                bool mem) {
   using namespace std;
   // warmup rounds
   for (int warmup = 0; warmup < 3; ++warmup) fn();

   // Collect per-rep wall times
   std::vector<double> repTimes;
   repTimes.reserve(repetitions);

   uint64_t memStart = 0;
   if (mem) memStart = getCurrentRSS();
   startAll();
   double totalStart = gettime();
   size_t performedRep = 0;
   for (; performedRep < repetitions; ++performedRep) {
      double t0 = gettime();
      fn();
      double t1 = gettime();
      repTimes.push_back((t1 - t0) * 1e3); // ms
   }
   double totalEnd = gettime();
   readAll();

   // Compute statistics
   std::vector<double> sorted = repTimes;
   std::sort(sorted.begin(), sorted.end());
   double minTime = sorted.front();
   double maxTime = sorted.back();
   double median = (sorted.size() % 2 == 1)
       ? sorted[sorted.size() / 2]
       : (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) / 2.0;
   double mean = std::accumulate(repTimes.begin(), repTimes.end(), 0.0) / repTimes.size();
   double sq_sum = 0;
   for (auto t : repTimes) sq_sum += (t - mean) * (t - mean);
   double stddev = repTimes.size() > 1 ? std::sqrt(sq_sum / (repTimes.size() - 1)) : 0.0;

   std::cout.precision(3);
   std::cout.setf(std::ios::fixed, std::ios::floatfield);
   if (writeHeader) {
      std::cout << setw(20) << "name"
                << "," << setw(printFieldWidth) << "median"
                << "," << setw(printFieldWidth) << "mean"
                << "," << setw(printFieldWidth) << "min"
                << "," << setw(printFieldWidth) << "max"
                << "," << setw(printFieldWidth) << "stddev"
                << "," << setw(printFieldWidth) << " CPUs"
                << "," << setw(printFieldWidth) << " IPC"
                << "," << setw(printFieldWidth) << " GHz"
                << "," << setw(printFieldWidth) << " Bandwidth"
                << ",";
      printHeader(std::cout);
      std::cout << std::endl;
   }

   auto runtime = totalEnd - totalStart;
   std::cout << setw(20) << s
             << "," << setw(printFieldWidth) << median
             << "," << setw(printFieldWidth) << mean
             << "," << setw(printFieldWidth) << minTime
             << "," << setw(printFieldWidth) << maxTime
             << "," << setw(printFieldWidth) << stddev
             << ",";
#ifdef __linux__
   if (!getenv("EXTERNALPROFILE")) {
      std::cout << setw(printFieldWidth)
                << ((*this)["task-clock"] / (runtime * 1e9)) << ",";
      std::cout << setw(printFieldWidth)
                << ((*this)["instr."] / (*this)["cycles"]) << ",";
      std::cout << setw(printFieldWidth)
                << ((*this)["cycles"] /
                    (this->events["cycles"][0].data.time_enabled -
                     this->events["cycles"][0].prev.time_enabled))
                << ",";
      std::cout << setw(printFieldWidth)
                << ((((*this)["all_rd"] * 64.0) / (1024 * 1024)) /
                    runtime)
                << ",";
   }
#endif

   printAll(std::cout, count * performedRep);
   if (mem) std::cout << (getCurrentRSS() - memStart) / (1024.0 * 1024) << "MB";
   std::cout << std::endl;

   // Emit per-rep times to stderr for post-processing
   std::cerr << "per-rep " << s << ":";
   for (size_t i = 0; i < repTimes.size(); ++i)
      std::cerr << " " << std::fixed << std::setprecision(3) << repTimes[i];
   std::cerr << std::endl;

   writeHeader = false;
}
