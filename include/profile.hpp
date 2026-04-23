#include "common/Compat.hpp"
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <asm/unistd.h>
#include <fstream>
#include <linux/perf_event.h>
#include <string>

/**
 * TOGGLE MACRO:
 * Uncomment #define USE_TMA to use the new Grouped TMA counter logic.
 * Keep it commented to use the original Legacy/Independent counter logic.
 */
// #define USE_TMA

#ifndef USE_TMA
// =============================================================================
// ORIGINAL CODE SECTION
// =============================================================================

#define NO_JEVENTS
#if (defined(__x86_64__) || defined(__i386__)) && !defined(NO_JEVENTS)
extern "C" {
#include "jevents.h"
}
#else
extern "C" {
inline char* get_cpu_str() {
   static char cpu_buffer[256] = "";
   if (cpu_buffer[0] == '\0') {
      std::ifstream file("/proc/cpuinfo");
      bool found = false;
      if (file.is_open()) {
         std::string line;
         while (std::getline(file, line)) {
            if (line.find("model name") != std::string::npos ||
                line.find("Model") != std::string::npos) {
               size_t pos = line.find(": ");
               if (pos != std::string::npos) {
                  std::string model = line.substr(pos + 2);
                  model.erase(std::find_if(model.rbegin(), model.rend(),
                                           [](unsigned char ch) {
                                              return !std::isspace(ch);
                                           })
                                  .base(),
                              model.end());
                  snprintf(cpu_buffer, sizeof(cpu_buffer), "%s", model.c_str());
                  found = true;
                  break;
               }
            }
         }
         file.close();
      }
      if (!found) {
         snprintf(cpu_buffer, sizeof(cpu_buffer), "generic-processor-stub");
      }
   }
   return cpu_buffer;
}}

inline int resolve_event(const char* str, struct perf_event_attr* pe) {
   (void)str;
   (void)pe;
   return -1;

}
#endif

#define GLOBAL 1
extern bool writeHeader;

struct PerfEvents {
   const size_t printFieldWidth = 12;
   size_t counters;

#ifdef __linux__
   struct read_format {
      uint64_t value = 0;
      uint64_t time_enabled = 0;
      uint64_t time_running = 0;
      uint64_t id = 0;
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
      else { counters = std::thread::hardware_concurrency(); }
#ifdef __linux__
      char* cpustr = get_cpu_str();
      std::string cpu(cpustr);
      if (cpu == "GenuineIntel-6-57-core") {
         add("cycles", "cpu/cpu-cycles/");
         add("LLC-misses", "cpu/cache-misses/");
         add("l1-misses", "MEM_UOPS_RETIRED.L1_MISS_LOADS");
         add("stores", "MEM_UOPS_RETIRED.ALL_STORES");
         add("loads", "MEM_UOPS_RETIRED.ALL_LOADS");
         add("instr.", "instructions");
      } else if (cpu == "GenuineIntel-6-55-core") {
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
         add("instr.", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
         add("br. misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
      }
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
         pe.exclude_kernel = true;
         pe.exclude_hv = true;
         pe.read_format =
             PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
      }
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
         pe.exclude_kernel = true;
         pe.exclude_hv = true;
         pe.read_format =
             PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
      }
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
            read(event.fd, &event.prev, sizeof(uint64_t) * 3);
#endif
         }
      }
   }

   void readAll() {
      for (auto& ev : events) {
         for (auto& event : ev.second) {
#ifdef __linux__
            read(event.fd, &event.data, sizeof(uint64_t) * 3);
            ioctl(event.fd, PERF_EVENT_IOC_DISABLE, 0);
#endif
         }
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

#else
// =============================================================================
// NEW TMA COUNTER VERSION
// =============================================================================

#define GLOBAL 1
extern bool writeHeader;

struct PerfEvents {
   const size_t printFieldWidth = 12;
   int group_leader_fd = -1;
   size_t counters;

   struct read_format {
      uint64_t value = 0;
      uint64_t time_enabled = 0;
      uint64_t time_running = 0;
   };

   struct event {
      struct perf_event_attr pe;
      int fd = -1;
      read_format prev;
      read_format data;

      double readCounter() {
         if (fd < 0) return 0;
         uint64_t diff_val = data.value - prev.value;
         uint64_t diff_run = data.time_running - prev.time_running;
         uint64_t diff_enb = data.time_enabled - prev.time_enabled;
         if (diff_run == 0) return 0;
         return (double)diff_val * (double)diff_enb / (double)diff_run;
      }
   };

   std::unordered_map<std::string, event> event_map;
   std::vector<std::string> ordered_names;

   PerfEvents() {
      counters = 1; // Simplify for global instrumentation

      // 1. Grouped TMA components (Must use PERF_TYPE_RAW for reliability)
      // These are placeholders for standard Intel TMA slots
      add_raw("total-slots", 0x003c); // Denominator
      add_raw("front-end", 0x019c);
      add_raw("bad-spec", 0x010d);
      add_raw("retired-slots", 0x02c2);

      // 2. Ungrouped diagnostic counters
      add_std("instr.", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
      add_std("cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
      add_std("task-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK);

      registerAll();
   }

   void add_std(std::string name, uint32_t type, uint64_t config) {
      ordered_names.push_back(name);
      auto& ev = event_map[name];
      memset(&ev.pe, 0, sizeof(struct perf_event_attr));
      ev.pe.type = type;
      ev.pe.config = config;
      ev.pe.size = sizeof(struct perf_event_attr);
      ev.pe.disabled = true;
      ev.pe.inherit = 1;
      ev.pe.read_format =
          PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
   }

   void add_raw(std::string name, uint64_t config) {
      ordered_names.push_back(name);
      auto& ev = event_map[name];
      memset(&ev.pe, 0, sizeof(struct perf_event_attr));
      ev.pe.type = PERF_TYPE_RAW;
      ev.pe.config = config;
      ev.pe.size = sizeof(struct perf_event_attr);
      ev.pe.disabled = true;
      ev.pe.inherit = 1;
      ev.pe.read_format =
          PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
   }

   void registerAll() {
      // Step A: Open total-slots as leader
      auto& leader = event_map["total-slots"];
      leader.fd = syscall(__NR_perf_event_open, &leader.pe, 0, -1, -1, 0);
      group_leader_fd = leader.fd;

      // Step B: Open others, linking TMA members to leader
      for (auto& name : ordered_names) {
         if (name == "total-slots") continue;
         auto& ev = event_map[name];
         int group_fd = (ev.pe.type == PERF_TYPE_RAW) ? group_leader_fd : -1;
         ev.fd = syscall(__NR_perf_event_open, &ev.pe, 0, -1, group_fd, 0);
      }
   }

   void startAll() {
      ioctl(group_leader_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
      for (auto& name : ordered_names)
         read(event_map[name].fd, &event_map[name].prev, sizeof(read_format));
      ioctl(group_leader_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
   }

   void readAll() {
      ioctl(group_leader_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
      for (auto& name : ordered_names)
         read(event_map[name].fd, &event_map[name].data, sizeof(read_format));
   }

   double operator[](std::string name) { return event_map[name].readCounter(); }

   void timeAndProfile(std::string s, uint64_t count, std::function<void()> fn,
                       uint64_t repetitions = 1, bool mem = false);
};
#endif

// Shared utility functions
inline double gettime() {
   struct timeval now_tv;
   gettimeofday(&now_tv, NULL);
   return ((double)now_tv.tv_sec) + ((double)now_tv.tv_usec) / 1000000.0;
}

size_t getCurrentRSS() {
   long rss = 0L;
   FILE* fp = fopen("/proc/self/statm", "r");
   if (!fp) return 0;
   if (fscanf(fp, "%*s%ld", &rss) != 1) {
      fclose(fp);
      return 0;
   }
   fclose(fp);
   return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);
}

#define USE_MIN_MODE

// Logic for report printing (switches based on macro)
void PerfEvents::timeAndProfile(std::string s, uint64_t count,
                                std::function<void()> fn, uint64_t repetitions,
                                bool mem) {
   for (int i = 0; i < 3; i++) fn(); // Warmup
   uint64_t memStart = mem ? getCurrentRSS() : 0;

   double runtime = 0;
   double min_runtime = std::numeric_limits<double>::max();

#ifdef USE_MIN_MODE
   bool min_mode = true;
   // Execute and measure each repetition individually to find the minimum
   for (size_t i = 0; i < repetitions; i++) {
      startAll();
      double start = gettime();
      fn();
      double end = gettime();
      readAll(); // This also stops/disables counters internally

      double current_run = end - start;
      if (current_run < min_runtime) {
         min_runtime = current_run;
         // In this mode, readAll() captured the delta for exactly one 'fn'
         // call. We keep the state in the event objects for the calculations
         // below. Note: If you need to preserve data across the loop, you'd
         // clone the 'event_map' or 'events' here.
      }
   }
   runtime = min_runtime;
   // For scaling logic, we treat this as 1 repetition since we took the best 1
   double effective_reps = 1.0;
#else
   // Original Average Mode: Measure all repetitions in one batch
   startAll();
   double start = gettime();
   for (size_t i = 0; i < repetitions; i++) fn();
   double end = gettime();
   readAll();
   runtime = end - start;
   double effective_reps = static_cast<double>(repetitions);
#endif

   if (writeHeader) {
#ifdef USE_TMA
      std::cout << std::setw(20) << "name" << "," << std::setw(printFieldWidth)
                << (min_mode ? "min_time" : "time") << ","
                << std::setw(printFieldWidth) << "FE-Bound"
                << "," << std::setw(printFieldWidth) << "BE-Bound" << ","
                << std::setw(printFieldWidth) << "BadSpec"
                << "," << std::setw(printFieldWidth) << "Retiring" << ","
                << std::setw(printFieldWidth) << "IPC" << std::endl;
#else
      // std::cout << std::setw(20) << "name" << ", time, CPUs, IPC, GHz,";
      std::cout << std::setw(20) << "name"
                << "," << std::setw(printFieldWidth) << "time"
                << "," << std::setw(printFieldWidth) << "CPUs"
                << "," << std::setw(printFieldWidth) << "IPC"
                << "," << std::setw(printFieldWidth) << "GHz"
                << "," << std::setw(printFieldWidth) << "Bandwidth" << ",";
      for (auto& n : ordered_names)
         std::cout << std::setw(printFieldWidth) << n << ",";
      std::cout << std::endl;
#endif
      writeHeader = false;
   }

   std::cout << std::setw(20) << s << ",";
#ifdef USE_TMA
   double slots = (*this)["total-slots"];
   double fe = ((*this)["front-end"] / slots) * 100.0;
   double spec = ((*this)["bad-spec"] / slots) * 100.0;
   double ret = ((*this)["retired-slots"] / slots) * 100.0;
   double be = 100.0 - (fe + spec + ret);
   double ipc = (*this)["instr."] / (*this)["cycles"];
   std::cout << std::setw(printFieldWidth)
             << (runtime * 1e3 / (min_mode ? 1 : repetitions)) << ",";
   std::cout << std::fixed << std::setprecision(2) << std::setw(printFieldWidth)
             << fe << "%," << std::setw(printFieldWidth) << be << "%,"
             << std::setw(printFieldWidth) << spec << "%,"
             << std::setw(printFieldWidth) << ret << "%,"
             << std::setw(printFieldWidth) << ipc;
#else
   std::cout << std::setw(printFieldWidth) << (runtime * 1e3 / repetitions)
             << ",";

   // if (!getenv("EXTERNALPROFILE")) {

   std::cout << std::setw(printFieldWidth)
             << ((*this)["task-clock"] / (runtime * 1e9)) << ",";
   std::cout << std::setw(printFieldWidth)
             << ((*this)["instr."] / (*this)["cycles"]) << ",";
   std::cout << std::setw(printFieldWidth)
             << ((*this)["cycles"] /
                 (this->events["cycles"][0].data.time_enabled -
                  this->events["cycles"][0].prev.time_enabled))
             << ",";
   std::cout << std::setw(printFieldWidth)
             << ((((*this)["all_rd"] * 64.0) / (1024 * 1024)) / runtime) << ",";

   printAll(std::cout, count * repetitions);
   //}

   // ... [Original scale/printing logic omitted for brevity, identical to your
   // top block]
#endif
   std::cout << std::endl;
}
