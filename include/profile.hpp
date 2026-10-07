#include "common/Compat.hpp"
#ifdef CANARY_IN_BENCH
#include "canary.hpp"
#endif
#include <algorithm>
#include <cerrno>
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
         // Scale the raw count by (enabled/running) to correct for counter
         // multiplexing.  If the counter never ran (time_running delta == 0),
         // e.g. a per-CPU counter on a CPU no thread used, it counted nothing
         // and must read as 0 -- otherwise 0/0 yields NaN and poisons the
         // per-CPU aggregate.
         auto deltaRunning =
             (double)(data.time_running - prev.time_running);
         if (deltaRunning <= 0.0)
            return 0;
         return (double)(data.value - prev.value) *
                (double)(data.time_enabled - prev.time_enabled) /
                deltaRunning;
#else
         return 0;
#endif
      }
   };
   std::unordered_map<std::string, std::vector<event>> events;
   std::vector<std::string> ordered_names;

   // Opt-in grouped mode (PERF_GROUP=<G0..G8, G3a, G3b>): cycles + instr. and
   // one event group, opened as one perf group so they are scheduled together
   // (no multiplexing), plus events opened on their own (task-clock, msr).
   // Unset, the per-CPU default lists below are used unchanged.
   std::string perfGroup;
   std::vector<std::string> groupNames; // leader first
   std::vector<std::string> unresolved;
   int groupLeaderFd = -1;

   PerfEvents() {
      if (GLOBAL)
         counters = 1;
      else {
         counters = std::thread::hardware_concurrency();
      }
#ifdef __linux__
      if (const char* g = getenv("PERF_GROUP");
          g && *g && !getenv("EXTERNALPROFILE")) {
         perfGroup = g;
         addGroup();
         add("task-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK);
         registerAll();
         return;
      }
#endif
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

      // burrata (Neoverse N1) has NO level-3 cache in the core-visible
      // hierarchy: sysfs shows only index0 (L1d 64K/4-way), index1 (L1i 64K)
      // and index2 (L2 1M/8-way) per CPU -- there is no level=3 index. The
      // Ampere Altra SLC is owned by the DSU (mesh) and is not part of the
      // core's cache hierarchy, which is also why its PMU is a separate,
      // system-wide, per-cluster block (40 instances) that this per-thread
      // loop cannot read.
      //
      // So the last level a core can see and we can count is L2. LLC-misses is
      // therefore re-purposed to l2d_cache_refill (0x17): the L2 refill count,
      // i.e. traffic leaving the core. This is NOT L1D-demand-miss-equivalent:
      // l2d_cache_refill includes prefetch fills, so it is an off-core traffic
      // proxy rather than demand misses (verified 2026-10-07, see the pmu
      // report). It replaces PERF_COUNT_HW_CACHE_MISSES, which this kernel maps
      // to the SAME core L1D read-miss event as l1-misses below (both read
      // 0.110), so the column used to duplicate l1-misses exactly.
      //
      // Column name and ordinal are unchanged, so no header-version migration
      // is needed; the name now means "last core-visible cache level (L2) on
      // N1" and is not the same quantity as LLC-misses on the x86 machines.
      add("LLC-misses", PERF_TYPE_RAW, 0x17);
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
         // Skylake-X / Cascade Lake (model 0x55; covers dubliner, Xeon Gold
         // 6238L). Generic PERF_TYPE_HARDWARE/HW_CACHE forms for the base
         // counters resolve with no perfmon JSON. The store/load/bandwidth/
         // stall counters need the skylakex perfmon JSON
         // (GenuineIntel-6-55-core.json) in ~/.cache/pmu-events/ to resolve;
         // without it they read 0 (readCounter guard).
         add("cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
         add("LLC-misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
         add("l1-misses", PERF_TYPE_HW_CACHE,
             PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                 (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
         add("l1-hits", PERF_TYPE_HW_CACHE,
             PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                 (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16));
         add("instr.", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
         add("br. misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
        //add("all_rd", "offcore_requests.all_data_rd");
         //add("stores", "mem_inst_retired.all_stores");
         //add("loads", "mem_inst_retired.all_loads");
         add("mem_stall", "cycle_activity.stalls_mem_any");
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
         // Zen3 has no memory-stall event, and generic STALLED_CYCLES_BACKEND
         // opens but barely counts there (0.015-0.136 per tuple against 9-44
         // cycles on roquefort, 2026-10-07-counters-default-3). Use dispatch
         // stalls on retire-queue tokens instead: verified to resolve and count
         // through this path on roquefort. NOTE: this is dispatch stalls, not
         // memory stalls, so the roquefort mem_stall column is not directly
         // comparable with the other machines'. Zen4 renamed the dispatch-stall
         // events, so it keeps the generic event (not verified on Zen4).
         if (cpu == "AuthenticAMD-25-1-core")
            add("mem_stall",
                "de_dis_dispatch_token_stalls2.retire_token_stall");
         else
            add("mem_stall", PERF_TYPE_HARDWARE,
                PERF_COUNT_HW_STALLED_CYCLES_BACKEND);
      } else if (cpu == "GenuineIntel-6-8F-core") {
         // Sapphire Rapids (SPR): Xeon Silver 4509Y (manchego).  Lean, proven
         // set: keep the generic PERF_TYPE_HARDWARE/HW_CACHE forms for the
         // basic counters, and use perfmon JSON events only where the generic
         // form doesn't work (stores/loads on SPR need the JSON; all_rd is the
         // 64B/request bandwidth proxy).  SPR has no offcore_requests.all_data_rd
         // (SKX); closest completed-request analog is OFFCORE_REQUESTS.DATA_RD
         // (all data reads incl. prefetch).  On SPR, per-CPU counters for a very
         // short measured window (a single short rep) can be left unscheduled
         // (time_running=0); readCounter() guards against the resulting 0/0 NaN,
         // and normal multi-rep runs (-r 5) schedule all counters correctly.
         add("cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
         add("LLC-misses", "cpu/cache-misses/");
         add("l1-misses", PERF_TYPE_HW_CACHE,
             PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                 (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
         add("l1-hits", PERF_TYPE_HW_CACHE,
             PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                 (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16));
         add("stores", "mem_inst_retired.all_stores");
         add("loads", "mem_inst_retired.all_loads");
         add("all_rd", "offcore_requests.data_rd");
         add("instr.", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
         add("br. misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
         // Generic PERF_COUNT_HW_STALLED_CYCLES_BACKEND reports 0 on SPR (the
         // hardware generic backend-stall counter is not wired up there).
         // CYCLE_ACTIVITY.STALLS_MEM_ANY also does not exist on SPR (it is not
         // in the 6-8F JSON, so resolve_event fails and the column silently
         // reads 0); SPR splits that family into CYCLES_MEM_ANY and the
         // MEMORY_ACTIVITY.STALLS_* set. Use memory_activity.stalls_l1d_miss:
         // a real stall event, verified to resolve and count on manchego.
         // NOTE: this is stalls-on-L1D-miss, not the old "any memory stall"
         // definition dubliner/Cascade Lake and Zen use, so the manchego
         // mem_stall column is not directly comparable with those machines'.
         add("mem_stall", "memory_activity.stalls_l1d_miss");
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
         if (resolve_event(const_cast<char*>(str.c_str()), &pe) < 0) {
            std::cerr << "Error resolving perf event " << str << std::endl;
            unresolved.push_back(str);
         }
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

#ifdef __linux__
   [[noreturn]] void groupFail(const std::string& why) {
      std::cerr << "PERF_GROUP=" << perfGroup << ": " << why << std::endl;
      std::exit(2);
   }

   // Event groups, from PMU_EVENT_GROUPS.md. Each is sized to fit next to
   // cycles + instr. without multiplexing (4 free counters on Cascade Lake
   // and Zen 3, 8 on Sapphire Rapids, 5 on Neoverse N1). G0's branch events
   // are the generic ones, so G0 also works on CPUs without a table here.
   void addGroup() {
      const std::string& g = perfGroup;
      auto member = [&](const std::string& name) { groupNames.push_back(name); };
      auto hw = [&](const char* name, uint64_t type, uint64_t config) {
         add(name, type, config);
         member(name);
      };
      [[maybe_unused]] auto named = [&](const char* name, const char* ev) {
         add(name, ev);
         member(name);
      };
      // anchors; cycles leads the group
      hw("cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
      hw("instr.", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
      const size_t anchors = groupNames.size();
      if (g == "G0") {
         hw("branches", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS);
         hw("br-misp", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
      }
#ifdef __aarch64__
      // architectural PMUv3 common events (raw codes; jevents is stubbed here)
      auto raw = [&](const char* name, uint64_t code) {
         hw(name, PERF_TYPE_RAW, code);
      };
      if (g == "G0") {
         raw("inst-spec", 0x1B);
      } else if (g == "G1") {
         raw("ld-spec", 0x70);
         raw("l1d-refill", 0x03);
         raw("l2d-refill", 0x17); // includes prefetch fills
      } else if (g == "G2") {
         raw("stall-backend", 0x24);
      } else if (g == "G4") {
         raw("unaligned-ld-spec", 0x68);
      } else if (g == "G6") {
         raw("l1i-refill", 0x01);
         raw("stall-frontend", 0x23);
      } else if (g == "G7") {
         raw("ase-spec", 0x74);
         raw("dp-spec", 0x73);
         raw("vfp-spec", 0x75);
      } else if (g == "G8") {
         raw("st-spec", 0x71);
         raw("bus-access", 0x19);
         raw("l1d-tlb-refill", 0x05);
      }
#else
      const std::string cpu(get_cpu_str());
      if (cpu == "GenuineIntel-6-55-core") { // Cascade Lake (dubliner)
         if (g == "G0") {
            hw("ref-cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES);
         } else if (g == "G1") {
            named("loads", "mem_inst_retired.all_loads");
            named("l1-miss", "mem_load_retired.l1_miss");
            named("l2-miss", "mem_load_retired.l2_miss");
            named("l3-miss", "mem_load_retired.l3_miss");
         } else if (g == "G2") {
            named("stall-l1d", "cycle_activity.stalls_l1d_miss");
            named("stall-l2", "cycle_activity.stalls_l2_miss");
            named("stall-l3", "cycle_activity.stalls_l3_miss");
            named("stall-mem", "cycle_activity.stalls_mem_any");
         } else if (g == "G3a") {
            named("l1d-pend", "l1d_pend_miss.pending"); // counter-restricted
         } else if (g == "G3b") {
            named("l1d-pend-cyc", "l1d_pend_miss.pending_cycles");
            named("offcore-out", "offcore_requests_outstanding.all_data_rd");
            named("offcore-out-cyc",
                  "offcore_requests_outstanding.cycles_with_data_rd");
         } else if (g == "G4") {
            named("stall-total", "cycle_activity.stalls_total");
            named("st-fwd-block", "ld_blocks.store_forward");
            named("alias-4k", "ld_blocks_partial.address_alias");
            named("split-loads", "mem_inst_retired.split_loads");
         } else if (g == "G5") {
            named("license0", "core_power.lvl0_turbo_license");
            named("license1", "core_power.lvl1_turbo_license");
            named("license2", "core_power.lvl2_turbo_license");
            named("ms-uops", "idq.ms_uops");
         } else if (g == "G6") {
            named("dsb-uops", "idq.dsb_uops");
            named("mite-uops", "idq.mite_uops");
            named("dsb-miss", "frontend_retired.dsb_miss");
            named("icache-stall", "icache_16b.ifdata_stall");
         } else if (g == "G7") {
            named("port0", "uops_dispatched_port.port_0");
            named("port1", "uops_dispatched_port.port_1");
            named("port5", "uops_dispatched_port.port_5");
         } else if (g == "G8") {
            named("stores", "mem_inst_retired.all_stores");
            named("offcore-rd", "offcore_requests.all_data_rd");
            named("offcore-all", "offcore_requests.all_requests");
            named("dtlb-walk", "dtlb_load_misses.walk_completed");
         }
      } else if (cpu == "GenuineIntel-6-8F-core") { // Sapphire Rapids (manchego)
         if (g == "G0") {
            hw("ref-cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES);
         } else if (g == "G1") {
            named("loads", "mem_inst_retired.all_loads");
            named("l1-miss", "mem_load_retired.l1_miss");
            named("l2-miss", "mem_load_retired.l2_miss");
            named("l3-miss", "mem_load_retired.l3_miss");
         } else if (g == "G2") {
            named("stall-l1d", "memory_activity.stalls_l1d_miss");
            named("stall-l2", "memory_activity.stalls_l2_miss");
            named("stall-l3", "memory_activity.stalls_l3_miss");
         } else if (g == "G3a") {
            named("l1d-pend", "l1d_pend_miss.pending");
         } else if (g == "G3b") {
            named("l1d-pend-cyc", "l1d_pend_miss.pending_cycles");
            named("offcore-out", "offcore_requests_outstanding.all_data_rd");
            named("offcore-out-cyc",
                  "offcore_requests_outstanding.cycles_with_data_rd");
         } else if (g == "G4") {
            named("stall-total", "cycle_activity.stalls_total");
            named("st-fwd-block", "ld_blocks.store_forward");
            named("alias-4k", "ld_blocks.address_alias");
            named("split-loads", "mem_inst_retired.split_loads");
         } else if (g == "G5") { // no license events on SPR
            named("ms-uops", "idq.ms_uops");
            named("ms-retired", "uops_retired.ms");
         } else if (g == "G6") {
            named("dsb-uops", "idq.dsb_uops");
            named("mite-uops", "idq.mite_uops");
            named("dsb-miss", "frontend_retired.dsb_miss");
            named("icache-stall", "icache_data.stalls");
         } else if (g == "G7") {
            named("port0", "uops_dispatched.port_0");
            named("port1", "uops_dispatched.port_1");
            named("port5-11", "uops_dispatched.port_5_11");
         } else if (g == "G8") {
            named("stores", "mem_inst_retired.all_stores");
            named("offcore-rd", "offcore_requests.data_rd");
            named("dtlb-walk", "dtlb_load_misses.walk_completed");
         }
      } else if (cpu == "AuthenticAMD-25-1-core") { // Zen 3 (roquefort)
         if (g == "G0") {
            // msr PMU: outside the group (a group can't span PMUs) and without
            // exclusion bits, which the msr PMU rejects
            for (const char* ev : {"msr/aperf/", "msr/mperf/"}) {
               std::string name = ev[4] == 'a' ? "aperf" : "mperf";
               add(name, ev);
               auto& pe = events[name][0].pe;
               pe.exclude_kernel = 0;
               pe.exclude_hv = 0;
            }
         } else if (g == "G1") {
            named("loads", "ls_dispatch.ld_dispatch");
            hw("l1-miss", PERF_TYPE_HW_CACHE,
               PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                   (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
            named("l2-miss", "l2_cache_req_stat.ls_rd_blk_c");
            named("dram-fills", "ls_dmnd_fills_from_sys.mem_io_local");
         } else if (g == "G2") { // no memory-stall event on Zen 3: proxies
            named("ldq-stall",
                  "de_dis_dispatch_token_stalls1.load_queue_rsrc_stall");
            named("retire-stall",
                  "de_dis_dispatch_token_stalls2.retire_token_stall");
         } else if (g == "G4") {
            named("st-fwd", "ls_stlf");
            named("st-fwd-block", "ls_bad_status2.stli_other");
            named("alias-4k", "ls_misal_loads.ma4k");
            named("misal-64", "ls_misal_loads.ma64");
         } else if (g == "G6") {
            named("opcache-acc", "op_cache_hit_miss.all_op_cache_accesses");
            named("opcache-miss", "op_cache_hit_miss.op_cache_miss");
            named("icache-miss", "ic_tag_hit_miss.instruction_cache_miss");
            named("fetch-stall", "ic_fetch_stall.ic_stall_any");
         } else if (g == "G7") {
            named("sse-avx", "ex_ret_mmx_fp_instr.sse_instr"); // int + FP
         } else if (g == "G8") {
            named("stores", "ls_dispatch.store_dispatch");
            named("mab-loads", "ls_mab_alloc.loads");
            named("pf-dram-fills", "ls_hw_pf_dc_fills.mem_io_local");
            named("dtlb-miss", "ls_l1_d_tlb_miss.all");
         }
      }
#endif
      if (!unresolved.empty())
         groupFail("cannot resolve " + unresolved.front());
      if (groupNames.size() == anchors)
         groupFail("no such group on this CPU (G0..G8, G3a, G3b; see "
                   "PMU_EVENT_GROUPS.md)");
   }
#endif

   void registerAll() {
#ifdef __linux__
      if (!perfGroup.empty()) {
         // the group: cycles leads; it can't run partially, so a group that
         // doesn't fit fails to open (or reports .run = 0) instead of rotating
         for (auto& name : groupNames) {
            auto& event = events[name][0];
            event.fd = syscall(__NR_perf_event_open, &event.pe, 0, -1,
                               groupLeaderFd, 0);
            if (event.fd < 0)
               groupFail("cannot open " + name + ": " + strerror(errno));
            if (groupLeaderFd < 0) groupLeaderFd = event.fd;
         }
         // everything else (task-clock, msr) on its own
         for (auto& name : ordered_names) {
            if (std::find(groupNames.begin(), groupNames.end(), name) !=
                groupNames.end())
               continue;
            auto& event = events[name][0];
            event.fd = syscall(__NR_perf_event_open, &event.pe, 0, -1, -1, 0);
            if (event.fd < 0)
               groupFail("cannot open " + name + ": " + strerror(errno));
         }
         return;
      }
#endif
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

   bool inGroup(const std::string& name) const {
      return std::find(groupNames.begin(), groupNames.end(), name) !=
             groupNames.end();
   }

   void startAll() {
#ifdef __linux__
      if (!perfGroup.empty()) {
         // enable the whole group at once, the rest one by one, then take
         // the start readings
         ioctl(groupLeaderFd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
         for (auto& name : ordered_names)
            if (!inGroup(name))
               ioctl(events[name][0].fd, PERF_EVENT_IOC_ENABLE, 0);
         for (auto& name : ordered_names) {
            auto& event = events[name][0];
            if (read(event.fd, &event.prev, sizeof(uint64_t) * 3) !=
                sizeof(uint64_t) * 3)
               groupFail("cannot read " + name);
         }
         return;
      }
#endif
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
#ifdef __linux__
      if (!perfGroup.empty()) {
         // read everything before disabling anything: a disabled leader
         // stops its members while their enabled time keeps running
         for (auto& name : ordered_names) {
            auto& event = events[name][0];
            if (read(event.fd, &event.data, sizeof(uint64_t) * 3) !=
                sizeof(uint64_t) * 3)
               groupFail("cannot read " + name);
         }
         ioctl(groupLeaderFd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
         for (auto& name : ordered_names)
            if (!inGroup(name))
               ioctl(events[name][0].fd, PERF_EVENT_IOC_DISABLE, 0);
         return;
      }
#endif
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
      for (auto& name : ordered_names) {
         out << std::setw(printFieldWidth) << name << ",";
         // grouped mode: each event's share of time actually counted
         if (!perfGroup.empty())
            out << std::setw(printFieldWidth) << name + ".run" << ",";
      }
   }

   // time_running / time_enabled over the last measurement; 1 = counted the
   // whole time, < 1 = multiplexed or not scheduled
   double runShare(const std::string& name) {
#ifdef __linux__
      auto& event = events[name][0];
      double enabled = (double)(event.data.time_enabled - event.prev.time_enabled);
      return enabled > 0
                 ? (double)(event.data.time_running - event.prev.time_running) /
                       enabled
                 : 0;
#else
      compat::unused(name);
      return 0;
#endif
   }

   void printAll(std::ostream& out, double n) {
      // grouped mode prints 6 decimals so rare events don't round to 0
      auto oldPrecision = out.precision();
      if (!perfGroup.empty()) out.precision(6);
      for (auto& name : ordered_names) {
         double aggr = 0;
         for (auto& event : events[name]) aggr += event.readCounter();
         out << std::setw(printFieldWidth) << aggr / n << ",";
         if (!perfGroup.empty()) {
            double share = runShare(name);
            out << std::setw(printFieldWidth) << share << ",";
            if (share < 0.999)
               std::cerr << "PERF_GROUP=" << perfGroup << ": " << name
                         << " counted " << share
                         << " of the time (multiplexed or not scheduled)"
                         << std::endl;
         }
      }
      out.precision(oldPrecision);
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

// Tier-2 run-timeline: emit a monotonic, within-process stage marker to stderr
// so the corpus can reconstruct load/measure timing and the gaps between them.
// Format: TIMELINE\t<event>\t<label>\t<monotonic_ms>
inline void tl(const char* event, const char* label = "") {
   auto now = std::chrono::steady_clock::now();
   long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()).count();
   fprintf(stderr, "TIMELINE\t%s\t%s\t%lld\n", event, label, ms);
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
#ifdef CANARY_IN_BENCH
   if (canary::g_enabled) canary::before(s.c_str());
#endif
   tl("measure_start", s.c_str());

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

#ifdef CANARY_IN_BENCH
   if (canary::g_enabled) canary::after(s.c_str());
#endif
   tl("measure_end", s.c_str());

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
