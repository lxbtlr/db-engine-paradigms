#pragma once
/// NUMA_DEBUG-only trace of Typer (TBB) pipelines.
///
/// A query is traced between NUMA_TRACE_QUERY(label) and the end of its
/// scope; every parallel construct of ParallelHelper.hpp / GroupBy.hpp opens
/// a pipeline (NT_SCOPE) and records each task body it runs (NT_TASK):
///   - OS thread (gettid), arena slot, sched_getcpu(), regionOfCpu(cpu)
///   - start time relative to the pipeline start (region arrival)
///   - numa_parallel_* only: launcher iterations per thread, morsels per
///     shard split into home / stolen (dispatch home = regionOf(slot)) and
///     local / remote (CPU region)
/// endQuery() prints one "[NT] pipe" line per pipeline and one "[NT] query"
/// summary with the cross-pipeline slot / region changes per OS thread.
/// Without NUMA_DEBUG every macro expands to nothing.

#ifdef NUMA_DEBUG
#include "common/runtime/Concurrency.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <sched.h>
#include <set>
#include <string>
#include <sys/syscall.h>
#include <tbb/task_arena.h>
#include <unistd.h>
#include <vector>

namespace numa_trace {
constexpr size_t N = runtime::NUM_NUMA_REGIONS;

inline uint64_t nowNs() {
   return std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
       .count();
}

struct Task {
   long tid;
   int slot;
   int cpu;
   size_t region; // regionOfCpu(cpu)
   uint64_t startNs; // since pipeline start
};

struct Pipeline {
   std::string name;
   bool numa = false; // numa_parallel_scan / numa_parallel_reduce
   uint64_t t0 = 0, t1 = 0;
   std::vector<Task> tasks;
   // per shard: morsels taken by a task whose dispatch home is that shard /
   // by another; by a task whose CPU is on that region / elsewhere
   std::array<size_t, N> home{}, stolen{}, local{}, remote{};
};

struct State {
   std::mutex m;
   bool active = false;
   std::string query;
   std::vector<std::unique_ptr<Pipeline>> pipes;
   Pipeline* cur = nullptr;
};
inline State& state() {
   static State s;
   return s;
}

inline void beginQuery(const std::string& label) {
   auto& s = state();
   std::lock_guard<std::mutex> l(s.m);
   s.active = true;
   // "q1 h  t8  " -> "q1h_t8"
   s.query.clear();
   for (char c : label)
      if (c != ' ') s.query += c;
   auto t = s.query.find('t', 1);
   if (t != std::string::npos) s.query.insert(t, "_");
   s.pipes.clear();
   s.cur = nullptr;
}

inline void endQuery();

struct Query {
   explicit Query(const std::string& label) { beginQuery(label); }
   ~Query() { endQuery(); }
};

/// Opens a pipeline for its lifetime (nested scopes restore the outer one).
struct Scope {
   Pipeline* p = nullptr;
   Pipeline* prev = nullptr;
   Scope(const char* name, bool numa = false) {
      auto& s = state();
      std::lock_guard<std::mutex> l(s.m);
      if (!s.active) return;
      s.pipes.push_back(std::make_unique<Pipeline>());
      p = s.pipes.back().get();
      p->name = name;
      p->numa = numa;
      prev = s.cur;
      s.cur = p;
      p->t0 = nowNs();
   }
   ~Scope() {
      if (!p) return;
      auto& s = state();
      std::lock_guard<std::mutex> l(s.m);
      p->t1 = nowNs();
      s.cur = prev;
   }
};

/// Record one task body of pipeline p (nullptr: the current pipeline).
inline Task task(Pipeline* p = nullptr) {
   Task t;
   t.tid = syscall(SYS_gettid);
   t.slot = tbb::this_task_arena::current_thread_index();
   t.cpu = sched_getcpu();
   t.region = runtime::regionOfCpu(static_cast<size_t>(t.cpu));
   auto& s = state();
   std::lock_guard<std::mutex> l(s.m);
   if (!p) p = s.cur;
   if (!p) return t;
   t.startNs = nowNs() - p->t0;
   p->tasks.push_back(t);
   return t;
}

/// Add one numa_parallel_* task's morsel counts (indexed by shard) to
/// pipeline p (nullptr: the current pipeline).
inline void morsels(Pipeline* p, size_t dispatchHome, size_t cpuRegion,
                    const std::array<size_t, N>& taken) {
   auto& s = state();
   std::lock_guard<std::mutex> l(s.m);
   if (!p) p = s.cur;
   if (!p) return;
   for (size_t n = 0; n < N; ++n) {
      (n == dispatchHome ? p->home : p->stolen)[n] += taken[n];
      (n == cpuRegion ? p->local : p->remote)[n] += taken[n];
   }
}

template <typename A> std::string join(const A& a) {
   std::string r;
   for (size_t i = 0; i < a.size(); ++i) {
      if (i) r += ",";
      r += std::to_string(a[i]);
   }
   return r;
}

inline void endQuery() {
   auto& s = state();
   std::lock_guard<std::mutex> l(s.m);
   if (!s.active) return;
   s.active = false;
   const char* q = s.query.c_str();

   // last (slot, region) seen per OS thread, for cross-pipeline changes
   std::map<long, std::pair<int, size_t>> last;
   size_t transitions = 0, slotChanges = 0, regionChanges = 0;
   size_t pipesRegionChange = 0, pipesSlotChange = 0, intraRegion = 0;
   size_t slotCpuMismatch = 0, nTasks = 0;
   std::set<long> allTids, migratedTids;

   for (size_t pi = 0; pi < s.pipes.size(); ++pi) {
      auto& p = *s.pipes[pi];
      std::map<long, std::pair<std::set<int>, std::set<size_t>>> seen;
      std::map<long, size_t> iters;
      std::array<int64_t, N> firstUs, lastUs;
      std::array<size_t, N> tasksPerRegion{};
      firstUs.fill(-1);
      lastUs.fill(-1);
      for (auto& t : p.tasks) {
         seen[t.tid].first.insert(t.slot);
         seen[t.tid].second.insert(t.region);
         iters[t.tid]++;
         allTids.insert(t.tid);
         int64_t us = static_cast<int64_t>(t.startNs / 1000);
         if (firstUs[t.region] < 0 || us < firstUs[t.region])
            firstUs[t.region] = us;
         lastUs[t.region] = std::max(lastUs[t.region], us);
         tasksPerRegion[t.region]++;
         if (t.slot >= 0 && runtime::regionOf(static_cast<size_t>(t.slot)) !=
                                t.region)
            slotCpuMismatch++;
      }
      nTasks += p.tasks.size();
      bool rc = false, sc = false;
      for (auto& kv : seen) {
         auto& slots = kv.second.first;
         auto& regions = kv.second.second;
         if (regions.size() > 1) intraRegion++;
         int slot = *slots.begin();
         size_t region = *regions.begin();
         auto it = last.find(kv.first);
         if (it != last.end()) {
            transitions++;
            if (it->second.first != slot) {
               slotChanges++;
               sc = true;
            }
            if (it->second.second != region) {
               regionChanges++;
               rc = true;
               migratedTids.insert(kv.first);
            }
         }
         last[kv.first] = {*slots.rbegin(), *regions.rbegin()};
      }
      pipesRegionChange += rc;
      pipesSlotChange += sc;

      size_t maxIters = 0, multi = 0;
      for (auto& kv : iters) {
         maxIters = std::max(maxIters, kv.second);
         multi += kv.second > 1;
      }
      fprintf(stderr,
              "[NT] pipe q=%s i=%zu name=%s numa=%d us=%llu tasks=%zu tids=%zu"
              " region_tasks=%s first_us=%s last_us=%s region_change=%d"
              " slot_change=%d",
              q, pi, p.name.c_str(), p.numa ? 1 : 0,
              (unsigned long long)((p.t1 - p.t0) / 1000), p.tasks.size(),
              seen.size(), join(tasksPerRegion).c_str(),
              join(firstUs).c_str(), join(lastUs).c_str(), rc ? 1 : 0,
              sc ? 1 : 0);
      if (p.numa)
         fprintf(stderr,
                 " iters_max=%zu tids_multi=%zu home=%s stolen=%s local=%s"
                 " remote=%s",
                 maxIters, multi, join(p.home).c_str(), join(p.stolen).c_str(),
                 join(p.local).c_str(), join(p.remote).c_str());
      fprintf(stderr, "\n");
   }
   fprintf(stderr,
           "[NT] query q=%s pipes=%zu tasks=%zu tids=%zu transitions=%zu"
           " slot_changes=%zu region_changes=%zu pipes_slot_change=%zu"
           " pipes_region_change=%zu tids_migrated=%zu intra_pipe_region=%zu"
           " slot_cpu_mismatch=%zu\n",
           q, s.pipes.size(), nTasks, allTids.size(), transitions, slotChanges,
           regionChanges, pipesSlotChange, pipesRegionChange,
           migratedTids.size(), intraRegion, slotCpuMismatch);
   s.pipes.clear();
   s.cur = nullptr;
}
} // namespace numa_trace

#define NT_CAT2(a, b) a##b
#define NT_CAT(a, b) NT_CAT2(a, b)
#define NUMA_TRACE_QUERY(label) numa_trace::Query NT_CAT(nt_q_, __LINE__)(label)
#define NT_SCOPE(name) numa_trace::Scope NT_CAT(nt_s_, __LINE__)(name)
#define NT_TASK() numa_trace::task()
/// Temporary scope for an expression: (NT_SCOPE_TMP("x"), expr)
#define NT_SCOPE_TMP(name) numa_trace::Scope(name)
#else
#define NUMA_TRACE_QUERY(label)
#define NT_SCOPE(name)
#define NT_TASK()
#define NT_SCOPE_TMP(name) (void)0
#endif
