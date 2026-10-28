#pragma once
#include "common/runtime/Concurrency.hpp"
#include <pthread.h>
#include <sched.h>
#include <tbb/task_arena.h>
#include <tbb/task_scheduler_observer.h>

/// Pins each TBB worker thread to a CPU using the same policy as
/// WorkerGroup::run (spread by default, packed with THREAD_PIN_PACKED).
class PinningObserver : public tbb::task_scheduler_observer {
public:
   explicit PinningObserver(tbb::task_arena& arena)
       : tbb::task_scheduler_observer(arena) {
      observe(true);
   }
   void on_scheduler_entry(bool /*is_worker*/) override {
      int slot = tbb::this_task_arena::current_thread_index();
      if (slot < 0) return;
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(runtime::cpuOfThread(static_cast<size_t>(slot)), &cpuset);
      pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
   }
   ~PinningObserver() override { observe(false); }
};
