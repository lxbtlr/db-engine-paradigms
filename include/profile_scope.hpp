#pragma once
// Scope an external profiler to the measured repetitions of
// PerfEvents::timeAndProfile: the data load and the three warmup runs are
// left out, so VTune / perf see the same window run_tpch times.
//
//   VTune: build with -DPROF_ITT, ittnotify.h on the include path and
//          libittnotify.a -ldl linked (scripts/vtune_q1.sh injects all three
//          through CMAKE_CXX_FLAGS / CMAKE_CXX_STANDARD_LIBRARIES), then
//          collect with `vtune -start-paused`. Each measured window is also
//          an ITT task named after the run label ("q1 v  t1"), so the GUI
//          can filter by task when one process runs several queries.
//   perf:  `perf stat --delay=-1 --control fifo:CTL,ACK -- run_tpch ...`
//          with PERF_CTL_FIFO=CTL and PERF_ACK_FIFO=ACK in the environment.
//
// Both are opt-in: without PROF_ITT no ITT call is compiled in, and without
// PERF_CTL_FIFO no fifo is opened, so default builds and runs are unchanged.
// The calls sit outside the per-rep timing loop.
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#ifdef PROF_ITT
#include <ittnotify.h>
#endif

namespace profscope {
#ifdef PROF_ITT
inline __itt_domain* domain() {
   static __itt_domain* d = __itt_domain_create("run_tpch");
   return d;
}
#endif

// perf --control fifo: send one command and wait for perf's "ack", so the
// counters are really on (or off) before the caller continues. perf holds
// both fifos open read-write, so neither open() blocks.
inline void perfCtl(const char* cmd) {
   static int ctlFd = -2, ackFd = -1;
   if (ctlFd == -2) {
      const char* ctl = getenv("PERF_CTL_FIFO");
      ctlFd = (ctl && *ctl) ? open(ctl, O_WRONLY | O_CLOEXEC) : -1;
      if (ctl && *ctl && ctlFd < 0)
         fprintf(stderr, "profscope: cannot open PERF_CTL_FIFO %s: %s\n", ctl,
                 strerror(errno));
      const char* ack = getenv("PERF_ACK_FIFO");
      if (ctlFd >= 0 && ack && *ack) ackFd = open(ack, O_RDONLY | O_CLOEXEC);
   }
   if (ctlFd < 0) return;
   if (write(ctlFd, cmd, strlen(cmd)) < 0) return;
   if (ackFd >= 0) {
      char buf[16];
      if (read(ackFd, buf, sizeof(buf)) < 0) return;
   }
}

inline void begin(const char* label) {
#ifdef PROF_ITT
   __itt_resume();
   __itt_task_begin(domain(), __itt_null, __itt_null,
                    __itt_string_handle_create(label));
#else
   (void)label;
#endif
   perfCtl("enable\n");
}

inline void end() {
   perfCtl("disable\n");
#ifdef PROF_ITT
   __itt_task_end(domain());
   __itt_pause();
#endif
}
} // namespace profscope
