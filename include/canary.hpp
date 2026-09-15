// canary.hpp — frozen reference measurement, built as a STANDALONE tool.
//
// Rationale (see skills/canary): on an exclusive node, ambient state is a weak
// quality signal and reference measurements are a strong one. A tiny fixed
// workload, frozen forever, run before and after a benchmark job, turns "was
// the machine behaving?" from an inference over proxies into a direct
// measurement.
//
// The canary deliberately does NOT live inside the benchmark binary. It is a
// separate executable (CMake target `canary`, src/tools/canary.cpp) invoked by
// the run-job prologue (phase=before) and epilogue (phase=after), so run_tpch
// carries no canary code in its critical path at all.
//
// IDENTITY & REPRODUCIBILITY
//   The canary's version string is "<workload-version>@<git-commit>":
//     - workload-version (CANARY_WORKLOAD_VERSION) freezes the kernels below.
//       The array sizes, iteration counts and algorithms are PART of the
//       canary identity. Never change them after a version has been run into
//       the corpus — a modified canary is a new canary and its history
//       restarts (same discipline as probe.version).
//     - git-commit (GIT_COMMIT_SHORT) is embedded at build time by CMake, so
//       every binary reports exactly which source it was built from. This
//       makes the canary both version-controlled and reproducible.
//
// Two kernels, responding to different machine misbehaviour:
//   - "compute": cache-resident, compute-bound (dependent int-mul chain over a
//     1 MiB buffer that stays in L2). Sensitive to clock/thermal/contention on
//     the ALU pipe.
//   - "memory": memory-bound (dependent pointer chase over 128 MiB, > all
//     caches). Sensitive to DRAM latency / bandwidth / NUMA / ECC scrubbing.
//   The ratio between them tells you WHICH kind of machine problem occurred.
//
// All kernels are single-threaded (fixed thread count for the life of the
// project) and deterministic. Timings are reported in MILLISECONDS.

#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#ifndef CANARY_WORKLOAD_VERSION
#define CANARY_WORKLOAD_VERSION "canary-v1"
#endif

#ifndef GIT_COMMIT_SHORT
#define GIT_COMMIT_SHORT "unknown"
#endif

namespace canary {

// Master switch for the IN-BENCH path (set from run.cpp when --canary is
// passed). Only compiled into run_tpch when CANARY_IN_BENCH is ON; the
// standalone canary binary never sets it.
inline bool g_enabled = false;

// ── frozen workload parameters (part of the canary identity) ─────────────
// compute-bound: 1 MiB cache-resident gather + dependent multiply chain
static constexpr size_t kComputeBytes = 1u << 20;        // 1 MiB (L2-resident)
static constexpr size_t kComputeIters = 40000000u;       // fixed iteration count
// memory-bound: 128 MiB dependent pointer chase (exceeds every cache)
static constexpr size_t kMemBytes     = 128u << 20;      // 128 MiB
static constexpr size_t kMemIters     = 2000000u;        // fixed iteration count

// fixed rep count for the canary's own statistics
static constexpr int kReps = 5;

struct Stats {
   double median_ms = 0, mean_ms = 0, min_ms = 0, max_ms = 0, stddev_ms = 0;
   int reps = 0;
};

inline const char* version() { return CANARY_WORKLOAD_VERSION "@" GIT_COMMIT_SHORT; }

// Prevent the compiler from removing any kernel work. Kept out-of-line-ish via
// a volatile sink so the frozen loop bodies are the thing being measured.
inline void escape(volatile uint64_t* p) { (void)p; }

inline double now_ms() {
   using namespace std::chrono;
   return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// One rep of the compute-bound kernel: dependent imul/add chain over a
// cache-resident gather. ~0.2-0.3 s per rep on a modern core.
inline double runComputeOnce() {
   static std::vector<uint64_t> buf(kComputeBytes / sizeof(uint64_t), 1);
   const size_t mask = buf.size() - 1;   // power of two
   volatile uint64_t sink = 0;
   double t0 = now_ms();
   for (size_t i = 0; i < kComputeIters; ++i) {
      sink += buf[(i * 2654435761u) & mask];  // cache-resident gather
      sink *= 0x9E3779B97F4A7C15ull;          // dependent chain (latency bound)
   }
   double t1 = now_ms();
   escape(&sink);
   return t1 - t0;
}

// One rep of the memory-bound kernel: dependent pointer chase over 128 MiB.
// p = buf[p] is a serial dependency on DRAM latency. ~0.1-0.2 s per rep.
inline double runMemoryOnce() {
   static std::vector<size_t> chase(kMemBytes / sizeof(size_t));
   static bool inited = false;
   if (!inited) {
      // full-period permutation: step is odd (coprime with the power-of-two
      // length), so the chase covers every cache line, no repeats.
      for (size_t i = 0; i < chase.size(); ++i)
         chase[i] = (i * 2654435761u + 1) & (chase.size() - 1);
      inited = true;
   }
   size_t p = 0;
   volatile size_t sink = 0;
   double t0 = now_ms();
   for (size_t i = 0; i < kMemIters; ++i) p = chase[p];
   double t1 = now_ms();
   sink = p; escape(reinterpret_cast<volatile uint64_t*>(&sink));
   return t1 - t0;
}

// Measure a kernel kReps times and reduce to moments. A single warmup rep runs
// first (un-timed) so the measured reps observe steady-state (cold-TLB/page
// effects land in the warmup, not in the moments).
inline Stats measure(double (*kernel)()) {
   kernel();  // warmup
   std::vector<double> t;
   t.reserve(kReps);
   for (int i = 0; i < kReps; ++i) t.push_back(kernel());
   std::vector<double> sorted = t;
   std::sort(sorted.begin(), sorted.end());
   Stats s;
   s.reps = kReps;
   s.min_ms = sorted.front();
   s.max_ms = sorted.back();
   s.median_ms = sorted[sorted.size() / 2];
   s.mean_ms = std::accumulate(t.begin(), t.end(), 0.0) / t.size();
   double sq = 0;
   for (double x : t) sq += (x - s.mean_ms) * (x - s.mean_ms);
   s.stddev_ms = t.size() > 1 ? std::sqrt(sq / (t.size() - 1)) : 0.0;
   return s;
}

// Emit one observation as a parseable line to the given stream. Timings are
// MILLISECONDS; the corpus stores them as nanoseconds (x1e6).
//
//   CANARY\t<version>\t<kind>\t<phase>\t<label>\t<median>\t<mean>\t<min>\t<max>\t<stddev>\t<reps>\t<threads>
inline void emit(std::ostream& out, const char* kind, const char* phase,
                 const char* label, const Stats& s) {
   out << "CANARY\t" << version() << "\t" << kind << "\t" << phase
       << "\t" << label
       << "\t" << s.median_ms << "\t" << s.mean_ms << "\t" << s.min_ms
       << "\t" << s.max_ms << "\t" << s.stddev_ms << "\t" << s.reps
       << "\t1\n";
}

// Run both frozen kernels and emit their observations for a given phase.
// `label` is the job/batch identifier this canary brackets (empty for a plain
// prologue/epilogue canary). Emits one line per kind to std::cerr so callers
// can tee it to a sidecar without touching stdout.
inline void run(const char* phase, const char* label, std::ostream& out) {
   Stats c = measure(runComputeOnce);
   emit(out, "compute", phase, label, c);
   Stats m = measure(runMemoryOnce);
   emit(out, "memory", phase, label, m);
}

// IN-BENCH bracketing (only used when CANARY_IN_BENCH is ON): bracket the
// START/END of a timed cell with both kernels, emitting to stderr so the
// stdout CSV stays intact. These are compiled into run_tpch only under that
// option; otherwise the canary lives solely in the standalone binary.
inline void before(const char* label) {
   Stats c = measure(runComputeOnce);
   emit(std::cerr, "compute", "before", label, c);
   Stats m = measure(runMemoryOnce);
   emit(std::cerr, "memory", "before", label, m);
}
inline void after(const char* label) {
   Stats c = measure(runComputeOnce);
   emit(std::cerr, "compute", "after", label, c);
   Stats m = measure(runMemoryOnce);
   emit(std::cerr, "memory", "after", label, m);
}

}  // namespace canary
