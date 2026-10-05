// run_widthbench: how many instructions / micro-ops per cycle an aarch64 core
// sustains, to check llvm-mca's Neoverse N1 model (dispatch width 3, which
// makes the scalar VectorWise loops front-end bound in study/NEON_PORT_STUDY.md)
// against the core. Arm documents N1 as 4-wide decode/rename and 8-wide issue.
//
// Each test is an inline-asm loop of BODY repeated, plus subs + b.ne.
// "chain" is 32 dependent 1-cycle adds: exactly 32 cycles per iteration, so
// its time calibrates ns per cycle without PMU access (cycles = ns / chain
// ns-per-add). Every other test reports instructions per cycle (loop
// control included) and the ceiling each hypothesis predicts:
//   test      body (repeated 8x)                     ceiling if limited by
//   nop       4 nop                                  decode/rename width
//   alu       4 independent add (4 chains, lat 1)    integer ALUs (N1: 3)
//   alu_simd  3 add + 4 vector add (4 chains, lat 2) ALU 3 + ASIMD 2 = 5
//   mix_ld    2 add + 2 vector add + 2 ldr (L1)      ALU + ASIMD + LD = 6
//   store     2 str + 2 add (str = address + data    6 uops per 4 insns
//             uops, per Arm's N1 optimization guide)
//   ldp       2 ldp + 2 add                          loads, 2 regs each
// With a 3-wide front end every test except chain tops out at 3 IPC; with
// 4-wide decode at 4 IPC; 8-wide dispatch only shows as uops (store, ldp).
//
// -d prints each test's loop body as assembly (for study/neon_port/width_mca.sh).
// Usage: run_widthbench [-n iterations] [-d]     CSV on stdout
#if !defined(__aarch64__)
#error "run_widthbench is aarch64 only"
#endif
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

#define X4(s) s s s s
#define X8(s) X4(s) X4(s)
#define X32(s) X8(s) X8(s) X8(s) X8(s)

// loop bodies; %[p] is a 64-byte aligned scratch buffer
#define B_CHAIN X32("add x9, x9, #1\n\t")
#define B_NOP X32("nop\n\t")
#define B_ALU X8("add x9, x9, #1\n\tadd x10, x10, #1\n\tadd x11, x11, #1\n\tadd x12, x12, #1\n\t")
#define B_ALU_SIMD                                                                           \
   X8("add x9, x9, #1\n\tadd x10, x10, #1\n\tadd x11, x11, #1\n\t"                          \
      "add v0.4s, v0.4s, v31.4s\n\tadd v1.4s, v1.4s, v31.4s\n\t"                            \
      "add v2.4s, v2.4s, v31.4s\n\tadd v3.4s, v3.4s, v31.4s\n\t")
#define B_MIX_LD                                                                             \
   X8("add x9, x9, #1\n\tadd x10, x10, #1\n\t"                                              \
      "add v0.4s, v0.4s, v31.4s\n\tadd v1.4s, v1.4s, v31.4s\n\t"                            \
      "ldr x13, [%[p]]\n\tldr x14, [%[p], #64]\n\t")
#define B_STORE                                                                              \
   X8("str x9, [%[p], #128]\n\tstr x10, [%[p], #192]\n\tadd x11, x11, #1\n\tadd x12, x12, #1\n\t")
#define B_LDP                                                                                \
   X8("ldp x13, x14, [%[p]]\n\tldp x15, x16, [%[p], #64]\n\tadd x11, x11, #1\n\tadd x12, x12, #1\n\t")

#define CLOBBERS                                                                             \
   "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "v0", "v1", "v2", "v3", "cc",     \
       "memory"

#define LOOP(name, body)                                                                     \
   static void name(long n, void* p) {                                                       \
      asm volatile("movi v31.4s, #1\n\t"                                                     \
                   "1:\n\t" body "subs %[n], %[n], #1\n\t"                                   \
                   "b.ne 1b\n\t"                                                             \
                   : [n] "+r"(n)                                                             \
                   : [p] "r"(p)                                                              \
                   : CLOBBERS, "v31");                                                       \
   }
LOOP(t_chain, B_CHAIN)
LOOP(t_nop, B_NOP)
LOOP(t_alu, B_ALU)
LOOP(t_alu_simd, B_ALU_SIMD)
LOOP(t_mix_ld, B_MIX_LD)
LOOP(t_store, B_STORE)
LOOP(t_ldp, B_LDP)

namespace {
struct Test {
   const char* name;
   void (*fn)(long, void*);
   int insns;       // per iteration, loop control included
   int uops;        // per iteration with N1's cracking (str: address + data)
   const char* asmBody;
   const char* expect;
};
const Test kTests[] = {
    {"chain", t_chain, 34, 34, B_CHAIN, "1 IPC (calibration)"},
    {"nop", t_nop, 34, 34, B_NOP, "front-end width"},
    {"alu", t_alu, 34, 34, B_ALU, "3 (N1 integer ALUs)"},
    {"alu_simd", t_alu_simd, 58, 58, B_ALU_SIMD, "min(front end, 5)"},
    {"mix_ld", t_mix_ld, 50, 50, B_MIX_LD, "min(front end, 6)"},
    {"store", t_store, 34, 50, B_STORE, "uops: min(dispatch, store pipes)"},
    {"ldp", t_ldp, 34, 34, B_LDP, "min(front end, load pipes)"},
};

double nsPerIter(void (*fn)(long, void*), long n, void* p) {
   double best = 1e300;
   for (int t = 0; t < 7; ++t) {
      const auto t0 = std::chrono::steady_clock::now();
      fn(n, p);
      const auto t1 = std::chrono::steady_clock::now();
      best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
   }
   return best / double(n);
}

void printBody(const Test& t) {
   // the bodies use %[p]; llvm-mca needs a register
   std::string s = t.asmBody;
   for (size_t at; (at = s.find("%[p]")) != std::string::npos;) s.replace(at, 4, "x0");
   for (size_t at; (at = s.find("\t")) != std::string::npos;) s.erase(at, 1);
   std::printf("### %s\n%ssubs x1, x1, #1\nb.ne 0\n", t.name, s.c_str());
}
} // namespace


int main(int argc, char** argv) {
   long n = 2000000;
   bool dump = false;
   for (int opt; (opt = getopt(argc, argv, "n:d")) != -1;) {
      if (opt == 'n') n = std::atol(optarg);
      else if (opt == 'd') dump = true;
      else {
         std::fprintf(stderr, "usage: %s [-n iterations] [-d]\n", argv[0]);
         return 2;
      }
   }
   if (dump) {
      for (const auto& t : kTests) printBody(t);
      return 0;
   }
   alignas(64) static long buf[64];
   const double nsChain = nsPerIter(t_chain, n, buf);
   const double nsPerCycle = nsChain / 32.0; // 32 dependent adds per iteration
   std::printf("test,ns_per_iter,cycles_per_iter,ipc,uops_per_cycle,ghz,expect\n");
   for (const auto& t : kTests) {
      const double ns = t.fn == t_chain ? nsChain : nsPerIter(t.fn, n, buf);
      const double cyc = ns / nsPerCycle;
      std::printf("%s,%.3f,%.2f,%.2f,%.2f,%.3f,%s\n", t.name, ns, cyc, t.insns / cyc,
                  t.uops / cyc, 1.0 / nsPerCycle, t.expect);
   }
   return 0;
}
