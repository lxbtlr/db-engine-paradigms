// canary.cpp — standalone frozen reference measurement.
//
// A tiny fixed workload (compute-bound + memory-bound kernels, see
// include/canary.hpp) whose runtime is a direct statement about machine
// behaviour at that moment. It is deliberately a SEPARATE executable from the
// benchmark (run_tpch) so the benchmark's critical path carries no canary code.
//
// The run-job prologue invokes  `canary --phase before --out <job>.canary-before`
// and the epilogue invokes    `canary --phase after  --out <job>.canary-after`
// exactly like capture_env.sh brackets a job. ingest-tpch reads both sidecars
// and links the observations to the job's ingest_batch.
//
// Usage: canary [--phase before|after] [--out FILE] [--label STR]
//   --out   write CANARY lines to FILE (default: stdout)
//   --label job identifier carried in each line's label field (default: empty)
//   --phase 'before' or 'after' (default: before)
//
// The kernels are frozen forever: changing them makes a NEW canary version
// whose corpus history restarts (same discipline as probe.version).

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "canary.hpp"

int main(int argc, char* argv[]) {
   std::string phase = "before";
   std::string outfile;
   std::string label;

   for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--phase" && i + 1 < argc) {
         phase = argv[++i];
      } else if (a == "--out" && i + 1 < argc) {
         outfile = argv[++i];
      } else if (a == "--label" && i + 1 < argc) {
         label = argv[++i];
      } else if (a == "-h" || a == "--help") {
         std::cerr << "usage: canary [--phase before|after] [--out FILE] [--label STR]\n";
         return 0;
      } else {
         std::cerr << "canary: unknown argument '" << a << "'\n";
         return 2;
      }
   }
   if (phase != "before" && phase != "after") {
      std::cerr << "canary: --phase must be 'before' or 'after' (got '" << phase << "')\n";
      return 2;
   }

   std::ostream* out = &std::cout;
   if (!outfile.empty()) out = new std::ofstream(outfile);

   std::cerr << "canary version: " << canary::version() << "\n";
   canary::run(phase.c_str(), label.c_str(), *out);

   if (!outfile.empty()) {
      out->flush();
      delete out;
   }
   return 0;
}
