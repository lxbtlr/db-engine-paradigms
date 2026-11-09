#!/usr/bin/env bash
# Builds and runs run_hashbench (src/benchmarks/primitives/hashbench.cpp:
# every hash the engine has on the same forms and keys: scalar MurmurHash,
# its SIMD tiers (avx512 / avx2 / neon, VW_SIMD_HASH), CRC32Hash with
# VW_CRC32_FAST and the VW_CRC32_VPCLMUL kernels) on this machine, after the
# SimdHash and GroupConcat tests of test_all. Built with the TARGET_MACHINE
# preset of this host, as flag_ablation.sh builds.
#
# Environment (defaults in brackets):
#   COMPILER  gcc | clang                               [gcc]
#   MODES     l1 | stream | all                         [all]
#   REPS      vectors per trial (l1), -r                [20000]
#   DENSITY   selection density % for *_sel, -d         [50]
#   AUTOVEC   ON | OFF: -DAUTOVECTORIZE                 [OFF]
#   CPU       core the benchmark is pinned to           [0]
#
# Output: results/hashbench_<host>_<stamp>/ with machine.txt, tests.txt,
# hashbench.csv, build.log.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_HOST=${BENCH_HOST:-$(hostname -s 2>/dev/null || hostname)}
COMPILER=${COMPILER:-gcc}
MODES=${MODES:-all}
REPS=${REPS:-20000}
DENSITY=${DENSITY:-50}
AUTOVEC=${AUTOVEC:-OFF}
CPU=${CPU:-0}
case "$BENCH_HOST" in
  dubliner|roquefort|manchego|burrata|kafir|rpi5) MACHINE=$BENCH_HOST ;;
  *) MACHINE=custom ;;
esac
OUT="$ROOT/results/hashbench_${BENCH_HOST}_$(date +%Y%m%d_%H%M%S)"
B="$ROOT/build_flags/$BENCH_HOST/hashbench_${COMPILER}_av$AUTOVEC"
mkdir -p "$OUT"

{
  echo "host: $(hostname)  MACHINE: $MACHINE  CPU: $CPU  COMPILER: $COMPILER"
  echo "date: $(date -Is)"
  echo "git:  $(git -C "$ROOT" rev-parse --short HEAD)"
  lscpu | grep -E 'Model name|MHz' || true
  echo "isa flags: $(grep -o -w -E 'avx2|avx512f|avx512dq|vpclmulqdq|asimd|crc32' /proc/cpuinfo | sort -u | tr '\n' ' ')"
  echo "MODES: $MODES  REPS: $REPS  DENSITY: $DENSITY  AUTOVEC: $AUTOVEC"
} | tee "$OUT/machine.txt"

cmake -S "$ROOT" -B "$B" -DCMAKE_BUILD_TYPE=Release -DCOMPILER="$COMPILER" \
  -DTARGET_MACHINE="$MACHINE" -DTARGET_ARCH= -DAUTOVECTORIZE="$AUTOVEC" \
  -DVW_SIMD_HASH=ON -DVW_CRC32_FAST=ON > "$OUT/build.log" 2>&1 &&
  cmake --build "$B" -j "$(nproc)" --target run_hashbench test_all >> "$OUT/build.log" 2>&1 ||
  { echo "build failed (see build.log)" >&2; tail -40 "$OUT/build.log" >&2; exit 1; }
grep -m1 -E 'CXX_FLAGS|ARCH' "$B/CMakeCache.txt" >> "$OUT/machine.txt" || true

"$B/test_all" --gtest_filter='SimdHash*:GroupConcat*' > "$OUT/tests.txt" 2>&1 ||
  { echo "SimdHash / GroupConcat tests failed (see tests.txt)" >&2; tail -20 "$OUT/tests.txt" >&2; exit 1; }

taskset -c "$CPU" "$B/run_hashbench" -m "$MODES" -r "$REPS" -d "$DENSITY" \
  > "$OUT/hashbench.csv" 2>> "$OUT/machine.txt" ||
  { echo "run_hashbench failed" >&2; exit 1; }
column -t -s, "$OUT/hashbench.csv"
