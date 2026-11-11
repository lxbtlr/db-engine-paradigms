#!/usr/bin/env bash
# Builds and runs run_selbench (src/benchmarks/primitives/selbench.cpp: the
# per-row cost of the selection kernels: scalar _bf templates, the old
# *_avx512 kernels, the VW_SIMD_SEL AVX-512 kernels and the AVX2 kernels of
# SimdSelAvx2.hpp, for contiguous and gathered input and the range pairs) on
# this machine. Built with the TARGET_MACHINE preset of this host, as
# flag_ablation.sh builds; every kernel the target's ISA allows is compiled,
# independent of the VW_SIMD_SEL options.
#
# Environment (defaults in brackets):
#   COMPILER  gcc | clang                               [gcc]
#   MODE      l1 | stream | all, run_selbench -m        [all]
#   REPS      -r                                        [5000]
#   SELS      selectivity percentages, -s               [1,10,50,90,99]
#   DENSITY   inSel density percent for selsel, -d      [50]
#   CPU       core the benchmark is pinned to           [0]
#
# Output: results/selbench_<host>_<stamp>/ with machine.txt, selbench.csv,
# build.log.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_HOST=${BENCH_HOST:-$(hostname -s 2>/dev/null || hostname)}
COMPILER=${COMPILER:-gcc}
MODE=${MODE:-all}
REPS=${REPS:-5000}
SELS=${SELS:-1,10,50,90,99}
DENSITY=${DENSITY:-50}
CPU=${CPU:-0}
case "$MODE" in l1|stream|all) ;; *) echo "MODE must be l1, stream or all" >&2; exit 2 ;; esac
case "$BENCH_HOST" in
  dubliner|roquefort|manchego|burrata|kafir|rpi5) MACHINE=$BENCH_HOST ;;
  *) MACHINE=custom ;;
esac
OUT="$ROOT/results/selbench_${BENCH_HOST}_$(date +%Y%m%d_%H%M%S)"
B="$ROOT/build_flags/$BENCH_HOST/selbench_$COMPILER"
mkdir -p "$OUT"

{
  echo "host: $(hostname)  MACHINE: $MACHINE  CPU: $CPU  COMPILER: $COMPILER"
  echo "date: $(date -Is)"
  echo "git:  $(git -C "$ROOT" rev-parse --short HEAD)"
  lscpu | grep -E 'Model name|MHz' || true
  echo "isa flags: $(grep -o -w -E 'avx2|avx512f|avx512vl|avx512bw|asimd' /proc/cpuinfo | sort -u | tr '\n' ' ')"
  echo "MODE: $MODE  REPS: $REPS  SELS: $SELS  DENSITY: $DENSITY"
} | tee "$OUT/machine.txt"

cmake -S "$ROOT" -B "$B" -DCMAKE_BUILD_TYPE=Release -DCOMPILER="$COMPILER" \
  -DTARGET_MACHINE="$MACHINE" -DTARGET_ARCH= > "$OUT/build.log" 2>&1 &&
  cmake --build "$B" -j "$(nproc)" --target run_selbench >> "$OUT/build.log" 2>&1 ||
  { echo "build failed (see build.log)" >&2; tail -40 "$OUT/build.log" >&2; exit 1; }
grep -m1 -E 'CXX_FLAGS|ARCH' "$B/CMakeCache.txt" >> "$OUT/machine.txt" || true

taskset -c "$CPU" "$B/run_selbench" -m "$MODE" -r "$REPS" -s "$SELS" \
  -d "$DENSITY" > "$OUT/selbench.csv" ||
  { echo "run_selbench failed" >&2; exit 1; }
echo "$(($(wc -l < "$OUT/selbench.csv") - 1)) rows in $OUT/selbench.csv"
