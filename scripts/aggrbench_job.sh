#!/usr/bin/env bash
# Builds and runs run_aggrbench -U (src/benchmarks/primitives/aggrbench.cpp,
# HashGroup::aggUpdate alone on Q1's aggregate shape: fused, kinds, bucket
# (VW_AGGR_FUSED_BUCKET) and direct, swept over the groups per vector) on
# this machine. Built with the TARGET_MACHINE preset of this host, as
# flag_ablation.sh builds.
#
# Environment (defaults in brackets):
#   COMPILER  gcc | clang                               [gcc]
#   DISTS     key distributions, -d per run             [q1,runs]
#   SELS      selection percentages, -s                 [98,100]
#   GROUP_LIST  -g list                                 [its defaults]
#   REPS      vectors per trial, -r                     [2000]
#   CPU       core the benchmark is pinned to           [0]
#
# Output: results/aggrbench_<host>_<stamp>/ with machine.txt, aggrbench.csv,
# build.log.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_HOST=${BENCH_HOST:-$(hostname -s 2>/dev/null || hostname)}
COMPILER=${COMPILER:-gcc}
DISTS=${DISTS:-q1,runs}
SELS=${SELS:-98,100}
REPS=${REPS:-2000}
CPU=${CPU:-0}
case "$BENCH_HOST" in
  dubliner|roquefort|manchego|burrata|kafir|rpi5) MACHINE=$BENCH_HOST ;;
  *) MACHINE=custom ;;
esac
OUT="$ROOT/results/aggrbench_${BENCH_HOST}_$(date +%Y%m%d_%H%M%S)"
B="$ROOT/build_flags/$BENCH_HOST/aggrbench_$COMPILER"
mkdir -p "$OUT"

{
  echo "host: $(hostname)  MACHINE: $MACHINE  CPU: $CPU  COMPILER: $COMPILER"
  echo "date: $(date -Is)"
  echo "git:  $(git -C "$ROOT" rev-parse --short HEAD)"
  lscpu | grep -E 'Model name|MHz' || true
  echo "isa flags: $(grep -o -w -E 'avx2|avx512f|asimd' /proc/cpuinfo | sort -u | tr '\n' ' ')"
  echo "DISTS: $DISTS  SELS: $SELS  GROUP_LIST: ${GROUP_LIST:-defaults}  REPS: $REPS"
} | tee "$OUT/machine.txt"

cmake -S "$ROOT" -B "$B" -DCMAKE_BUILD_TYPE=Release -DCOMPILER="$COMPILER" \
  -DTARGET_MACHINE="$MACHINE" -DTARGET_ARCH= > "$OUT/build.log" 2>&1 &&
  cmake --build "$B" -j "$(nproc)" --target run_aggrbench >> "$OUT/build.log" 2>&1 ||
  { echo "build failed (see build.log)" >&2; tail -40 "$OUT/build.log" >&2; exit 1; }
grep -m1 -E 'CXX_FLAGS|ARCH' "$B/CMakeCache.txt" >> "$OUT/machine.txt" || true

args=(-U -s "$SELS" -r "$REPS")
[ -n "${GROUP_LIST:-}" ] && args+=(-g "$GROUP_LIST")
first=1
for d in ${DISTS//,/ }; do
  taskset -c "$CPU" "$B/run_aggrbench" "${args[@]}" -d "$d" > "$OUT/aggrbench_$d.csv" ||
    { echo "run_aggrbench -d $d failed" >&2; exit 1; }
  if [ $first = 1 ]; then cat "$OUT/aggrbench_$d.csv"; first=0
  else tail -n +2 "$OUT/aggrbench_$d.csv"; fi
done > "$OUT/aggrbench.csv"
grep -q MISMATCH "$OUT/aggrbench.csv" && echo "MISMATCH rows in aggrbench.csv" >&2
column -t -s, "$OUT/aggrbench.csv"
