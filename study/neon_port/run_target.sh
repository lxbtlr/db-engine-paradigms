#!/usr/bin/env bash
# Build run_neonbench (+ _autovec) for a target machine with gcc and clang and
# run every mode pinned to one core / its memory node. Writes
#   <out>/machine.txt, <out>/<compiler>_<binary>.csv
#
# Usage: study/neon_port/run_target.sh <TARGET_MACHINE> [out dir]
#   e.g. study/neon_port/run_target.sh burrata
# Env: CPU=<core> (default 0), COMPILERS="gcc clang", ARGS (extra
# run_neonbench flags, e.g. "-t 64").
set -euo pipefail
M=${1:?usage: run_target.sh <TARGET_MACHINE> [out dir]}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${2:-$ROOT/results/neonbench_${M}_$(date +%Y%m%dT%H%M)}
CPU=${CPU:-0}
COMPILERS=${COMPILERS:-gcc clang}
mkdir -p "$OUT"
{ uname -a; lscpu; } > "$OUT/machine.txt" 2>&1 || true

if command -v numactl >/dev/null; then
   NODE=$(numactl --hardware | awk -v c="$CPU" '/cpus:/ { for (i = 4; i <= NF; ++i) if ($i == c) { print $2; exit } }')
   PIN=(numactl --physcpubind="$CPU" --membind="${NODE:-0}")
else
   PIN=(taskset -c "$CPU")
fi

for cc in $COMPILERS; do
   B=$ROOT/build_neonbench_${M}_$cc
   cmake -S "$ROOT" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCOMPILER="$cc" \
         -DTARGET_MACHINE="$M" > "$OUT/${cc}_cmake.log"
   WIDTH=; [[ $(uname -m) == aarch64 ]] && WIDTH=run_widthbench
   cmake --build "$B" --target run_neonbench run_neonbench_autovec $WIDTH >> "$OUT/${cc}_cmake.log"
   if [[ -n $WIDTH ]]; then
      # front-end / dispatch width (calibrates the llvm-mca cycle figures)
      "${PIN[@]}" "$B/run_widthbench" > "$OUT/${cc}_widthbench.csv"
      if command -v perf >/dev/null; then
         perf stat -x, -e cycles,instructions,stall_frontend,stall_backend \
            "${PIN[@]}" "$B/run_widthbench" > /dev/null 2> "$OUT/${cc}_widthbench_perf.csv" || true
      fi
   fi
   for bin in run_neonbench run_neonbench_autovec; do
      echo "== $cc $bin" >&2
      "${PIN[@]}" "$B/$bin" ${ARGS:-} > "$OUT/${cc}_${bin}.csv" 2> "$OUT/${cc}_${bin}.err"
   done
done
echo "results: $OUT" >&2
