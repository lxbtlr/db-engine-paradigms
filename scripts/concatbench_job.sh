#!/usr/bin/env bash
# Builds and runs run_concatbench (src/benchmarks/primitives/concatbench.cpp,
# VW_GROUP_CONCAT_SIMD's compound key concatenation against the per-column
# copy) on this machine, after the GroupConcat tests of test_all. Built with
# the TARGET_MACHINE preset of this host (manchego: Sapphire Rapids, VBMI;
# dubliner: native Cascade Lake, AVX2; burrata: Neoverse-N1, NEON), as
# flag_ablation.sh builds.
#
# Environment (defaults in brackets):
#   COMPILER  gcc | clang                          [gcc]
#   SHAPES    run_concatbench -k list               [its presets]
#   SELS      selection percentages, -s             [98,50]
#   MODES     l1 | stream | both                    [both]
#   N         rows per column (stream), -N          [8388608]
#   REPS      best of, -r                           [5]
#   CPU       core the benchmark is pinned to       [0]
#   HASH      1: -H, concat plus the packed-key hash  [0]
#   PAD       1: -P, padded against unpadded keys    [0]
#
# Output: results/concatbench_<host>_<stamp>/ with machine.txt, tests.txt,
# concatbench.csv, build.log.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_HOST=${BENCH_HOST:-$(hostname -s 2>/dev/null || hostname)}
COMPILER=${COMPILER:-gcc}
SELS=${SELS:-98,50}
MODES=${MODES:-both}
N=${N:-8388608}
REPS=${REPS:-5}
CPU=${CPU:-0}
case "$BENCH_HOST" in
  dubliner|roquefort|manchego|burrata|kafir|rpi5) MACHINE=$BENCH_HOST ;;
  *) MACHINE=custom ;;
esac
OUT="$ROOT/results/concatbench_${BENCH_HOST}_$(date +%Y%m%d_%H%M%S)"
B="$ROOT/build_flags/$BENCH_HOST/concatbench_$COMPILER"
mkdir -p "$OUT"

{
  echo "host: $(hostname)  MACHINE: $MACHINE  CPU: $CPU  COMPILER: $COMPILER"
  echo "date: $(date -Is)"
  echo "git:  $(git -C "$ROOT" rev-parse --short HEAD)"
  lscpu | grep -E 'Model name|MHz' || true
  echo "isa flags: $(grep -o -w -E 'avx2|avx512f|avx512vbmi|asimd' /proc/cpuinfo | sort -u | tr '\n' ' ')"
  echo "SHAPES: ${SHAPES:-presets}  SELS: $SELS  MODES: $MODES  N: $N  REPS: $REPS  HASH: ${HASH:-0}  PAD: ${PAD:-0}"
} | tee "$OUT/machine.txt"

cmake -S "$ROOT" -B "$B" -DCMAKE_BUILD_TYPE=Release -DCOMPILER="$COMPILER" \
  -DTARGET_MACHINE="$MACHINE" -DTARGET_ARCH= > "$OUT/build.log" 2>&1 &&
  cmake --build "$B" -j "$(nproc)" --target run_concatbench test_all >> "$OUT/build.log" 2>&1 ||
  { echo "build failed (see build.log)" >&2; tail -40 "$OUT/build.log" >&2; exit 1; }
grep -m1 -E 'CXX_FLAGS|ARCH' "$B/CMakeCache.txt" >> "$OUT/machine.txt" || true

"$B/test_all" --gtest_filter='GroupConcat*' > "$OUT/tests.txt" 2>&1 ||
  { echo "GroupConcat tests failed (see tests.txt)" >&2; tail -20 "$OUT/tests.txt" >&2; exit 1; }

args=(-s "$SELS" -m "$MODES" -N "$N" -r "$REPS")
[ -n "${SHAPES:-}" ] && args+=(-k "$SHAPES")
[ "${HASH:-0}" = 1 ] && args+=(-H)
[ "${PAD:-0}" = 1 ] && args+=(-P)
taskset -c "$CPU" "$B/run_concatbench" "${args[@]}" > "$OUT/concatbench.csv" ||
  { echo "run_concatbench failed" >&2; exit 1; }
column -t -s, "$OUT/concatbench.csv"
