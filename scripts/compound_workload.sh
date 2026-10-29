#!/usr/bin/env bash
# Record the compound projection chains the benchmark plans build, for
# VW_COMPOUND_SET=workload (cmake/CompoundKernels.cmake).
#
#   scripts/compound_workload.sh <run_tpch> <tpch dir> [out]
#
# <run_tpch> must be a VW_PROJ_COMPOUND build with chain kernels
# (VW_COMPOUND_SET=workload or VW_COMPOUND_DEPTH > 2) at the depth to record:
# plans are built with chains of up to its VW_COMPOUND_DEPTH, and each chain
# is logged whole, whether or not that build has a kernel for it. Runs the
# TPC-H queries (QUERIES) and Q1 with the dd chain (DD_W widths, both shapes),
# one rep each, Tectorwise, one thread, then writes the sorted chain list to
# out (default src/vectorwise/primitives/compound_workload.txt).
set -euo pipefail

[ $# -ge 2 ] || { sed -n '2,13p' "$0" >&2; exit 2; }
BIN=$1
DATA=$2
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=${3:-$ROOT/src/vectorwise/primitives/compound_workload.txt}
QUERIES=${QUERIES:-"1 3 5 6 9 18"}
DD_W=${DD_W:-"2 16 64"}

LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT
run() { VW_COMPOUND_RECORD=$LOG EXTERNALPROFILE=1 "$BIN" -p "$DATA" -e v -t 1 -r 1 "$@" > /dev/null 2>&1; }
for q in $QUERIES; do run -q "$q"; done
for w in $DD_W; do
  for k in chained independent; do run -q 1 -w "$w" -k "$k"; done
done

{
  echo "# Compound projection chains recorded by scripts/compound_workload.sh"
  echo "# ($(basename "$BIN"), queries: $QUERIES; Q1 dd W: $DD_W, both shapes)."
  echo "# One chain per line, format in cmake/CompoundKernels.cmake."
  sort -u "$LOG"
} > "$OUT"
echo "$(grep -vc '^#' "$OUT") chains -> $OUT"
