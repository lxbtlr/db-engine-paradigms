#!/bin/bash
set -euo pipefail

TPCH_PATH="${1:?Usage: $0 <tpch-data-path> [threads] [reps]}"
THREADS="${2:-1}"
REPS="${3:-5}"
BUILD_BASE="data_depth"
WIDTHS="2 4 6 8 10 12 14 16 20 24"

echo "W,engine,output"

for W in $WIDTHS; do
    DIR="${BUILD_BASE}_W${W}"
    mkdir -p "$DIR"
    pushd "$DIR" >/dev/null
    cmake clean
    cmake ../.. -DLIVE_SET_WIDTH=$W 2>&1 | grep -v "^--" >/dev/null || true
    make -j 22 run_tpch 2>&1 | tail -1

    # Hyper
    OUTPUT=$(./run_tpch -p "$TPCH_PATH" -q 1 -e h -t "$THREADS" -r "$REPS" 2>/dev/null | grep "q1 hyper")
    echo "${W},hyper,${OUTPUT}"

    # VW
    OUTPUT=$(./run_tpch -p "$TPCH_PATH" -q 1 -e v -t "$THREADS" -r "$REPS" 2>/dev/null | grep "q1 vectorwise")
    echo "${W},vw,${OUTPUT}"

    # Packed
    OUTPUT=$(./run_tpch -p "$TPCH_PATH" -q 1 -e p -t "$THREADS" -r "$REPS" 2>/dev/null | grep "q1 packed" || true)
    if [ -n "$OUTPUT" ]; then
        echo "${W},packed,${OUTPUT}"
    fi

    popd >/dev/null
done
