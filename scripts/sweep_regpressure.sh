#!/bin/bash
set -euo pipefail

TPCH_PATH="${1:?Usage: $0 <tpch-data-path> [threads] [reps] [outfile]}"
THREADS="${2:-1}"
REPS="${3:-5}"
_outfile="${4:-sweep_results_$(date +%Y%m%d_%H%M%S).csv}"
# Make absolute so pushd into build dirs doesn't break the path
[[ "$_outfile" = /* ]] || _outfile="$PWD/$_outfile"
OUTFILE="$_outfile"
BUILD_BASE="/home/alexb/swole/db-engines/build/dd"
WIDTHS="2 4 6 8 10 12 14 16 20 24"

# Write params header
{
    echo "# sweep_regpressure run: $(date)"
    echo "# tpch_path: $TPCH_PATH"
    echo "# threads: $THREADS"
    echo "# reps: $REPS"
    echo "# widths: $WIDTHS"
    echo "W,engine,output"
} > "$OUTFILE"

echo "Writing results to: $OUTFILE"
echo "Params: threads=$THREADS reps=$REPS widths=[$WIDTHS]"

for W in $WIDTHS; do
    echo "--- W=$W ---"
    DIR="${BUILD_BASE}_W${W}"
    mkdir -p "$DIR"
    pushd "$DIR" >/dev/null

    #rm -rf CMake*
    cmake ../.. -DLIVE_SET_WIDTH=$W 2>&1 | grep -v "^--" >/dev/null || true
    make -j 22 run_tpch 2>&1 | tail -1

    # Hyper
    OUTPUT=$(./run_tpch -p "$TPCH_PATH" -q 1 -e h -t "$THREADS" -r "$REPS" 2>/dev/null | grep "q1 hyper")
    LINE="${W},hyper,${OUTPUT}"
    echo "$LINE"
    echo "$LINE" >> "$OUTFILE"

    # VW
    OUTPUT=$(./run_tpch -p "$TPCH_PATH" -q 1 -e v -t "$THREADS" -r "$REPS" 2>/dev/null | grep "q1 vectorwise")
    LINE="${W},vw,${OUTPUT}"
    echo "$LINE"
    echo "$LINE" >> "$OUTFILE"

    # Packed
    OUTPUT=$(./run_tpch -p "$TPCH_PATH" -q 1 -e p -t "$THREADS" -r "$REPS" 2>/dev/null | grep "q1 packed" || true)
    if [ -n "$OUTPUT" ]; then
        LINE="${W},packed,${OUTPUT}"
        echo "$LINE"
        echo "$LINE" >> "$OUTFILE"
    fi

    popd >/dev/null
done

echo "Done. Results saved to: $OUTFILE"
