#!/bin/bash
set -euo pipefail

# sweep_tier.sh — sweep (W, Tier) grid for the tiered-reload kernel.
#
# Usage:
#   sweep_tier.sh <tpch-data-path> [threads] [reps] [outfile] [settle]
#
# Sweeps W ∈ {8, 16, 32} × Tier ∈ {l1, l2, llc, dram}, single thread default,
# 5 reps default.  Uses -P for perf-stat passthrough (handled by profile.hpp).

TPCH_PATH="${1:?Usage: $0 <tpch-data-path> [threads] [reps] [outfile] [settle]}"
THREADS="${2:-1}"
REPS="${3:-5}"
_outfile="${4:-sweep_tier_$(date +%Y%m%d_%H%M%S).csv}"
SETTLE="${5:-3}"
# Make absolute so cd into build dir doesn't break the path
[[ "$_outfile" = /* ]] || _outfile="$PWD/$_outfile"
OUTFILE="$_outfile"

BINARY="./run_tpch"
if [ ! -f "$BINARY" ]; then
    echo "Error: run_tpch not found in current directory. Run from build dir."
    exit 1
fi

WIDTHS="8 16 32"
TIERS="l1 l2 llc dram"

# Write params header
{
    echo "# sweep_tier run: $(date)"
    echo "# tpch_path: $TPCH_PATH"
    echo "# threads: $THREADS"
    echo "# reps: $REPS"
    echo "# settle: $SETTLE"
    echo "# widths: $WIDTHS"
    echo "# tiers: $TIERS"
    echo "W,tier,output"
} > "$OUTFILE"

echo "Writing results to: $OUTFILE"
echo "Params: threads=$THREADS reps=$REPS settle=$SETTLE widths=[$WIDTHS] tiers=[$TIERS]"

for W in $WIDTHS; do
    for TIER in $TIERS; do
        echo "--- W=$W tier=$TIER ---"
        OUTPUT=$($BINARY -p "$TPCH_PATH" -q 1 -e h -t "$THREADS" -r "$REPS" \
                          -w "$W" -c "$TIER" -s "$SETTLE" -P 2>/dev/null \
                 | grep "q1dd" || true)
        if [ -z "$OUTPUT" ]; then
            echo "  (no output)"
            continue
        fi
        LINE="${W},${TIER},${OUTPUT}"
        echo "  $OUTPUT"
        echo "$LINE" >> "$OUTFILE"
    done
done

echo ""
echo "Done. Results saved to: $OUTFILE"
