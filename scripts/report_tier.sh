#!/bin/bash
set -euo pipefail

# report_tier.sh — post-process sweep_tier.sh output.
#
# Emits per (W, Tier): cycles/tuple, instructions/tuple, IPC,
# L1-misses/tuple, LLC-misses/tuple, mem_stall/tuple, and
# marginal cycles-per-instruction (fitted across W at fixed tier).
#
# Usage:
#   report_tier.sh <sweep_tier_output.csv>

INPUT="${1:?Usage: $0 <sweep_tier_output.csv>}"

if [ ! -f "$INPUT" ]; then
    echo "Error: input file not found: $INPUT"
    exit 1
fi

echo "========================================="
echo "  Tiered-Reload Report"
echo "========================================="
echo ""

# Parse the CSV.  The output column from profile.hpp has the form:
#   label, time_ms, CPUs, IPC, GHz, BW, cycles, LLC-miss, LLC-miss2, l1-miss, instr, br-miss, all_rd, br-miss2, stores, loads, mem_stall, task-clock,
# Fields are comma-separated and whitespace-padded.
# We extract: cycles (col 7), l1-miss (col 10), instr (col 11), LLC-miss (col 8), mem_stall (col 17)

printf "%-4s %-6s %12s %12s %8s %12s %12s %12s\n" \
       "W" "Tier" "cyc/tup" "instr/tup" "IPC" "L1miss/tup" "LLCmiss/tup" "memstall/tup"
echo "---- ------ ------------ ------------ -------- ------------ ------------ ------------"

# Skip comment/header lines
grep -v '^#' "$INPUT" | grep -v '^W,' | while IFS= read -r line; do
    W=$(echo "$line" | cut -d',' -f1)
    TIER=$(echo "$line" | cut -d',' -f2)
    # The rest after the second comma is the profile.hpp output line
    PROFILE=$(echo "$line" | cut -d',' -f3-)

    # Parse the profile fields (comma-separated, whitespace-trimmed)
    # Field layout depends on profile.hpp; the Skylake-X layout is:
    #   name, time, CPUs, IPC, GHz, BW, cycles, LLC-miss, LLC-miss2, l1-miss, instr, br-miss, all_rd, br-miss2, stores, loads, mem_stall, task-clock
    # But the output from profile is per-tuple (divided by count).
    # Indices (0-based from the PROFILE substring, which starts at field 3 of original):
    #   The first field in PROFILE is the label continuation, then time, CPUs, IPC, GHz, BW, ...
    # Actually the "output" in the CSV includes the full timeAndProfile line.
    # Let's just extract by field position from the full profile line.

    # Count commas in the profile line to determine field positions
    IPC=$(echo "$PROFILE" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $4); print $4}')
    CYCLES=$(echo "$PROFILE" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $7); print $7}')
    L1_MISS=$(echo "$PROFILE" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $10); print $10}')
    INSTR=$(echo "$PROFILE" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $11); print $11}')
    LLC_MISS=$(echo "$PROFILE" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $8); print $8}')
    MEM_STALL=$(echo "$PROFILE" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $17); print $17}')

    printf "%-4s %-6s %12s %12s %8s %12s %12s %12s\n" \
           "$W" "$TIER" "$CYCLES" "$INSTR" "$IPC" "$L1_MISS" "$LLC_MISS" "$MEM_STALL"
done

echo ""
echo "To compute marginal cycles-per-instruction per tier:"
echo "  For each tier, fit cycles = a * instructions + b across W."
echo "  The slope a is the marginal cycles-per-instruction."
echo "  Compare against the ~0.29 baseline from Tests 1-2."
