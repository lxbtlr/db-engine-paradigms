#!/usr/bin/env bash
# Where does Vectorwise time go? perf-profiles run_tpch (engine v) per query and
# ranks functions by self time, so the next kernels to tune are picked from
# data rather than guessed.
#
# Output: results/vw_profile_<timestamp>/
#   <build>_q<N>_t<T>.txt   perf report (self %, top symbols)
#   summary.csv             build,query,threads,rank,self_pct,symbol  (top 25)
#
# Environment:
#   TPCH_PATH  (required) run_tpch -p path, e.g. .../tpch/sf10/
#   BUILDS     "build_ablation/gcc_base build_ablation/gcc_sel_char"
#   QUERIES    "1 3 5 6 9 18"
#   THREADS    "1"          space-separated; 1 keeps symbols clean of scheduling noise
#   REPS       20           run_tpch -r; the TPC-H load is sampled too, so more
#                           reps push the load share down
#   FREQ       2999         perf sampling frequency (Hz)
#   PERF       perf         perf binary
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TPCH_PATH:?set TPCH_PATH to the run_tpch data dir (e.g. .../tpch/sf10/)}"
BUILDS=${BUILDS:-"build_ablation/gcc_base build_ablation/gcc_sel_char"}
QUERIES=${QUERIES:-"1 3 5 6 9 18"}
THREADS=${THREADS:-"1"}
REPS=${REPS:-20}
FREQ=${FREQ:-2999}
PERF=${PERF:-perf}
OUT="$ROOT/results/vw_profile_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/driver.log"; }

command -v "$PERF" > /dev/null || { echo "perf not found (set PERF=)"; exit 2; }
echo "build,query,threads,rank,self_pct,symbol" > "$OUT/summary.csv"

for b in $BUILDS; do
  bin="$ROOT/$b/run_tpch"
  [ -x "$bin" ] || { log "skip $b: no run_tpch"; continue; }
  tag=$(basename "$b")
  for q in $QUERIES; do
    for t in $THREADS; do
      base="$OUT/${tag}_q${q}_t${t}"
      log "$tag q$q threads=$t"
      # The data load runs in the same process and is sampled too (loader /
      # page-fault symbols); REPS repetitions make query time dominate.
      if ! "$PERF" record -F "$FREQ" -g -o "$base.data" -- \
            "$bin" -p "$TPCH_PATH" -e v -q "$q" -r "$REPS" -t "$t" -s 1 \
            > "$base.run.log" 2>&1; then
        log "  run failed (see $base.run.log)"; continue
      fi
      # self time per symbol, demangled; children off so callers don't absorb
      # the primitives they call through function pointers
      "$PERF" report -i "$base.data" --no-children --sort symbol --percent-limit 0.3 \
          --stdio --demangle 2> /dev/null | grep -v '^#' | grep -v '^$' > "$base.txt"
      awk -v b="$tag" -v q="$q" -v t="$t" '
        /^ *[0-9.]+%/ && n < 25 {
          pct = $1; sub(/%/, "", pct)
          sym = $0; sub(/^ *[0-9.]+% +\[[^]]*\] +/, "", sym); gsub(/,/, ";", sym)
          printf "%s,q%s,%s,%d,%s,%s\n", b, q, t, ++n, pct, sym }' "$base.txt" >> "$OUT/summary.csv"
      rm -f "$base.data"   # large; the text report is what we need
    done
  done
done

log "results: $OUT (summary.csv, per-query reports)"
