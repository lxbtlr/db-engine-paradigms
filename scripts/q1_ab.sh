#!/usr/bin/env bash
# A/B check: does HEAD change the default VectorWise Q1 path vs a base commit?
#
#   1. builds run_tpch at BASE (git worktree) and at HEAD with identical flags
#   2. compares the machine code of every function in the vectorwise library
#      (per-symbol, addresses/relocations stripped) and lists the ones that
#      differ, flagging the ones Q1 executes
#   3. times Q1 (engine v) alternating BASE/HEAD binaries ROUNDS times
#
# Environment:
#   BASE        01adaeb            commit to compare against
#   TPCH_PATH   (from simd_sel_ablation.sh default) run_tpch -p path
#   COMPILER    gcc
#   MACHINE     dubliner
#   EXTRA       ""                 extra cmake -D flags for BOTH builds
#   THREADS     1                  run_tpch -t
#   REPS        30                 run_tpch -r per invocation
#   ROUNDS      5                  alternating BASE/HEAD invocations
#   NUMA        "numactl --cpunodebind=0 --membind=0"   prefix for runs ("" = none)
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE=${BASE:-01adaeb}
TPCH_PATH=${TPCH_PATH:-/tank/alexb/swole/tpch/sf1}
COMPILER=${COMPILER:-gcc}
MACHINE=${MACHINE:-dubliner}
EXTRA=${EXTRA:-}
THREADS=${THREADS:-1}
REPS=${REPS:-30}
ROUNDS=${ROUNDS:-5}
NUMA=${NUMA-"numactl --cpunodebind=0 --membind=0"}
OUT="$ROOT/results/q1_ab_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/driver.log"; }

WT="$ROOT/../q1_ab_base_$BASE"
log "BASE=$BASE HEAD=$(git -C "$ROOT" rev-parse --short HEAD) compiler=$COMPILER extra='$EXTRA'"
[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ] && \
  log "WARNING: working tree has uncommitted changes; HEAD build uses them"

# ------------------------------------------------------------------ builds
if [ ! -d "$WT" ]; then
  git -C "$ROOT" worktree add --detach "$WT" "$BASE" > "$OUT/worktree.log" 2>&1 || { log "worktree failed"; exit 1; }
fi
# submodule (header-only) + test-only build fixes, as in q3_race_check.sh
if [ ! -e "$WT/3rdparty/simde/simde" ] && [ -d "$ROOT/3rdparty/simde/simde" ]; then
  rm -rf "$WT/3rdparty/simde" && ln -s "$ROOT/3rdparty/simde" "$WT/3rdparty/simde"
fi
grep -q '<cstdint>' "$WT/include/common/runtime/Mmap.hpp" || \
  sed -i 's|#include <cassert>|#include <cassert>\n#include <cstdint>|' "$WT/include/common/runtime/Mmap.hpp"

build() { # <src> <builddir> <tag>
  # shellcheck disable=SC2086
  cmake -S "$1" -B "$2" -DCMAKE_BUILD_TYPE=Release -DCOMPILER="$COMPILER" \
        -DTARGET_MACHINE="$MACHINE" $EXTRA > "$OUT/$3_cmake.log" 2>&1 &&
  cmake --build "$2" -j "$(nproc)" --target run_tpch > "$OUT/$3_build.log" 2>&1
}
BB="$WT/build_q1ab"; HB="$ROOT/build_q1ab_head"
build "$WT" "$BB" base || { log "BASE build failed (see $OUT/base_build.log)"; exit 1; }
build "$ROOT" "$HB" head || { log "HEAD build failed (see $OUT/head_build.log)"; exit 1; }
# identical compile flags?
grep -h 'CMAKE_CXX_FLAGS:\|CMAKE_BUILD_TYPE:' "$BB/CMakeCache.txt" > "$OUT/base_flags.txt"
grep -h 'CMAKE_CXX_FLAGS:\|CMAKE_BUILD_TYPE:' "$HB/CMakeCache.txt" > "$OUT/head_flags.txt"
diff -q "$OUT/base_flags.txt" "$OUT/head_flags.txt" > /dev/null && log "cache flags identical" || log "cache flags DIFFER (see *_flags.txt)"
for b in "$BB" "$HB"; do
  grep -rh -- '-D[A-Z_]*' "$b"/CMakeFiles/vectorwise.dir/flags.make 2>/dev/null | head -1
done > "$OUT/defines.txt"

# ---------------------------------------------------- per-function codegen
# normalized disassembly per symbol: no addresses, no raw bytes, no offsets
disasm() { # <builddir> -> "symbol<TAB>md5" lines
  find "$1/CMakeFiles/vectorwise.dir" -name '*.o' | sort | while read -r o; do
    objdump -d --no-show-raw-insn -C "$o" | awk '
      /^[0-9a-f]+ <.*>:$/ { if (name) print name "\t" body; name = substr($0, index($0, "<") + 1); sub(/>:$/, "", name); body = ""; next }
      /^ +[0-9a-f]+:/ { l = $0; sub(/^ +[0-9a-f]+:[ \t]+/, "", l); gsub(/[0-9a-f]+ <[^>]*>/, "<addr>", l); body = body l ";" }
      END { if (name) print name "\t" body }'
  done | while IFS=$'\t' read -r name body; do printf '%s\t%s\n' "$name" "$(printf '%s' "$body" | md5sum | cut -c1-12)"; done | sort
}
disasm "$BB" > "$OUT/base_funcs.tsv"
disasm "$HB" > "$OUT/head_funcs.tsv"
join -t $'\t' -a1 -a2 -e MISSING -o 0,1.2,2.2 "$OUT/base_funcs.tsv" "$OUT/head_funcs.tsv" | \
  awk -F'\t' '$2 != $3' > "$OUT/changed_funcs.tsv"
# functions Q1 VectorWise executes (profile: HashGroup, aggr, proj, sel, Scan...)
Q1RE='HashGroup|aggr_|proj_|sel_col_val|selsel_|Scan|Select|Project|FixedAggr|Expression|Aggregates|ResultWriter|Hashmap'
log "functions with different code: $(wc -l < "$OUT/changed_funcs.tsv") (all), $(grep -cE "$Q1RE" "$OUT/changed_funcs.tsv") on Q1-relevant names"
grep -E "$Q1RE" "$OUT/changed_funcs.tsv" | cut -f1 | head -40 | sed 's/^/    /' | tee -a "$OUT/driver.log"

# ------------------------------------------------------------------ timing
echo "variant,round,median_ms,min_ms" > "$OUT/timing.csv"
for r in $(seq 1 "$ROUNDS"); do
  for v in base head; do
    bin="$BB/run_tpch"; [ "$v" = head ] && bin="$HB/run_tpch"
    # shellcheck disable=SC2086
    $NUMA "$bin" -p "$TPCH_PATH" -e v -q 1 -r "$REPS" -t "$THREADS" -s 2 > "$OUT/${v}_r$r.csv" 2> "$OUT/${v}_r$r.err"
    awk -F, -v v="$v" -v r="$r" '{l=$1; gsub(/^ +| +$/,"",l)} l ~ /^q1 v/ {m=$2; n=$4; gsub(/ /,"",m); gsub(/ /,"",n); print v "," r "," m "," n}' \
      "$OUT/${v}_r$r.csv" >> "$OUT/timing.csv"
  done
done
log "Q1 (engine v, t=$THREADS) per round:"; column -t -s, "$OUT/timing.csv" | tee -a "$OUT/driver.log"
awk -F, 'NR>1 { s[$1]+=$3; n[$1]++; if (!($1 in mn) || $4 < mn[$1]) mn[$1]=$4 }
  END { for (v in s) printf "  %s: mean of medians %.2f ms, best %.2f ms\n", v, s[v]/n[v], mn[v] }' "$OUT/timing.csv" | tee -a "$OUT/driver.log"
log "results: $OUT   (worktree kept at $WT; remove with: git worktree remove --force $WT)"
