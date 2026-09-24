#!/usr/bin/env bash
# Smoke test for the nondeterministic TPCH.q3 (vectorwise) failure.
#
# Answers two questions:
#   1. Is it a race?        TPCH.q3 at several thread counts, several times each,
#                           on an existing build (default: build_ablation/gcc_base).
#   2. Is it pre-existing?  Same test on a clean worktree of PRE_COMMIT (default
#                           e841b30, the commit before the SIMD selection work).
#
# Environment:
#   DATADIR       (required for part 2) CMake DATADIR; tests read $DATADIR/tpch/sf1/
#   BUILD         build_ablation/gcc_base      existing build dir with test_all
#   THREADS_LIST  "1 2 4 8 22 44 $(nproc)"
#   REPEAT        3                            runs per thread count
#   PRE_COMMIT    e841b30                      set PRE_COMMIT= to skip part 2
#   COMPILER      gcc                          compiler family for the PRE_COMMIT build
#   MACHINE       dubliner                     CMake TARGET_MACHINE
#   WORKTREE      <repo>/../q3_pre_<commit>     worktree location for PRE_COMMIT
#   KEEP_WORKTREE 0                            1 = keep worktree and build afterwards
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD=${BUILD:-$ROOT/build_ablation/gcc_base}
THREADS_LIST=${THREADS_LIST:-"1 2 4 8 22 44 $(nproc)"}
REPEAT=${REPEAT:-3}
PRE_COMMIT=${PRE_COMMIT-e841b30}
COMPILER=${COMPILER:-gcc}
MACHINE=${MACHINE:-dubliner}
WORKTREE=${WORKTREE:-$ROOT/../q3_pre_${PRE_COMMIT}}
KEEP_WORKTREE=${KEEP_WORKTREE:-0}
OUT="$ROOT/results/q3_check_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

log() { echo "[$(date +%T)] $*" | tee -a "$OUT/driver.log"; }
echo "variant,threads,run,result,got,expected" > "$OUT/summary.csv"

# run_q3 <variant> <test_all binary>: TPCH.q3 for every thread count x REPEAT
run_q3() {
  local variant=$1 bin=$2 t r f res got exp
  for t in $THREADS_LIST; do
    for r in $(seq 1 "$REPEAT"); do
      f="$OUT/${variant}_t${t}_r${r}.log"
      threads=$t SIMDsel=0 "$bin" --gtest_filter='TPCH.q3' > "$f" 2>&1
      if grep -q '^\[  PASSED  \] 1 test' "$f"; then
        res=pass got="" exp=""
      elif grep -q '^\[  FAILED  \]' "$f"; then
        res=fail
        got=$(grep -m1 -A1 'revenue\[i\]' "$f" | grep -o 'Which is: .*' | cut -d' ' -f3)
        exp=$(grep -m1 -A1 'expected\[make_tuple' "$f" | grep -o 'Which is: .*' | cut -d' ' -f3)
        grep -q 'q3 hyper' "$f" && res="fail(hyper)"
        grep -q 'q3 vectorwise' "$f" && res="fail(vectorwise)"
      else
        res=crash got="" exp=""
      fi
      echo "$variant,$t,$r,$res,$got,$exp" >> "$OUT/summary.csv"
      log "$variant threads=$t run=$r: $res ${got:+got=$got expected=$exp}"
    done
  done
}

verdict() { # verdict <variant>
  awk -F, -v v="$1" '$1 == v { n[$2]++; if ($4 == "pass") p[$2]++ }
    END { for (t in n) printf "  %-10s threads=%-4s %d/%d pass\n", v, t, p[t] + 0, n[t] }' \
    "$OUT/summary.csv" | sort -t= -k2,2n
}

# -------------------------------------------------------------- part 1
if [ ! -x "$BUILD/test_all" ]; then
  log "no $BUILD/test_all; build it first (or set BUILD=)"; exit 2
fi
log "part 1: current code ($BUILD, git $(git -C "$ROOT" rev-parse --short HEAD))"
run_q3 current "$BUILD/test_all"

# -------------------------------------------------------------- part 2
if [ -n "$PRE_COMMIT" ]; then
  if [ -z "${DATADIR:-}" ]; then
    log "part 2 skipped: set DATADIR (dir containing tpch/sf1/) or PRE_COMMIT= to silence"
  else
    log "part 2: clean worktree of $PRE_COMMIT at $WORKTREE"
    if [ ! -d "$WORKTREE" ]; then
      git -C "$ROOT" worktree add --detach "$WORKTREE" "$PRE_COMMIT" > "$OUT/worktree.log" 2>&1 \
        || { log "git worktree add failed (see $OUT/worktree.log)"; exit 1; }
    fi
    # Newer compilers need the two test-build fixes made after that commit.
    # They touch only test/support code, not query execution.
    grep -q '<cstdint>' "$WORKTREE/include/common/runtime/Mmap.hpp" \
      || sed -i 's|#include <cassert>|#include <cassert>\n#include <cstdint>|' "$WORKTREE/include/common/runtime/Mmap.hpp"
    sed -i 's|m.insert({i, {i}});|m.insert({int(i), {i}});|' "$WORKTREE/src/test/common/Mmap.cpp"
    PB="$WORKTREE/build_q3"
    if cmake -S "$WORKTREE" -B "$PB" -DCMAKE_BUILD_TYPE=Release -DCOMPILER="$COMPILER" \
          -DTARGET_MACHINE="$MACHINE" -DDATADIR="$DATADIR" > "$OUT/pre_cmake.log" 2>&1 \
       && cmake --build "$PB" -j "$(nproc)" --target test_all > "$OUT/pre_build.log" 2>&1; then
      run_q3 "pre_$PRE_COMMIT" "$PB/test_all"
    else
      log "PRE_COMMIT build failed (see $OUT/pre_cmake.log / pre_build.log)"
    fi
    if [ "$KEEP_WORKTREE" != 1 ]; then
      git -C "$ROOT" worktree remove --force "$WORKTREE" && log "removed worktree $WORKTREE"
    fi
  fi
fi

# -------------------------------------------------------------- verdict
{
  echo; echo "pass counts:"
  verdict current
  [ -n "$PRE_COMMIT" ] && verdict "pre_$PRE_COMMIT"
  echo
  echo "reading it:"
  echo "  current passes at threads=1 but fails at higher counts  -> race in the parallel plan"
  echo "  current fails even at threads=1                         -> deterministic bug (data/plan)"
  echo "  pre_$PRE_COMMIT also fails                                -> pre-existing, not from the SIMD work"
  echo "  pre_$PRE_COMMIT passes, current fails                     -> regression in e.g. $PRE_COMMIT..HEAD; bisect"
} | tee -a "$OUT/driver.log"
log "results: $OUT (summary.csv, per-run logs)"
