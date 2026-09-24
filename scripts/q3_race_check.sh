#!/usr/bin/env bash
# Smoke test for the nondeterministic TPCH.q3 (vectorwise) failure.
#
# Answers two questions:
#   1. Is it a race?        TPCH.q3 at several thread counts, several times each,
#                           on an existing build (default: build_ablation/gcc_base).
#   2. Is it pre-existing?  Same test on a clean worktree of PRE_COMMIT (default
#                           e841b30, the commit before the SIMD selection work).
#   3. Is it the restrict?  Same test on HEAD with only the lookup_sel_
#                           __restrict__ removed.
#
# Environment:
#   DATADIR       (required for part 2) CMake DATADIR; tests read $DATADIR/tpch/sf1/
#   BUILD         build_ablation/gcc_base      existing build dir with test_all
#   THREADS_LIST  "1 2 4 8 22 44 $(nproc)"
#   REPEAT        3                            runs per thread count
#   PRE_COMMIT    e841b30                      set PRE_COMMIT= to skip part 2
#   COMPILER      gcc                          compiler family for the PRE_COMMIT build
#   MACHINE       dubliner                     CMake TARGET_MACHINE
#   SKIP_NORES    0                            1 = skip part 3 (HEAD without lookup_sel_ RES)
#   KEEP_WORKTREE 0                            1 = keep worktree and build afterwards
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD=${BUILD:-$ROOT/build_ablation/gcc_base}
THREADS_LIST=${THREADS_LIST:-"1 2 4 8 22 44 $(nproc)"}
REPEAT=${REPEAT:-3}
PRE_COMMIT=${PRE_COMMIT-e841b30}
COMPILER=${COMPILER:-gcc}
MACHINE=${MACHINE:-dubliner}
KEEP_WORKTREE=${KEEP_WORKTREE:-0}
OUT="$ROOT/results/q3_check_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

log() { echo "[$(date +%T)] $*" | tee -a "$OUT/driver.log"; }
echo "variant,threads,run,result,wrong_groups,got,expected" > "$OUT/summary.csv"

# run_q3 <variant> <test_all binary>: TPCH.q3 for every thread count x REPEAT
run_q3() {
  local variant=$1 bin=$2 t r f res got exp nwrong
  for t in $THREADS_LIST; do
    for r in $(seq 1 "$REPEAT"); do
      f="$OUT/${variant}_t${t}_r${r}.log"
      threads=$t SIMDsel=0 "$bin" --gtest_filter='TPCH.q3' > "$f" 2>&1
      if grep -q '^\[  PASSED  \] 1 test' "$f"; then
        res=pass got="" exp="" nwrong=""
      elif grep -q '^\[  FAILED  \]' "$f"; then
        res=fail
        # first wrong group: "group (k, d, p): revenue X expected Y"
        got=$(grep -m1 -o 'revenue [0-9.-]* expected [0-9.-]*' "$f" | cut -d' ' -f2)
        exp=$(grep -m1 -o 'revenue [0-9.-]* expected [0-9.-]*' "$f" | cut -d' ' -f4)
        nwrong=$(grep -m1 -o '[0-9]* of [0-9]* groups have the wrong revenue' "$f" | cut -d' ' -f1,3 | tr ' ' '/')
        if [ -z "$got" ]; then # older test binaries: ASSERT_EQ "Which is:" format
          got=$(grep -m1 -A1 'revenue\[i\]' "$f" | grep -o 'Which is: .*' | cut -d' ' -f3)
          exp=$(grep -m1 -A1 'expected\[make_tuple' "$f" | grep -o 'Which is: .*' | cut -d' ' -f3)
        fi
        grep -q 'q3 hyper' "$f" && res="fail(hyper)"
        grep -q 'q3 vectorwise' "$f" && res="fail(vectorwise)"
      else
        res=crash got="" exp="" nwrong=""
      fi
      echo "$variant,$t,$r,$res,$nwrong,$got,$exp" >> "$OUT/summary.csv"
      log "$variant threads=$t run=$r: $res ${nwrong:+wrong=$nwrong }${got:+first: got=$got expected=$exp}"
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
# Reuse the DATADIR the existing build was configured with.
if [ -z "${DATADIR:-}" ] && [ -f "$BUILD/CMakeCache.txt" ]; then
  DATADIR=$(sed -n 's/^DATADIR:[A-Z]*=//p' "$BUILD/CMakeCache.txt")
  [ -n "$DATADIR" ] && log "DATADIR from $BUILD/CMakeCache.txt: $DATADIR"
fi
log "part 1: current code ($BUILD, git $(git -C "$ROOT" rev-parse --short HEAD))"
run_q3 current "$BUILD/test_all"

# build_variant <name> <commit> <patch-function>: worktree of <commit>, apply the
# patch function inside it, build test_all, run the sweep. Worktree is removed
# afterwards unless KEEP_WORKTREE=1.
build_variant() {
  local name=$1 commit=$2 patch=$3 wt="$ROOT/../q3_$1" pb
  log "$name: worktree of $commit at $wt"
  if [ ! -d "$wt" ]; then
    git -C "$ROOT" worktree add --detach "$wt" "$commit" > "$OUT/${name}_worktree.log" 2>&1       || { log "$name: git worktree add failed (see $OUT/${name}_worktree.log)"; return 1; }
  fi
  # Worktrees do not populate submodules. simde is header-only: reuse the main
  # checkout's copy, or fall back to fetching it.
  if [ ! -e "$wt/3rdparty/simde/simde" ]; then
    if [ -d "$ROOT/3rdparty/simde/simde" ]; then
      rm -rf "$wt/3rdparty/simde" && ln -s "$ROOT/3rdparty/simde" "$wt/3rdparty/simde"
    else
      git -C "$wt" submodule update --init 3rdparty/simde >> "$OUT/${name}_worktree.log" 2>&1
    fi
  fi
  ( cd "$wt" && $patch ) || { log "$name: patch step failed"; return 1; }
  pb="$wt/build_q3"
  if cmake -S "$wt" -B "$pb" -DCMAKE_BUILD_TYPE=Release -DCOMPILER="$COMPILER"         -DTARGET_MACHINE="$MACHINE" -DDATADIR="$DATADIR" > "$OUT/${name}_cmake.log" 2>&1      && cmake --build "$pb" -j "$(nproc)" --target test_all > "$OUT/${name}_build.log" 2>&1; then
    run_q3 "$name" "$pb/test_all"
  else
    log "$name: build failed (see $OUT/${name}_cmake.log / ${name}_build.log)"
  fi
  if [ "$KEEP_WORKTREE" != 1 ]; then
    # drop the simde symlink first so cleanup never touches the main checkout
    [ -L "$wt/3rdparty/simde" ] && rm "$wt/3rdparty/simde"
    git -C "$ROOT" worktree remove --force "$wt" && log "$name: removed $wt"
  fi
}

# Newer compilers need the two test-build fixes made after e841b30. They touch
# only test/support code, not query execution.
patch_pre() {
  grep -q '<cstdint>' include/common/runtime/Mmap.hpp     || sed -i 's|#include <cassert>|#include <cassert>\n#include <cstdint>|' include/common/runtime/Mmap.hpp
  sed -i 's|m.insert({i, {i}});|m.insert({int(i), {i}});|' src/test/common/Mmap.cpp
}

# HEAD with only the lookup_sel_ __restrict__ (cc50900 / b2b3d36) removed.
patch_nores() {
  sed -i -z 's|pos_t lookup_sel_(pos_t n, pos_t\* RES target, pos_t\* RES sel,\n *pos_t\* RES source)|pos_t lookup_sel_(pos_t n, pos_t* target, pos_t* sel, pos_t* source)|'     src/vectorwise/primitives/Projection.cpp
  sed -i 's|pos_t RES lookup_sel_(|pos_t lookup_sel_(|' src/vectorwise/primitives/Projection.cpp
  if grep -A1 'lookup_sel_(' src/vectorwise/primitives/Projection.cpp | grep -q RES; then
    echo "RES still present in lookup_sel_" >&2; return 1
  fi
  grep -n 'lookup_sel_(' src/vectorwise/primitives/Projection.cpp
}

if [ -z "${DATADIR:-}" ]; then
  log "parts 2-3 skipped: no DATADIR (set it to the dir containing tpch/sf1/)"
else
  # ------------------------------------------------------------ part 2
  [ -n "$PRE_COMMIT" ] && build_variant "pre_$PRE_COMMIT" "$PRE_COMMIT" patch_pre
  # ------------------------------------------------------------ part 3
  [ "${SKIP_NORES:-0}" = 1 ] || build_variant head_noRES HEAD patch_nores
fi

# -------------------------------------------------------------- verdict
{
  echo; echo "pass counts:"
  verdict current
  [ -n "$PRE_COMMIT" ] && verdict "pre_$PRE_COMMIT"
  verdict head_noRES
  echo
  echo "reading it:"
  echo "  current passes at threads=1 but fails at higher counts  -> race in the parallel plan"
  echo "  current fails even at threads=1                         -> deterministic bug (data/plan)"
  echo "  pre_$PRE_COMMIT also fails                                -> pre-existing, not from the SIMD work"
  echo "  pre_$PRE_COMMIT passes, current fails                     -> regression in $PRE_COMMIT..HEAD; bisect"
  echo "  head_noRES passes, current fails                           -> the lookup_sel_ __restrict__ is the cause"
} | tee -a "$OUT/driver.log"
log "results: $OUT (summary.csv, per-run logs)"
