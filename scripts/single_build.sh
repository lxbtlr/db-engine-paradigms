#!/usr/bin/env bash
# Build, test and time one TPC-valid build: one compiler x one config from
# flag_ablation.sh's join_valid group. The build is what gets registered in
# the results database, so a config outside join_valid is refused and any
# build, test or timing failure fails the run (exit 1, results kept).
#
# Runs flag_ablation.sh with CONFIGS=$CONFIG COMPILERS=$COMPILER, so the
# build tree, cmake flags, TPC-H tests and timing are identical to an
# ablation run of that config. Output has the same layout, under
# results/single_build_<host>_<timestamp>/ (timing.csv, matrix.csv,
# machine.txt, driver.log, <compiler>_<config>/{defines.txt,tpch_test.log,...})
# plus build.txt (config, compiler, commit, effective defines).
#
# Environment:
#   CONFIG     required; one name from flag_ablation.sh's join_valid group
#   COMPILER   gcc       gcc or clang
#   QUERIES, REPS, ROUNDS, SETTLE, TEST_THREADS, TIMEOUT, BENCH_HOST, ...
#              passed through to flag_ablation.sh (see its header)
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABL="$ROOT/scripts/flag_ablation.sh"
CONFIG=${CONFIG:-}
COMPILER=${COMPILER:-gcc}

die() { echo "single_build: $*" >&2; exit 2; }
[ -n "$CONFIG" ] || die "CONFIG is required"
case "$CONFIG" in *[[:space:]]*) die "CONFIG takes one config, got '$CONFIG'" ;; esac
case "$COMPILER" in gcc|clang) ;; *) die "COMPILER must be gcc or clang, got '$COMPILER'" ;; esac
valid=$(CONFIGS=join_valid PRINT_CONFIGS=1 bash "$ABL") || die "cannot list join_valid configs"
case " $valid " in *" $CONFIG "*) ;; *) die "$CONFIG is not TPC-valid (join_valid: $valid)" ;; esac

export RESULTS_PREFIX=single_build MICROBENCH=0 CONFIGS="$CONFIG" COMPILERS="$COMPILER"
unset HASH_BASE SKIP_BUILD SUMMARIZE_ONLY PRINT_CONFIGS
bash "$ABL"
rc=$?

# the results directory this run just made
BENCH_HOST=${BENCH_HOST:-$(hostname -s 2>/dev/null || hostname)}
out=$(ls -d "$ROOT"/results/single_build_"$BENCH_HOST"_* 2>/dev/null | sort | tail -n 1)
[ -n "$out" ] || { echo "single_build: no results directory" >&2; exit 1; }
# flag_ablation.sh only logs a missing data directory; a registered build
# must have passed the TPC-H tests and been timed
if [ "$rc" -eq 0 ]; then
  grep -q -E '^\[  PASSED  \]' "$out/${COMPILER}_${CONFIG}/tpch_test.log" 2>/dev/null ||
    { echo "single_build: TPC-H tests did not run" | tee -a "$out/driver.log" >&2; rc=1; }
  [ "$(wc -l < "$out/timing.csv")" -gt 1 ] ||
    { echo "single_build: no timings" | tee -a "$out/driver.log" >&2; rc=1; }
fi
{
  echo "config: $CONFIG"
  echo "compiler: $COMPILER"
  echo "commit: $(git -C "$ROOT" rev-parse HEAD)"
  echo "exit: $rc"
  echo "defines: $(cat "$out/${COMPILER}_${CONFIG}/defines.txt" 2>/dev/null)"
} > "$out/build.txt"
exit "$rc"
