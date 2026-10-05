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
# plus build.txt (config, compiler, autovectorize, run settings, commit,
# effective defines). Settings that differ from their defaults are appended
# to the registered config name (see flag_ablation.sh: <config>_autovec_t4...).
#
# Environment:
#   CONFIG     required; one name from flag_ablation.sh's join_valid group
#              (any flag_ablation.sh config with BENCH=ssb)
#   BENCH      tpch      tpch or ssb (run_ssb, SSB.* tests, _ssb names)
#   COMPILER   gcc       gcc or clang
#   AUTOVEC    OFF       ON = build with -DAUTOVECTORIZE=ON
#   QUERIES, REPS, ROUNDS, SETTLE, TEST_THREADS, TIMEOUT, ENGINE, THREADS,
#   VECTOR_SIZE, SF, PIN, CPU, SIMDhash, SIMDjoin, SIMDsel, SIMDproj, VW_FLAGS,
#   PERF, BENCH_HOST, ... (TESTS must stay all)
#              passed through to flag_ablation.sh (see its header)
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABL="$ROOT/scripts/flag_ablation.sh"
CONFIG=${CONFIG:-}
COMPILER=${COMPILER:-gcc}
AUTOVEC=${AUTOVEC:-OFF}
BENCH=${BENCH:-tpch}

die() { echo "single_build: $*" >&2; exit 2; }
[ -n "$CONFIG" ] || die "CONFIG is required"
case "$CONFIG" in *[[:space:]]*) die "CONFIG takes one config, got '$CONFIG'" ;; esac
case "$COMPILER" in gcc|clang) ;; *) die "COMPILER must be gcc or clang, got '$COMPILER'" ;; esac
case "$AUTOVEC" in ON|OFF) ;; *) die "AUTOVEC must be ON or OFF, got '$AUTOVEC'" ;; esac
case "$BENCH" in tpch|ssb) ;; *) die "BENCH must be tpch or ssb, got '$BENCH'" ;; esac
# a registered build has passed every test of its benchmark
[ "${TESTS:-all}" = all ] || die "TESTS=$TESTS: a registered build runs all $BENCH tests"
# join_valid is the TPC-H audit; SSB builds may use any config
if [ "$BENCH" = tpch ]; then
  valid=$(CONFIGS=join_valid PRINT_CONFIGS=1 bash "$ABL") || die "cannot list join_valid configs"
  case " $valid " in *" $CONFIG "*) ;; *) die "$CONFIG is not TPC-valid (join_valid: $valid)" ;; esac
fi

export RESULTS_PREFIX=single_build MICROBENCH=0 CONFIGS="$CONFIG" COMPILERS="$COMPILER" AUTOVEC BENCH
unset HASH_BASE SKIP_BUILD SUMMARIZE_ONLY PRINT_CONFIGS TESTS
bash "$ABL"
rc=$?
# exit 2: flag_ablation.sh refused a setting before making a results directory
[ "$rc" -eq 2 ] && exit 2

# the results directory this run just made
BENCH_HOST=${BENCH_HOST:-$(hostname -s 2>/dev/null || hostname)}
out=$(ls -d "$ROOT"/results/single_build_"$BENCH_HOST"_* 2>/dev/null | sort | tail -n 1)
[ -n "$out" ] || { echo "single_build: no results directory" >&2; exit 1; }
# the build's directory: <compiler>_<config><suffix of the run settings>
build=$(ls -d "$out/${COMPILER}_${CONFIG}"*/ 2>/dev/null)
[ "$(echo "$build" | grep -c .)" -eq 1 ] || { echo "single_build: expected one ${COMPILER}_${CONFIG}* directory in $out" >&2; exit 1; }
NAME=$(basename "$build"); NAME=${NAME#"${COMPILER}_"}
# flag_ablation.sh only logs a missing data directory; a registered build
# must have passed its benchmark's tests and been timed
if [ "$rc" -eq 0 ]; then
  grep -q -E '^\[  PASSED  \]' "$out/${COMPILER}_${NAME}/${BENCH}_test.log" 2>/dev/null ||
    { echo "single_build: $BENCH tests did not run" | tee -a "$out/driver.log" >&2; rc=1; }
  [ "$(wc -l < "$out/timing.csv")" -gt 1 ] ||
    { echo "single_build: no timings" | tee -a "$out/driver.log" >&2; rc=1; }
fi
{
  echo "config: $NAME"
  echo "compiler: $COMPILER"
  echo "autovectorize: $AUTOVEC"
  echo "bench: $BENCH"
  echo "engine: ${ENGINE:-v}  threads: ${THREADS:-1}  vector_size: ${VECTOR_SIZE:-1024}  sf: ${SF:-1}  pin: ${PIN:-auto}"
  echo "SIMDhash: ${SIMDhash:-}  SIMDjoin: ${SIMDjoin:-}  SIMDsel: ${SIMDsel:-}  SIMDproj: ${SIMDproj:-}"
  echo "vw_flags: $(cat "$out/vw_flags.txt" 2>/dev/null)"
  echo "cpu: ${CPU:-0}  perf: ${PERF:-none}"
  echo "commit: $(git -C "$ROOT" rev-parse HEAD)"
  echo "exit: $rc"
  echo "defines: $(cat "$out/${COMPILER}_${NAME}/defines.txt" 2>/dev/null)"
} > "$out/build.txt"
exit "$rc"
