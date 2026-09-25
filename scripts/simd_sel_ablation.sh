#!/usr/bin/env bash
# Ablation driver for the VectorWise kernel options: AVX-512 selection
# (VW_SIMD_SEL*), hashing (VW_SIMD_HASH vs VW_USE_CRC32) and join-probe
# prefetching (VW_JOIN_PREFETCH). See config_flags for the configs.
#
# For every compiler x config it:
#   1. configures + builds test_all, run_selbench, run_hashbench, run_tpch into build_ablation/<compiler>_<config>
#   2. checks via nm/objdump that SIMD kernels / prefetches exist exactly when their option is ON
#   3. runs the differential unit tests (SimdSel*, SimdSelRedirect*, SimdHash*, GTEQ*)
#   4. runs the TPC-H correctness tests (TPCH.*) with SIMDsel=0 and SIMDsel=1   [needs DATADIR sf1]
#   5. runs run_selbench [base, sel_pos16] and run_hashbench [base] (per-tuple kernel cost)
#   6. runs run_tpch -e v on Q1,3,5,6,9,18 with SIMDsel=0 and SIMDsel=1         [needs TPCH_PATH]
# and writes everything plus summary.csv / speedup.csv to results/simd_sel_<timestamp>/.
#
# Environment (all optional):
#   COMPILERS   "gcc clang"            compiler families (CMake COMPILER=...)
#   CONFIGS     "base sel sel_scalarload sel_hwgather sel_char sel_pos16"
#   MACHINE     dubliner               CMake TARGET_MACHINE
#   BUILD_TYPE  Release
#   JOBS        $(nproc)
#   DATADIR     (unset)                CMake DATADIR; test_all reads $DATADIR/tpch/sf1/
#   TPCH_PATH   (unset)                run_tpch -p path (e.g. .../tpch/sf10/)
#   THREADS     "1,$(nproc)"           run_tpch -t
#   REPS        10                     run_tpch -r
#   VEC         1024                   run_tpch -v and run_selbench -v
#   SETTLE      5                      run_tpch -s
#   PIN_CPU     0                      run_selbench is pinned here with taskset
#   SKIP_BUILD  0                      1 = reuse existing build dirs
#   EXTRA_CMAKE ""                     -D flags appended to EVERY config (they override
#                                      the config's own flags), e.g. a tuned baseline:
#                                      "-DVW_GROUP_AGGR=ON -DVW_GROUP_AGGR_SEL=ON -DVW_POS_16=ON
#                                       -DVW_USE_CRC32=ON -DHUGE_2MB_MALLOC_HUGE=ON"
#   TEST_THREADS <first of THREADS>    worker threads for test_all (the TPC-H tests
#                                      read env "threads"; unset they use every hw thread)
#   TIMEOUT     1800                   seconds per test / benchmark step (0 = none)
set -u -o pipefail

DATADIR="/tank/alexb/swole/"
TPCH_PATH="/tank/alexb/swole/tpch/sf1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILERS=${COMPILERS:-"gcc clang"}
CONFIGS=${CONFIGS:-"base sel sel_scalarload sel_hwgather sel_char sel_pos16"}
MACHINE=${MACHINE:-dubliner}
BUILD_TYPE=${BUILD_TYPE:-Release}
JOBS=${JOBS:-$(nproc)}
THREADS=${THREADS:-"1,$(nproc)"}
REPS=${REPS:-10}
VEC=${VEC:-1024}
SETTLE=${SETTLE:-5}
PIN_CPU=${PIN_CPU:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
TEST_THREADS=${TEST_THREADS:-${THREADS%%,*}}
TIMEOUT=${TIMEOUT:-1800}
# run a step with a time limit; a timeout counts as a failure (exit 124)
tlimit() { if [ "$TIMEOUT" -gt 0 ]; then timeout --kill-after=30 "$TIMEOUT" "$@"; else "$@"; fi; }
OUT="$ROOT/results/simd_sel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
FAILS=0

log() { echo "[$(date +%T)] $*" | tee -a "$OUT/driver.log"; }
fail() { log "FAIL: $*"; FAILS=$((FAILS + 1)); }

# flags <sel> <gather> <char> <pos16> <crc32> <simdhash> <joinpf> [pfdist] [compress] [unroll] [width]
# Every option is passed explicitly: CMake caches them across reconfigures.
flags() {
  echo "-DVW_SIMD_SEL=$1 -DVW_SIMD_SEL_GATHER=$2 -DVW_SIMD_SEL_CHAR=$3 -DVW_POS_16=$4" \
       "-DVW_USE_CRC32=$5 -DVW_SIMD_HASH=$6 -DVW_JOIN_PREFETCH=$7 -DVW_JOIN_PREFETCH_DIST=${8:-16}" \
       "-DVW_SIMD_SEL_COMPRESS=${9:-reg} -DVW_SIMD_SEL_UNROLL=${10:-1} -DVW_SIMD_SEL_WIDTH=${11:-512}"
}
config_flags() {
  case "$1" in
    #                     sel gather     char pos16 crc32 hash pf  dist compress unroll width
    base)           flags OFF scalar     OFF  OFF   OFF   OFF  OFF ;;
    sel)            flags ON  scalar     OFF  OFF   OFF   OFF  OFF ;;
    sel_scalarload) flags ON  scalarload OFF  OFF   OFF   OFF  OFF ;;
    sel_hwgather)   flags ON  hwgather   OFF  OFF   OFF   OFF  OFF ;;
    sel_mem)        flags ON  scalar     OFF  OFF   OFF   OFF  OFF 16 mem 1 ;;
    sel_u2)         flags ON  scalar     OFF  OFF   OFF   OFF  OFF 16 reg 2 ;;
    sel_mem_u2)     flags ON  scalar     OFF  OFF   OFF   OFF  OFF 16 mem 2 ;;
    sel_w256)       flags ON  scalar     OFF  OFF   OFF   OFF  OFF 16 reg 1 256 ;;
    sel_mem_w256)   flags ON  scalar     OFF  OFF   OFF   OFF  OFF 16 mem 1 256 ;;
    sel_char)       flags ON  scalar     ON   OFF   OFF   OFF  OFF ;;
    sel_pos16)      flags ON  scalar     ON   ON    OFF   OFF  OFF ;;
    char_only)      flags OFF scalar     ON   OFF   OFF   OFF  OFF ;;
    crc32)          flags OFF scalar     OFF  OFF   ON    OFF  OFF ;;
    simd_hash)      flags OFF scalar     OFF  OFF   OFF   ON   OFF ;;
    join_pf)        flags OFF scalar     OFF  OFF   OFF   OFF  ON  16 ;;
    join_pf32)      flags OFF scalar     OFF  OFF   OFF   OFF  ON  32 ;;
    all)            flags ON  scalar     ON   OFF   OFF   ON   ON  16 ;;
    all_crc32)      flags ON  scalar     ON   OFF   ON    OFF  ON  16 ;;
    *) echo "unknown config $1" >&2; exit 2 ;;
  esac
}
# on <cfg> <OPTION>: is OPTION=ON in this config?
# the last -D for an option wins, so EXTRA_CMAKE overrides the config
on() {
  echo "$(config_flags "$1") ${EXTRA_CMAKE:-}" | tr ' ' '\n' | grep -- "^-D$2=" | tail -1 | grep -q "=ON$"
}

# ---------------------------------------------------------------- machine info
{
  echo "host: $(hostname)"
  echo "date: $(date -Is)"
  echo "git:  $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null) $(git -C "$ROOT" status --porcelain 2>/dev/null | wc -l) dirty files"
  lscpu | grep -E 'Model name|Socket|Core|Thread|NUMA node\(s\)|MHz' || true
  echo -n "avx512 flags: "; grep -o -w -E 'avx512(f|bw|vl|dq|cd|vbmi2)' /proc/cpuinfo | sort -u | tr '\n' ' '; echo
  echo -n "microcode: "; grep -m1 microcode /proc/cpuinfo | awk '{print $3}'
  echo -n "gather_data_sampling: "; cat /sys/devices/system/cpu/vulnerabilities/gather_data_sampling 2>/dev/null || echo n/a
} | tee "$OUT/machine.txt"
grep -q -w avx512f /proc/cpuinfo || log "WARNING: no avx512f on this CPU; SIMD tests will be skipped and the ON builds fall back to scalar"

log "run_tpch -t $THREADS; test_all threads=$TEST_THREADS; timeout ${TIMEOUT}s per step"
[ -n "${EXTRA_CMAKE:-}" ] && log "EXTRA_CMAKE (all configs): $EXTRA_CMAKE"
echo "compiler,config,simdsel,query,threads,median_ms" > "$OUT/summary.csv"

for comp in $COMPILERS; do
  for cfg in $CONFIGS; do
    tag="${comp}_${cfg}"
    B="$ROOT/build_ablation/$tag"
    D="$OUT/$tag"
    mkdir -p "$D"
    log "===== $tag ====="

    # ------------------------------------------------------------- 1. build
    if [ "$SKIP_BUILD" != 1 ]; then
      extra=""
      [ -n "${DATADIR:-}" ] && extra="-DDATADIR=$DATADIR"
      # shellcheck disable=SC2046
      if ! cmake -S "$ROOT" -B "$B" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DCOMPILER="$comp" \
            -DTARGET_MACHINE="$MACHINE" $(config_flags "$cfg") ${EXTRA_CMAKE:-} $extra > "$D/cmake.log" 2>&1; then
        fail "$tag cmake (see $D/cmake.log)"; continue
      fi
      # Build each target on its own so one broken target (e.g. test_all)
      # does not keep the benchmarks from running.
      : > "$D/build.log"
      for tgt in run_selbench run_hashbench run_tpch test_all; do
        echo "### target $tgt" >> "$D/build.log"
        if ! cmake --build "$B" -j "$JOBS" --target "$tgt" >> "$D/build.log" 2>&1; then
          fail "$tag build of $tgt (see $D/build.log)"
        fi
      done
      grep -E "VW_SIMD_(SEL|HASH).*(not available|needs AVX)" "$D/build.log" && fail "$tag: ISA fallback pragma fired (ARCH_FLAGS lacks AVX-512?)"
    fi

    # ------------------------------------------------------- 2. symbol check
    # SIMD kernels / prefetches must be present exactly when their option is ON.
    check_count() { # <what> <count> <yes|no>
      if [ "$3" = yes ]; then [ "$2" -gt 0 ] || fail "$tag: expected $1 (count $2)"
      else [ "$2" -eq 0 ] || fail "$tag: unexpected $1 (count $2)"; fi
    }
    yn() { if on "$cfg" "$1"; then echo yes; else echo no; fi; }
    sel_o=$(find "$B" -name 'Selection.cpp.o' | head -1)
    hash_o=$(find "$B" -path '*primitives*' -name 'Hash.cpp.o' | head -1)
    ops_o=$(find "$B" -path '*vectorwise.dir*' -name 'Operators.cpp.o' | head -1)
    if [ -z "$sel_o" ] || [ -z "$hash_o" ] || [ -z "$ops_o" ]; then
      fail "$tag: object files not found"
    else
      nsel=$(nm -C "$sel_o" | grep -c 'simd::sel_col_val<' || true)
      nchar=$(nm -C "$sel_o" | grep -c 'simd::sel_char_eq_col_val<' || true)
      nhash=$(nm -C "$hash_o" | grep -c 'simd_hash::' || true)
      npf=$(objdump -d "$ops_o" | grep -c 'prefetcht0' || true)
      echo "sel_col_val: $nsel  sel_char_eq: $nchar  simd_hash: $nhash  prefetcht0: $npf" | tee "$D/symbols.txt"
      check_count "SIMD selection kernels" "$nsel" "$(yn VW_SIMD_SEL)"
      check_count "Char kernels" "$nchar" "$(yn VW_SIMD_SEL_CHAR)"
      if on "$cfg" VW_USE_CRC32; then check_count "SIMD hash kernels" "$nhash" no
      else check_count "SIMD hash kernels" "$nhash" "$(yn VW_SIMD_HASH)"; fi
      check_count "join prefetches" "$npf" "$(yn VW_JOIN_PREFETCH)"
    fi

    # ---------------------------------------------------------- 3. unit tests
    if [ ! -x "$B/test_all" ]; then
      fail "$tag: test_all not built, skipping unit + TPC-H tests"
    elif ! threads=$TEST_THREADS tlimit "$B/test_all" --gtest_filter='SimdSel*:SimdSelRedirect*:SimdHash*:GTEQ*' > "$D/unit.log" 2>&1; then
      fail "$tag unit tests (see $D/unit.log)"
    else
      log "$tag unit tests: $(grep -E '^\[  PASSED  \]|SKIPPED' "$D/unit.log" | tr '\n' ' ')"
    fi

    # ------------------------------------------------- 4. TPC-H correctness
    if [ -x "$B/test_all" ] && [ -n "${DATADIR:-}" ] && [ -d "$DATADIR/tpch/sf1" ]; then
      for s in 0 1; do
        if ! threads=$TEST_THREADS SIMDsel=$s tlimit "$B/test_all" --gtest_filter='TPCH.*' > "$D/tpch_test_simdsel$s.log" 2>&1; then
          fail "$tag TPC-H tests SIMDsel=$s (see $D/tpch_test_simdsel$s.log)"
        else
          log "$tag TPC-H tests SIMDsel=$s passed"
        fi
      done
    else
      log "$tag: skipping TPC-H correctness tests (set DATADIR with tpch/sf1/)"
    fi

    # ------------------------------------------------------ 5. microbenchmark
    if { [ "$cfg" = base ] || [ "$cfg" = sel_pos16 ]; } && [ -x "$B/run_selbench" ]; then
      log "$tag run_selbench"
      tlimit taskset -c "$PIN_CPU" "$B/run_selbench" -v "$VEC" > "$D/selbench.csv" 2> "$D/selbench.err" \
        || fail "$tag run_selbench"
    fi

    if [ "$cfg" = base ] && [ -x "$B/run_hashbench" ]; then
      log "$tag run_hashbench"
      tlimit taskset -c "$PIN_CPU" "$B/run_hashbench" -v "$VEC" > "$D/hashbench.csv" 2> "$D/hashbench.err" \
        || fail "$tag run_hashbench"
    fi

    # --------------------------------------------------------- 6. end to end
    if [ -n "${TPCH_PATH:-}" ] && [ -x "$B/run_tpch" ]; then
      for s in 0 1; do
        log "$tag run_tpch SIMDsel=$s"
        if SIMDsel=$s tlimit "$B/run_tpch" -p "$TPCH_PATH" -e v -q 1,3,5,6,9,18 -r "$REPS" \
              -t "$THREADS" -v "$VEC" -s "$SETTLE" > "$D/tpch_simdsel$s.csv" 2> "$D/tpch_simdsel$s.err"; then
          # timeAndProfile rows: "<setw(20) label>,<padded median>,..." where the
          # label is "q3 v  t8"; the thread count is parsed from the label.
          awk -F, -v c="$comp" -v g="$cfg" -v s="$s" '
            { lbl = $1; gsub(/^ +| +$/, "", lbl) }
            lbl ~ /^q[0-9]+ v/ { med = $2; gsub(/ /, "", med);
              split(lbl, p, " "); t = p[3]; sub(/^t/, "", t);
              printf "%s,%s,%s,%s,%s,%s\n", c, g, s, p[1], t, med }' \
            "$D/tpch_simdsel$s.csv" >> "$OUT/summary.csv"
        else
          fail "$tag run_tpch SIMDsel=$s (see $D/tpch_simdsel$s.err)"
        fi
      done
    else
      log "$tag: skipping run_tpch (set TPCH_PATH)"
    fi
  done
done

# ------------------------------------------------------------ speedup table
# speedup = base/SIMDsel=0 median divided by each (config, SIMDsel) median.
awk -F, 'NR > 1 { key = $1 "," $4 "," $5; m[$1 "," $2 "," $3 "," $4 "," $5] = $6;
                  if ($2 == "base" && $3 == 0) ref[key] = $6; rows[NR] = $0 }
         END { print "compiler,config,simdsel,query,threads,median_ms,speedup_vs_base";
               for (i in rows) { split(rows[i], f, ","); key = f[1] "," f[4] "," f[5];
                 sp = (key in ref && f[6] > 0) ? ref[key] / f[6] : "";
                 print rows[i] "," sp } }' "$OUT/summary.csv" | sort -t, -k1,1 -k4,4 -k5,5n -k2,2 -k3,3 > "$OUT/speedup.csv"

log "results: $OUT  (summary.csv, speedup.csv, */selbench.csv)"
if [ "$FAILS" -gt 0 ]; then log "$FAILS FAILURE(S)"; exit 1; fi
log "ALL CHECKS PASSED"
