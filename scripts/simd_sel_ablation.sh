#!/usr/bin/env bash
# Ablation driver for the AVX-512 selection kernels (VW_SIMD_SEL, VW_SIMD_SEL_CHAR).
#
# For every compiler x config it:
#   1. configures + builds test_all, run_selbench, run_tpch into build_ablation/<compiler>_<config>
#   2. checks via nm that the SIMD kernels are linked in exactly when the option is ON
#   3. runs the differential unit tests (SimdSel*, SimdSelRedirect*, GTEQ*)
#   4. runs the TPC-H correctness tests (TPCH.*) with SIMDsel=0 and SIMDsel=1   [needs DATADIR sf1]
#   5. runs run_selbench (per-tuple kernel cost, CSV)                            [base + sel_pos16 builds]
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
OUT="$ROOT/results/simd_sel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
FAILS=0

log() { echo "[$(date +%T)] $*" | tee -a "$OUT/driver.log"; }
fail() { log "FAIL: $*"; FAILS=$((FAILS + 1)); }

config_flags() {
  case "$1" in
    # every option is set explicitly: CMake caches them across reconfigures
    base)         echo "-DVW_SIMD_SEL=OFF -DVW_SIMD_SEL_GATHER=scalar     -DVW_SIMD_SEL_CHAR=OFF -DVW_POS_16=OFF" ;;
    sel)          echo "-DVW_SIMD_SEL=ON  -DVW_SIMD_SEL_GATHER=scalar     -DVW_SIMD_SEL_CHAR=OFF -DVW_POS_16=OFF" ;;
    sel_scalarload) echo "-DVW_SIMD_SEL=ON  -DVW_SIMD_SEL_GATHER=scalarload -DVW_SIMD_SEL_CHAR=OFF -DVW_POS_16=OFF" ;;
    sel_hwgather) echo "-DVW_SIMD_SEL=ON  -DVW_SIMD_SEL_GATHER=hwgather   -DVW_SIMD_SEL_CHAR=OFF -DVW_POS_16=OFF" ;;
    sel_char)     echo "-DVW_SIMD_SEL=ON  -DVW_SIMD_SEL_GATHER=scalar     -DVW_SIMD_SEL_CHAR=ON  -DVW_POS_16=OFF" ;;
    sel_pos16)    echo "-DVW_SIMD_SEL=ON  -DVW_SIMD_SEL_GATHER=scalar     -DVW_SIMD_SEL_CHAR=ON  -DVW_POS_16=ON" ;;
    char_only)    echo "-DVW_SIMD_SEL=OFF -DVW_SIMD_SEL_GATHER=scalar     -DVW_SIMD_SEL_CHAR=ON  -DVW_POS_16=OFF" ;;
    *) echo "unknown config $1" >&2; exit 2 ;;
  esac
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
            -DTARGET_MACHINE="$MACHINE" $(config_flags "$cfg") $extra > "$D/cmake.log" 2>&1; then
        fail "$tag cmake (see $D/cmake.log)"; continue
      fi
      # Build each target on its own so one broken target (e.g. test_all)
      # does not keep the benchmarks from running.
      : > "$D/build.log"
      for tgt in run_selbench run_tpch test_all; do
        echo "### target $tgt" >> "$D/build.log"
        if ! cmake --build "$B" -j "$JOBS" --target "$tgt" >> "$D/build.log" 2>&1; then
          fail "$tag build of $tgt (see $D/build.log)"
        fi
      done
      grep -E "VW_SIMD_SEL.*not available" "$D/build.log" && fail "$tag: ISA fallback pragma fired (ARCH_FLAGS lacks AVX-512?)"
    fi

    # ------------------------------------------------------- 2. symbol check
    obj=$(find "$B" -name 'Selection.cpp.o' | head -1)
    if [ -z "$obj" ]; then fail "$tag: Selection.cpp.o not found"; else
      nsel=$(nm -C "$obj" | grep -c 'simd::sel_col_val<' || true)
      nchar=$(nm -C "$obj" | grep -c 'simd::sel_char_eq_col_val<' || true)
      echo "simd::sel_col_val symbols: $nsel, simd::sel_char_eq_col_val symbols: $nchar" | tee "$D/symbols.txt"
      case "$cfg" in
        base)      [ "$nsel" -eq 0 ] && [ "$nchar" -eq 0 ] || fail "$tag: SIMD symbols present in base build" ;;
        sel|sel_scalarload|sel_hwgather) [ "$nsel" -gt 0 ] && [ "$nchar" -eq 0 ] || fail "$tag: expected sel kernels only" ;;
        char_only) [ "$nsel" -eq 0 ] && [ "$nchar" -gt 0 ] || fail "$tag: expected char kernels only" ;;
        *)         [ "$nsel" -gt 0 ] && [ "$nchar" -gt 0 ] || fail "$tag: expected sel + char kernels" ;;
      esac
    fi

    # ---------------------------------------------------------- 3. unit tests
    if [ ! -x "$B/test_all" ]; then
      fail "$tag: test_all not built, skipping unit + TPC-H tests"
    elif ! "$B/test_all" --gtest_filter='SimdSel*:SimdSelRedirect*:GTEQ*' > "$D/unit.log" 2>&1; then
      fail "$tag unit tests (see $D/unit.log)"
    else
      log "$tag unit tests: $(grep -E '^\[  PASSED  \]|SKIPPED' "$D/unit.log" | tr '\n' ' ')"
    fi

    # ------------------------------------------------- 4. TPC-H correctness
    if [ -x "$B/test_all" ] && [ -n "${DATADIR:-}" ] && [ -d "$DATADIR/tpch/sf1" ]; then
      for s in 0 1; do
        if ! SIMDsel=$s "$B/test_all" --gtest_filter='TPCH.*' > "$D/tpch_test_simdsel$s.log" 2>&1; then
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
      taskset -c "$PIN_CPU" "$B/run_selbench" -v "$VEC" > "$D/selbench.csv" 2> "$D/selbench.err" \
        || fail "$tag run_selbench"
    fi

    # --------------------------------------------------------- 6. end to end
    if [ -n "${TPCH_PATH:-}" ] && [ -x "$B/run_tpch" ]; then
      for s in 0 1; do
        log "$tag run_tpch SIMDsel=$s"
        if SIMDsel=$s "$B/run_tpch" -p "$TPCH_PATH" -e v -q 1,3,5,6,9,18 -r "$REPS" \
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
